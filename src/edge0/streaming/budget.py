"""Memory-budgeted streaming profile (pure calculation).

Task 4 of ``docs/plans/jetson-orin-nano.md``: turn an OBSERVED amount of
memory into explicit byte allowances for everything the streaming stack
retains or transiently allocates, and refuse impossible profiles before
inference instead of OOMing during it.

Everything here is pure: the caller supplies observations (available
RAM, process limits), checkpoint-derived payload sizes and the declared
workload; this module only does arithmetic and policy.  No psutil, no
torch, no engine imports — see ``engine/ling.py`` for the integration
that gathers the inputs.

Accounting rules (Gate C):

* Start from observed available RAM, tightened by any process limit.
* Deduct, each exactly once: an OS/application growth reserve; KV cache
  at the DECLARED maximum context; the prefill transient (whole-layer
  load-drop materializes every expert of one layer at once when
  ``full_layer_prefill`` is on); in-flight expert builds (bounded by
  the resolved ``max_inflight``, not the thread count alone); pinned
  staging (Task 5 — zero today); the batched expert gather's per-call
  transient cap (Task 6 — zero unless ``EDGE0_QMM_BATCHED=1``); and an
  allocator/driver allowance (fraction of the observed total, applied
  once).
* What remains is the retained-cache allowance, split between the
  shared LRU and the prefetch buffer in payload bytes computed from
  checkpoint metadata — slot counts alone are meaningless.
* ``SharedExpertCache(0)`` and ``PrefetchBuffer(0)`` DISABLE eviction
  (unbounded).  A calculated zero is therefore never passed through:
  any allowance that cannot fit the minimum working set raises
  ``BudgetError`` with the itemized deductions.
* The budget only ever LOWERS the profile's requested capacities.  A
  budget roomier than the tested profile keeps the tested values.
* Full dequantized-weight caching (``EDGE0_TORCH_WEIGHT_CACHE``) is
  reported as permitted only when its measured payload fits in the
  allowance that remains AFTER the caches — it is never turned on by
  this module, only vetoed.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetError(ValueError):
    """The declared workload cannot fit the observed memory."""


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BudgetError(f"{name} must be an int, got {value!r}")
    if value <= 0:
        raise BudgetError(f"{name} must be positive, got {value}")
    return value


def _non_negative_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BudgetError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise BudgetError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class Observation:
    """What was actually observed on the target, at the observation
    point (immediately before engine construction — allocations already
    resident at that point are outside ``available_bytes`` by
    definition and must not be deducted again)."""

    available_bytes: int
    #: A tighter limit than system RAM when one exists (cgroup,
    #: rlimit); None means no separate limit was observed.
    process_limit_bytes: int | None = None

    def usable(self) -> int:
        avail = _positive_int("available_bytes", self.available_bytes)
        if self.process_limit_bytes is None:
            return avail
        return min(avail,
                   _positive_int("process_limit_bytes",
                                 self.process_limit_bytes))


@dataclass(frozen=True)
class ExpertFootprint:
    """Per-expert payload from checkpoint metadata (see
    ``bundle_bytes_from_entries``)."""

    bundle_bytes: int
    num_experts: int
    num_moe_layers: int


@dataclass(frozen=True)
class WorkloadDecl:
    """The declared operating envelope.  KV for ``max_context_tokens``
    is deducted up front: a context the budget did not declare is a
    different workload and must be re-resolved, not discovered OOM."""

    max_context_tokens: int
    kv_bytes_per_token: int


@dataclass(frozen=True)
class Reserves:
    os_growth_bytes: int = 512 * 1024 * 1024
    #: Allocator + driver allowance as a fraction of the usable
    #: observation, deducted once.
    allocator_fraction: float = 0.10
    #: Task 5's pinned staging pool; zero until it exists.
    pinned_staging_bytes: int = 0
    #: Task 6's batched expert gather: the per-call transient cap
    #: (``EDGE0_QMM_BATCHED_MAX_BYTES``) when that path is on, else zero.
    kernel_transient_bytes: int = 0


@dataclass(frozen=True)
class ResolvedBudget:
    """The resolved policy, with its arithmetic shown."""

    usable_bytes: int
    deductions: tuple  # ((name, bytes), ...) in application order
    cache_allowance_bytes: int
    cache_slots: int
    cache_bytes: int
    prefetch_cap: int
    prefetch_bytes: int
    #: Hard ceiling for ``PrefetchBuffer.set_cap`` growth
    #: (``prefetch_all`` raises the cap to num_experts + 32 otherwise).
    max_prefetch_cap: int
    max_inflight: int
    weight_cache_permitted: bool
    weight_cache_bytes: int | None
    headroom_bytes: int
    notes: tuple = ()

    def as_dict(self) -> dict:
        reasons = {}
        if self.weight_cache_bytes is None:
            reasons["weight_cache_bytes"] = (
                "no dequantized-weight cache was requested "
                "(EDGE0_TORCH_WEIGHT_CACHE unset)")
        return {
            "unavailable_reasons": reasons,
            "usable_bytes": self.usable_bytes,
            "deductions": [list(d) for d in self.deductions],
            "cache_allowance_bytes": self.cache_allowance_bytes,
            "cache_slots": self.cache_slots,
            "cache_bytes": self.cache_bytes,
            "prefetch_cap": self.prefetch_cap,
            "prefetch_bytes": self.prefetch_bytes,
            "max_prefetch_cap": self.max_prefetch_cap,
            "max_inflight": self.max_inflight,
            "weight_cache_permitted": self.weight_cache_permitted,
            "weight_cache_bytes": self.weight_cache_bytes,
            "headroom_bytes": self.headroom_bytes,
            "notes": list(self.notes),
        }


def dense_cache_request(*, full_requested: bool, cap_bytes,
                        candidate_total: int,
                        max_fill_transient_bytes: int = 0) -> dict:
    """What the budget prices for the dense dequantized-weight cache
    (Task 6).

    The price is the RESIDENT bytes the cache keeps plus the largest
    single build transient: the weights are filled lazily at the first
    forward, one chunked module at a time, so exactly one transient is
    live but it must still fit alongside everything resident.

    ``full`` (``EDGE0_TORCH_WEIGHT_CACHE=1``) prices the loader's measured
    total of every ``QuantizedLinear`` (the GB10's 4.1 GB figure only when
    no model is loaded yet) and keeps its veto; ``capped``
    (``EDGE0_TORCH_WEIGHT_CACHE_BYTES``) prices ``min(cap, total)`` and is
    later shrunk to the headroom instead of vetoed; ``off`` prices nothing.
    """
    fill = _non_negative_int("max_fill_transient_bytes",
                             max_fill_transient_bytes)
    if full_requested:
        resident = candidate_total or 4_100_000_000
        return {"mode": "full", "priced_bytes": resident + fill,
                "resident_bytes": resident, "fill_transient_bytes": fill,
                "cap_bytes": None}
    if cap_bytes:
        resident = min(int(cap_bytes), candidate_total)
        return {"mode": "capped", "priced_bytes": resident + fill,
                "resident_bytes": resident, "fill_transient_bytes": fill,
                "cap_bytes": int(cap_bytes)}
    return {"mode": "off", "priced_bytes": 0, "resident_bytes": 0,
            "fill_transient_bytes": 0, "cap_bytes": None}


def effective_capped_bytes(*, cap_bytes: int, headroom_bytes: int) -> int:
    """The capped cache never exceeds the budget's post-cache headroom."""
    return max(0, min(int(cap_bytes), int(headroom_bytes)))


def bundle_bytes_from_entries(entries: dict, key_prefix: str,
                              num_experts: int,
                              projections=("gate_proj", "up_proj",
                                           "down_proj"),
                              parts=("weight", "scales", "biases")) -> int:
    """Per-expert payload bytes for one layer, from a safetensors header.

    ``entries`` maps tensor name to ``{"size": bytes, ...}`` (the
    ``SafetensorsMmap.entries`` shape) or directly to an int size.
    Expert tensors are stacked ``[num_experts, ...]``; the per-expert
    payload is the summed tensor size divided by the expert count.
    """
    _positive_int("num_experts", num_experts)
    total = 0
    for proj in projections:
        for part in parts:
            name = f"{key_prefix}.{proj}.{part}"
            if name not in entries:
                raise BudgetError(
                    f"checkpoint metadata is missing {name!r}; cannot "
                    "compute the expert payload from slot counts alone")
            meta = entries[name]
            size = meta["size"] if isinstance(meta, dict) else meta
            total += _positive_int(f"size of {name}", size)
    if total % num_experts:
        # Not fatal — round up so the budget never undercounts.
        return total // num_experts + 1
    return total // num_experts


def resolve_budget(
    observation: Observation,
    experts: ExpertFootprint,
    workload: WorkloadDecl,
    *,
    requested_cache_slots: int,
    requested_prefetch_cap: int,
    full_layer_prefill: bool,
    inflight_builds: int,
    min_cache_slots: int | None = None,
    weight_cache_bytes: int | None = None,
    reserves: Reserves = Reserves(),
) -> ResolvedBudget:
    """Resolve the streaming memory policy for one engine build.

    Raises ``BudgetError`` when the declared workload cannot fit — with
    the itemized deductions, so the failure is a review artifact rather
    than a mystery OOM.  ``min_cache_slots`` defaults to the larger of
    2 and the per-step working set the caller declares; the shared LRU
    is never resolved below it and never to zero.
    """
    bundle = _positive_int("bundle_bytes", experts.bundle_bytes)
    n_experts = _positive_int("num_experts", experts.num_experts)
    _positive_int("num_moe_layers", experts.num_moe_layers)
    slots_req = _positive_int("requested_cache_slots", requested_cache_slots)
    cap_req = _positive_int("requested_prefetch_cap", requested_prefetch_cap)
    inflight = _positive_int("inflight_builds", inflight_builds)
    min_slots = (2 if min_cache_slots is None
                 else _positive_int("min_cache_slots", min_cache_slots))
    if weight_cache_bytes is not None:
        _non_negative_int("weight_cache_bytes", weight_cache_bytes)
    if not 0.0 <= reserves.allocator_fraction < 1.0:
        raise BudgetError(
            f"allocator_fraction must be in [0, 1), got "
            f"{reserves.allocator_fraction!r}")

    usable = observation.usable()
    kv_bytes = (_non_negative_int("max_context_tokens",
                                  workload.max_context_tokens)
                * _positive_int("kv_bytes_per_token",
                                workload.kv_bytes_per_token))
    prefill_transient = bundle * n_experts if full_layer_prefill else 0
    inflight_bytes = bundle * inflight
    deductions = (
        ("os_growth_reserve", _non_negative_int(
            "os_growth_bytes", reserves.os_growth_bytes)),
        ("allocator_allowance", int(usable * reserves.allocator_fraction)),
        ("kv_cache_at_declared_context", kv_bytes),
        ("full_layer_prefill_transient", prefill_transient),
        ("inflight_expert_builds", inflight_bytes),
        ("pinned_staging", _non_negative_int(
            "pinned_staging_bytes", reserves.pinned_staging_bytes)),
        ("kernel_transient", _non_negative_int(
            "kernel_transient_bytes", reserves.kernel_transient_bytes)),
    )
    allowance = usable - sum(d[1] for d in deductions)
    if allowance <= 0:
        detail = ", ".join(f"{n}={b:,}" for n, b in deductions)
        raise BudgetError(
            f"declared workload does not fit: usable={usable:,} bytes, "
            f"deductions leave {allowance:,} ({detail})")

    # Shared LRU first (correctness path), prefetch from the remainder.
    cache_slots = min(slots_req, allowance // bundle)
    if cache_slots < min_slots:
        raise BudgetError(
            f"cache allowance {allowance:,} bytes fits only "
            f"{cache_slots} expert bundles of {bundle:,} bytes — below "
            f"the minimum working set of {min_slots}; a zero/starved "
            f"cache is rejected, not constructed (SharedExpertCache(0) "
            f"would disable eviction entirely)")
    cache_bytes = cache_slots * bundle
    prefetch_cap = min(cap_req, (allowance - cache_bytes) // bundle)
    notes = []
    if prefetch_cap < 1:
        # An explicit bounded floor, never PrefetchBuffer(0): one slot
        # keeps the buffer's eviction semantics intact and costs one
        # bundle, which min_cache_slots' margin covers.
        prefetch_cap = 1
        notes.append("prefetch_cap floored to 1 (0 disables eviction)")
    prefetch_bytes = prefetch_cap * bundle
    if cache_slots < slots_req:
        notes.append(
            f"cache_slots lowered {slots_req} -> {cache_slots} by budget")
    if prefetch_cap < cap_req:
        notes.append(
            f"prefetch_cap lowered {cap_req} -> {prefetch_cap} by budget")

    headroom = allowance - cache_bytes - prefetch_bytes
    permitted = (weight_cache_bytes is not None
                 and weight_cache_bytes <= headroom)
    return ResolvedBudget(
        usable_bytes=usable,
        deductions=deductions,
        cache_allowance_bytes=allowance,
        cache_slots=cache_slots,
        cache_bytes=cache_bytes,
        prefetch_cap=prefetch_cap,
        prefetch_bytes=prefetch_bytes,
        # prefetch_all may not grow the buffer past what the budget
        # priced in; without a budget it grows to num_experts + 32.
        max_prefetch_cap=prefetch_cap,
        max_inflight=inflight,
        weight_cache_permitted=permitted,
        weight_cache_bytes=weight_cache_bytes,
        headroom_bytes=headroom,
        notes=tuple(notes),
    )
