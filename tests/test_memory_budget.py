"""Task 4: memory-budgeted streaming profile (pure calculation + the
cache-side enforcement hooks).  No backend, no weights, no psutil."""

from __future__ import annotations

import pytest

from edge0.streaming import budget as B
from edge0.streaming.cache import PrefetchBuffer

MB = 1024 * 1024
GB = 1024 * MB

#: A comfortable Orin-like scenario: ~3.5 GB observed, 2 MB bundles.
OBS = B.Observation(available_bytes=3500 * MB)
EXPERTS = B.ExpertFootprint(bundle_bytes=2 * MB, num_experts=128,
                            num_moe_layers=23)
WORKLOAD = B.WorkloadDecl(max_context_tokens=1024,
                          kv_bytes_per_token=1_100_000)


def resolve(obs=OBS, experts=EXPERTS, workload=WORKLOAD, **kw):
    kw.setdefault("requested_cache_slots", 64)
    kw.setdefault("requested_prefetch_cap", 48)
    kw.setdefault("full_layer_prefill", True)
    kw.setdefault("inflight_builds", 4)
    kw.setdefault("min_cache_slots", 16)
    return B.resolve_budget(obs, experts, workload, **kw)


# ---------------------------------------------------------------- happy path

def test_roomy_budget_keeps_requested_profile():
    r = resolve()
    assert r.cache_slots == 64
    assert r.prefetch_cap == 48
    assert r.max_prefetch_cap == 48       # prefetch_all may not grow past it
    assert r.max_inflight == 4
    assert r.notes == ()
    # every deduction appears exactly once and the arithmetic closes
    assert r.cache_allowance_bytes == r.usable_bytes - sum(
        b for _, b in r.deductions)
    assert r.headroom_bytes == (r.cache_allowance_bytes
                                - r.cache_bytes - r.prefetch_bytes)
    names = [n for n, _ in r.deductions]
    assert names == ["os_growth_reserve", "allocator_allowance",
                     "kv_cache_at_declared_context",
                     "full_layer_prefill_transient",
                     "inflight_expert_builds", "pinned_staging"]


def test_deductions_use_declared_values():
    r = resolve()
    d = dict(r.deductions)
    assert d["kv_cache_at_declared_context"] == 1024 * 1_100_000
    assert d["full_layer_prefill_transient"] == 128 * 2 * MB
    assert d["inflight_expert_builds"] == 4 * 2 * MB
    r2 = resolve(full_layer_prefill=False)
    assert dict(r2.deductions)["full_layer_prefill_transient"] == 0


def test_overflow_scale_budget_never_raises_requested():
    r = resolve(obs=B.Observation(available_bytes=10**18))
    assert r.cache_slots == 64            # requested is the ceiling
    assert r.prefetch_cap == 48


def test_process_limit_tightens_observation():
    obs = B.Observation(available_bytes=3500 * MB,
                        process_limit_bytes=2500 * MB)
    assert obs.usable() == 2500 * MB
    assert resolve(obs=obs).usable_bytes == 2500 * MB


# ------------------------------------------------------------ tight budgets

def test_tight_budget_lowers_and_records():
    r = resolve(obs=B.Observation(available_bytes=2150 * MB))
    assert 16 <= r.cache_slots < 64
    assert any("cache_slots lowered" in n for n in r.notes)
    assert r.cache_bytes == r.cache_slots * EXPERTS.bundle_bytes


def test_prefetch_floor_is_one_never_zero():
    # Just enough for the LRU minimum, nothing left for prefetch:
    # PrefetchBuffer(0) disables eviction, so the floor is 1, recorded.
    r = resolve(obs=B.Observation(available_bytes=2100 * MB),
                min_cache_slots=16)
    assert r.prefetch_cap >= 1
    if r.prefetch_cap == 1:
        assert any("floored" in n for n in r.notes)


def test_exhausted_budget_rejected_with_itemization():
    with pytest.raises(B.BudgetError) as exc:
        resolve(obs=B.Observation(available_bytes=600 * MB))
    msg = str(exc.value)
    assert "does not fit" in msg and "kv_cache_at_declared_context" in msg


def test_starved_cache_rejected_not_constructed_as_zero():
    # Allowance fits a few bundles but fewer than the working set.
    with pytest.raises(B.BudgetError) as exc:
        resolve(obs=B.Observation(available_bytes=2080 * MB),
                min_cache_slots=32)
    assert "SharedExpertCache(0)" in str(exc.value)


def test_declared_context_is_priced_not_ignored():
    ok = resolve()
    with pytest.raises(B.BudgetError):
        resolve(workload=B.WorkloadDecl(max_context_tokens=4096,
                                        kv_bytes_per_token=1_100_000))
    assert ok.cache_slots == 64


# ------------------------------------------------------- invalid inputs

@pytest.mark.parametrize("field,value", [
    ("requested_cache_slots", 0), ("requested_cache_slots", -1),
    ("requested_prefetch_cap", 0), ("inflight_builds", 0),
    ("min_cache_slots", 0),
])
def test_invalid_requests_rejected(field, value):
    with pytest.raises(B.BudgetError):
        resolve(**{field: value})


def test_invalid_observation_rejected():
    with pytest.raises(B.BudgetError):
        resolve(obs=B.Observation(available_bytes=0))
    with pytest.raises(B.BudgetError):
        resolve(obs=B.Observation(available_bytes=None))  # missing
    with pytest.raises(B.BudgetError):
        resolve(obs=B.Observation(available_bytes=True))  # bool is not int


def test_invalid_reserves_rejected():
    with pytest.raises(B.BudgetError):
        resolve(reserves=B.Reserves(allocator_fraction=1.0))
    with pytest.raises(B.BudgetError):
        resolve(reserves=B.Reserves(os_growth_bytes=-1))


def test_invalid_workload_rejected():
    with pytest.raises(B.BudgetError):
        resolve(workload=B.WorkloadDecl(max_context_tokens=-1,
                                        kv_bytes_per_token=1_100_000))
    with pytest.raises(B.BudgetError):
        resolve(workload=B.WorkloadDecl(max_context_tokens=1024,
                                        kv_bytes_per_token=0))


# ------------------------------------------------- payload from metadata

def _entries(per_tensor: int):
    names = [f"model.layers.1.mlp.experts.{p}.{part}"
             for p in ("gate_proj", "up_proj", "down_proj")
             for part in ("weight", "scales", "biases")]
    return {n: {"size": per_tensor} for n in names}


def test_bundle_bytes_from_entries():
    got = B.bundle_bytes_from_entries(
        _entries(128 * 1000), "model.layers.1.mlp.experts", 128)
    assert got == 9 * 1000


def test_bundle_bytes_rounds_up_on_remainder():
    got = B.bundle_bytes_from_entries(
        _entries(1001), "model.layers.1.mlp.experts", 128)
    assert got == (9 * 1001) // 128 + 1


def test_bundle_bytes_missing_tensor_rejected():
    entries = _entries(1000)
    entries.pop("model.layers.1.mlp.experts.down_proj.biases")
    with pytest.raises(B.BudgetError) as exc:
        B.bundle_bytes_from_entries(
            entries, "model.layers.1.mlp.experts", 128)
    assert "down_proj.biases" in str(exc.value)


def test_bundle_bytes_varying_sizes_change_slots():
    small = resolve(experts=B.ExpertFootprint(
        bundle_bytes=1 * MB, num_experts=128, num_moe_layers=23))
    big = resolve(experts=B.ExpertFootprint(
        bundle_bytes=16 * MB, num_experts=128, num_moe_layers=23),
        obs=B.Observation(available_bytes=4500 * MB))
    assert small.cache_slots == 64
    assert big.cache_slots < 64


# ------------------------------------------------------- weight cache veto

def test_weight_cache_permitted_only_within_headroom():
    denied = resolve(weight_cache_bytes=4_100_000_000)
    assert not denied.weight_cache_permitted
    granted = resolve(obs=B.Observation(available_bytes=10 * GB),
                      weight_cache_bytes=4_100_000_000)
    assert granted.weight_cache_permitted
    assert resolve().weight_cache_bytes is None


# ----------------------------------------------- PrefetchBuffer max_cap

def test_prefetch_buffer_max_cap_clamps_growth():
    buf = PrefetchBuffer(4, max_cap=6)
    assert buf.cap == 4
    buf.set_cap(160)                      # prefetch_all's n + 32 growth
    assert buf.cap == 6
    for i in range(10):
        buf.put(i, object())
    assert len(buf._buf) == 6


def test_prefetch_buffer_default_semantics_unchanged():
    buf = PrefetchBuffer(4)
    buf.set_cap(160)
    assert buf.cap == 160                 # historical unbounded growth
    zero = PrefetchBuffer(0)
    for i in range(100):
        zero.put(i, object())
    assert len(zero._buf) == 100          # 0 still means no eviction


def test_prefetch_buffer_constructor_clamped_by_max_cap():
    assert PrefetchBuffer(48, max_cap=6).cap == 6
