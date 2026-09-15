"""Task 6 pricing in the memory budget (pure, no backend, no weights).

* ``Reserves.kernel_transient_bytes`` deducts the batched gather's
  per-call transient cap once, like every other transient.
* ``budget.dense_cache_request`` turns the two dense-weight-cache knobs
  (``EDGE0_TORCH_WEIGHT_CACHE=1`` full, ``..._BYTES=<n>`` capped) and the
  loader's measured candidate total into what the budget prices and how
  the policy is finalized afterwards (``engine.ling`` calls it) -- the
  full cache keeps its veto, the capped one shrinks to the headroom.
"""

from __future__ import annotations

import pytest

from edge0.streaming import budget as b
from edge0.streaming.budget import (
    dense_cache_request as _dense_cache_request,
)
from edge0.streaming.budget import (
    effective_capped_bytes as _effective_capped_bytes,
)

OBS = b.Observation(available_bytes=8_000_000_000)
EXPERTS = b.ExpertFootprint(bundle_bytes=1_300_000, num_experts=128,
                            num_moe_layers=23)
WORKLOAD = b.WorkloadDecl(max_context_tokens=1024, kv_bytes_per_token=1_100_000)


def _resolve(**kw):
    kw.setdefault("requested_cache_slots", 64)
    kw.setdefault("requested_prefetch_cap", 48)
    kw.setdefault("full_layer_prefill", True)
    kw.setdefault("inflight_builds", 8)
    return b.resolve_budget(OBS, EXPERTS, WORKLOAD, **kw)


def test_kernel_transient_is_deducted_once_and_itemized():
    plain = _resolve()
    with_kernel = _resolve(reserves=b.Reserves(kernel_transient_bytes=256 << 20))
    names = dict(with_kernel.deductions)
    assert names["kernel_transient"] == 256 << 20
    assert dict(plain.deductions)["kernel_transient"] == 0
    assert (plain.cache_allowance_bytes - with_kernel.cache_allowance_bytes
            == 256 << 20)
    serialized = dict(map(tuple, with_kernel.as_dict()["deductions"]))
    assert serialized["kernel_transient"] == 256 << 20


def test_kernel_transient_must_be_a_non_negative_int():
    with pytest.raises(b.BudgetError, match="kernel_transient_bytes"):
        _resolve(reserves=b.Reserves(kernel_transient_bytes=-1))
    with pytest.raises(b.BudgetError, match="kernel_transient_bytes"):
        _resolve(reserves=b.Reserves(
            kernel_transient_bytes=1.5))  # type: ignore[arg-type]


# ---- dense cache request (engine helper) ----------------------------------------------


def test_dense_cache_request_full_uses_the_measured_total():
    req = _dense_cache_request(full_requested=True, cap_bytes=None,
                               candidate_total=1_400_000_000)
    assert req == {"mode": "full", "priced_bytes": 1_400_000_000,
                   "resident_bytes": 1_400_000_000,
                   "fill_transient_bytes": 0, "cap_bytes": None}


def test_dense_cache_request_prices_the_build_transient_on_top():
    """The weights are filled lazily at the first forward, one chunked
    module at a time: exactly one transient is live, and it must still fit
    beside everything resident."""
    req = _dense_cache_request(full_requested=True, cap_bytes=None,
                               candidate_total=1_400_000_000,
                               max_fill_transient_bytes=106_000_000)
    assert req["resident_bytes"] == 1_400_000_000
    assert req["fill_transient_bytes"] == 106_000_000
    assert req["priced_bytes"] == 1_506_000_000
    capped = _dense_cache_request(full_requested=False,
                                  cap_bytes=500_000_000,
                                  candidate_total=1_400_000_000,
                                  max_fill_transient_bytes=106_000_000)
    assert capped["priced_bytes"] == 606_000_000
    off = _dense_cache_request(full_requested=False, cap_bytes=None,
                               candidate_total=1_400_000_000,
                               max_fill_transient_bytes=106_000_000)
    assert off["priced_bytes"] == 0 and off["fill_transient_bytes"] == 0
    with pytest.raises(b.BudgetError, match="max_fill_transient_bytes"):
        _dense_cache_request(full_requested=True, cap_bytes=None,
                             candidate_total=1, max_fill_transient_bytes=-1)


def test_dense_cache_request_full_without_candidates_keeps_the_gb10_figure():
    req = _dense_cache_request(full_requested=True, cap_bytes=None,
                               candidate_total=0)
    assert req["mode"] == "full" and req["priced_bytes"] == 4_100_000_000


def test_dense_cache_request_capped_prices_min_of_cap_and_total():
    req = _dense_cache_request(full_requested=False, cap_bytes=1_000_000_000,
                               candidate_total=1_400_000_000)
    assert req == {"mode": "capped", "priced_bytes": 1_000_000_000,
                   "resident_bytes": 1_000_000_000,
                   "fill_transient_bytes": 0, "cap_bytes": 1_000_000_000}
    req = _dense_cache_request(full_requested=False, cap_bytes=2_000_000_000,
                               candidate_total=1_400_000_000)
    assert req["priced_bytes"] == 1_400_000_000


def test_dense_cache_request_off():
    assert _dense_cache_request(full_requested=False, cap_bytes=None,
                                candidate_total=5)["mode"] == "off"
    assert _dense_cache_request(full_requested=False, cap_bytes=0,
                                candidate_total=5)["mode"] == "off"


def test_dense_cache_request_full_wins_over_a_cap():
    req = _dense_cache_request(full_requested=True, cap_bytes=10,
                               candidate_total=99)
    assert req["mode"] == "full" and req["priced_bytes"] == 99


def test_capped_cache_effective_cap_is_min_of_cap_and_headroom():
    assert _effective_capped_bytes(cap_bytes=1_000, headroom_bytes=5_000) == 1_000
    assert _effective_capped_bytes(cap_bytes=1_000, headroom_bytes=300) == 300
    assert _effective_capped_bytes(cap_bytes=1_000, headroom_bytes=-5) == 0
