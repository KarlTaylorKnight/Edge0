"""Capped exact dequantized-weight cache (Task 6, workstation side).

``EDGE0_TORCH_WEIGHT_CACHE=1`` keeps EVERY dense weight dequantized
(a measured 4.1 GB resident delta on the GB10; edge0-8b's dequantized
dense weights are 1.40 GB in bf16).  ``EDGE0_TORCH_WEIGHT_CACHE_BYTES=<n>``
keeps only the modules that fit a byte cap, admitted in registration
order with each module's own build transient counted, and the engine
re-finalizes the cap against the Task 4 budget's headroom.

The cached weight is built with the SAME per-chunk expression as the
on-the-fly path, so the dequantized weight is bit-identical (asserted on
CUDA below); the cached forward then issues one GEMM where the
on-the-fly path issues one per 4096-row chunk, which cuBLAS may split
differently for modules wider than a chunk -- outputs are identical here
and on the CPU, and the reference check is what decides on the device.

No MLX, no checkpoint; modules are built from hand-packed codes.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EDGE0_BACKEND", "cuda")
torch = pytest.importorskip("torch")

from edge0.backends.cuda import nn as cnn  # noqa: E402
from tests.test_quant_paths import _dequant64, _pack  # noqa: E402


def _linear(seed, out_f=32, in_f=128, group=64, dtype=torch.bfloat16):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (out_f, in_f), generator=gen)
    scales = (torch.randn(out_f, in_f // group, generator=gen) * 0.1).to(dtype)
    biases = (torch.randn(out_f, in_f // group, generator=gen) * 0.05).to(dtype)
    mod = cnn.QuantizedLinear(_pack(codes, 4), scales, biases, in_f)
    return mod, codes, scales, biases


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(cnn, "CACHE_DEQUANTIZED", False)
    p = cnn.WeightCachePolicy(cap_bytes=None)
    monkeypatch.setattr(cnn, "WEIGHT_CACHE", p)
    return p


# ---- env parsing --------------------------------------------------------------------


def test_cap_env_parsing(monkeypatch):
    monkeypatch.delenv("EDGE0_TORCH_WEIGHT_CACHE_BYTES", raising=False)
    assert cnn._read_weight_cache_cap() is None
    monkeypatch.setenv("EDGE0_TORCH_WEIGHT_CACHE_BYTES", "1000000")
    assert cnn._read_weight_cache_cap() == 1_000_000
    monkeypatch.setenv("EDGE0_TORCH_WEIGHT_CACHE_BYTES", "0")
    assert cnn._read_weight_cache_cap() == 0
    for bad in ("1GB", "-5", "1.5"):
        monkeypatch.setenv("EDGE0_TORCH_WEIGHT_CACHE_BYTES", bad)
        with pytest.raises(ValueError, match="EDGE0_TORCH_WEIGHT_CACHE_BYTES"):
            cnn._read_weight_cache_cap()


def test_dequantized_bytes_prices_out_in_itemsize():
    mod, *_ = _linear(0, out_f=32, in_f=128)
    assert cnn.dequantized_bytes(mod) == 32 * 128 * 2          # bf16 scales
    mod32, *_ = _linear(0, out_f=32, in_f=128, dtype=torch.float32)
    assert cnn.dequantized_bytes(mod32) == 32 * 128 * 4


# ---- admission -----------------------------------------------------------------------


def test_policy_admits_in_registration_order_until_the_cap(policy):
    mods = [_linear(i, out_f=32, in_f=128)[0] for i in range(4)]   # 8192 B each
    fill = cnn.fill_transient_bytes(mods[0])       # 4 * 32 * 128 * 4
    for i, m in enumerate(mods):
        policy.register(f"layers.{i}.q", m)
    summary = policy.finalize(cap_bytes=8192 * 2 + fill)
    assert [m.cache_weight for m in mods] == [True, True, False, False]
    assert summary["effective_cap_bytes"] == 8192 * 2 + fill
    assert summary["admitted_bytes"] == 8192 * 2
    assert summary["max_fill_transient_bytes"] == fill
    assert [a["path"] for a in summary["admitted"]] == ["layers.0.q", "layers.1.q"]
    assert all(a["fill_transient_bytes"] == fill for a in summary["admitted"])
    assert [s["path"] for s in summary["skipped"]] == ["layers.2.q", "layers.3.q"]
    assert summary["candidate_bytes_total"] == 8192 * 4
    assert summary["finalized"] is True


def test_admission_counts_each_modules_own_build_transient(policy):
    """Resident bytes alone would admit a module whose chunked fill then
    OOMs; the cap must cover the build too."""
    mod, *_ = _linear(11, out_f=32, in_f=128)
    resident = cnn.dequantized_bytes(mod)          # 8192
    fill = cnn.fill_transient_bytes(mod)           # 65536: 8x the resident
    assert fill > resident
    policy.register("m", mod)
    assert policy.finalize(cap_bytes=resident + fill - 1)["admitted"] == []
    assert mod.cache_weight is False
    assert policy.finalize(cap_bytes=resident + fill)["admitted_bytes"] == resident
    assert mod.cache_weight is True


def _priced_fill(rows, in_f, group=64, itemsize=2):
    groups = -(-in_f // group)
    return (4 * rows * in_f * 4 + 2 * rows * groups * (itemsize + 4)
            + rows * in_f * itemsize)


def test_fill_transient_is_bounded_by_the_chunk_not_the_module():
    chunk = cnn.QuantizedLinear.ROWS_PER_CHUNK
    narrow, *_ = _linear(12, out_f=100, in_f=128)
    wide, *_ = _linear(13, out_f=3 * chunk, in_f=128)
    wider, *_ = _linear(14, out_f=9 * chunk, in_f=128)
    assert cnn.fill_transient_bytes(narrow) == _priced_fill(100, 128)
    assert cnn.fill_transient_bytes(wide) == _priced_fill(chunk, 128)
    # three times the rows, same transient: the chunk bounds it
    assert cnn.fill_transient_bytes(wider) == cnn.fill_transient_bytes(wide)
    assert cnn.dequantized_bytes(wider) == 3 * cnn.dequantized_bytes(wide)


def test_policy_first_fit_skips_an_oversize_module_but_admits_later_ones(policy):
    big, *_ = _linear(1, out_f=256, in_f=128)      # 65536 B resident
    small, *_ = _linear(2, out_f=32, in_f=128)      # 8192 B resident
    policy.register("lm_head", big)
    policy.register("layers.0.q", small)
    cap = cnn.dequantized_bytes(small) + cnn.fill_transient_bytes(small)
    s = policy.finalize(cap_bytes=cap)
    assert big.cache_weight is False and small.cache_weight is True
    assert s["skipped"][0]["reason"].startswith("does not fit")
    assert "build transient" in s["skipped"][0]["reason"]


def test_policy_off_when_cap_is_none_or_zero(policy):
    mod, *_ = _linear(3)
    policy.register("m", mod)
    assert policy.finalize(cap_bytes=None)["enabled"] is False
    assert mod.cache_weight is False
    assert policy.finalize(cap_bytes=0)["enabled"] is False
    assert mod.cache_weight is False


def test_policy_from_env(monkeypatch):
    monkeypatch.setenv("EDGE0_TORCH_WEIGHT_CACHE_BYTES", "4096")
    p = cnn.WeightCachePolicy.from_env()
    assert p.cap_bytes == 4096 and p.enabled


def test_policy_begin_clears_candidates(policy):
    mod, *_ = _linear(4)
    policy.register("m", mod)
    policy.begin()
    assert policy.finalize(cap_bytes=1 << 30)["admitted"] == []


# ---- exactness --------------------------------------------------------------------------


def test_cached_output_is_bit_identical_to_on_the_fly(policy):
    mod, codes, scales, biases = _linear(5, out_f=5000, in_f=256)  # > one chunk
    gen = torch.Generator().manual_seed(15)
    x = torch.randn(3, 256, generator=gen).to(torch.bfloat16)
    with torch.no_grad():
        plain = mod(x)
        policy.register("m", mod)
        policy.finalize(cap_bytes=1 << 31)
        assert mod.cache_weight
        first = mod(x)
        second = mod(x)
    assert torch.equal(plain, first) and torch.equal(first, second)
    assert mod._dequantized is not None and mod._dequantized[0] == torch.bfloat16
    ref = x.double() @ _dequant64(codes, scales, biases, 64).T
    assert torch.allclose(first.double(), ref, rtol=2e-2, atol=1e-1)


def test_activation_itemsize_mismatch_falls_through_without_caching(policy):
    """An admitted module priced at bf16 must NOT cache a float32 weight:
    that would silently double the bytes the budget reserved.

    This is not hypothetical -- in the 8B engine 400 of 470 dense calls
    per step arrive as float32, so most admitted modules never fill.  The
    reservation is therefore conservative (it never under-reserves), and
    the bench report states filled bytes next to admitted bytes so the
    two are not confused.
    """
    mod, *_ = _linear(6)                    # priced at bf16 (2 bytes)
    policy.register("m", mod)
    policy.finalize(cap_bytes=1 << 20)
    assert mod.cache_weight                 # admitted...
    x = torch.randn(2, 128)                 # ...but called with float32
    with torch.no_grad():
        out = mod(x)
    assert mod._dequantized is None         # nothing cached at the wrong size
    assert out.dtype == torch.float32
    with torch.no_grad():                   # and the matching dtype does fill
        mod(x.to(torch.bfloat16))
    assert mod._dequantized is not None
    assert mod._dequantized[0] == torch.bfloat16


def test_refinalize_with_a_smaller_cap_drops_the_cached_weight(policy):
    a, *_ = _linear(7)
    b, *_ = _linear(8)
    policy.register("a", a)
    policy.register("b", b)
    policy.finalize(cap_bytes=1 << 20)
    x = torch.randn(2, 128).to(torch.bfloat16)
    with torch.no_grad():
        a(x)
        b(x)
    assert a._dequantized is not None and b._dequantized is not None
    fill = cnn.fill_transient_bytes(a)
    s = policy.finalize(cap_bytes=8192 + fill)   # only the first fits now
    assert a.cache_weight and not b.cache_weight
    assert b._dequantized is None           # released, not stale
    assert s["admitted_bytes"] == 8192


def test_full_cache_env_still_wins(policy, monkeypatch):
    monkeypatch.setattr(cnn, "CACHE_DEQUANTIZED", True)
    mod, *_ = _linear(9)
    x = torch.randn(2, 128).to(torch.bfloat16)
    with torch.no_grad():
        mod(x)
    assert mod._dequantized is not None


def test_summary_is_json_friendly(policy):
    mod, *_ = _linear(10)
    policy.register("layers.0.q", mod)
    s = policy.finalize(cap_bytes=1 << 20)
    import json
    json.dumps(s, allow_nan=False)
    assert s["requested_cap_bytes"] == 1 << 20
    assert set(s) >= {"enabled", "requested_cap_bytes", "effective_cap_bytes",
                      "admitted_bytes", "admitted", "skipped",
                      "candidate_bytes_total", "candidate_count", "finalized"}


# ---- loader registration ------------------------------------------------------------------


def test_install_quantized_registers_candidates(policy):
    from edge0.backends.cuda import io as cio
    gen = torch.Generator().manual_seed(20)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(128, 32, bias=False)
            self.emb = torch.nn.Embedding(10, 128)

    model = Tiny()
    codes = torch.randint(0, 16, (32, 128), generator=gen)
    scales = (torch.randn(32, 2, generator=gen) * 0.1).to(torch.bfloat16)
    biases = (torch.randn(32, 2, generator=gen) * 0.05).to(torch.bfloat16)
    ecodes = torch.randint(0, 16, (10, 128), generator=gen)
    state = {
        "proj.weight": _pack(codes, 4), "proj.scales": scales,
        "proj.biases": biases,
        "emb.weight": _pack(ecodes, 4),
        "emb.scales": scales[:10], "emb.biases": biases[:10],
    }
    replaced = cio._install_quantized(model, state, None)
    assert replaced == {"proj", "emb"}
    s = policy.finalize(cap_bytes=1 << 20)
    assert [a["path"] for a in s["admitted"]] == ["proj"]   # embeddings are not cached
    assert s["admitted_bytes"] == 32 * 128 * 2


# ---- CUDA: the fill transient the admission arithmetic promises --------------------
# Skips without a device; FAILS with --require-cuda.


@pytest.fixture(scope="module")
def torch_cuda(request):
    from tests.test_torch_cuda_smoke import _torch_with_cuda
    return _torch_with_cuda(request)


def test_cached_fill_peak_stays_within_the_priced_transient(torch_cuda, policy):
    """The regression this pins: dequantizing the whole weight in one shot
    peaked at ~8x the resident cache (3.7 GB for edge0-8b's lm_head), which
    does not fit an 8 GB Orin's headroom even though the resident bytes do.
    The chunked fill must stay inside resident + fill_transient_bytes."""
    mod, *_ = _linear(30, out_f=3 * cnn.QuantizedLinear.ROWS_PER_CHUNK + 7,
                      in_f=512)
    mod = mod.cuda()
    policy.register("wide", mod)
    policy.finalize(cap_bytes=1 << 34)
    assert mod.cache_weight
    resident = cnn.dequantized_bytes(mod)
    priced = resident + cnn.fill_transient_bytes(mod)
    x = torch.randn(2, 512).to(torch.bfloat16).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = mod(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert peak <= priced + out.numel() * out.element_size(), (peak, priced)
    assert peak >= resident            # the cache really was built


def test_chunked_fill_is_bit_identical_to_a_whole_weight_dequantization(
        torch_cuda, policy):
    """Chunking bounds the transient without changing a single element:
    same expression per chunk as the on-the-fly path."""
    from edge0.backends.cuda.quant import _dequantize
    mod, *_ = _linear(31, out_f=2 * cnn.QuantizedLinear.ROWS_PER_CHUNK + 5,
                      in_f=256)
    mod = mod.cuda()
    policy.register("wide", mod)
    policy.finalize(cap_bytes=1 << 34)
    x = torch.randn(2, 256).to(torch.bfloat16).cuda()
    with torch.no_grad():
        mod(x)
    one_shot = _dequantize(mod.weight, mod.scales, mod.biases,
                           mod.group_size, mod.bits).to(torch.bfloat16)
    assert mod._dequantized is not None
    assert torch.equal(mod._dequantized[1], one_shot)


def test_full_cache_env_uses_the_same_chunked_fill(torch_cuda, monkeypatch,
                                                   policy):
    """EDGE0_TORCH_WEIGHT_CACHE=1 shares the branch, so it inherits the
    bounded transient (it was the original 3.7 GB offender)."""
    monkeypatch.setattr(cnn, "CACHE_DEQUANTIZED", True)
    mod, *_ = _linear(32, out_f=3 * cnn.QuantizedLinear.ROWS_PER_CHUNK,
                      in_f=512)
    mod = mod.cuda()
    assert mod.cache_weight is False          # the ENV path, not the policy
    priced = cnn.dequantized_bytes(mod) + cnn.fill_transient_bytes(mod)
    x = torch.randn(2, 512).to(torch.bfloat16).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = mod(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert peak <= priced + out.numel() * out.element_size(), (peak, priced)
