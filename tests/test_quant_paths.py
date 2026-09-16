"""Opt-in quantized paths for the torch backend (Task 6, workstation side).

Every test here computes its reference independently of the code under
test: int4 codes are packed by hand (8 per uint32 word, least
significant bits first, the layout ``backends/cuda/quant.py`` documents
and ``tests/test_cuda_backend.py`` verified against ``mx.quantize``),
dequantized by the documented formula ``code * scale + bias`` in
float64, and multiplied in float64.  No MLX anywhere: the suite runs on
a Jetson.  Tolerances are written down here, before any Orin run.

CPU cases run everywhere; the CUDA cases reuse the acceptance fixture
of ``tests/test_torch_cuda_smoke.py`` (skip without a device, FAIL with
``--require-cuda``).  Nothing here is a performance statement.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EDGE0_BACKEND", "cuda")
torch = pytest.importorskip("torch")

from edge0.backends.cuda import quant as cq  # noqa: E402

E, OUT, D, GS = 8, 16, 128, 64


# ---- hand-computed helpers ------------------------------------------------------


def _pack(codes, bits):
    """Pack integer codes along the last axis, ``32 // bits`` per uint32,
    least significant bits first (independent of ``quant._dequantize``)."""
    per_word = 32 // bits
    assert codes.shape[-1] % per_word == 0
    grouped = codes.reshape(*codes.shape[:-1], -1, per_word).to(torch.int64)
    shifts = torch.arange(per_word, dtype=torch.int64) * bits
    words = (grouped << shifts).sum(-1)
    return words.to(torch.int32).view(torch.uint32)


def _dequant64(codes, scales, biases, group):
    shp = codes.shape
    return (codes.reshape(*shp[:-1], -1, group).double()
            * scales.double().unsqueeze(-1)
            + biases.double().unsqueeze(-1)).reshape(shp)


def _gather_ref64(x, w64, idx, transpose):
    """``out[b] = x[b] @ W[idx[b]].T`` over the broadcast batch, float64."""
    bshape = torch.broadcast_shapes(x.shape[:-2], idx.shape)
    M = x.shape[-2]
    xb = x.double().expand(*bshape, M, x.shape[-1]).reshape(-1, M, x.shape[-1])
    ib = idx.expand(bshape).reshape(-1)
    outs = []
    for row, e in zip(xb, ib.tolist()):
        wm = w64[e]
        outs.append(row @ (wm.T if transpose else wm))
    return torch.stack(outs).reshape(*bshape, M, -1)


def _quantized(seed, bits=4, shape=(E, OUT, D), group=GS, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 1 << bits, shape, generator=gen)
    scales = (torch.randn(*shape[:-1], shape[-1] // group, generator=gen)
              * 0.1).to(dtype)
    biases = (torch.randn(*shape[:-1], shape[-1] // group, generator=gen)
              * 0.05).to(dtype)
    return codes, _pack(codes, bits), scales, biases


SHAPES = [
    ((3, D), (3,)),            # default convention: broadcasts to (3, 3, OUT)
    ((5, 1, 1, D), (5, 3)),    # streaming/layer.py unsorted path
    ((15, 1, D), (15,)),       # streaming/layer.py sorted path
    ((2, 1, 4, D), (2, 3)),
    ((1, 2, D), (4, 1)),
]


@pytest.fixture
def batched(monkeypatch):
    """Turn the batched gather on for one test."""
    monkeypatch.setattr(cq, "BATCHED", True)
    monkeypatch.setattr(cq, "BATCHED_MAX_BYTES", cq.DEFAULT_BATCHED_MAX_BYTES)
    return cq


@pytest.fixture
def reference_spy(monkeypatch):
    calls = []
    real = cq._gather_qmm_reference

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(cq, "_gather_qmm_reference", spy)
    return calls


# ---- gate ------------------------------------------------------------------------


def test_batched_gate_is_on_unless_disabled(monkeypatch):
    """Default flipped on the Orin's Task 6 acceptance (decode +21% on the
    target, reference check PASSING the registered bound); ``0`` selects
    the reference loop and any other value is treated as unset, so a typo
    cannot silently pick a path."""
    monkeypatch.delenv("EDGE0_QMM_BATCHED", raising=False)
    assert cq._read_batched_gate() is True
    monkeypatch.setenv("EDGE0_QMM_BATCHED", "0")
    assert cq._read_batched_gate() is False
    monkeypatch.setenv("EDGE0_QMM_BATCHED", "1")
    assert cq._read_batched_gate() is True
    monkeypatch.setenv("EDGE0_QMM_BATCHED", "no")
    assert cq._read_batched_gate() is True
    assert cq.DEFAULT_BATCHED_MAX_BYTES == 256 << 20
    monkeypatch.setenv("EDGE0_QMM_BATCHED_MAX_BYTES", "1024")
    assert cq._read_batched_max_bytes() == 1024
    monkeypatch.setenv("EDGE0_QMM_BATCHED_MAX_BYTES", "lots")
    with pytest.raises(ValueError, match="EDGE0_QMM_BATCHED_MAX_BYTES"):
        cq._read_batched_max_bytes()


def test_reference_path_is_selectable_and_2bit_still_falls_through(
        reference_spy, monkeypatch):
    """``EDGE0_QMM_BATCHED=0`` reaches the reference loop, and a layout the
    batched path does not cover falls through to it even when the (now
    default-on) gate is set."""
    codes, packed, s, b = _quantized(0)
    x = torch.randn(3, D)
    monkeypatch.setattr(cq, "BATCHED", False)
    cq.gather_qmm(x, packed, s, b, torch.tensor([1, 2, 3]), transpose=True,
                  group_size=GS, bits=4)
    assert reference_spy == [1]
    monkeypatch.setattr(cq, "BATCHED", True)
    codes2, packed2, s2, b2 = _quantized(0, bits=2)
    cq.gather_qmm(x, packed2, s2, b2, torch.tensor([1, 2, 3]), transpose=True,
                  group_size=GS, bits=2)
    assert reference_spy == [1, 1]


# ---- batched gather: parity ------------------------------------------------------


@pytest.mark.parametrize("x_shape,idx_shape", SHAPES)
def test_batched_matches_hand_reference_and_reference_loop(batched,
                                                          reference_spy,
                                                          x_shape, idx_shape):
    codes, packed, s, b = _quantized(1)
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(x_shape, generator=gen)
    idx = torch.randint(0, E, idx_shape, generator=gen)
    got = cq.gather_qmm(x, packed, s, b, idx, transpose=True, group_size=GS,
                        bits=4)
    assert reference_spy == []                     # the batched path ran
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert got.shape == ref.shape and got.dtype == x.dtype
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-4)
    loop = cq._gather_qmm_reference(x, packed, s, b, idx, True, GS, 4)
    assert torch.allclose(got, loop, rtol=1e-5, atol=1e-5)


def test_batched_no_transpose(batched, reference_spy):
    codes, packed, s, b = _quantized(2, shape=(E, D, OUT * 4))
    gen = torch.Generator().manual_seed(12)
    x = torch.randn(3, 1, 1, D, generator=gen)
    idx = torch.randint(0, E, (3, 2), generator=gen)
    got = cq.gather_qmm(x, packed, s, b, idx, transpose=False, group_size=GS,
                        bits=4)
    assert reference_spy == []
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, False)
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-4)


def test_batched_sorted_hint_and_repeated_experts_with_padding(batched):
    """The sorted call pattern with uneven group sizes (Rmax > 1, M > 1):
    padding rows must never leak into real rows."""
    codes, packed, s, b = _quantized(3)
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(9, 2, D, generator=gen)
    idx = torch.tensor([0, 0, 0, 0, 3, 3, 5, 7, 7])
    got = cq.gather_qmm(x, packed, s, b, idx, transpose=True, group_size=GS,
                        bits=4, sorted_indices=True)
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-4)
    # unsorted order of the same rows gives the same per-row answers
    perm = torch.tensor([8, 2, 6, 0, 4, 1, 7, 3, 5])
    got_p = cq.gather_qmm(x[perm], packed, s, b, idx[perm], transpose=True,
                          group_size=GS, bits=4)
    assert torch.allclose(got_p, got[perm], rtol=1e-6, atol=1e-6)


def test_batched_negative_scales_and_code_extremes(batched):
    gen = torch.Generator().manual_seed(4)
    codes = torch.randint(0, 16, (E, OUT, D), generator=gen)
    codes[0] = 0                                   # all-zero codes
    codes[1] = 15                                  # all-max codes
    codes[2, :, :GS] = 15
    codes[2, :, GS:] = 0
    packed = _pack(codes, 4)
    scales = torch.randn(E, OUT, D // GS, generator=gen) * 4.0   # +/- and large
    scales[3] = -3.0
    biases = torch.randn(E, OUT, D // GS, generator=gen) * 20.0
    x = torch.randn(4, 1, 1, D, generator=gen)
    idx = torch.tensor([[0, 1], [2, 3], [3, 0], [1, 2]])
    got = cq.gather_qmm(x, packed, scales, biases, idx, transpose=True,
                        group_size=GS, bits=4)
    ref = _gather_ref64(x, _dequant64(codes, scales, biases, GS), idx, True)
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-3)


def test_batched_index_boundaries(batched):
    codes, packed, s, b = _quantized(5)
    x = torch.randn(2, 1, 1, D)
    last = torch.tensor([[E - 1, 0], [E - 1, E - 1]])
    got = cq.gather_qmm(x, packed, s, b, last, transpose=True, group_size=GS,
                        bits=4)
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), last, True)
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-4)
    with pytest.raises(IndexError):
        cq.gather_qmm(x, packed, s, b, torch.tensor([[E, 0], [1, 2]]),
                      transpose=True, group_size=GS, bits=4)
    with pytest.raises(IndexError):
        cq._gather_qmm_reference(x, packed, s, b, torch.tensor([[E, 0], [1, 2]]),
                                 True, GS, 4)


def test_batched_bf16_checkpoint_dtypes(batched):
    codes, packed, s, b = _quantized(6, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(16)
    x = torch.randn(4, 1, 1, D, generator=gen).to(torch.bfloat16)
    idx = torch.randint(0, E, (4, 2), generator=gen)
    got = cq.gather_qmm(x, packed, s, b, idx, transpose=True, group_size=GS,
                        bits=4)
    assert got.dtype == torch.bfloat16
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert torch.allclose(got.double(), ref, rtol=2e-2, atol=1e-1)
    loop = cq._gather_qmm_reference(x, packed, s, b, idx, True, GS, 4)
    assert torch.allclose(got.float(), loop.float(), rtol=1e-2, atol=1e-2)


# ---- batched gather: guards and fall-through ------------------------------------------


@pytest.mark.parametrize("bits", [2, 8])
def test_batched_falls_through_for_2_and_8_bit(batched, reference_spy, bits):
    codes, packed, s, b = _quantized(7, bits=bits)
    x = torch.randn(4, 1, 1, D)
    idx = torch.randint(0, E, (4, 2))
    got = cq.gather_qmm(x, packed, s, b, idx, transpose=True, group_size=GS,
                        bits=bits)
    assert reference_spy == [1]
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-4)


def test_batched_falls_through_when_the_transient_exceeds_the_cap(
        batched, reference_spy, monkeypatch):
    codes, packed, s, b = _quantized(8)
    x = torch.randn(4, 1, 1, D)
    idx = torch.randint(0, E, (4, 2))
    monkeypatch.setattr(cq, "BATCHED_MAX_BYTES", 1)
    got = cq.gather_qmm(x, packed, s, b, idx, transpose=True, group_size=GS,
                        bits=4)
    assert reference_spy == [1]
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert torch.allclose(got.double(), ref, rtol=1e-4, atol=1e-4)


def test_batched_rejects_non_affine(batched):
    codes, packed, s, b = _quantized(9)
    with pytest.raises(NotImplementedError):
        cq.gather_qmm(torch.randn(2, D), packed, s, b, torch.tensor([0, 1]),
                      transpose=True, group_size=GS, bits=4, mode="mxfp4")


def test_batched_transient_bytes_formula():
    # decode shape of edge0-8b: 8 distinct experts, one row each, M=1,
    # in=2048 (up/gate) or 512 (down), out=512 / 2048
    n = cq.batched_transient_bytes(n_unique=8, rows_max=1, m=1, d_in=2048,
                                   n_out=512, bits=4)
    per_expert_elems = 512 * 2048
    assert n >= 8 * per_expert_elems * 4 * 3      # >= three float32-sized copies
    assert n <= 8 * per_expert_elems * 4 * 6      # not absurdly loose
    assert cq.batched_transient_bytes(1, 1, 1, 64, 8) > 0
    with pytest.raises(ValueError):
        cq.batched_transient_bytes(0, 1, 1, 64, 8)


# ---- batched gather on an actual CUDA device ------------------------------------------
# Skips without a device; FAILS with --require-cuda (the acceptance contract
# of tests/test_torch_cuda_smoke.py).  Numbers measured here are workstation
# evidence for the instrumentation, never Orin evidence.


@pytest.fixture(scope="module")
def torch_cuda(request):
    from tests.test_torch_cuda_smoke import _torch_with_cuda
    return _torch_with_cuda(request)


@pytest.mark.parametrize("x_shape,idx_shape", SHAPES)
def test_batched_on_cuda_matches_hand_reference(torch_cuda, batched,
                                                x_shape, idx_shape):
    codes, packed, s, b = _quantized(21)
    gen = torch.Generator().manual_seed(31)
    x = torch.randn(x_shape, generator=gen)
    idx = torch.randint(0, E, idx_shape, generator=gen)
    got = cq.gather_qmm(x.cuda(), packed.cuda(), s.cuda(), b.cuda(),
                        idx.cuda(), transpose=True, group_size=GS, bits=4)
    torch.cuda.synchronize()
    assert got.device.type == "cuda"
    assert torch.isfinite(got).all()
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert torch.allclose(got.cpu().double(), ref, rtol=1e-4, atol=1e-4)


def test_batched_on_cuda_padding_and_bf16(torch_cuda, batched):
    codes, packed, s, b = _quantized(22, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(32)
    x = torch.randn(9, 2, D, generator=gen).to(torch.bfloat16)
    idx = torch.tensor([4, 4, 4, 1, 1, 7, 0, 0, 0])
    got = cq.gather_qmm(x.cuda(), packed.cuda(), s.cuda(), b.cuda(),
                        idx.cuda(), transpose=True, group_size=GS, bits=4,
                        sorted_indices=True)
    torch.cuda.synchronize()
    assert got.dtype == torch.bfloat16
    ref = _gather_ref64(x, _dequant64(codes, s, b, GS), idx, True)
    assert torch.allclose(got.cpu().double(), ref, rtol=2e-2, atol=1e-1)


def _decode_shape_inputs(seed):
    """The edge0-8b decode shape of one up/gate projection: 8 distinct
    experts for one token, out 512, in 2048."""
    n_experts, out_f, in_f = 128, 512, 2048
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n_experts, out_f, in_f), generator=gen)
    packed = _pack(codes, 4)
    scales = (torch.randn(n_experts, out_f, in_f // GS, generator=gen) * 0.1
              ).to(torch.bfloat16)
    biases = (torch.randn(n_experts, out_f, in_f // GS, generator=gen) * 0.05
              ).to(torch.bfloat16)
    x = torch.randn(1, 1, 1, in_f, generator=gen).to(torch.bfloat16)
    idx = torch.tensor([[3, 17, 42, 64, 65, 99, 100, 127]])
    return packed, scales, biases, x, idx


def test_batched_transient_on_cuda_is_within_the_priced_bound(torch_cuda,
                                                              batched):
    packed, scales, biases, x, idx = (t.cuda() for t in _decode_shape_inputs(41))
    priced = cq.batched_transient_bytes(n_unique=8, rows_max=1, m=1,
                                        d_in=2048, n_out=512, bits=4,
                                        group_size=GS, scale_itemsize=2)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = cq.gather_qmm(x, packed, scales, biases, idx, transpose=True,
                        group_size=GS, bits=4)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    assert peak <= priced, f"measured transient {peak} > priced {priced}"
    assert peak >= priced // 4, f"pricing is > 4x loose: {peak} vs {priced}"


def _count_cuda_kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()                                   # warm up (allocator, cuBLAS)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for ev in (prof.events() or [])
               if ev.device_type == torch.autograd.DeviceType.CUDA)


def test_batched_launches_fewer_kernels_than_the_loop(torch_cuda, monkeypatch):
    """Launch count is the architecture-independent part of the win: the
    reference loop launches ~8x (dequant chain + matmul) per distinct
    expert, the batched path launches one chain + one bmm."""
    packed, scales, biases, x, idx = (t.cuda() for t in _decode_shape_inputs(42))
    monkeypatch.setattr(cq, "BATCHED_MAX_BYTES", cq.DEFAULT_BATCHED_MAX_BYTES)
    monkeypatch.setattr(cq, "BATCHED", False)
    loop = _count_cuda_kernels(lambda: cq.gather_qmm(
        x, packed, scales, biases, idx, transpose=True, group_size=GS, bits=4))
    monkeypatch.setattr(cq, "BATCHED", True)
    batched_n = _count_cuda_kernels(lambda: cq.gather_qmm(
        x, packed, scales, biases, idx, transpose=True, group_size=GS, bits=4))
    assert batched_n * 3 <= loop, (loop, batched_n)


def test_batched_is_within_one_bfloat16_ulp_at_the_real_decode_shapes(
        torch_cuda, batched):
    """Parity at the shapes the 8B tier actually decodes with, in the
    checkpoint's dtypes.  The two paths dequantize identically and differ
    only in how cuBLAS reduces the matmul, so the bf16 result must land on
    the same value or its neighbour -- one ulp, not a tolerance chosen
    after the fact.  End to end through 24 layers this accumulates: the
    workstation reference check measured max |dlogit| 1.01 with all 32
    token choices identical."""
    for n_experts, out_f, in_f, name in ((128, 512, 1536, "up/gate"),
                                         (128, 1536, 512, "down")):
        gen = torch.Generator().manual_seed(80 + out_f)
        codes = torch.randint(0, 16, (n_experts, out_f, in_f), generator=gen)
        packed = _pack(codes, 4).cuda()
        scales = (torch.randn(n_experts, out_f, in_f // GS, generator=gen)
                  * 0.05).to(torch.bfloat16).cuda()
        biases = (torch.randn(n_experts, out_f, in_f // GS, generator=gen)
                  * 0.05).to(torch.bfloat16).cuda()
        x = torch.randn(1, 1, 1, in_f, generator=gen).to(torch.bfloat16).cuda()
        idx = torch.tensor([[3, 17, 42, 64, 65, 99, 100, 127]]).cuda()
        got = cq.gather_qmm(x, packed, scales, biases, idx, transpose=True,
                            group_size=GS, bits=4)
        loop = cq._gather_qmm_reference(x, packed, scales, biases, idx,
                                        True, GS, 4)
        torch.cuda.synchronize()
        assert got.dtype == torch.bfloat16
        diff = (got.float() - loop.float()).abs()
        # one bfloat16 ulp of each element (8 mantissa bits)
        ulp = torch.ldexp(torch.ones_like(diff),
                          torch.floor(torch.log2(
                              loop.float().abs().clamp_min(1e-30))) - 7)
        assert (diff <= ulp).all(), (
            name, diff.max().item(), (diff / ulp).max().item())
