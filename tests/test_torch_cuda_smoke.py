"""Torch-only CUDA acceptance smoke (Task 3, Gate B).

Runs representative ``edge0.backends.cuda`` operations on an actual CUDA
device, synchronizes, and compares against deterministic references that
are computed independently of the code under test (hand-packed int4
codes dequantized by the documented formula, float64 matmul on the CPU).
No MLX import anywhere in this module: this suite must be runnable on a
Jetson where MLX does not exist.

Contract:

* plain ``pytest tests/test_torch_cuda_smoke.py`` skips when torch or a
  CUDA device is missing (hosts, CI) — unchanged behavior for everyone
  else;
* ``pytest tests/test_torch_cuda_smoke.py --require-cuda`` FAILS in the
  same situations.  This is the acceptance invocation on the Orin: a
  quiet skip or a CPU fallback must not read as CUDA evidence.

Passing here shows the toolchain executes kernels correctly on the
device.  It is NOT model correctness (that is the 32-token reference
check) and NOT a performance statement.
"""

from __future__ import annotations

import pytest


def _torch_with_cuda(request):
    required = request.config.getoption("--require-cuda")
    # ``edge0.backends`` binds to EDGE0_BACKEND at first import and
    # defaults to mlx, which does not exist on a Jetson.  This suite is
    # torch-only by contract, so select the cuda backend before any
    # test body imports ``edge0.backends.cuda.*`` (an explicit
    # EDGE0_BACKEND from the environment still wins).
    import os
    os.environ.setdefault("EDGE0_BACKEND", "cuda")
    try:
        import torch
    except ImportError:
        if required:
            pytest.fail("--require-cuda: torch is not importable in this "
                        "environment", pytrace=False)
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        if required:
            pytest.fail("--require-cuda: torch.cuda.is_available() is False "
                        f"(torch {torch.__version__}, built for CUDA "
                        f"{torch.version.cuda})", pytrace=False)
        pytest.skip("no CUDA device")
    # A device can be visible yet unable to run kernels (driver policy,
    # missing SASS/PTX for this arch).  Prove execution before any test
    # trusts an op's output.
    try:
        probe = (torch.arange(4, device="cuda", dtype=torch.float32) * 2).sum()
        torch.cuda.synchronize()
        value = float(probe.item())
    except Exception as exc:  # noqa: BLE001 - report whatever CUDA raised
        if required:
            pytest.fail(f"--require-cuda: CUDA device visible but kernel "
                        f"execution failed: {exc!r}", pytrace=False)
        pytest.skip(f"CUDA present but not executing: {exc!r}")
    assert value == 12.0
    return torch


@pytest.fixture(scope="module")
def torch_cuda(request):
    return _torch_with_cuda(request)


def _pack_int4(torch, codes):
    """Pack int codes in [0, 15] along the last axis, 8 per uint32 word,
    least significant bits first — the layout documented in
    ``backends/cuda/quant.py`` and verified against ``mx.quantize``."""
    assert codes.shape[-1] % 8 == 0
    grouped = codes.reshape(*codes.shape[:-1], -1, 8).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = (grouped << shifts).sum(-1)
    return words.to(torch.int32).view(torch.uint32)


def test_device_identity_and_finite(torch_cuda):
    torch = torch_cuda
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(64, 128, generator=gen)
    b = torch.randn(128, 32, generator=gen)
    out = a.cuda() @ b.cuda()
    torch.cuda.synchronize()
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    ref = (a.double() @ b.double()).float()
    assert torch.allclose(out.cpu(), ref, rtol=1e-4, atol=1e-4)


def test_unsigned_word_view_matches_cpu(torch_cuda):
    """The int4 path stores packed words as uint32 and dequantizes through
    an int32 view with shifts and masks; the top bit must survive the
    round trip on the device exactly as it does on the CPU."""
    torch = torch_cuda
    words = torch.tensor([0x80000001, 0xFFFFFFFF, 0x7FFFFFFF, 0],
                         dtype=torch.int64).to(torch.int32).view(torch.uint32)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    def unpack(w, s):
        return (w.view(torch.int32).unsqueeze(-1) >> s) & 0xF
    got = unpack(words.cuda(), shifts.cuda())
    torch.cuda.synchronize()
    assert got.device.type == "cuda"
    assert torch.equal(got.cpu(), unpack(words, shifts))


def test_rmsnorm_runs_on_cuda(torch_cuda):
    """`torch.nn.RMSNorm` is the API the backend's nn module subclasses
    (Gate A lists it as a required import)."""
    torch = torch_cuda
    from edge0.backends.cuda import nn as cnn
    norm = cnn.RMSNorm(64, eps=1e-6).cuda()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(3, 64, generator=gen).cuda()
    out = norm(x)
    torch.cuda.synchronize()
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()
    xr = x.cpu().double()
    ref = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)).float()
    assert torch.allclose(out.cpu(), ref, rtol=1e-4, atol=1e-5)


def test_gather_qmm_cuda_matches_hand_reference(torch_cuda):
    """End-to-end int4 expert gather on the device against a reference
    dequantized by the documented formula (code * scale + bias) in
    float64 on the CPU — independent of ``quant._dequantize``."""
    torch = torch_cuda
    from edge0.backends.cuda import quant as cq
    E, O, D, GS = 4, 8, 64, 32
    gen = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 16, (E, O, D), generator=gen)
    packed = _pack_int4(torch, codes)
    scales = (torch.randn(E, O, D // GS, generator=gen) * 0.1)
    # negative scales and biases on purpose: nothing may assume otherwise
    biases = torch.randn(E, O, D // GS, generator=gen) * 0.05
    x = torch.randn(3, 1, D, generator=gen)
    idx = torch.tensor([2, 0, 3], dtype=torch.int64)

    got = cq.gather_qmm(x.cuda(), packed.cuda(), scales.cuda(), biases.cuda(),
                        idx.cuda(), transpose=True, group_size=GS, bits=4)
    torch.cuda.synchronize()
    assert got.device.type == "cuda"
    assert torch.isfinite(got).all()

    w = (codes.double().reshape(E, O, D // GS, GS)
         * scales.double().unsqueeze(-1)
         + biases.double().unsqueeze(-1)).reshape(E, O, D)
    ref = torch.stack([x[i, :, :].double() @ w[e].T
                       for i, e in enumerate(idx.tolist())]).float()
    assert got.shape == ref.shape
    assert torch.allclose(got.cpu(), ref, rtol=1e-4, atol=1e-4)


def test_gather_sort_roundtrip_on_cuda(torch_cuda):
    torch = torch_cuda
    from edge0.backends.cuda import quant as cq
    gen = torch.Generator().manual_seed(3)
    T, K, D = 6, 2, 16
    x = torch.randn(T, 1, 1, D, generator=gen).cuda()
    idx = torch.randint(0, 4, (T, K), generator=gen).cuda()
    xs, idx_sorted, inv = cq.gather_sort(x, idx)
    torch.cuda.synchronize()
    assert xs.device.type == "cuda"
    assert torch.equal(idx_sorted, idx.reshape(-1).long().sort(stable=True).values)
    restored = cq.scatter_unsort(xs, inv, shape=(T, K))
    assert torch.equal(restored[:, 0], x.squeeze(1))


def test_core_ops_and_sampler_on_cuda(torch_cuda):
    """The core facade ops the decode loop leans on (softmax, topk,
    argmax, take_along_axis) on the CUDA device against CPU torch."""
    torch = torch_cuda
    from edge0.backends.cuda import core
    gen = torch.Generator().manual_seed(4)
    logits = torch.randn(1, 512, generator=gen)
    lc = logits.cuda()
    sm = core.softmax(lc, axis=-1)
    torch.cuda.synchronize()
    assert sm.device.type == "cuda"
    assert torch.allclose(sm.cpu(), torch.softmax(logits, dim=-1),
                          rtol=1e-5, atol=1e-6)
    assert int(core.argmax(lc, axis=-1).item()) == int(logits.argmax(-1))
    vals, _ = torch.topk(logits, 8)
    got_topk = core.topk(lc, 8, axis=-1)
    assert torch.allclose(torch.sort(got_topk.cpu(), descending=True).values,
                          vals, rtol=0, atol=0)
