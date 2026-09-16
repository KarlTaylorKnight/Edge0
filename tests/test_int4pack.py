"""torch's built-in int4 kernel for the dense quantized linears (Task 6,
design A, opt-in ``EDGE0_INT4PACK=1``).

The MLX affine int4 layout (8 codes per uint32, least significant bits
first, ``w = code * scale + bias``) is repacked into the layout of
``torch.ops.aten._weight_int4pack_mm`` (uint8 pairs, high nibble first,
tiled by ``_convert_weight_to_int4pack``) with ``zero = bias + 8 * scale``
so that the kernel's ``(q - 8) * scale + zero`` equals the affine form.
The mapping is NOT exact: ``zero`` is rounded to bfloat16 and the kernel
dequantizes in bfloat16, so weights differ from the reference's own
bfloat16-rounded weights by about one bfloat16 ulp for a third of the
entries (measured on torch 2.14.0+cu130).  Tolerances below were
registered from that workstation measurement BEFORE any Orin run; the
Orin's 32-token reference check decides whether the path is acceptable.

The pure repack step is tested for an exact round trip on the CPU; the
kernel cases need a CUDA device (skip without one, FAIL with
``--require-cuda``).  No MLX anywhere.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EDGE0_BACKEND", "cuda")
torch = pytest.importorskip("torch")

from edge0.backends.cuda import int4pack as i4  # noqa: E402
from edge0.backends.cuda import nn as cnn  # noqa: E402
from tests.test_quant_paths import _dequant64, _pack  # noqa: E402

#: Pre-registered tolerances (see the module docstring).
OUTPUT_RTOL = 1e-2           # max |y - y_exact| <= OUTPUT_RTOL * max |y_exact|
WEIGHT_ABS_FRACTION = 2 ** -6  # max |w_kernel - w_exact| <= 2^-6 * (max|w| + max|zero|)


def _within(got, ref, rtol=OUTPUT_RTOL) -> bool:
    """The registered output criterion: max |got - ref| relative to the
    output scale (bf16 outputs make an elementwise floor meaningless)."""
    diff = (got.double() - ref.double()).abs().max().item()
    return diff <= rtol * ref.double().abs().max().item()


# ---- pure repack --------------------------------------------------------------------


def test_gate_and_constants(monkeypatch):
    monkeypatch.delenv("EDGE0_INT4PACK", raising=False)
    assert i4._read_int4pack_gate() is False
    monkeypatch.setenv("EDGE0_INT4PACK", "1")
    assert i4._read_int4pack_gate() is True
    assert i4.INNER_K_TILES == 8
    assert i4.SUPPORTED_GROUP_SIZES == (32, 64, 128, 256)


@pytest.mark.parametrize("seed", [0, 1])
def test_uint32_to_uint8_repack_round_trips_exactly(seed):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (24, 256), generator=gen)
    codes[0, :8] = 0
    codes[1, :8] = 15
    codes[2, :8] = torch.tensor([0, 15, 1, 14, 2, 13, 3, 12])
    packed_u32 = _pack(codes, 4)
    u8 = i4.codes_to_uint8(packed_u32)
    assert u8.dtype == torch.uint8 and tuple(u8.shape) == (24, 128)
    # high nibble = even column, low nibble = odd column
    assert int(u8[2, 0]) == (0 << 4) | 15
    assert int(u8[2, 1]) == (1 << 4) | 14
    assert torch.equal(i4.uint8_to_codes(u8), codes)
    assert torch.equal(i4.codes_to_uint32(i4.uint8_to_codes(u8)), packed_u32)


def test_scales_and_zeros_mapping():
    scales = torch.tensor([[0.5, -0.25], [1.0, 2.0]], dtype=torch.bfloat16)   # [n=2, k/g=2]
    biases = torch.tensor([[1.0, 3.0], [-4.0, 0.5]], dtype=torch.bfloat16)
    saz = i4.scales_and_zeros(scales, biases)
    assert saz.dtype == torch.bfloat16 and tuple(saz.shape) == (2, 2, 2)  # [k/g, n, 2]
    assert saz[0, 0, 0].item() == 0.5 and saz[0, 0, 1].item() == 1.0 + 8 * 0.5
    assert saz[1, 0, 0].item() == -0.25 and saz[1, 0, 1].item() == 3.0 + 8 * -0.25
    assert saz[0, 1, 1].item() == -4.0 + 8 * 1.0
    # the rounding the kernel will see is reported, never hidden
    err = i4.zero_rounding_error(scales, biases)
    assert err >= 0.0


@pytest.mark.parametrize("in_f,out_f,group,bits,device,dtype,ok", [
    (1536, 512, 64, 4, "cuda", torch.bfloat16, True),
    (2048, 1536, 64, 4, "cuda", torch.bfloat16, True),
    (1536, 512, 128, 4, "cuda", torch.bfloat16, True),
    (1600, 512, 64, 4, "cuda", torch.bfloat16, False),    # k % 128 != 0
    (1536, 12, 64, 4, "cuda", torch.bfloat16, False),     # n % 8 != 0
    (1536, 512, 48, 4, "cuda", torch.bfloat16, False),    # unsupported group
    (1536, 512, 64, 8, "cuda", torch.bfloat16, False),    # 8-bit layout
    (1536, 512, 64, 2, "cuda", torch.bfloat16, False),    # 2-bit layout
    (1536, 512, 64, 4, "cpu", torch.bfloat16, False),     # no CUDA kernel
    (1536, 512, 64, 4, "cuda", torch.float16, False),     # kernel wants bf16
    (1536, 512, 64, 4, "cuda", torch.float32, False),
])
def test_is_supported_guards(in_f, out_f, group, bits, device, dtype, ok):
    supported, reason = i4.is_supported(in_features=in_f, out_features=out_f,
                                        group_size=group, bits=bits,
                                        device_type=device, x_dtype=dtype)
    assert supported is ok, reason
    if not ok:
        assert reason


def test_extra_bytes_prices_the_duplicate_payload():
    # the repacked payload duplicates the int4 words plus scales_and_zeros
    assert i4.int4pack_extra_bytes(out_features=512, in_features=1536,
                                   group_size=64) == 512 * 1536 // 2 + (1536 // 64) * 512 * 2 * 2


# ---- CUDA --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_cuda(request):
    from tests.test_torch_cuda_smoke import _torch_with_cuda
    return _torch_with_cuda(request)


def _dense(seed, out_f, in_f, group=64):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (out_f, in_f), generator=gen)
    scales = (torch.randn(out_f, in_f // group, generator=gen) * 0.05).to(torch.bfloat16)
    biases = (torch.randn(out_f, in_f // group, generator=gen) * 0.05).to(torch.bfloat16)
    return codes, _pack(codes, 4), scales, biases


@pytest.mark.parametrize("out_f,in_f,m", [(512, 1536, 1), (1536, 2048, 4),
                                          (4608, 1536, 37), (2048, 512, 1)])
def test_int4pack_linear_matches_exact_reference_within_registered_tolerance(
        torch_cuda, out_f, in_f, m):
    codes, packed, scales, biases = _dense(30 + m, out_f, in_f)
    tiled, saz = i4.repack(packed.cuda(), scales.cuda(), biases.cuda(), 64)
    gen = torch.Generator().manual_seed(40 + m)
    x = (torch.randn(2, m, in_f, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    y = i4.int4pack_linear(x, tiled, 64, saz)
    torch.cuda.synchronize()
    assert y.device.type == "cuda" and y.dtype == torch.bfloat16
    assert tuple(y.shape) == (2, m, out_f)
    y_exact = x.cpu().double() @ _dequant64(codes, scales, biases, 64).T
    err = (y.cpu().double() - y_exact).abs().max().item()
    assert err <= OUTPUT_RTOL * y_exact.abs().max().item(), err


def test_int4pack_per_weight_error_is_within_the_registered_bound(torch_cuda):
    out_f, in_f = 512, 1536
    codes, packed, scales, biases = _dense(50, out_f, in_f)
    tiled, saz = i4.repack(packed.cuda(), scales.cuda(), biases.cuda(), 64)
    eye = torch.eye(in_f, dtype=torch.bfloat16, device="cuda")   # exact in bf16
    w_kernel = i4.int4pack_linear(eye, tiled, 64, saz).t().cpu().double()
    torch.cuda.synchronize()
    w_exact = _dequant64(codes, scales, biases, 64)
    zero = biases.double() + 8 * scales.double()
    bound = WEIGHT_ABS_FRACTION * (w_exact.abs().max().item() + zero.abs().max().item())
    assert (w_kernel - w_exact).abs().max().item() <= bound
    # and it is genuinely approximate: report, don't hide
    assert (w_kernel != w_exact.to(torch.bfloat16).double()).any()


def test_quantized_linear_uses_the_kernel_only_when_eligible(torch_cuda,
                                                           monkeypatch):
    monkeypatch.setattr(cnn, "CACHE_DEQUANTIZED", False)
    monkeypatch.setattr(cnn, "WEIGHT_CACHE", cnn.WeightCachePolicy(None))
    monkeypatch.setattr(i4, "INT4PACK", True)
    calls = []
    real = i4.int4pack_linear

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(i4, "int4pack_linear", spy)
    codes, packed, scales, biases = _dense(60, 512, 1536)
    mod = cnn.QuantizedLinear(packed, scales, biases, 1536).cuda()
    x = (torch.randn(3, 1536) * 0.5).to(torch.bfloat16).cuda()
    monkeypatch.setattr(i4, "INT4PACK", False)
    with torch.no_grad():
        ref = mod(x)                       # gate off: the reference path
    assert calls == [] and mod._int4pack is None and mod.int4pack_bytes() == 0
    monkeypatch.setattr(i4, "INT4PACK", True)
    with torch.no_grad():
        y = mod(x)                         # repacked lazily on the first eligible call
        y2 = mod(x)
    assert mod._int4pack is not None
    assert calls == [1, 1]
    assert torch.equal(y, y2)
    # registered criterion: max |y - ref| relative to the output scale
    # (both outputs are bfloat16, so an elementwise floor makes no sense)
    assert _within(y, ref)
    assert (y != ref).any()                # approximate, and honest about it
    with torch.no_grad():
        f32 = mod(x.float())               # float32 activations: reference path
    assert calls == [1, 1]
    assert f32.dtype == torch.float32
    assert i4.int4pack_extra_bytes(512, 1536, 64) == mod.int4pack_bytes()


def test_exact_caches_take_precedence_over_the_kernel(torch_cuda, monkeypatch):
    monkeypatch.setattr(cnn, "CACHE_DEQUANTIZED", False)
    policy = cnn.WeightCachePolicy(None)
    monkeypatch.setattr(cnn, "WEIGHT_CACHE", policy)
    monkeypatch.setattr(i4, "INT4PACK", True)
    calls = []
    monkeypatch.setattr(i4, "int4pack_linear",
                        lambda *a, **k: calls.append(1) or i4._int4pack_mm(*a, **k))
    codes, packed, scales, biases = _dense(61, 512, 1536)
    mod = cnn.QuantizedLinear(packed, scales, biases, 1536).cuda()
    policy.register("m", mod)
    policy.finalize(cap_bytes=1 << 30)
    x = (torch.randn(3, 1536) * 0.5).to(torch.bfloat16).cuda()
    with torch.no_grad():
        mod(x)
    assert calls == [] and mod._dequantized is not None


def test_int4pack_under_a_lora_wrapper(torch_cuda, monkeypatch):
    monkeypatch.setattr(cnn, "CACHE_DEQUANTIZED", False)
    monkeypatch.setattr(cnn, "WEIGHT_CACHE", cnn.WeightCachePolicy(None))
    monkeypatch.setattr(i4, "INT4PACK", True)
    from edge0.adapters.lora import LoraLinear
    codes, packed, scales, biases = _dense(62, 512, 1536)
    mod = cnn.QuantizedLinear(packed, scales, biases, 1536).cuda()
    gen = torch.Generator().manual_seed(63)
    a = (torch.randn(16, 1536, generator=gen) * 0.02).to(torch.float16).cuda()
    b = (torch.randn(512, 16, generator=gen) * 0.02).to(torch.float16).cuda()
    wrapped = LoraLinear(mod, a, b, 2.0)
    x = (torch.randn(3, 1536) * 0.5).to(torch.bfloat16).cuda()
    with torch.no_grad():
        y = wrapped(x)
    assert y.dtype == torch.bfloat16 and y.device.type == "cuda"
    monkeypatch.setattr(i4, "INT4PACK", False)
    mod._int4pack = None
    with torch.no_grad():
        ref = wrapped(x)
    assert _within(y, ref)
