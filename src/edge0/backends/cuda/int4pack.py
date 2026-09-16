"""torch's built-in int4 kernel for the dense quantized linears (Task 6,
design A, opt-in ``EDGE0_INT4PACK=1``).

``torch.ops.aten._weight_int4pack_mm`` (the gpt-fast / torchao path)
runs a group-wise int4 x bfloat16 matmul on CUDA.  It wants the weight
in its own tiled layout (``_convert_weight_to_int4pack`` of a
``[n, k/2]`` uint8 tensor, high nibble first) and computes
``w = (q - 8) * scale + zero`` from a ``[k/group, n, 2]`` bfloat16
``scales_and_zeros`` tensor.  The MLX affine layout this repo stores
(8 codes per uint32, least significant bits first, ``w = code * scale
+ bias``) maps onto it with ``zero = bias + 8 * scale``.

That mapping is NOT exact: ``zero`` is rounded to bfloat16 and the
kernel dequantizes in bfloat16, so its weights differ from the
reference's own bfloat16-rounded weights by about one bfloat16 ulp for
roughly a third of the entries (measured on torch 2.14.0+cu130, see
``tests/test_int4pack.py`` for the pre-registered bounds).  The exact
paths (the on-the-fly dequantization and both dequantized-weight
caches) therefore take precedence in ``nn.QuantizedLinear``; this path
is used only when the gate is on, nothing exact is cached for the
module, and every guard in ``is_supported`` holds.  Whether the
approximation is acceptable is decided by the 32-token reference check
on the target device, not here.

Costs: the repacked payload duplicates the int4 words it covers
(``int4pack_extra_bytes``) -- the original buffers stay for the
fall-through paths -- and the kernel accepts only 2-D bfloat16
activations, so inputs are reshaped around the call.
"""

from __future__ import annotations

import os

import torch

INT4PACK_ENV = "EDGE0_INT4PACK"
#: ``_convert_weight_to_int4pack`` tiling; the kernel then needs
#: ``in_features % (INNER_K_TILES * 16) == 0``.
INNER_K_TILES = 8
SUPPORTED_GROUP_SIZES = (32, 64, 128, 256)


def _read_int4pack_gate() -> bool:
    return os.environ.get(INT4PACK_ENV, "") == "1"


#: Read once at import, like ``quant.BATCHED``; tests patch the attribute.
INT4PACK = _read_int4pack_gate()


# ---- pure repack (CPU or CUDA, exact) -------------------------------------------


def codes_to_uint8(weight_u32: torch.Tensor) -> torch.Tensor:
    """MLX-packed ``[..., k/8]`` uint32 (8 codes per word, least
    significant bits first) -> ``[..., k/2]`` uint8 with the EVEN column's
    code in the high nibble and the ODD column's in the low nibble (the
    order ``_convert_weight_to_int4pack`` expects)."""
    words = weight_u32.view(torch.int32)
    shifts = torch.arange(8, device=words.device, dtype=torch.int32) * 4
    codes = ((words.unsqueeze(-1) >> shifts) & 0xF)
    codes = codes.reshape(*words.shape[:-1], words.shape[-1] * 8)
    return ((codes[..., 0::2] << 4) | codes[..., 1::2]).to(torch.uint8)


def uint8_to_codes(packed_u8: torch.Tensor) -> torch.Tensor:
    """Inverse of ``codes_to_uint8``: ``[..., k/2]`` uint8 -> ``[..., k]``
    int64 codes in [0, 15]."""
    words = packed_u8.to(torch.int64)
    hi = (words >> 4) & 0xF
    lo = words & 0xF
    return torch.stack([hi, lo], dim=-1).reshape(*words.shape[:-1],
                                                 words.shape[-1] * 2)


def codes_to_uint32(codes: torch.Tensor) -> torch.Tensor:
    """``[..., k]`` codes -> the MLX packing (8 per uint32, LSB first)."""
    grouped = codes.reshape(*codes.shape[:-1], -1, 8).to(torch.int64)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * 4
    words = (grouped << shifts).sum(-1)
    return words.to(torch.int32).view(torch.uint32)


def scales_and_zeros(scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """``[n, k/group]`` scales/biases -> ``[k/group, n, 2]`` bfloat16
    ``(scale, zero)`` with ``zero = bias + 8 * scale`` so that the kernel's
    ``(q - 8) * scale + zero`` is the affine ``q * scale + bias``."""
    s = scales.to(torch.float32)
    z = biases.to(torch.float32) + 8.0 * s
    return torch.stack([s.t(), z.t()], dim=-1).to(torch.bfloat16).contiguous()


def zero_rounding_error(scales: torch.Tensor, biases: torch.Tensor) -> float:
    """Largest |bf16(zero) - zero| the mapping introduces (reported, never
    hidden: it is one of the two reasons the path is approximate)."""
    z = biases.double() + 8.0 * scales.double()
    return float((z.to(torch.bfloat16).double() - z).abs().max().item())


def is_supported(*, in_features: int, out_features: int, group_size: int,
                 bits: int, device_type: str, x_dtype) -> tuple[bool, str]:
    """Guarded dispatch: every combination the kernel does not handle
    falls through to the reference implementation."""
    if bits != 4:
        return False, f"{bits}-bit layout (kernel is int4 only)"
    if group_size not in SUPPORTED_GROUP_SIZES:
        return False, f"group_size {group_size} not in {SUPPORTED_GROUP_SIZES}"
    if in_features % (INNER_K_TILES * 16) != 0:
        return False, (f"in_features {in_features} is not a multiple of "
                       f"{INNER_K_TILES * 16}")
    if out_features % 8 != 0:
        return False, f"out_features {out_features} is not a multiple of 8"
    if device_type != "cuda":
        return False, f"device {device_type!r} (CUDA kernel only)"
    if x_dtype is not torch.bfloat16:
        return False, f"activation dtype {x_dtype} (kernel takes bfloat16)"
    return True, ""


def int4pack_extra_bytes(out_features: int, in_features: int,
                         group_size: int) -> int:
    """Bytes the repacked payload adds next to the original buffers."""
    return (out_features * in_features // 2
            + (in_features // group_size) * out_features * 2 * 2)


# ---- kernel (CUDA) --------------------------------------------------------------------


class KernelUnavailable(RuntimeError):
    """The device has no usable image for the kernel (e.g. a wheel whose
    SASS list excludes this architecture).  Every guard passed, so the
    caller falls through to the reference path and stops retrying."""


def repack(weight_u32: torch.Tensor, scales: torch.Tensor,
           biases: torch.Tensor, group_size: int):
    """Build the kernel's operands from the MLX layout."""
    packed_u8 = codes_to_uint8(weight_u32)
    tiled = torch.ops.aten._convert_weight_to_int4pack(packed_u8, INNER_K_TILES)
    return tiled, scales_and_zeros(scales, biases)


def probe(device=None) -> tuple[bool, str]:
    """Run the kernel once on a tiny tensor to prove the device actually
    has an image for it.

    ``is_supported`` checks shapes and dtypes; only execution proves the
    build covers this architecture.  The Orin's torch wheel warns that
    ``sm_87`` is outside its SASS list (CC 8.x binary compatibility is
    expected to carry it, but the plan's rule is that execution evidence
    gates, not the warning), so the dispatch probes before it commits.
    """
    try:
        codes = torch.randint(0, 16, (8, 256), device=device)
        packed = codes_to_uint32(codes)
        scales = torch.ones(8, 4, dtype=torch.bfloat16, device=device)
        biases = torch.zeros(8, 4, dtype=torch.bfloat16, device=device)
        tiled, saz = repack(packed, scales, biases, 64)
        x = torch.ones(1, 256, dtype=torch.bfloat16, device=device)
        out = _int4pack_mm(x, tiled, 64, saz)
        if device is not None and getattr(device, "type", device) == "cuda":
            torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            return False, "kernel probe produced non-finite output"
    except Exception as exc:  # noqa: BLE001 - any failure disables the path
        return False, f"kernel probe failed: {type(exc).__name__}: {exc}"
    return True, ""


def _int4pack_mm(x2d: torch.Tensor, tiled: torch.Tensor, group_size: int,
                 saz: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten._weight_int4pack_mm(x2d, tiled, group_size, saz)


def int4pack_linear(x: torch.Tensor, tiled: torch.Tensor, group_size: int,
                    saz: torch.Tensor, bias: torch.Tensor | None = None):
    """``F.linear`` on the repacked weight: reshape to the 2-D bfloat16
    input the kernel takes and back."""
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    y = _int4pack_mm(x2d, tiled, group_size, saz)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y.reshape(*x.shape[:-1], y.shape[-1])
