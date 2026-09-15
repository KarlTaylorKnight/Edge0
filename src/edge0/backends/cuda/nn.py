"""CUDA backend: module-factory namespace (contract: Module, Linear,
RMSNorm, silu, gelu).

``torch.nn`` already provides all four natively (``nn.Linear``,
``nn.functional.silu``/``gelu``); the only real gap is ``RMSNorm``,
which is standard ``torch.nn.RMSNorm`` since torch 2.4 -- re-exported
here under the contract's names rather than assuming call sites import
``torch.nn`` directly (see ``backends/__init__.py``'s enforcement that
framework code only ever imports ``edge0.backends.{core,nn,io,quant}``).

The quantized modules keep the 4-bit payload resident and dequantize on
the fly.  Three opt-in alternatives exist for the dense linears, tried
in this order in ``QuantizedLinear.forward`` (exact ones first):

* ``EDGE0_TORCH_WEIGHT_CACHE=1`` -- keep EVERY weight dequantized
  (bit-identical to on-the-fly; ~1.9 GB in bf16 for edge0-8b's dense
  modules, 4.1 GB in float32 as measured on the GB10);
* ``EDGE0_TORCH_WEIGHT_CACHE_BYTES=<n>`` -- keep only the modules that
  fit a byte cap (``WeightCachePolicy``: first-fit in registration
  order, re-finalized by the engine against the Task 4 budget's
  headroom, recorded in the bench report);
* ``EDGE0_INT4PACK=1`` -- torch's built-in int4 kernel (see
  ``int4pack.py``; approximate at bfloat16 rounding, guarded).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as _tnn
import torch.nn.functional as F

from edge0.backends.cuda import int4pack as _i4

Module = _tnn.Module


class Linear(_tnn.Linear):
    """``torch.nn.Linear`` that, like ``mlx.nn.Linear``, accepts a plain
    tensor assigned to ``weight`` / ``bias`` (``prerouter/install.py`` does
    ``head.fc1.weight = w``); torch itself insists on a Parameter.

    Parameters never require grad, as MLX arrays carry no autograd state:
    modules built after ``load_model`` (the prerouter heads) would otherwise
    make every forward that touches them record a graph."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)

    def __setattr__(self, name, value):
        if (name in ("weight", "bias") and isinstance(value, torch.Tensor)
                and not isinstance(value, _tnn.Parameter)):
            value = _tnn.Parameter(value, requires_grad=False)
        super().__setattr__(name, value)


class RMSNorm(_tnn.RMSNorm):
    """``mlx.nn.RMSNorm(dims, eps)`` signature on top of ``torch.nn.RMSNorm``.

    Subclassed rather than wrapped so the parameter keeps the name
    ``weight``, as in MLX and the checkpoints. Numerics match
    ``mx.fast.rms_norm`` (eps inside the sqrt); see
    ``tests/test_cuda_backend.py``.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__(dims, eps=eps)
        self.requires_grad_(False)


def _quant_params(weight, scales, in_features):
    """(bits, group_size) of an MLX affine-quantized tensor, from shapes:
    packed width = in * bits / 32, scales width = in / group_size. Works
    for per-path overrides (the edge0-35b routers are 8-bit)."""
    return (weight.shape[-1] * 32 // in_features,
            in_features // scales.shape[-1])


#: Dequantized weights kept per module instead of redone every call.
#: Off by default: the whole point of the quantized layers is that the
#: 4-bit payload is what stays resident. Worth turning on where memory is
#: plentiful: on a GB10 it takes the edge0-8b decode step from 206 ms to
#: 124 ms a token, for a measured 4.1 GB more resident
#: (EDGE0_TORCH_WEIGHT_CACHE=1). For this tier the dequantized weights
#: themselves are 1.40 GB in bf16 (``WeightCachePolicy`` prices them from
#: the loaded modules); the difference is allocator segments the fill
#: leaves behind, which is why the fill is chunked.
CACHE_DEQUANTIZED = os.environ.get("EDGE0_TORCH_WEIGHT_CACHE", "") == "1"

WEIGHT_CACHE_BYTES_ENV = "EDGE0_TORCH_WEIGHT_CACHE_BYTES"


def _read_weight_cache_cap() -> int | None:
    """``None`` when unset (policy off); ``0`` also means off."""
    raw = os.environ.get(WEIGHT_CACHE_BYTES_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{WEIGHT_CACHE_BYTES_ENV} must be an integer byte count, got "
            f"{raw!r}") from None
    if value < 0:
        raise ValueError(f"{WEIGHT_CACHE_BYTES_ENV} must be >= 0, got {value}")
    return value


def dequantized_bytes(module, itemsize: int | None = None) -> int:
    """Bytes a module's dequantized weight occupies once cached (activation
    itemsize defaults to the checkpoint's scales dtype, bf16 for the
    shipped tiers)."""
    if itemsize is None:
        itemsize = module.scales.element_size()
    return module.out_features * module.in_features * itemsize


def fill_transient_bytes(module) -> int:
    """Peak EXTRA bytes of building one cached weight, on top of the
    resident result.

    The fill runs ``ROWS_PER_CHUNK`` rows at a time, so the transient is
    bounded by the chunk, not by the module: ``_dequantize`` keeps four
    float32-sized tensors of one chunk live at its peak, plus the
    float32 casts of that chunk's scales and biases, plus the chunk's
    converted result before it is written into the buffer.  Measured at
    edge0-8b's widest module (lm_head, 157184 x 1536) on torch
    2.14.0+cu130: 556.9 MiB of total peak against a 460.5 MiB resident
    result.  Whole-weight dequantization instead peaks at 3702 MiB for
    the same module, which is why the fill is chunked.
    """
    rows = min(QuantizedLinear.ROWS_PER_CHUNK, module.out_features)
    in_features = module.in_features
    itemsize = module.scales.element_size()
    groups = -(-in_features // module.group_size)
    chain = 4 * rows * in_features * 4
    scale_terms = 2 * rows * groups * (itemsize + 4)
    chunk_result = rows * in_features * itemsize
    return chain + scale_terms + chunk_result


_UNSET: Any = object()


class WeightCachePolicy:
    """Capped, exact dequantized-weight cache: which ``QuantizedLinear``
    modules keep their dequantized weight.

    Candidates are registered by the loader in model order; ``finalize``
    admits them first-fit until ``cap_bytes`` is used, sets
    ``module.cache_weight`` and records the decision.  Re-finalizing with a
    smaller cap (the engine does this once the Task 4 budget resolves)
    releases weights the new selection no longer covers, so nothing stale
    survives.  ``None`` / ``0`` means off: no module is admitted.
    """

    def __init__(self, cap_bytes: int | None = None):
        self.cap_bytes = cap_bytes
        self._candidates: list[tuple[str, Any, int]] = []
        self._summary: dict | None = None

    @classmethod
    def from_env(cls) -> "WeightCachePolicy":
        return cls(_read_weight_cache_cap())

    @property
    def enabled(self) -> bool:
        return bool(self.cap_bytes)

    def begin(self) -> None:
        """A new model load starts: forget the previous model's modules."""
        self._candidates = []
        self._summary = None

    def register(self, path: str, module) -> None:
        self._candidates.append((path, module, dequantized_bytes(module)))

    def candidate_bytes_total(self) -> int:
        return sum(nbytes for _, _, nbytes in self._candidates)

    def max_fill_transient_bytes(self, admitted_only: bool = True) -> int:
        """Largest build transient among the modules that will be filled
        (zero when none is admitted).  The fills happen lazily at the first
        forward, AFTER the budget's observation point, so this is what the
        budget must still have room for."""
        chosen = [m for _, m, _ in self._candidates
                  if not admitted_only or getattr(m, "cache_weight", False)]
        return max((fill_transient_bytes(m) for m in chosen), default=0)

    def finalize(self, cap_bytes=_UNSET) -> dict:
        cap = self.cap_bytes if cap_bytes is _UNSET else cap_bytes
        enabled = bool(cap)
        admitted, skipped, used = [], [], 0
        for path, module, nbytes in self._candidates:
            # Each module must fit with its OWN build transient: the fills
            # are sequential, so only one is ever live, but a module whose
            # fill does not fit would OOM at the first forward even though
            # its resident bytes fit the cap.
            fill = fill_transient_bytes(module)
            if enabled and used + nbytes + fill <= cap:
                module.cache_weight = True
                used += nbytes
                admitted.append({"path": path, "bytes": nbytes,
                                 "fill_transient_bytes": fill})
                continue
            if getattr(module, "cache_weight", False):
                module._dequantized = None      # dropped: release, not stale
            module.cache_weight = False
            skipped.append({
                "path": path, "bytes": nbytes,
                "reason": ("policy off" if not enabled else
                           f"does not fit: {used + nbytes + fill:,} "
                           f"(incl. {fill:,} build transient) > cap {cap:,}"),
            })
        self._summary = {
            "enabled": enabled,
            "requested_cap_bytes": (self.cap_bytes if self.cap_bytes is not None
                                    else cap),
            "effective_cap_bytes": cap,
            "admitted_bytes": used,
            "max_fill_transient_bytes": self.max_fill_transient_bytes(),
            "admitted": admitted,
            "skipped": skipped,
            "candidate_bytes_total": self.candidate_bytes_total(),
            "candidate_count": len(self._candidates),
            "finalized": True,
        }
        return self._summary

    def summary(self) -> dict:
        if self._summary is not None:
            return self._summary
        return {
            "enabled": self.enabled,
            "requested_cap_bytes": self.cap_bytes,
            "effective_cap_bytes": None,
            "admitted_bytes": 0,
            "max_fill_transient_bytes": 0,
            "admitted": [],
            "skipped": [],
            "candidate_bytes_total": self.candidate_bytes_total(),
            "candidate_count": len(self._candidates),
            "finalized": False,
        }


#: Process-wide policy, filled by ``io.load_model`` (one model per process).
WEIGHT_CACHE = WeightCachePolicy.from_env()


class QuantizedLinear(_tnn.Module):
    """``mlx.nn.QuantizedLinear`` layout (``weight`` packed uint32,
    ``scales``, ``biases``) dequantized on the fly -- the 4-bit payload
    stays resident, not a bf16 copy (unless ``CACHE_DEQUANTIZED`` or the
    ``WeightCachePolicy`` admitted this module)."""

    weight: torch.Tensor      # buffers (declared for type checkers)
    scales: torch.Tensor
    biases: torch.Tensor

    def __init__(self, weight, scales, biases, in_features, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = weight.shape[0]
        self.bits, self.group_size = _quant_params(weight, scales, in_features)
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("biases", biases)
        self.bias = None if bias is None else _tnn.Parameter(bias, False)
        self._dequantized = None          # (dtype, weight), see forward
        #: set by WeightCachePolicy.finalize; priced at the scales' itemsize
        self.cache_weight = False
        self._priced_itemsize = scales.element_size()
        self._int4pack = None             # (tiled weight, scales_and_zeros)
        #: falsy while usable; the reason string once the kernel proved
        #: unusable on this device (never retried afterwards)
        self._int4pack_disabled = ""

    ROWS_PER_CHUNK = 4096

    def int4pack_bytes(self) -> int:
        """Extra resident bytes of the repacked int4 payload (0 until the
        kernel path has been used)."""
        if self._int4pack is None:
            return 0
        return _i4.int4pack_extra_bytes(self.out_features, self.in_features,
                                        self.group_size)

    def _int4pack_eligible(self, x) -> bool:
        if self._int4pack_disabled or x.dtype is not torch.bfloat16 \
                or x.device.type != "cuda":
            return False
        if self._int4pack is not None:
            return True
        if self.weight.device.type != "cuda" or x.shape[-1] != self.in_features:
            return False
        ok, _reason = _i4.is_supported(
            in_features=self.in_features, out_features=self.out_features,
            group_size=self.group_size, bits=self.bits,
            device_type=x.device.type, x_dtype=x.dtype)
        if not ok:
            return False
        # Shapes and dtypes are not enough: prove the device has an image
        # for the kernel before committing to it (once per module).
        usable, reason = _i4.probe(x.device)
        if not usable:
            self._int4pack_disabled = reason
            return False
        return True

    def forward(self, x):
        """Dequantize ``ROWS_PER_CHUNK`` output rows at a time: the full
        float weight of a large projection is never materialized (edge0-8b's
        lm_head alone would be ~1 GB per call). Same arithmetic per
        element as dequantizing everything first."""
        from edge0.backends.cuda.quant import _dequantize
        if CACHE_DEQUANTIZED or (self.cache_weight
                                 and x.element_size() == self._priced_itemsize):
            # Same weight every call: dequantize once.  Fill it CHUNKED
            # into a preallocated buffer -- element for element the same
            # arithmetic as the on-the-fly path below, but bounding the
            # build transient to one chunk: whole-weight dequantization of
            # edge0-8b's lm_head peaks at 3.7 GB for a 0.46 GB result,
            # which does not fit an 8 GB Orin's headroom
            # (``fill_transient_bytes`` prices what this costs instead).
            if self._dequantized is None or self._dequantized[0] != x.dtype:
                w = torch.empty(self.out_features, self.in_features,
                                dtype=x.dtype, device=self.weight.device)
                for r in range(0, self.out_features, self.ROWS_PER_CHUNK):
                    sl = slice(r, r + self.ROWS_PER_CHUNK)
                    w[sl] = _dequantize(self.weight[sl], self.scales[sl],
                                        self.biases[sl], self.group_size,
                                        self.bits).to(x.dtype)
                self._dequantized = (x.dtype, w)
            b = None if self.bias is None else self.bias.to(x.dtype)
            return F.linear(x, self._dequantized[1], b)
        if _i4.INT4PACK and self._int4pack_eligible(x):
            try:
                if self._int4pack is None:
                    self._int4pack = _i4.repack(self.weight, self.scales,
                                                self.biases, self.group_size)
                tiled, saz = self._int4pack
                return _i4.int4pack_linear(x, tiled, self.group_size, saz,
                                           bias=self.bias)
            except RuntimeError as exc:   # no image, or a shape the kernel
                self._int4pack = None     # rejects at runtime: fall through
                self._int4pack_disabled = f"{type(exc).__name__}: {exc}"
        outs = []
        for r in range(0, self.out_features, self.ROWS_PER_CHUNK):
            sl = slice(r, r + self.ROWS_PER_CHUNK)
            w = _dequantize(self.weight[sl], self.scales[sl], self.biases[sl],
                            self.group_size, self.bits).to(x.dtype)
            b = None if self.bias is None else self.bias[sl].to(x.dtype)
            outs.append(F.linear(x, w, b))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


class QuantizedEmbedding(_tnn.Module):
    """``mlx.nn.QuantizedEmbedding`` layout; only the looked-up rows are
    dequantized. Output dtype: ``dtype``, else the scales' (the
    checkpoint's)."""

    weight: torch.Tensor      # buffers (declared for type checkers)
    scales: torch.Tensor
    biases: torch.Tensor

    def __init__(self, weight, scales, biases, embedding_dim, dtype=None):
        super().__init__()
        self.num_embeddings = weight.shape[0]
        self.embedding_dim = embedding_dim
        self.bits, self.group_size = _quant_params(weight, scales,
                                                   embedding_dim)
        self.out_dtype = dtype or scales.dtype
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("biases", biases)

    def forward(self, ids):
        from edge0.backends.cuda.quant import _dequantize
        # Gather on the int32 view of the packed rows: CUDA has no index
        # kernel for uint32 ("index_cuda not implemented for UInt32"), and
        # _dequantize reads the words as int32 anyway.
        rows = _dequantize(self.weight.view(torch.int32)[ids],
                           self.scales[ids],
                           self.biases[ids], self.group_size, self.bits)
        return rows.to(self.out_dtype)


def silu(x):
    return F.silu(x)


def gelu(x):
    return F.gelu(x)
