"""CUDA backend: quantized gather kernels (torch reference implementation).

Semantics match ``mx.gather_qmm`` as checked against real MLX 0.30.4
(Metal, the version this repo pins) in ``tests/test_cuda_backend.py``:

* Packing: each uint32 word holds ``32 // bits`` codes, least significant
  bits first, along the last (input) axis. Codes are unsigned; a group of
  ``group_size`` codes dequantizes as ``w = code * scale + bias``. Scales
  can be negative -- nothing here assumes otherwise.
* Broadcasting: the output batch shape is
  ``broadcast(x.shape[:-2], rhs_indices.shape)`` and
  ``out[b] = x[b] @ W[rhs_indices[b]].T``. Both call patterns in
  ``streaming/layer.py`` rely on this: the unsorted path passes
  ``x[..., 1, 1, D]`` against ``rhs_indices[..., K]``, the sorted path
  passes ``x[T*K, 1, D]`` against ``rhs_indices[T*K]``.
* ``sorted_indices`` is a kernel hint in MLX; it never changes the values.
  Ignored here.

Two implementations share those semantics:

* the reference loop: one distinct expert at a time, each dequantized in
  full to float32 and multiplied -- correct, testable, and the one every
  other path is compared against.  ``EDGE0_QMM_BATCHED=0`` selects it;
* the batched path (Task 6, **default on this backend**): the same
  float32 arithmetic with ONE dequantization over the distinct experts
  and ONE batched matmul over a padded per-expert slab, instead of a
  python loop of ``len(unique)`` dequantize+matmul pairs.  Only affine
  4-bit takes it; 2/8-bit layouts and any call whose priced transient
  exceeds ``EDGE0_QMM_BATCHED_MAX_BYTES`` (default 256 MiB) fall through
  to the reference loop.  The transient is bounded and priced
  (``batched_transient_bytes``) so the Task 4 budget deducts it.

The default was flipped on the Orin's Task 6 acceptance evidence, which
is what Gate D asks for before a path becomes default: measured benefit
on the target (decode +21%, three independent launches) plus the
pre-registered 32-token reference check PASSING there (32/32 token
choices identical, max |delta logit| 0.777 against the registered bound
of 1.0) -- and a second GPU generation agreeing in direction (+59% on an
RTX PRO 6000).  The arithmetic is the same expression per element; it
lands within one bfloat16 ulp of the reference loop at the real decode
shapes, which is why the check passes rather than merely not crashing.
"""

from __future__ import annotations

import os

import torch

BATCHED_ENV = "EDGE0_QMM_BATCHED"
BATCHED_MAX_BYTES_ENV = "EDGE0_QMM_BATCHED_MAX_BYTES"
DEFAULT_BATCHED_MAX_BYTES = 256 << 20


def _read_batched_gate() -> bool:
    """On by default (see the module docstring for the evidence that
    flipped it); ``EDGE0_QMM_BATCHED=0`` selects the reference loop.
    Any other value is treated as unset, so a typo cannot silently
    change the path."""
    return os.environ.get(BATCHED_ENV, "").strip() != "0"


def _read_batched_max_bytes() -> int:
    raw = os.environ.get(BATCHED_MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_BATCHED_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{BATCHED_MAX_BYTES_ENV} must be an integer byte count, got "
            f"{raw!r}") from None
    if value < 0:
        raise ValueError(f"{BATCHED_MAX_BYTES_ENV} must be >= 0, got {value}")
    return value


#: Read once at import, like ``nn.CACHE_DEQUANTIZED``; tests patch the
#: module attributes.  A malformed cap fails the import visibly.
BATCHED = _read_batched_gate()
BATCHED_MAX_BYTES = _read_batched_max_bytes()


def _dequantize(w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                group_size: int, bits: int) -> torch.Tensor:
    """Packed ``[..., rows, in * bits / 32]`` uint32 -> float32 ``[..., rows, in]``."""
    per_word = 32 // bits
    # int32 is enough: >> sign-extends, but the mask keeps only the low
    # ``bits`` bits, which the sign bits never reach (int64 doubled the
    # transient memory -- 1.9 GB of codes for edge0-8b's lm_head).
    words = w.view(torch.int32)
    shifts = torch.arange(per_word, device=w.device, dtype=torch.int32) * bits
    codes = (words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    codes = codes.reshape(*w.shape[:-1], w.shape[-1] * per_word)
    grouped = codes.reshape(*codes.shape[:-1], -1, group_size).to(torch.float32)
    deq = (grouped * scales.to(torch.float32).unsqueeze(-1)
           + biases.to(torch.float32).unsqueeze(-1))
    return deq.reshape(codes.shape)


def batched_transient_bytes(n_unique: int, rows_max: int, m: int, d_in: int,
                            n_out: int, bits: int = 4, group_size: int = 64,
                            scale_itemsize: int = 2) -> int:
    """Upper bound on the extra bytes ``_gather_qmm_batched`` allocates for
    one call (freed on return): the gathered packed words, the int32 code
    tensors and float32 dequantized copies the ``_dequantize`` chain keeps
    live (four float32-sized tensors of ``n_unique * n_out * d_in``), the
    gathered scales/biases and their float32 casts, and the padded
    input/output slabs.  Pure arithmetic, checked against allocator peaks
    in ``tests/test_quant_paths.py``."""
    for name, value in (("n_unique", n_unique), ("rows_max", rows_max),
                        ("m", m), ("d_in", d_in), ("n_out", n_out),
                        ("group_size", group_size),
                        ("scale_itemsize", scale_itemsize)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    per_expert = n_out * d_in
    packed = n_unique * per_expert * bits // 8
    chain = 4 * n_unique * per_expert * 4
    groups = -(-d_in // group_size)
    scale_terms = 2 * n_unique * n_out * groups * (scale_itemsize + 4)
    slabs = 2 * n_unique * rows_max * m * (d_in + n_out) * 4
    return packed + chain + scale_terms + slabs


def _gather_qmm_reference(x, w, scales, biases, rhs_indices, transpose,
                          group_size, bits):
    """One distinct expert at a time (the reference every other path is
    compared against).  Dequantizing a copy per (token, expert) pair
    peaked at several GB per MoE layer during prefill, hence per expert.
    An index past the last expert raises here; in MLX it silently reads
    out of bounds."""
    idx = rhs_indices.to(torch.long)
    bshape = torch.broadcast_shapes(x.shape[:-2], idx.shape)
    M = x.shape[-2]
    xf = x.to(torch.float32).expand(*bshape, *x.shape[-2:]).reshape(-1, M, x.shape[-1])
    flat = idx.expand(bshape).reshape(-1)
    n_out = w.shape[-2] if transpose else w.shape[-1] * (32 // bits)
    out = torch.empty(flat.numel(), M, n_out, dtype=torch.float32, device=x.device)
    for e in torch.unique(flat).tolist():
        rows = (flat == e).nonzero().squeeze(-1)
        deq = _dequantize(w[e], scales[e], biases[e], group_size, bits)
        out[rows] = torch.matmul(xf[rows], deq.T if transpose else deq)
    return out.reshape(*bshape, M, n_out).to(x.dtype)


def _gather_qmm_batched(x, w, scales, biases, rhs_indices, transpose,
                        group_size, bits):
    """Same result as the reference loop with one dequantization and one
    batched matmul; ``None`` when the priced transient exceeds the cap
    (the caller then runs the reference loop)."""
    idx = rhs_indices.to(torch.long)
    bshape = torch.broadcast_shapes(x.shape[:-2], idx.shape)
    M = x.shape[-2]
    D = x.shape[-1]
    n_out = w.shape[-2] if transpose else w.shape[-1] * (32 // bits)
    flat = idx.expand(bshape).reshape(-1)
    R = flat.numel()
    if R == 0:
        return torch.empty(*bshape, M, n_out, dtype=x.dtype, device=x.device)
    uniq, inverse = torch.unique(flat, return_inverse=True)
    n_experts = w.shape[0]
    if int(uniq.max()) >= n_experts:
        raise IndexError(
            f"expert index {int(uniq.max())} out of range for {n_experts} "
            f"experts")
    U = uniq.numel()
    counts = torch.bincount(inverse, minlength=U)
    rows_max = int(counts.max())
    priced = batched_transient_bytes(U, rows_max, M, D, n_out, bits,
                                     group_size, scales.element_size())
    if priced > BATCHED_MAX_BYTES:
        return None

    xf = x.to(torch.float32).expand(*bshape, M, D).reshape(-1, M, D)
    # One dequantization over the distinct experts.  Index the int32 view:
    # CUDA has no index kernel for uint32 (see nn.QuantizedEmbedding), and
    # _dequantize reads the words as int32 anyway.
    deq = _dequantize(w.view(torch.int32)[uniq], scales[uniq], biases[uniq],
                      group_size, bits)                        # [U, out, in]
    wmat = deq.transpose(1, 2) if transpose else deq          # [U, in, out]
    order = torch.argsort(inverse, stable=True)               # rows by expert
    grouped = inverse[order]
    if rows_max == 1:
        # decode: every distinct expert has exactly one row, no padding
        y = torch.matmul(xf[order], wmat)                     # [U, M, out]
        out = torch.empty(R, M, n_out, dtype=torch.float32, device=x.device)
        out[order] = y
        return out.reshape(*bshape, M, n_out).to(x.dtype)
    starts = torch.cumsum(counts, 0) - counts                 # [U]
    slot = torch.arange(R, device=flat.device) - starts[grouped]
    slab = xf.new_zeros(U, rows_max, M, D)
    slab[grouped, slot] = xf[order]
    y = torch.matmul(slab.reshape(U, rows_max * M, D), wmat)
    y = y.reshape(U, rows_max, M, n_out)[grouped, slot]        # [R, M, out]
    out = torch.empty(R, M, n_out, dtype=torch.float32, device=x.device)
    out[order] = y
    return out.reshape(*bshape, M, n_out).to(x.dtype)


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
               group_size=64, bits=4, mode="affine",
               sorted_indices=False):
    """Quantized matmul over a gathered subset of experts (``mx.gather_qmm``)."""
    if mode != "affine" or bits not in (2, 4, 8):
        raise NotImplementedError(
            f"reference gather_qmm covers affine 2/4/8-bit only "
            f"(got mode={mode!r}, bits={bits!r})")
    if BATCHED and bits == 4:
        out = _gather_qmm_batched(x, w, scales, biases, rhs_indices,
                                  transpose, group_size, bits)
        if out is not None:
            return out
    return _gather_qmm_reference(x, w, scales, biases, rhs_indices,
                                 transpose, group_size, bits)


def gather_sort(x, indices):
    """Sort token rows by expert id for ``gather_qmm(sorted_indices=True)``:
    returns ``(x_sorted, indices_sorted, inv_order)``, as mlx-lm's
    ``_gather_sort`` does."""
    m = indices.shape[-1]
    flat = indices.reshape(-1).to(torch.long)
    order = torch.argsort(flat, stable=True)
    inv_order = torch.argsort(order)
    return x.flatten(0, -3)[order // m], flat[order], inv_order


def scatter_unsort(x, inv_order, shape=None):
    """Undo ``gather_sort``; ``shape`` re-splits the leading axis."""
    x = x[inv_order]
    if shape is not None:
        x = x.unflatten(0, tuple(shape))
    return x


def swiglu(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SiLU gated activation: silu(gate) * up."""
    return torch.nn.functional.silu(gate) * up
