# Task 6 handoff: quantized-kernel development on a workstation GPU

Written 15 September 2026 on the target Orin, for kernel development on
a separate CUDA workstation (RTX 6000-class). Read
[`jetson-orin-nano.md`](jetson-orin-nano.md) Task 6 and Gate D first —
this document adds the measured profile that picks the work and the
division of labor between the two machines. Branch:
`feat/jetson-orin-nano-foundation` (Tasks 1–5 complete; Task 3–5
statuses in the plan carry the Orin evidence).

## Why this split

Kernel iteration (nvcc/CUTLASS builds, parity sweeps) is slow and
memory-cramped on the 8 GB Orin. Correctness of an int4 kernel is
arch-portable and can be developed anywhere torch + CUDA runs; sm_87
SASS cross-compiles from any x86 CUDA 12/13 toolchain
(`-gencode arch=compute_87,code=sm_87` — no Jetson needed to build).
What does NOT transfer is performance: an RTX 6000 (discrete GDDR,
different SM generation) has a different bottleneck shape than the
unified-LPDDR5 sm_87 board, exactly the way the plan forbids projecting
GB10 numbers. So: **develop and parity-test there, decide and accept
here.**

## The measured Orin profile (what to optimize)

Decode is compute-bound (GPU 99%; transfers measured near-irrelevant —
plan Task 5 status). Per-op shares of one decode step, synchronized
timers, unoptimized profile (`EDGE0_TORCH_WEIGHT_CACHE=0`; shares are
the signal, absolute times are sync-inflated):

| Op | Share | Calls per step | What it is |
|---|---:|---:|---|
| `backends/cuda/quant.py::gather_qmm` | **37%** | 69 (23 layers × 3 projections) | routed experts: python loop over ≤8 experts, each dequantized in full to float32, then matmul |
| `backends/cuda/nn.py::QuantizedLinear.forward` | **26%** | 235 | dense attention / shared expert / router / lm_head: dequantize per call, then linear |
| everything else | 37% | — | attention, routing, sampling, expert loads, sync overhead |

Both consumers read the same storage format: MLX affine int4,
group_size 64, codes packed 8-per-uint32 little-end-first along the
input axis, `w = code * scale + bias`, scales/biases bf16, **scales can
be negative and there is no zero-point symmetry** — see the layout
notes at the top of `backends/cuda/quant.py`, verified against real
`mx.quantize` output in `tests/test_cuda_backend.py`.

## Candidate designs (in rising effort order)

A. **Repack to torch's built-in int4 kernel.** `torch.ops.aten.
   _weight_int4pack_mm` (the gpt-fast / torchao path) runs group-wise
   int4×bf16 on CUDA. One-time repack at load (MLX layout → tensor-core
   tiled layout; MLX's `scale/bias` maps to the kernel's
   `scales_and_zeros`) would accelerate `QuantizedLinear` with no
   custom extension. Check sm_87 support in the pinned torch build
   FIRST (the Orin runs the official `2.14.0+cu130` aarch64 wheel);
   verify the affine (bias, not zero-point) mapping is exact, not
   approximate.
B. **Batch the expert loop.** `gather_qmm` currently dequantizes and
   matmuls one distinct expert at a time in python. A batched variant
   (dequantize the ≤8 gathered experts into one `[E, out, in]`
   transient, one `bmm`) cuts launch count ~8× at the cost of a
   bounded, budget-priceable transient. Pure torch, no extension —
   cheap to try before any custom kernel.
C. **Custom fused kernel** (CUDA C++/CUTLASS extension): int4 dequant
   in registers fused into the matmul, covering both the gathered
   (`rhs_indices`) and dense cases. Highest ceiling, highest cost; pin
   the extension/compiler combination and prove runtime loading on the
   Orin stack per the plan.

Recommendation: try A and B first — they are hours, not days, and their
Orin acceptance can run immediately.

## Workstation setup

```bash
git clone <repo> && cd Edge0
git checkout feat/jetson-orin-nano-foundation
python3.12 -m venv .venv
.venv/bin/pip install "numpy>=1.24" "safetensors>=0.4" "tokenizers>=0.15" \
    "psutil>=5.9" "pytest>=8.0" transformers
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -e . --no-deps        # keep MLX out
.venv/bin/python -m pytest -q               # expect: all pass, MLX suites skip
.venv/bin/python -m pytest tests/test_torch_cuda_smoke.py --require-cuda -q
```

Real weights are optional for kernel work (the parity tests build
synthetic quantized tensors); for end-to-end runs fetch
`Edge0/Edge0-8B-A1B-preview` (~4.2 GB) per the README and export
`EDGE0_BACKEND=cuda EDGE0_8B_MODEL=...`.

## Deliverables (what to bring back)

1. The kernel/path behind a **guarded dispatch**: only supported
   dtype/shape/stride/transpose/group-size/device combinations take the
   new path; everything else — including 2-bit and 8-bit layouts —
   falls through to the current reference implementation unchanged.
   Off by default (env-gated), like every Task 5 knob.
2. **Parity tests in the smoke-suite style** (hand-computed references,
   no MLX): every broadcast/gather shape already covered in
   `tests/test_cuda_backend.py`, index boundaries, negative scales,
   quantization extremes, all checkpoint dtypes. Tolerances DEFINED IN
   THE TEST before any Orin run (the Task 3 reference check shows the
   format for pre-registering them).
3. Build pinned and cross-arch: compile for the workstation's arch AND
   `compute_87`; record the exact toolchain versions. For design A,
   record the repack transform and its inverse test.
4. Scratch/transient memory of the new path measured and stated in
   bytes, so the Task 4 budget can price it (`streaming/budget.py`
   takes it as a deduction or per-call transient).

## What stays on the Orin (do not claim from the workstation)

* Which design wins, and any tok/s number: re-run the profile above and
  the 3-launch benchmark protocol (`docs/nvidia.md`, Task 3 status) on
  the Orin only.
* Runtime loading of any compiled extension on the L4T R39 / CUDA 13.2
  / torch 2.14.0+cu130 stack.
* The 32-token reference check (`scripts/orin_reference_check.py`)
  with the new path enabled — token choices must stay explainable
  under the pre-registered comparison, same as Task 3.
* The Task 5 re-measurement: once compute shrinks, transfers matter
  more — `EDGE0_CACHE_SLOTS` / `EDGE0_PREDICT_PREFETCH` (+7.7%
  measured pre-Task-6) and the deferred pinned-slot pipeline get
  re-evaluated then.
