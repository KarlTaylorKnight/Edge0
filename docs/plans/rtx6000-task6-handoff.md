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

## Workstation results (15 September 2026)

Workstation: RTX PRO 6000 Blackwell (sm_120, 96 GB), Windows 11; venv
Python 3.12 + **torch 2.14.0+cu130** (the Orin's torch version, x86
build) for every GPU measurement; a torch-CPU venv (Python 3.11) and a
Python 3.9 venv for portability; WSL Ubuntu 24.04 for the Linux checks.
Nothing below is an Orin number.

### Delivered (all opt-in, off by default, guarded, priced)

| Design | Knob | Files | Status |
|---|---|---|---|
| B batched expert gather | `EDGE0_QMM_BATCHED=1`, `EDGE0_QMM_BATCHED_MAX_BYTES` | `backends/cuda/quant.py` | exact (float32, parity 1e-4 to a float64 hand reference, 1e-5 to the loop); 4-bit affine only, 2/8-bit and over-cap calls fall through; transient priced by `batched_transient_bytes` and asserted against allocator peaks in `tests/test_quant_paths.py`; kernel launches per call (reference loop vs batched) counted with `torch.profiler`, >= 3x fewer asserted |
| A' capped exact weight cache | `EDGE0_TORCH_WEIGHT_CACHE_BYTES=<n>` | `backends/cuda/nn.py` (`WeightCachePolicy`), `backends/cuda/io.py`, `engine/ling.py` | first-fit in checkpoint order, each module admitted only if its resident bytes AND its own build transient fit; engine shrinks the cap to the budget headroom. The dequantized weight is bit-identical to the on-the-fly path (same per-chunk expression, asserted on CUDA); the cached forward issues one GEMM instead of one per 4096-row chunk, which only the two modules wider than a chunk (lm_head, layer-0 dense gate/up) can notice. **The fill is chunked**: whole-weight dequantization of lm_head (157184 x 1536) peaks at 3702 MiB for a 460.5 MiB result and would OOM the Orin's stated headroom; chunked it measures 556.9 MiB, and `fill_transient_bytes` prices that. The full cache shares the branch and inherits the fix; it is priced from the loader's measured total (235 cacheable linears, 1.402 GB bf16 — routers are not quantized here and `word_embeddings` is a lookup-only `QuantizedEmbedding`) plus one fill transient |
| A torch int4 kernel | `EDGE0_INT4PACK=1` | `backends/cuda/int4pack.py`, `backends/cuda/nn.py` | runs on torch 2.14.0+cu130; **approximate**: per-weight error median 3.7e-4 / p99 3.7e-3 / max 1.1e-2 (n=512, k=1536, MLX-like statistics), 68% of weights identical to the reference's bf16 weights, outputs within 3.5e-3 of the output scale; bounds registered in `tests/test_int4pack.py` (1e-2 of output scale; 2^-6 (max\|w\| + max\|zero\|) per weight). Constraints verified on this build: bf16 activations only, `in_features % 128 == 0` (every edge0-8b dense k qualifies: 256, 512, 1536, 2048, 4608), `out_features % 8 == 0` up to the 157184-row lm_head; the dequantization is exactly `(q - 8) * scale + zero` on an identity probe. **sm_87 is not answered from a table**: `int4pack.probe` executes the kernel once per module on the target device before committing, and any runtime failure disables the path for that module — so a wheel without an image for the Orin's arch falls through instead of raising. Extra resident bytes: the repacked payload, 0.351 GB for all dense linears; originals stay for the fall-through paths |
| C custom fused kernel | — | — | not written; toolchain proven: WSL `nvcc` 13.0.88 + g++ 13.3.0 compile `-gencode arch=compute_87,code=sm_87 -gencode arch=compute_120,code=sm_120` and the binary runs on this GPU; Windows alternative: VS 2022 BuildTools + CUDA 12.8 |

Budget: `streaming/budget.py` gained `Reserves.kernel_transient_bytes`
(the batched gather's cap, deducted once when the path is on) and
`dense_cache_request`, which prices the dense cache as resident bytes
plus one build transient (`full` keeps its veto, `capped` shrinks to the
headroom, `off` prices nothing); `engine/ling.py` calls it and
re-finalizes the policy after the budget resolves.
Report: `examples/bench.py` records `caches.quant_paths` (batched gate
and cap, the policy's admitted/skipped modules with their bytes and fill
transients, int4pack repacked modules and extra bytes) and echoes the
four knobs.
Reference check: `scripts/orin_reference_check.py --test-env KEY=VALUE`
/ `--reference-env` run the two phases with different knobs on the same
device. The report now carries **host identity** (GPU name, compute
capability, torch build, `/proc/device-tree/model`, machine, hashed
hostname) and each phase's **resolved** knobs, not just the CLI
overrides — a workstation report can no longer be mistaken for an Orin
one, and a same-device run whose two phases resolve identical knobs
warns that it is comparing a configuration with itself. The schema is
`edge0-reference-check/2`.

### Same-device reference checks on the workstation (real weights)

`scripts/orin_reference_check.py models/edge0-8b --test-device cuda
--reference-device cuda --test-env <KNOB>` — the reference path
teacher-forces the tokens the knob's path generated, on the same GPU, so
the only variable is the knob. 32 greedy tokens, edge0-8b, RTX PRO 6000:

| Knob | Token choices | Divergences | max abs logit diff | Verdict against the registered bound (1.0) |
|---|---|---|---|---|
| `EDGE0_TORCH_WEIGHT_CACHE_BYTES=2000000000` | 32/32 identical | 0 | **0 exactly** | PASS — bit-identical through the whole model, the strongest form of the exactness claim |
| `EDGE0_QMM_BATCHED=1` | 32/32 identical | 0 | 1.01 | FAIL of that bound, reported as such |
| `EDGE0_INT4PACK=1` | 32/32 identical | 0 | 2.06 | FAIL of that bound, reported as such |

All three keep every token choice with zero near-ties consumed. The two
that exceed the bound are **not** loosened: as in Task 3, the bound's
failure is recorded. For the batched gather the mechanism is identified —
at the real decode shapes it lands within one bfloat16 ulp of the
reference loop (`tests/test_quant_paths.py` asserts exactly that), and 24
layers of one-ulp differences accumulate; note the Task 3 CUDA-vs-CPU
baseline measured 2.06 by the same mechanism, so 1.01 sits below the
noise the tier already exhibits. For the int4 kernel the difference is
expected by construction (bf16 zero-point and bf16 dequantization). The
Orin's own reference check decides acceptability; a top-k-scoped
criterion is the change to propose in review first, per the Task 3 status.

**A caution about this machine's shape.** Decode here runs at roughly
1 tok/s against the Orin's 0.55 — under 2x, on a GPU with vastly more
compute. That is the handoff's point restated as data: the bottleneck is
launch and host overhead, not arithmetic, so a workstation speedup would
not transfer. No tok/s comparison between configurations is reported
here for that reason; only launch counts, bytes and parity.

### The finding that reshapes the dense half of Task 6

**Most dense `QuantizedLinear` calls run with float32 activations, so
both dense paths engage on a small minority of them.** Traced on the real
model, one prefill plus one decode step: **400 of 470 calls arrive as
float32, only 70 as bfloat16.** Both dense knobs require bf16 — the cache
because it is priced at the checkpoint's scales itemsize, the int4 kernel
because `_weight_int4pack_mm` accepts nothing else. So:

* the weight cache admits 235 modules / 1.402 GB but **actually fills 35
  modules / 0.174 GB**; the reservation is conservative and never
  under-reserves, and the bench report now states `filled_bytes` beside
  `admitted_bytes` so the two are never confused;
* the int4 kernel repacks **exactly those same 35 modules** — that is
  where its `repacked_modules: 35` comes from, not from a shape guard;
* neither knob moves decode measurably on this GPU (0.95 and 0.96 against
  a 0.92 baseline), which is what covering 70 of 470 calls predicts.

This is the thing to settle before spending more effort on design A or C:
the dense half of the 26% is currently gated by **activation dtype, not by
the kernel**. Whether those float32 call sites can run bf16 (and what that
costs numerically) is a model-path question for the Orin session, and it
governs how much of that 26% any kernel can reach. The batched expert
gather is unaffected — it covers the routed 37% and engages on every call.

### What the benchmarks showed (workstation only, 3 launches x 2 runs each)

| Configuration | decode tok/s (6 runs) | CUDA alloc peak | Engaged |
|---|---|---|---|
| baseline | 0.90-0.95 (mean 0.92) | 1.02 GiB | — |
| `EDGE0_QMM_BATCHED=1` | 1.41-1.52 (mean 1.46) | 1.02 GiB | every gather call |
| `EDGE0_TORCH_WEIGHT_CACHE_BYTES=1.5e9` | 0.94-0.96 (mean 0.95) | 1.18 GiB | 35/235 modules filled |
| `EDGE0_INT4PACK=1` | 0.95-0.96 (mean 0.96) | 1.07 GiB | 35 modules repacked |
| batched + weight cache | 1.46-1.47 (mean 1.46) | 1.18 GiB | both |

Every configuration ran to completion with the budget active, no OOM.
**Do not project the batched gather's 1.59x onto the Orin.** The
mechanism (fewer kernel launches) is the same one that dominates the
Orin's decode, but this is a different SM generation with different
launch costs, and the plan forbids projecting between machines — the
whole reason this workstation decodes at only ~1 tok/s against the
Orin's 0.55 despite far more compute. The number is here as evidence the
path works end to end, not as a prediction.

### How the Orin accepts a knob

1. `EDGE0_BACKEND=cuda EDGE0_TORCH_DEVICE=cuda python -m pytest tests/test_quant_paths.py tests/test_weight_cache_policy.py tests/test_int4pack.py tests/test_task6_budget.py --require-cuda -q` on the board (the CUDA cases run for real there).
2. `scripts/orin_reference_check.py edge0-8b --test-device cuda --reference-device cuda --test-env <KNOB>=<value> --output "$EDGE0_RUN_DIR/ref-<knob>.json"` — the reference path vs the knob on the same device; the exact knobs must match 32/32 tokens with logits at float32-noise level, the int4 kernel is judged by the pre-registered bounds and the near-tie rule.
3. The three-launch benchmark protocol (Task 3 status) with `EDGE0_MEMORY_BUDGET=auto` so the transient / cache bytes are priced, comparing against the Task 5 baseline; then the Task 5 re-measurement.
