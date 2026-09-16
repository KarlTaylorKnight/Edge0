# NVIDIA / CUDA support — status

## Jetson Orin Nano bring-up

**Status (14 September 2026): edge0-8b runs on a physical Orin Nano 8 GB.**
The unoptimized torch reference path generated coherent 32-token greedy
output on the device's GPU, with every token choice identical to the
torch-CPU reference on the same board.  The claim is qualified to the
tested workload (37-token chat-templated prompt, 32 greedy tokens,
weight cache and prewarm off) and to what the correctness gate below
actually shows — see [Task 3 baseline](#task-3-orin-baseline-summary).

Before attempting inference on an Orin Nano, capture a machine-readable
capability report with the read-only probe:

```bash
EDGE0_BACKEND=cuda python scripts/jetson_probe.py \
  --model-dir "$EDGE0_8B_MODEL" \
  --output artifacts/orin-probe.json
```

The command exits non-zero and lists actionable blockers when it cannot confirm
a Jetson Orin, CUDA-enabled PyTorch, and NVMe-backed model storage. It reports
facts only; passing the probe does not imply that inference fits in 8 GB.

The staged implementation and measurement gates are documented in
[`docs/plans/jetson-orin-nano.md`](plans/jetson-orin-nano.md) (change
justifications in [`jetson-orin-nano-review.md`](plans/jetson-orin-nano-review.md)).
Start with the 8B tier and retain the existing correctness path while
optimizing memory and I/O. The benchmark's machine-readable report, which the
Orin baseline uses, is described in [Benchmark reporting](#benchmark-reporting)
below.

### Task 3 Orin baseline summary

Target: Jetson Orin Nano Developer Kit (Super), 8 GB (7.85 GB observed),
NVMe/ext4, L4T R39.2 (kernel 6.8.12-tegra), 25W power mode (observed, not
changed), Python 3.12.3.  Torch: the official **2.14.0+cu130 aarch64
wheel** — the same build the GB10 work used.  It warns that its SASS list
excludes `sm_87`; empirically the `sm_80` binaries run on this Orin
(CC 8.x binary compatibility), and the smoke suite below is the evidence
gate that decides this, not the warning or the docs.  Environment:
edge0 installed `pip install -e . --no-deps` plus
numpy/safetensors/tokenizers/psutil/transformers (5.17.0) — no MLX
package on the device; the 8B torch import path needs none.

Evidence chain (raw reports in the untracked evidence set, sanitized
summary here):

* **Probe** (`scripts/jetson_probe.py --model-dir …`): `ready: true`,
  no blockers; checkpoint manifest sha256 `a83dee78…` (48 files,
  4.60 GB).
* **CUDA acceptance** — `pytest tests/test_torch_cuda_smoke.py
  --require-cuda -q`: 6/6 pass on the device (matmul vs float64
  reference, packed-uint32 int4 `gather_qmm` vs a hand-computed
  dequantization, RMSNorm, unsigned-word view semantics, core ops).
  The suite FAILS rather than skips when CUDA is missing (verified
  with `CUDA_VISIBLE_DEVICES=""`).
* **32-token greedy correctness** (`scripts/orin_reference_check.py`,
  CUDA generation vs torch-CPU teacher-forced logits; torch-CPU is the
  path checked against MLX layer-by-layer on the GB10): **32/32 token
  choices identical, zero near-ties consumed** (smallest top-1→top-2
  margin 0.067, median 4.6).  The pre-registered vocab-wide logit
  bound (max |Δ| ≤ 1.0) was **exceeded: max |Δ| = 2.06** — reported as
  the FAIL it is, not re-run with a looser bound.  Post-hoc analysis
  (recorded in the evidence set): median |Δ| 0.087, p99 0.52; every
  excursion above 1.0 sits on deep-tail tokens (worst: reference rank
  429); at the reference's top-8 tokens of every step the difference
  is ≤ 0.22.  This is accumulated bfloat16 reduction-order noise, not
  a kernel defect; a top-k-scoped logit criterion should be registered
  in review before the next comparison run, and MLX-generated fixtures
  remain the intended reference once available.
* **Short baseline** (`BENCH_TEMP=0 BENCH_SEED=0 examples/bench.py
  edge0-8b --ntok 32 --warmup 0`, probe linked, tegrastats at 1 s
  alongside, three independent process launches, two internal runs
  each, `EDGE0_TORCH_WEIGHT_CACHE=0 EDGE0_PREWARM=0`):

  | Launch | Prefill (37 tok) | Decode (32 tok) | CUDA alloc peak | Peak RSS |
  |---|---|---|---|---|
  | 1 | 15.9 / 11.7 s | 0.53 / 0.55 tok/s | 0.92 / 1.00 GiB | 5.31 GiB |
  | 2 | 13.4 / 11.4 s | 0.55 / 0.55 tok/s | 0.92 / 1.00 GiB | 5.69 GiB |
  | 3 | 14.9 / 13.2 s | 0.56 / 0.58 tok/s | 0.92 / 1.00 GiB | 5.76 GiB |

  Six runs, descriptive statistics: decode mean **0.55 tok/s** (min
  0.53, max 0.58); engine build ~11.4 s.  GPU utilization pegged at
  99% during decode (tegrastats); no OOM, no swap activity; RSS and
  CUDA peaks are separate views (unified memory — RSS includes
  touched mmapped expert pages) and are not added.  A desktop session
  (~3.5 GB) was resident throughout — headroom, not a guarantee.

This is the **unoptimized reference implementation**: `gather_qmm`
dequantizes every gathered expert on every call, exactly the cost
Tasks 4–6 of the plan exist to remove (the GB10 measured 2.5× from the
dequantized-weight cache alone; it is off here pending Task 4's byte
budget).  Do not read 0.55 tok/s as the platform's capability.

### Memory-budgeted profile (Task 4)

`EDGE0_MEMORY_BUDGET=auto` (or an explicit byte count) makes the 8B
engine resolve an explicit byte budget immediately before it builds the
streaming caches: observed available RAM (tightened by any address-space
rlimit), minus an OS-growth reserve, an allocator allowance, KV at the
DECLARED context (`EDGE0_BUDGET_CONTEXT`, default 1024 tokens), the
whole-layer prefill transient and in-flight expert builds — every
payload priced from the checkpoint's safetensors header, not from slot
counts.  What remains bounds the shared LRU and the prefetch buffer in
bytes; the budget only ever LOWERS the tested profile, and it enforces
two bounds the unbudgeted path leaves open (`PrefetchBuffer.max_cap`
against `prefetch_all()`'s growth, `LayerOptions.max_inflight` against
the speculative build queue).

Impossible profiles fail BEFORE inference with the itemized arithmetic:
a context whose KV cannot fit, a cache allowance below the working set
(a calculated zero is rejected — `SharedExpertCache(0)` /
`PrefetchBuffer(0)` mean *unbounded*, never "off"), or
`EDGE0_TORCH_WEIGHT_CACHE=1` whose measured ~4.1 GB does not fit the
post-cache headroom.  The resolved policy (or the rejection) lands in
the bench report's `caches.memory_budget` group.  On the Orin Nano
8 GB baseline workload, `auto` resolves ≈ 4.2 GB usable, keeps the
tested 64-slot / 48-cap profile with ≈ 1.9 GB headroom, and measures
identically to the unbudgeted baseline; a declared 4096-token context
is rejected (KV alone ≈ 4.5 GB), as is the weight cache.  Unset, the
engine behaves exactly as before Task 4.

### Bounded expert reuse and predicted prefetch (Task 5, opt-in)

The measured Orin profile found the tested 64-slot LRU never hits on
this tier (working set 184 bundles/step) — and also that decode is
compute-bound, so transfers barely show end-to-end until the Task 6
kernel work lands.  Two opt-in knobs, both scheduling-only (token
output is bit-identical; wrong or late predictions fall back to demand
loads, never a staged zero row):

* `EDGE0_CACHE_SLOTS=512` — request a working-set-sized shared LRU
  (the budget may lower it).  85% hit rate on the baseline workload.
* `EDGE0_PREDICT_PREFETCH=1` — the prerouter's per-step predictions
  are prefetched for non-staged layers while the current forward runs
  (98.8% of predicted builds consumed, zero waited on, zero wasted).

Together on the baseline workload: decode 0.568–0.623 tok/s (mean
0.594) vs 0.53–0.58 (mean 0.552) — **+7.7%**, for ~0.6 GiB more
resident, inside the resolved budget.  Off by default until the
post-Task-6 re-measurement; the pinned-buffer CUDA-stream pipeline is
deferred on the same evidence (see the plan's Task 5 status).

### Quantized paths (Task 6, developed on a workstation GPU)

The Orin profile (plan Task 6 scoping) put 37% of a decode step in
`quant.gather_qmm` and 26% in the dense `QuantizedLinear.forward`, both
dequantize-then-matmul.  Three paths were developed and parity-tested
on an RTX PRO 6000 (torch 2.14.0+cu130, the Orin's torch version) per
[`plans/rtx6000-task6-handoff.md`](plans/rtx6000-task6-handoff.md).  All are
guarded (every unsupported case falls through to the reference
implementation unchanged), priced for the Task 4 budget and
recorded in the bench report's `caches.quant_paths` group.  Which one the
Orin adopts, and any tok/s, is decided on the Orin — see
[the acceptance](#task-6-orin-acceptance) below, which made
`EDGE0_QMM_BATCHED` the **default** on this backend and left the two
dense knobs opt-in.

| Knob | What it does | Exact? | Extra memory |
|---|---|---|---|
| `EDGE0_QMM_BATCHED=1` | `gather_qmm` dequantizes the distinct experts of a call ONCE and runs ONE batched matmul over a padded per-expert slab instead of a Python loop of dequantize+matmul pairs (decode: 8 pairs -> 1 + 1). Affine 4-bit only; 2/8-bit and any call whose priced transient exceeds `EDGE0_QMM_BATCHED_MAX_BYTES` (default 256 MiB) take the reference loop. | yes: same float32 arithmetic, parity 1e-4 to a float64 hand reference and 1e-5 to the loop | per-call transient only, priced by `quant.batched_transient_bytes`; the cap is deducted once by the budget |
| `EDGE0_TORCH_WEIGHT_CACHE_BYTES=<n>` | keeps the dequantized bf16 weight of the dense `QuantizedLinear` modules that fit `n` bytes, admitted first-fit in checkpoint order with each module's own build transient counted (`nn.WeightCachePolicy`); the engine shrinks `n` to the budget's post-cache headroom; the admitted module list is in the report | the dequantized weight is bit-identical to the on-the-fly path (same per-chunk expression, asserted on CUDA); the cached forward then issues one GEMM where the reference issues one per 4096-row chunk, so for the two modules wider than a chunk (lm_head, layer-0 dense gate/up) cuBLAS may split differently — identical on this GPU and on the CPU, and the reference check decides on the device | the admitted bytes plus one build transient at a time: the fill is chunked, so it costs `nn.fill_transient_bytes` (edge0-8b's widest module, lm_head 157184 x 1536: 460.5 MiB resident, 556.9 MiB measured peak) rather than the 3702 MiB a whole-weight dequantization peaks at. edge0-8b's cacheable dense linears total 1.40 GB in bf16: lm_head 0.46 GiB, attention and shared experts the rest. Routers are not quantized in this checkpoint and `word_embeddings` is a `QuantizedEmbedding` that dequantizes only the rows it looks up, so neither is a candidate |
| `EDGE0_INT4PACK=1` | dense `QuantizedLinear` through torch's built-in `_weight_int4pack_mm` after a one-time repack of the MLX layout (uint32 LSB-first codes -> uint8 high-nibble-first pairs, `zero = bias + 8 * scale`), lazily on the first eligible call; CUDA + bf16 activations + 4-bit + group 32/64/128/256 + `in_features % 128 == 0` + `out_features % 8 == 0` only, and `int4pack.probe` must execute the kernel once on the device first (the Orin's wheel warns `sm_87` is outside its SASS list, so execution evidence gates, not the warning); a runtime failure disables the path for that module permanently | **no**: `zero` is rounded to bf16 and the kernel dequantizes in bf16; measured (identity-vector probe, n=512, k=1536): per-weight error median 3.7e-4, p99 3.7e-3, max 1.1e-2; 68% of weights equal the reference's bf16-rounded weights, the rest within ~1-2 bf16 ulps; outputs within 3.5e-3 of the output scale. Pre-registered test bounds: 1e-2 of output scale, 2^-6 (max|w| + max|zero|) per weight | duplicates the int4 payload it covers (`int4pack_extra_bytes`) next to the original buffers; for all of edge0-8b's dense linears that is 0.35 GB |

Precedence inside `QuantizedLinear.forward`: full cache, then the capped
cache, then the int4 kernel, then the chunked on-the-fly dequantization —
exact paths first.

**Both dense knobs need bfloat16 activations, and most calls are not.**
Traced on the real edge0-8b model over one prefill plus one decode step,
400 of 470 dense `QuantizedLinear` calls arrive as float32 and only 70 as
bfloat16. The cache therefore fills 35 of the 235 modules it admits
(0.174 GB of the 1.402 GB reserved) and the int4 kernel repacks exactly
those same 35. The reservation stays conservative on purpose — it never
under-reserves — and the report states `filled_bytes` next to
`admitted_bytes` so residency is never overstated. The routed path
(`EDGE0_QMM_BATCHED`) has no such gate and engages on every call.  The batched gather touches only the routed-expert
path; the dense knobs touch only the dense path; nothing changes token
choice by construction except the int4 kernel, whose acceptability the
32-token reference check decides (`scripts/orin_reference_check.py` now
takes `--test-env KEY=VALUE` / `--reference-env` so the reference path and an
opt-in path can be compared on the SAME device, isolating the path from
device noise).

Tests (no MLX, hand-computed references, tolerances written before any Orin
run): `tests/test_quant_paths.py` (every broadcast shape of
`tests/test_cuda_backend.py`, 2/8-bit fall-through, index boundaries,
negative scales, code extremes, bf16 dtypes, transient cap fall-back, the
allocator-measured transient against the price, kernel-launch counts),
`tests/test_weight_cache_policy.py` (admission, caps, re-finalize, bit
identity, loader registration), `tests/test_int4pack.py` (exact repack round
trip, the zero mapping, every guard, the registered kernel bounds, the LoRA
wrapper, exact-cache precedence), `tests/test_task6_budget.py` (pricing).
The CUDA cases skip without a device and FAIL under `--require-cuda`.

Custom fused kernel (design C): not written. Toolchain recorded for the next
session: WSL Ubuntu nvcc 13.0.88 + g++ 13.3 compile
`-gencode arch=compute_87,code=sm_87 -gencode arch=compute_120,code=sm_120`
fat binaries that run on this GPU; Windows has VS 2022 BuildTools + CUDA
12.8 as an alternative.

### Task 6 Orin acceptance

Run on the target (Orin Nano 8 GB, L4T R39.2, CUDA 13.2, torch
2.14.0+cu130, 25W, budget active) on 16 September 2026.  Same workload
and instrumentation as the Task 3 baseline: 37-token prompt, 32 greedy
tokens, `BENCH_TEMP=0 BENCH_SEED=0`, three independent launches x two
internal runs, tegrastats alongside.

**Environment and correctness.** Full suite 341 passed on-device; the
CUDA suites pass under `--require-cuda` (96 tests).  `int4pack.probe`
**executes on sm_87** — the kernel the wheel's SASS list does not
advertise runs here, answered by execution as the design intended.

Same-device 32-token reference checks (`--test-env`, so the only
variable is the knob; the registered bound is Task 3's max |Δlogit| ≤ 1.0):

| Path | Token choices | max abs logit diff | Verdict |
|---|---|---|---|
| `EDGE0_QMM_BATCHED=1` | 32/32 identical | **0.777** | **PASS** |
| `EDGE0_TORCH_WEIGHT_CACHE_BYTES=1.5e9` | 32/32 identical | 1.58 | FAIL of that bound, recorded |
| `EDGE0_INT4PACK=1` | 32/32 identical | 2.17 | FAIL of that bound, recorded |

**Decode (6 runs each, mean ± sd):**

| Configuration | decode tok/s | mean | vs baseline | CUDA peak / RSS |
|---|---|---|---|---|
| baseline (Task 3) | 0.530–0.580 | 0.553 | — | 1.00 / 5.3–5.8 GiB |
| batched gather | 0.623–0.699 | **0.668** | **+20.7%** | 1.00 / 5.6–5.7 GiB |
| batched + Task 5 knobs | 0.743–0.815 | **0.785** | **+41.8%** | 1.58 / 5.8–5.9 GiB |
| capped weight cache (screen) | 0.517 | 0.517 | +0.8% | 1.16 GiB |
| int4 kernel (screen) | 0.524–0.540 | 0.532 | +3.7% | 1.07 GiB |

**Decision.** `EDGE0_QMM_BATCHED` is now **on by default** on this
backend (`EDGE0_QMM_BATCHED=0` selects the reference loop): it is the
only path that both passed the registered correctness bound on the
target and delivered a measured benefit there, it engages on every
gather call, its transient is bounded and the budget deducts the cap.
The two dense knobs stay **opt-in**: they move decode by ~1–4% because
of the dtype gate below, and both exceed the registered bound.

**The Task 5 re-measurement Gate D asked for.** With compute reduced,
transfers matter more, exactly as predicted: the Task 5 knobs were worth
+7.7% before Task 6 and are worth **+17.5% on top of the batched gather**
now (0.668 → 0.785), with expert load wall falling 72.5 s → 17.0 s over
the same six runs and 30,159 cache hits against 5,169 loads.  They remain
opt-in because they are board-specific tuning that needs the budget
active (`EDGE0_MEMORY_BUDGET=auto EDGE0_CACHE_SLOTS=512
EDGE0_PREDICT_PREFETCH=1` is the recommended Orin profile, and is what
the 0.785 figure uses).

**Two corrections to the workstation's report, found only by running here:**

1. *The capped weight cache is not bit-identical on this GPU.*  The
   workstation measured max |Δlogit| exactly 0; the Orin measures 1.58.
   Isolated: the cached dequantized weight IS bit-identical to the
   chunked path's (verified directly), but feeding those same weights
   through one GEMM versus the reference's per-4096-row GEMMs differs by
   0.03125 on a 30.6 output scale — one bf16 ulp of accumulation-order
   difference in cuBLAS's kernel choice for the two shapes.  The
   exactness claim is a property of the workstation's cuBLAS heuristics,
   not of the code; on sm_87 the path is exact in the weights and
   near-exact in the outputs.
2. *The float32 activation gate is an MLA/RoPE promotion, not an
   accident.*  Confirmed here (35 bf16 vs 200 float32 dense calls per
   decode step) and traced to its origin: layer 3 is the first
   `BailingMLA`, whose RoPE path computes in float32 and concatenates
   `q_nope.to(q_pe.dtype)` with the float32 `q_pe` — deliberate MLX
   promotion parity, commented as such in the vendored model.  From that
   layer on, the residual stream stays float32, so only layers 0–2 (KDA)
   present bf16 activations to their dense linears.  That is the 35
   modules, and it is a numerics-parity decision, not a kernel
   limitation: reaching the rest of the dense 26% means deciding to
   diverge from MLX's promotion in the MLA path, which belongs in review
   with its own reference check, not in a kernel increment.

## Benchmark reporting

`examples/bench.py` keeps its two-run protocol and its human-readable output
and can additionally write a machine-readable report:

```bash
EDGE0_BACKEND=cuda EDGE0_TORCH_DEVICE=cuda \
  python examples/bench.py edge0-8b --ntok 32 --warmup 0 \
  --probe-json "$EDGE0_RUN_DIR/orin-probe.json" \
  --json-output "$EDGE0_RUN_DIR/orin-short-baseline.json"
```

The pure builder (`examples/benchmark_report.py`, standard library only,
imports neither torch nor MLX) turns measured numbers into the schema; the
benchmark-owned collectors (`examples/benchmark_measure.py`) talk to the
backend, the process and the machine behind lazy, injectable imports.

### What is measured

* **Protocol** `fixed-length-sample-and-step`, version 1. Per run:
  `engine.reset()`, a timed chat-templated prefill, `--warmup` **greedy**
  (argmax) steps that are not timed, then `--ntok` timed iterations of "sample
  one token from the current logits, run one engine step". The final step's
  logits are computed but never sampled and the loop does not stop at EOS.
  None of the numbers is a time-to-first-token or an EOS-aware completion
  latency; the report's `protocol` block says so.
* **Counts** per run: `prompt_tokens` (after chat templating),
  `warmup_tokens`, `timed_decode_tokens`,
  `total_generated_tokens = warmup + timed`,
  `decode_start_context_tokens = prompt + warmup`; the requested counts are
  kept separately (`requested_*`). `decode_tokens_per_second` divides the
  timed tokens only. A zero measured duration gives `null` plus a reason,
  never a zero rate; `--ntok 0` is refused.
* **Wall time** is `time.perf_counter()` and includes storage, transfer and
  CPU sampling costs by design. On an actual CUDA device (`EDGE0_BACKEND=cuda`
  *and* a resolved `cuda` device; `EDGE0_TORCH_DEVICE=cpu`/`mps` is not one)
  the benchmark calls `torch.cuda.synchronize(device)` after the engine reset,
  at the end of prefill, at the end of warmup and after the final timed step:
  phase boundaries only, no per-token synchronization. On MLX nothing changes,
  the engine's `core.eval(logits)` already materializes every step. Torch on
  `mps` is not synchronized by the benchmark (the plan mandates CUDA only), so
  its wall times may exclude the final step's in-flight device work; the
  report says so in `runtime.measurement.synchronization`.
* **Memory** is integer bytes everywhere, each value with its `method` and
  `scope`. Per run (`runs[i].memory`): `cuda_peak_allocated` from the torch
  caching allocator, reset immediately before prefill and read after the
  synchronized decode. It covers prefill + warmup + decode, includes whatever
  was resident at the reset (recorded as `allocated_at_reset`), does not
  recover an earlier load transient and does not see the CUDA context,
  driver or non-torch memory. `cuda_peak_reserved` is recorded next to it
  with its own scope: `reset_peak_memory_stats` does not drop the reserved
  peak below the segments the allocator already holds, so its floor is every
  segment cached at the reset (`reserved_at_reset`) and run 0 and run 1 are
  expected to be close. The headline value (the human `peak_active` line and
  the legacy `peak_gib`) is `cuda_peak_allocated`, the analogue of MLX's peak
  active memory; `runtime.measurement.headline_peak` names it. On MLX the
  same reset point feeds `mlx_peak_active` (`mlx.core.get_peak_memory`); for
  torch on any non-CUDA device (cpu, mps) the per-run `backend_peak` is
  `null` with the reason. Process-wide (`memory`): `process_peak_rss`, a
  lifetime high-water mark (`ru_maxrss`; `peak_wset` on Windows) that
  includes model load and every run, and the sampled peaks from a sampler
  thread started before the model loads (`--rss-sample-interval`, default
  0.25 s, `0` disables; spikes shorter than the interval are missed):
  `process_sampled_peak_rss`, plus on Linux (`/proc/self/status`)
  `process_sampled_peak_rss_anon`, `process_sampled_peak_rss_file` and
  `process_sampled_peak_swap`. Resident set size counts the resident
  file-backed pages of the mmap'd checkpoint (whole-layer prefill touches
  every expert of `model.safetensors`, reclaimable page cache) together with
  anonymous memory, so it is not the process's anonymous footprint and not
  comparable to the "peak anonymous memory" figure further down this page;
  the `*_rss_anon` peak is. A non-zero swap peak marks a swapping run, which
  cannot establish a resident-memory target. RSS and allocator peaks are
  separate views of the same shared DRAM on Jetson; never add them.
* **Unavailable values** are `null` with a reason: every data group
  (`identity`, `model`, `runtime`, `workload`, `caches`, `memory`, `probe`,
  each entry of `runs[]`, each statistic block of `summary` and every nested
  object that can hold a null) carries an `unavailable_reasons` map
  (`{field: reason}`), and memory metric objects carry `unavailable_reason`;
  `protocol` and the `summary` container hold no nullable fields. An invalid *required* measurement (negative or
  non-finite timing, count or byte value) aborts the run instead of being
  written. Serialization is strict JSON (`allow_nan=False`).

### Report layout (`schema_version` 1, independent of the probe's schema)

| Group | Contents |
|---|---|
| `protocol` | name, version and the labels above |
| `identity` | start/finish timestamps, git commit and tracked-file dirty flag of this checkout (untracked files do not count), `run_count`, `command.argv`, the requested `BENCH_*` / `EDGE0_*` / `MLX_CACHE_LIMIT_MB` / `LING_HIDDEN_CLIP` / `LING_PREWARM` / `PREROUTER_*` knobs as raw strings, snapshotted before the engine is built (the 8B engine sets a default `LING_HIDDEN_CLIP` the user did not request), the requested arguments |
| `model` | resolved local checkpoint path, tier, `model_type` / `architectures`, `config.json` SHA-256, the checkpoint manifest (`files`: every regular file's relative path and size, `algorithm`, and the `sha256` over that list, so the digest is reproducible from the retained list; weights are never content-hashed), SHA-256 of every non-weight file up to 32 MiB (config, chat template, `tokenizer.json`, vocab, merges), safetensors header metadata, tensor counts and header hashes, `tokenizer_files` (name, size, SHA-256), and the LoRA / prerouter adapters as the engine resolved them (`engine.cfg`, which may point at the repo's `artifacts/` fallback rather than the checkpoint directory: path, whether it is inside the checkpoint directory, size, SHA-256 up to 256 MiB, header metadata, r/alpha and prerouter settings) |
| `runtime` | backend and version, resolved device, the measurement adapter's description (synchronization policy, peak source, `headline_peak`), execution evidence (class and device of the logits the engine produced), Python / torch / CUDA or MLX versions, platform, Jetson power mode (`nvpmodel -q`, else the nvpmodel status file, else a reason) |
| `workload` | prompt source, SHA-256 of the prompt text, SHA-256 of the prompt token ids after chat templating (`prompt_token_ids_sha256`: two launches are the same workload only if prompt digest, token count and token-id digest all match), length, text and token count; requested counts; resolved seed, temperature, top-k, top-p, repetition penalty, prefill chunk and think flag; `model_env`, the model-read knobs resolved after the build exactly as the bailing_hybrid code parses them (`ling_hidden_clip`, `prerouter_feature_topk`, `prerouter_intra`), which change numerics and per-token work and must be equal on both sides of a comparison |
| `caches` | resolved `LayerOptions`, the actual shared-LRU and prefetch capacities read from the installed streaming layers, `weight_cache` (the torch backend's resolved `EDGE0_TORCH_WEIGHT_CACHE`), `prewarm` (the requested `EDGE0_PREWARM` / `LING_PREWARM` flag; only the 8B engine honors it), thread settings, the MLX cache limit, streaming statistics after the runs |
| `runs[]` | per-run counts, `prefill_seconds` / `prefill_tokens_per_second`, `decode_seconds` / `decode_tokens_per_second`, per-run memory metrics, execution evidence |
| `summary` | per metric: `n`, `n_valid`, the per-run `values` (nulls kept in run order), mean, min, max and sample standard deviation over the runs that have a value (null with a reason when none has one; the standard deviation needs two); descriptive only, two sequential runs in one process are not independent launches |
| `memory` | the process-lifetime RSS peak and the sampled RSS / anonymous / file-backed / swap peaks, with notes |
| `probe` | with `--probe-json`: the probe's own `schema_version`, the SHA-256 of the file as read, its `ready` flag, its timestamp and local path |

### Safety rails

* `--json-output` must be a new path. An existing destination, an invalid
  `--probe-json` (missing, not JSON, no integer `schema_version`),
  `--ntok <= 0` (also via `BENCH_NTOK`), `--warmup < 0`, a non-integer
  `BENCH_SEED` or a non-numeric `BENCH_TEMP` are rejected **before** the model
  loads, and none of that needs a backend: `bench.py` imports the framework
  lazily, so `--help` and the argument checks work on a host without torch
  or MLX. `BENCH_SEED=0` is a seed (both runs re-seed with it right after
  prefill); at `BENCH_TEMP=0` the sampler is greedy, so the output does not
  depend on the seed even though it is recorded. A tier name resolves to
  `EDGE0_<TIER>_MODEL` by the same rule as the CLI.
* The report is written atomically (temporary file in the destination
  directory, fsync, then linked or renamed into place without ever
  overwriting an existing file). A failed engine build, benchmark, report
  assembly or write exits non-zero, prints `[bench] FAILED during <stage>` and
  leaves no report at the destination; the engine and the sampler are closed
  in `finally` either way.
* The human output is unchanged, except that `peak_active` prints `n/a` when
  the backend peak is unavailable and measured CUDA numbers now include the
  synchronized device work.
* Keep the JSON outside tracked source: it records local paths and the prompt
  text. Share sanitized summaries and evidence digests.

### What this does and does not show

`tests/test_benchmark_report.py`, `tests/test_benchmark_measure.py` and
`tests/test_bench_cli.py` validate the instrumentation with a fake engine, a
fake array namespace and fake device APIs: dispatch, synchronization order,
count arithmetic, units, pre-load validation, failure paths and cleanup. They
run on a host with neither torch nor MLX. `tests/test_bench_backend.py`
repeats the core cases through the real `core` ops and sampler of whichever
backend is importable (on a host without MLX it selects the torch backend on
the CPU unless the environment already chose otherwise) and is skipped when
there is none. None of this is CUDA or Orin evidence. The Orin baseline
(Task 3 of the plan) needs the physical device, a validated environment and
the probe.

`edge0` does **not** run on NVIDIA through MLX, at any version tested:
the first forward pass fails, differently at each version (the table
below). It does run on NVIDIA through the torch backend in
`backends/cuda/` (`EDGE0_BACKEND=cuda`): both edge0-8b and edge0-35b
generate text on a GB10, layer for layer within 5.3e-7 (8b) and 3.7e-6
(35b, real weights) of MLX. The rest of this file is
that investigation, kept because it is what the next person attempting
this would otherwise repeat from scratch.

## Environment

- Hardware: DGX Spark, GB10 (`sm_121`, Grace-Blackwell, 128GB unified memory)
- Toolkit: CUDA 13.0 — use `mlx[cuda13]`, not `mlx[cuda12]`; the latter
  ships NVRTC 12.9, which does not compile against the CUDA 13 headers
  (`cuda_fp6.hpp` / `cuda_fp4.hpp`) on this platform.
- Tier tested: edge0-8b (Ling / Bailing hybrid)

## The failure chain (three MLX versions, three distinct errors)

Everything up to the model finishing construction works cleanly at
every version tested: install, checkpoint download, LoRA/prerouter
attach (`lora applied=153 not_found=0`, `prerouter installed: 16 heads`,
built in 0.7s). The failure is always at the first forward pass, and it
moves as the MLX version moves — meaning this is not one bug, it's the
edge of MLX's own CUDA-quantized-op support arriving in stages, with
`edge0`'s code (written and pinned against `0.30.4`) landing in a
different gap each time:

| `mlx` version | Result |
|---|---|
| `0.30.4` (this repo's current pin) | `RuntimeError: QMM NYI` — quantized matmul has no CUDA implementation at all |
| `0.31.1` | `GatherQMM has no CUDA implementation` — the gather-variant specifically still missing |
| `0.32.0` / `0.32.2` | Both ops now present — clears `GatherQMM`, advances from `ling.py:181` to `ling.py:190`, then `IndexError: SmallVector out of range` inside `core.eval(logits)` |

No CPU fallback was viable either: `mlx-cpu==0.30.4` fails to JIT on
g++13/aarch64, and `mlx-cpu==0.32.2` raises `There is no Stream(cpu, 3)`.

The `0.32.x` failure is the most interesting one and the most worth a
second look by someone who wants to pursue this further: both required
CUDA kernels are confirmed present, the crash is downstream inside
`core.eval`, and it reproduced identically on two separate versions —
consistent with a real, narrow incompatibility between `edge0`'s
`ling.py` code path (written for `0.30.4`'s API) and something that
changed by `0.32.x`, not with a fundamentally unsupported operation.

## A reliability caveat, independent of all of the above

`mx.default_device()` reporting `Device(gpu, 0)` is **not** evidence the
GPU is usable — MLX is lazy and this call never touches the driver. It
reported `gpu` in every run above, including the ones where the GPU was
later confirmed dead. The real signal is whether `cuInit()` succeeds.
On our machine `cuInit()` started returning 999 a few minutes after
each boot while `nvidia-smi` still looked healthy. We first read that
as the driver dying after an aborted CUDA process. It was not: a
host-level cgroup device policy on that machine (unrelated to `edge0`
or MLX) denies `/dev/nvidia-uvm` and `/dev/nvidia-caps/*`, and the CUDA
runtime needs both. `nvidia-smi` goes through NVML on `/dev/nvidiactl`,
so it never notices. If `cuInit()` gives 999, first try opening
`/dev/nvidia-uvm` from the same shell. `EPERM` there means a device
policy, not a broken GPU.

## Bottom line

MLX's own CUDA backend does not close the gap at any version currently
available. The torch backend does: `EDGE0_BACKEND=cuda` runs both shipped
tiers on a GB10 and on CPUs, checked against MLX throughout (next
section). On the real weights, edge0-35b on the GB10 generates the same
32 greedy tokens as MLX run on its CPU device.

## Torch backend: what exists and how it is checked

MLX is the ground truth throughout: the checks run on Apple Silicon, where
both backends are available, and compare the torch side against MLX
(`tests/test_cuda_backend.py`, `tests/test_backend_parity.py`; the latter
runs each case under both backends in subprocesses).

| Piece | Checked against |
|---|---|
| `quant.gather_qmm` (2/4/8-bit affine, all broadcast shapes the streaming layer uses) | `mx.gather_qmm` |
| `core` ops at their real call sites (routing, sampling), `nn.RMSNorm` / `gelu` | the same code on MLX: identical expert choices, identical sampler masks |
| `StreamingSwitchGLU`, every path (exact, whole-layer, hot, staged), on layer 1 of the real edge0-8b checkpoint (`EDGE0_8B_MODEL`) | MLX on the same inputs: 1.2-1.5% of output scale, about two bf16 ulps |
| `io.load_model` + `install_streaming_experts` on a small checkpoint in the exact published edge0-35b format | the source model on the same weights: 4e-7 relative, same argmax |
| `backends/cuda/_impl/bailing_hybrid.py` (torch port of the edge0-8b backbone) on the real checkpoint, every layer, chunked prefill + decode | the vendored MLX model on the MLX CPU device, float32: <= 1.5e-6 per layer, <= 1.7e-6 on the logits |
| the whole edge0-8b engine (`engine/ling.py` unchanged: LoRA, prerouter-staged decode, streaming, sampling) | the same engine on MLX: identical greedy tokens (`pytest -m slow`) |
| the same engine on Linux aarch64 (DGX Spark, torch on the CPU), 32 greedy tokens of a chat prompt | MLX on Apple Silicon, same checkpoint (same file hashes): identical 32 tokens; 1.2 GB peak anonymous memory |
| everything above with torch on an accelerator: `EDGE0_TORCH_DEVICE=mps` (Apple GPU), whole suite including the slow engine test, plus the 32-token run | the same references: all pass, identical 32 tokens. On a device a tensor left on the host fails loudly, as it would on CUDA; this is what found `load_model` leaving init-time buffers (the rotary `inv_freq`) on the host |
| **the edge0-8b backbone on a real GPU** — a GB10 (`sm_121`, DGX Spark), torch 2.14+cu130, float32, every layer fed MLX's input for that layer | MLX on the Apple CPU device: **5.3e-7 max per layer** (median 2.6e-7), 4.3e-7 on the logits, same argmax. TF32 off, `float32_matmul_precision=highest` |
| the whole edge0-8b engine on that GPU, 32 greedy tokens | MLX on Apple Silicon: 31 of 32 tokens identical, diverging at step 31. Not a GPU artifact: torch on the CPU differs from MLX by the same 4.5% median per-step logit distance (the staged prerouter path), and that step's top-2 margin is smaller than that noise |
| `backends/cuda/_impl/qwen3_5_moe.py` (torch port of the edge0-35b backbone), every layer, chunked prefill + decode, on a small model MLX wrote in the published format (bf16, 4-bit, 8-bit router and shared gate) | the vendored MLX model on the MLX CPU device, float32: <= 2.6e-7 per layer, <= 4.1e-7 on the logits |
| the whole edge0-35b engine (`engine/qwen.py` unchanged: streaming, staged decode, the class-level prerouter patch) on that small checkpoint | the same engine on MLX: identical greedy tokens, per-step logits within bf16 noise and tracking MLX *with* the prerouter (the prerouter moves them 5-11%) |
| **the edge0-35b backbone on the real 23 GB checkpoint**, every layer, chunked prefill + decode (`test_qwen35_port_matches_mlx_on_real_weights`, needs `EDGE0_35B_MODEL`) | the vendored MLX model on the MLX CPU device, float32: <= 2.4e-6 per GatedDeltaNet layer, <= 3.7e-6 per gated full-attention layer, <= 2.4e-6 on the logits, same argmax; 40 layers, no missing or unexpected tensor |
| the whole edge0-35b engine on the real weights (LoRA 310 targets, prerouter 33 heads, streamed experts), 32 greedy tokens | MLX on the MLX CPU device: **identical 32 tokens**. Against MLX on Metal, 29 of 32 -- and MLX-Metal disagrees with MLX-CPU at exactly those three positions, so the flip is its GPU precision |
| **the edge0-35b engine on a real GPU** — the same GB10, torch 2.14+cu130, as a Slurm job | **identical 32 tokens** to both MLX-CPU and torch on the Mac's CPU (18.4 s) |
| **the edge0-35b backbone on that GPU**, real weights, float32, every layer fed MLX's own input for that layer | MLX on the Apple CPU device: **1.6e-6 max per layer** (median 2.7e-8) — 5.4e-7 across the 30 GatedDeltaNet layers, 1.6e-6 across the 10 gated full-attention ones — 5.1e-7 on the logits, same argmax; 0 missing / 0 unexpected tensors, 40 streamed expert layers |

Why the MLX *CPU* device: on some Apple GPUs MLX runs float32 matmul and
SDPA at reduced precision (an M5 Max measured 7.5e-4 from float64; MLX on
the CPU and torch both 2e-7). Against MLX on the GPU the port looks up to
1000x worse in the attention layers, all of it on the MLX side.

`load_model` handles what the published checkpoints actually contain: MLX
quantization of nearly every linear and embedding (kept 4-bit resident via
`QuantizedLinear` / `QuantizedEmbedding`), 8-bit routers, and, for
edge0-35b, MLX's sanitize (conv1d stored `[C, k, 1]`, five RMSNorm kinds
stored as `w + 1`), which it undoes.

## Speed, and where it goes (GB10, edge0-8b decode)

The backend was written as a correctness reference, and it shows: the
default profile decodes at **316 ms a token (3.2 tok/s)** on a GB10. A
profile of three decode steps says the GPU is barely working -- 63 ms of
CUDA against 643 ms of CPU -- so this is launch and copy overhead, not
arithmetic:

* **1681 host-to-device copies per token.** The streaming layer rebuilds
  expert bundles from the mmap every step, which is the whole point on a
  Mac with 36 GB and pointless on a 128 GB box. The shared LRU already
  fixes it: `cache_slots=3072` (from the tier's 64) keeps every expert
  resident after warm-up -> **206 ms a token (4.9 tok/s)**, 1.9 GB. A
  config change, no code.
* **235 quantized-linear calls per token**, dequantizing the same
  attention and lm_head weights every time (103 ms a token).
  `EDGE0_TORCH_WEIGHT_CACHE=1` keeps the dequantized weight instead ->
  **124 ms a token (8.1 tok/s)**, 6.0 GB resident.

Together that is **2.5x** over the default profile, with the same greedy
tokens. The two paths are bit-for-bit identical on the CPU
(`test_quantized_linear_weight_cache_is_exact`); on CUDA, keeping the
whole weight changes cuBLAS's split against the chunked path, so they
differ within the engine's own noise -- teacher-forced they pick the same
token at all 32 steps and sit the same distance from MLX (4.8% vs 5.3%
median per-step), while free-running they can diverge a step earlier.

What is still on the table: a fused dequantize+matmul kernel (the
elementwise shift/mask/mul/add chain is most of the remaining CUDA time),
captured graphs or `torch.compile` for the per-step launch storm, and
bundles built directly on the device rather than copied per step.

## What is left

* **Performance.** `gather_qmm` and the quantized linears dequantize on
  every call and `core.compile` is eager: this is a correctness reference,
  not a fast path.

## A stale assumption this also corrects

`tests/test_repo_hygiene.py`'s module docstring says these hygiene tests
"can run on any platform — including the Linux CI job where MLX has no
wheels." That was true when written; it no longer is — MLX ships both a
CPU-only Linux wheel and CUDA-backed Linux wheels
(`manylinux_2_35_x86_64`/`aarch64`) as of at least `mlx==0.30.4`, the
exact version this repo already pins. Worth knowing regardless of
whether GPU runners are in reach for CI: the assumption that MLX-dependent
tests categorically cannot run in this repo's own CI is no longer correct.
