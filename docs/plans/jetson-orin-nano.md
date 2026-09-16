# Jetson Orin Nano CUDA implementation plan — revised

Review date: 14 September 2026. Source snapshot: `KarlTaylorKnight/Edge0` commit `dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe`. Change IDs link this plan to the accompanying [justifications](jetson-orin-nano-review.md).

## Goal and scope [C01, C02]

Establish reproducible Edge0-8B inference on a physical **8 GB Jetson Orin Nano with NVMe**, then improve the measured bottleneck while retaining the current Torch reference implementation and MLX behavior. Keep the branch stacked on `cuda-backend-base`, derived from upstream PR #19. Do not substitute the different MLX CUDA experiment in issue #20.

Task 1 is present in the reviewed branch. Task 2 is the next implementation increment and its stopping point. Tasks 3–7 remain later increments. No reviewed evidence establishes Orin inference support or an Orin performance number.

Preserve the existing model engines and facade. Add narrow measurement/configuration/streaming hooks as required, with regression tests. Edge0-35B, a TensorRT rewrite and new model families remain outside the initial target. Do not project GB10 results onto Orin.

Use RED → GREEN → REFACTOR for meaningful behaviors. Document changes and evidence after each increment. Retain raw measurements outside Git; a documented baseline means a reviewed summary plus retrievable evidence references, not committed raw artifacts.

## Acceptance gates

### Gate A — target and installation evidence [C08, C09]

Record board/module identity, nominal memory variant, observed RAM, architecture, L4T/JetPack, Python ABI, installed Torch build, CUDA runtime, GPU properties, storage backing and checkpoint identity. Verify the target is the intended Nano 8 GB model; a generic Orin match also accepts other Orin models.

The existing probe provides preliminary evidence. Its `ready` field does not certify the 8 GB SKU, enough free memory, a valid/complete checkpoint, package compatibility or a successful CUDA kernel. Missing required observations remain blockers. Resolve L4T to JetPack only from a supported mapping or package evidence; otherwise report it as unknown.

Before inference, establish a reproducible environment compatible with the detected board/software stack. Record exact package versions and wheel URLs/digests or container digest. Verify actual imports and required APIs, including `torch.nn.RMSNorm`, the unsigned dtypes used by the backend, tokenizer dependencies and CUDA execution. Use NVIDIA's [installation requirements](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and [compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html); select a combination supported for this Orin, rather than the newest table entry automatically.

The reviewed packaging unconditionally depends on MLX/MLX-LM and omits explicit Torch/Transformers dependencies. Audit the actual 8B import path and provide an isolated, tested Torch installation recipe. If dependency metadata must be separated, make that a small prerequisite change and preserve the default Apple installation path. Do not use the unrelated Mac wheel blocker as a diagnosis of Jetson compatibility.

### Gate B — actual CUDA and model correctness [C10, C12]

Require a Torch-only smoke test that executes representative backend operations on CUDA and synchronizes. The acceptance invocation must fail if CUDA is required but absent, rather than quietly skipping or falling back to CPU. Assert tensor/output device identity, finite output and comparison against a deterministic reference.

Then run Edge0-8B with a pinned checkpoint, tokenizer/chat template, LoRA/prerouter identity, prompt IDs, routing profile and numerical settings. Capture at least 32 greedy tokens and compare teacher-forced logits and token choices with identified reference data. Define tolerances before examining the Orin result. Any allowed near-tie token divergence must be explained by the logit comparison; do not retroactively loosen tolerances to pass.

Run live MLX/Torch comparison suites where MLX is supported. Provide small, reviewed deterministic fixtures with generating commit, input/model hashes, dtype and tolerance for use on Jetson. An optimized CUDA path must also agree with the current Torch correctness path on the same Orin and workload. Keep free-running output checks separate from teacher-forced numerical comparisons.

### Gate C — bounded shared memory [C06, C11]

Require no OOM across initialization, prefill, decode and repeated requests for the declared workload. Replace the unsupported universal “RSS below 6.5 GB” acceptance threshold with an explicit byte budget and observed system headroom. Record the selected reserve and why it is adequate for this device's services and workload; it is a tested policy, not a hardware guarantee.

Start from observed available RAM and any tighter process/container limit. Deduct allocations still to be made, OS/application growth reserve, KV/context growth, prefill and dequantization transients, pinned staging, in-flight work, retained caches and allocator/driver allowance. Account for each allocation once at the chosen observation point. Compute tensor payload sizes from checkpoint metadata; slot counts alone are insufficient.

Validate minimum available RAM, RAM/swap trends, process RSS and Torch allocator peaks during the run. RSS and CUDA peaks are separate views and must not be added as a total. Mark swap activity explicitly; a swapping run cannot establish a resident-memory target. Reject impossible profiles before inference when the calculation can identify them, and preserve evidence for unexpected OOMs.

### Gate D — measured optimization [C13, C14]

For expert transfers, reuse a bounded payload pool after initialization; do not require every model operation to become allocation-free. Preserve exact fallback for required experts that are late, absent or predicted incorrectly. Demonstrate correct asynchronous ownership and a measured benefit before enabling a new path by default.

Treat elimination of dense-layer dequantization as a separate, conditional objective. An expert gather kernel does not meet it. Any dense-weight cache must fit the byte budget; whole-model expansion is not a default Orin setting.

## Task 1 — retain the probe; record its limits [C09]

Keep `scripts/jetson_probe.py`, its stable schema and focused tests. Do not redo the completed increment. Add narrowly scoped corrections only if required evidence is missing or defective, with schema changes documented rather than silently redefining `ready`.

Treat its Linux storage/sysfs behavior as platform-specific. Use Linux for those tests, and distinguish fixtures exercised on a supported host from behavior on the actual Jetson. The report builder introduced next must remain independently testable without this platform dependency.

## Task 2 — functioning, machine-readable benchmark reporting [C03–C07]

**Scope:** reporting, backend-aware measurement, focused tests and documentation. No cache/kernel optimization and no mandatory physical-device run in this increment.

**Files:** modify `examples/bench.py`; add a pure helper such as `examples/benchmark_report.py`; add `tests/test_benchmark_report.py` and focused integration tests as appropriate; update `docs/nvidia.md` and this plan. Keep optional imports and metric collection outside the pure builder.

### Measurement behavior [C03, C04, C05]

The current benchmark calls peak-memory methods absent from the CUDA core. Add a benchmark-owned adapter: retain MLX metric behavior where available, use Torch CUDA metrics for an actual CUDA device, and use explicit unavailable values elsewhere. `EDGE0_BACKEND=cuda` alone does not prove the selected device is CUDA.

Measure wall time with a monotonic clock. On CUDA, synchronize the resolved device before each timed phase and before its endpoint. Finish prefill before recording its time; complete untimed warmup before starting decode; finish the final timed step before recording decode. Use phase boundaries, not new per-token synchronizations. Include storage, transfer and CPU sampling costs in wall time; kernel-only event timings are a separate diagnostic. [PyTorch timing and stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution).

Keep the current two-run protocol, defaults, sampling behavior and human-output format. The existing warmup is greedy; timed iterations perform sampling followed by an engine step, including a final forward whose logits are not sampled. The loop is fixed-length and does not stop at EOS. Label this protocol explicitly; do not call it TTFT or EOS-aware completion latency. Correct the warmup comment without changing it to sampled warmup.

For each run, record prompt tokens after chat templating; actual warmup tokens; timed decode tokens; total generated tokens; and context length at timed decode start. For the current loop, `total_generated = warmup + timed_decode` and `decode_start_context = prompt + warmup`. Timed throughput uses only timed decode tokens. Keep requested counts separately. Reject invalid negative counts and require a positive timed-token count; do not silently produce a successful empty benchmark.

### Report contract [C05, C06, C16]

Use a benchmark schema version independent of the probe schema. Include:

| Group | Required information |
|---|---|
| Run identity | Timestamp, benchmark protocol version, Git commit, dirty status, run count/index and command/configuration needed to reproduce |
| Model | Local resolved checkpoint path, model/revision identity, manifest digest where available, tokenizer/template and adapter/prerouter identity |
| Runtime | Backend name, resolved device, observed execution evidence, Python/Torch/CUDA versions, platform; observed power mode or an explicit unavailable reason |
| Workload | Prompt identity/digest and token count; counts defined above; resolved seed, temperature, top-k, top-p, repetition penalty, context/prefill settings |
| Caches | Resolved LRU/prefetch/staging settings, weight-cache and prewarm flags, and relevant thread settings; do not record only requested environment variables |
| Timings | Per-run prefill/decode seconds and their separate token/s values; all runs retained; summary calculation and variability defined |
| Memory | Integer byte values, method and scope for every metric, with explicit unavailable reasons |
| Probe link | Optional supplied probe file's schema version, content SHA-256 and local reference; linkage to that actual report, not only a schema number |

Record process-lifetime peak RSS with a platform-specific method and unit normalization. If only periodic samples are available, call it a sampled peak and record the interval. A single endpoint RSS value is not a peak. Do not subtract lifetime high-water marks to invent a per-run peak.

For CUDA, synchronize after engine reset, reset allocator peak statistics immediately before prefill, and read allocated/reserved peaks after synchronized decode. State that this covers prefill, warmup and decode, and includes allocations already resident at reset; it does not recover an earlier initialization transient. Start process/system observation before model load in physical baseline runs. Allocator peaks do not measure all driver or non-Torch memory. [Torch allocator scope](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management).

Keep hashing, power queries, collector setup and report construction outside timed intervals. Preconfigured lightweight memory/system samplers may run throughout the benchmark; record their interval and use identical instrumentation for comparisons. The pure builder performs no collection. Prepare a checkpoint manifest outside measurement; do not hash large weights before each alleged cold-cache trial. A digest identifies an input but does not reproduce it: retain the referenced manifest/prompt in the private evidence set.

Represent unavailable metrics as `null` with a reason. Distinguish unavailable metadata from a failed required measurement. Reject negative/nonfinite timings and memory; zero measured duration gives unavailable throughput, not a fabricated zero-rate success. Use strict JSON serialization (`allow_nan=False`). [Python JSON behavior](https://docs.python.org/3.10/library/json.html).

### CLI and acceptance tests [C07]

Add `--json-output PATH` and optional `--probe-json PATH`. Validate an explicitly requested probe before inference. Keep probe collection separate so generic benchmarks do not acquire Jetson prerequisites. Preserve human-output format, except measured numeric values may correctly change after synchronization.

Require a new output path and reject an existing destination before inference. Write complete JSON atomically through a temporary file in the destination directory; report output errors with a nonzero exit status. Close the engine and collectors in `finally`, including generation and write failures. Failed inference must not leave a success report at the requested path; identify failures separately.

Use TDD for schema/units, count arithmetic, requested versus resolved configuration, CPU/MLX/CUDA dispatch, phase synchronization order, missing metrics, malformed probes, nonfinite inputs, output failure and cleanup. Exercise the CLI and fake engine without weights. Test output-format compatibility without pinning real measured numbers. Host tests validate instrumentation; CUDA tests validate device behavior; neither substitutes for the Orin baseline.

**Done:** applicable checks pass, exact results and host limitations are reported, documentation describes the protocol, and Task 3's next command is supplied. Stop here in the current handoff.

**Status (14 September 2026):** implemented on this branch. `examples/benchmark_report.py` (pure builder, schema version 1), `examples/benchmark_measure.py` (backend-aware measurement adapter, process-memory sampler, git/power/checkpoint/adapter identity), `examples/bench.py` (`--json-output`, `--probe-json`, `--rss-sample-interval`, phase-boundary synchronization, pre-load validation, `finally` cleanup, lazy backend import), tests `tests/test_benchmark_report.py`, `tests/test_benchmark_measure.py`, `tests/test_bench_cli.py` (fakes only) and `tests/test_bench_backend.py` (real backend namespace, no weights), and the protocol/schema section in `docs/nvidia.md`. `pyproject.toml` gained `pythonpath = ["."]` so the plain `pytest` invocation collects the repo-root `scripts`/`examples` imports. Host checks only: no CUDA execution and no Orin run were performed; Task 3 starts with the commands below.

## Task 3 — establish the unoptimized Orin baseline [C08–C10, C15]

First complete Gate A's installation and model checks. Add a small Torch-only acceptance module, proposed as `tests/test_torch_cuda_smoke.py`, with a documented `--require-cuda` option or equivalent hard-fail contract. Add an on-device reference checker using the pinned fixtures described in Gate B. These are future deliverables, not commands that exist in the reviewed branch.

The reviewed plan's `tests/test_nvidia_backend_smoke.py` is an MLX smoke test. The existing `test_cuda_backend.py` also imports MLX and uses CPU tensors in its comparison helper. Keep those suites, but do not use their skip/pass result as CUDA execution evidence.

On the actual Orin, use an isolated validated environment and an untracked `$EDGE0_RUN_DIR`:

```bash
export EDGE0_BACKEND=cuda
export EDGE0_TORCH_DEVICE=cuda
export EDGE0_TORCH_WEIGHT_CACHE=0
export EDGE0_PREWARM=0
python scripts/jetson_probe.py --model-dir "$EDGE0_8B_MODEL" \
  --output "$EDGE0_RUN_DIR/orin-probe.json"
```

After implementing the dedicated acceptance test, its intended invocation is:

```bash
python -m pytest tests/test_torch_cuda_smoke.py --require-cuda -q
```

Run the separate 32-token greedy correctness check and retain its comparison evidence. After that passes and Task 2 reporting exists, a short benchmark is:

```bash
BENCH_TEMP=0 BENCH_SEED=0 python examples/bench.py edge0-8b \
  --ntok 32 --warmup 0 \
  --probe-json "$EDGE0_RUN_DIR/orin-probe.json" \
  --json-output "$EDGE0_RUN_DIR/orin-short-baseline.json"
```

For performance comparisons, freeze a workload manifest and use at least three independent process launches before and after optimization. Keep both internal runs from each launch and distinguish their order/cache state. Use the same prompt, counts, seed, checkpoint, numerical/routing settings and power profile on both sides. A benchmark reset is not an OS or expert-cache flush.

Capture `tegrastats` beside the run, from before loading until cleanup, with interval and timestamps. Record power mode, clock policy, cooling, storage/filesystem and thermal/throttling evidence. Observe existing settings before proposing changes. Do not assume a fresh process makes storage cold or drop system caches automatically. [NVIDIA tegrastats](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

If inference fails, preserve its stage, configuration, available diagnostic evidence and exit status. OOM diagnostics may not reveal an exact failing allocation, especially for an OS kill; state unknown when necessary. Reducing context or generation produces a new workload, so retain both results. If default inference cannot fit, a documented failed attempt permits a bounded-memory correction in Task 4; establish a successful reference baseline before asynchronous or kernel optimization.

**Status (14 September 2026, on the physical device):** executed on an
Orin Nano Developer Kit (Super) 8 GB, L4T R39.2, CUDA 13.2 driver,
25W mode, NVMe.  Gate A: validated venv (Python 3.12.3, official torch
2.14.0+cu130 aarch64 wheel — the GB10 combination; edge0 installed
`--no-deps`, no MLX on device; `torch.nn.RMSNorm`, `torch.uint32`,
tokenizer imports and CUDA kernel execution verified).  The wheel warns
`sm_87` is outside its SASS list; execution evidence, not the warning,
was used as the gate.  Probe: `ready: true`, no blockers.
`tests/test_torch_cuda_smoke.py` + `tests/conftest.py` (`--require-cuda`)
added: 6/6 pass on device; hard-fail contract verified with
`CUDA_VISIBLE_DEVICES=""`.  `scripts/orin_reference_check.py` added
(CUDA greedy vs torch-CPU teacher-forced logits, tolerances registered
in the script before the run): 32/32 token choices identical, no
near-ties consumed; the pre-registered vocab-wide logit bound (1.0)
was exceeded (max |Δ| 2.06, tail-token bfloat16 noise — ≤ 0.22 at every
step's top-8 tokens; analysis retained in the evidence set).  Recorded
as a FAIL of that bound, not loosened retroactively; a top-k-scoped
criterion is proposed for review before the next comparison run, and
MLX fixtures remain the intended reference.  Short baseline: three
independent launches × two runs (`--ntok 32 --warmup 0`, temp 0, seed 0,
weight cache and prewarm off, tegrastats alongside): decode 0.53–0.58
tok/s (mean 0.55), prefill 11.3–15.9 s at 37 prompt tokens, CUDA
allocator peak ≤ 1.00 GiB, process peak RSS 5.3–5.8 GiB, GPU 99%
utilized, no OOM/swap.  Raw JSON reports, npz logit captures and
tegrastats logs live in the untracked `$EDGE0_RUN_DIR` evidence set;
sanitized summary in `docs/nvidia.md`.  Task 4's profiling starts from
this baseline; the dominant cost is per-call expert dequantization in
`gather_qmm` (GB10 already measured 2.5× from the weight cache that
Task 4's byte budget must first make safe to enable here).

## Task 4 — memory-budgeted profile [C11]

Implement the pure budget calculation first, then integrate at existing configuration/cache-construction points. Inspect `models/base.py`, the 8B configuration, `streaming/install.py`, `streaming/cache.py` and `streaming/layer.py`; the CUDA backend namespace alone cannot enforce the profile.

Resolve byte limits for completed caches and in-flight builds, prefetch buffers, pinned memory, staged/assembled copies, full-layer prefill, transient dequantization and KV growth. Bound the producer queues as well as their completed outputs. Audit `prefetch_all()`, which can raise buffer capacity beyond its configured starting value.

**Existing zero semantics:** `SharedExpertCache(0)` and `PrefetchBuffer(0)` disable eviction. Never pass a calculated zero allowance to them as if it disabled caching. Initially reject such impossible settings; if cache-off operation is needed, implement an explicit bounded/disabled state with regression tests rather than silently changing shared MLX semantics.

Record resolved policy and rejected overrides. Keep full dequantized weight caching off unless the measured budget permits it. Test exhausted/tiny budgets, missing observations, overflow, invalid overrides, varying tensor sizes, inflight backpressure and repeated-request memory stability.

**Status (15 September 2026, implemented and validated on the device):**
`edge0/streaming/budget.py` (pure: `Observation` / `ExpertFootprint` /
`WorkloadDecl` / `Reserves` → `resolve_budget()` → `ResolvedBudget` with
itemized deductions; `bundle_bytes_from_entries()` prices the per-expert
payload from the safetensors header, not slot counts).  Enforcement:
`PrefetchBuffer` gained `max_cap` (a ceiling `set_cap` cannot exceed —
bounds `prefetch_all()`'s silent growth to `num_experts + 32`);
`LayerOptions.max_inflight` bounds QUEUED speculative prefetch builds
(demand loads are never dropped); both default to the historical
unbounded semantics when no budget is active.  Integration:
`EDGE0_MEMORY_BUDGET=auto|<bytes>` resolved in the 8B engine's
`load_installed` immediately before cache construction
(`EDGE0_BUDGET_CONTEXT` declares the context, default 1024 tokens;
`ModelConfig.kv_bytes_per_token` prices it — measured, 1.1 MB/token for
this tier).  The resolved policy raises `BudgetError` pre-inference for
impossible profiles — a calculated starved cache is rejected, never
constructed as `SharedExpertCache(0)`/`PrefetchBuffer(0)` — and is
recorded in the engine (`engine.memory_budget`) and the bench report's
caches group.  `EDGE0_TORCH_WEIGHT_CACHE=1` under a budget is vetoed
unless its measured 4.1 GB fits the post-cache headroom.  25 unit tests
in `tests/test_memory_budget.py`; the streaming package init became
lazy for backend-bound names so the pure modules test everywhere.
On-device validation (Orin Nano 8 GB): `auto` resolved usable ≈ 4.2 GB,
kept the tested 64/48 profile (real bundle ≈ 1.30 MiB/expert, full-layer
transient 162 MiB), benchmark equivalent to the unoptimized baseline
(0.52 tok/s, same peaks, RSS 5.55 GiB in the baseline range); a
4096-token declaration was rejected with itemized arithmetic (KV
4.5 GB > usable), and the weight cache was vetoed (4.1 GB > 2.2 GB
headroom).  Not yet done: staged/assembled-copy pricing (the 8B
`prod_k8` profile has staging off — budgeting a staged profile is
future work with Task 5), and a long repeated-request soak (two
same-process runs were stable; a sustained test belongs to Task 7).

## Task 5 — bounded asynchronous expert transfers [C12, C13]

The 8B `prod_k8` profile deliberately disables staged decode. Do not enable `staged_k8` or replace routing to activate prefetch. Integrate the new transfer cache with the existing exact expert-consumption path. Predictions may change scheduling, not required expert identity or contribution. Missing required data must wait/load through the reference path rather than use the existing staged zero-row behavior.

Start with bounded reusable pinned buffers and CUDA slots as a candidate design. Both consume the same physical DRAM pool on Jetson; measure their duplication and benefit. Keep alternative memory paths and GPUDirect Storage deferred until support and benefit are established. [NVIDIA Tegra memory architecture](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-for-tegra-appnote/index.html#memory-management).

Define slot ownership from free → filling → ready → in-use → reusable, with generation IDs. Publish readiness only after copy completion. Reuse a slot only after all GPU consumers finish; retain pinned source contents until its copy finishes. Bound queued work; handle stale completion events, failed reads, cancellation, reset, close and partial initialization. Use explicit stream dependencies and allocator lifetime handling as required. [PyTorch stream ownership](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-streams).

Test state transitions with fake events, then real CUDA stress/parity cases: wrong/late predictions, eviction during consumption, repeated requests and shutdown. Define counters and scopes for cache/prediction hits, bytes requested/read/copied, I/O wait and consumer stalls. Distinguish logical file reads from physical NVMe traffic. Demonstrate overlap and end-to-end improvement on the baseline workload before making the path default.

**Status (15 September 2026, measured on the device):** the measured
profile reshaped this task.  First finding: the tested 64-slot shared
LRU is smaller than this tier's per-step working set (23 MoE layers x
K=8 = 184 bundles), so on the baseline it NEVER hits (`hits: 0`,
~184 rebuilds per token).  Second finding: fixing that (512 slots, 65%
hit rate, loads 11,776 → 4,075 over the workload) moved end-to-end
decode by roughly nothing (0.49–0.57 vs 0.50–0.52 tok/s) — the
unoptimized decode is COMPUTE-bound (GPU pegged at 99%; per-call
dequantization).  Per this task's own gate, the pinned-buffer /
CUDA-stream slot pipeline (free→filling→ready→in-use→reusable,
generation IDs) is therefore **deferred** — it would optimize a cost
that is currently invisible end-to-end — and is to be revisited after
Task 6 shrinks compute, when transfers become a meaningful fraction.

What WAS implemented (opt-in, exact path untouched):
`EDGE0_CACHE_SLOTS=<int>` overrides the REQUESTED LRU size (the budget
may still lower it; zero/negative rejected), and
`LayerOptions.predict_prefetch` / `EDGE0_PREDICT_PREFETCH=1` routes
each step's prerouter predictions into the existing bounded
`prefetch()` for non-staged layers (`prod_k8` keeps staged decode OFF;
`stage_experts` is untouched; wrong/late predictions fall back to
demand loads through `_get_bundles` — never a staged zero row).  The
prefetch buffer is raised to one full predicted step (owners x K = 128)
and the per-layer in-flight bound defaults to K, both priced by the
Task 4 budget (in-flight transient = per-layer bound x producing
layers).  Reuse-safety of consumed bundles rests on Python/torch
reference counting (consumers hold the tensors they use); the manual
slot-generation lifecycle belongs to the deferred pinned design.

Measured (3 independent launches x 2 runs, identical workload and
instrumentation as the Task 3 baseline, budget active): decode
**0.568–0.623 tok/s, mean 0.594** vs baseline 0.53–0.58 mean 0.552
(**+7.7%**); hit rate 85% (10,059 hits / 1,717 loads); 2,038 of 2,063
predicted prefetches consumed (98.8%), 0 wasted, prefetch_wait 0.0 s —
the builds genuinely overlap the forward; load wall 26.5 s → ~6 s;
CUDA peak 1.45/1.58 GiB and RSS 5.48–5.82 GiB, both inside the
resolved budget (headroom ≈ 1.27 GB) with no OOM.  Token choices are
bit-identical to the Task 3 reference sequence with both knobs on.
Kept **off by default** per Gate D: the gain is real but small while
compute dominates; defaulting is a profile change to make together
with the Task 6 re-measurement.  Remaining for a later increment:
logical-vs-physical NVMe read accounting, CUDA stress cases for
eviction-during-consumption, and the deferred pinned-slot pipeline.

## Task 6 — conditional quantized-kernel work [C14]

**Scoping (15 September 2026, measured on the device):** per-op decode
shares — `gather_qmm` 37% (69 calls/step), dense `QuantizedLinear` 26%
(235 calls/step), everything else 37%. Both dequantization paths
qualify. Development is handed to a workstation GPU per
[`rtx6000-task6-handoff.md`](rtx6000-task6-handoff.md); decisions and
acceptance stay on the Orin.

**Workstation increment (15 September 2026, RTX PRO 6000, torch
2.14.0+cu130 — parity and pricing only, no Orin decision):** three
opt-in, guarded paths, documented in `docs/nvidia.md` ("Opt-in quantized
paths"): `EDGE0_QMM_BATCHED=1` (exact batched expert gather, transient
priced at 140 MB for the decode shape and deducted by the budget),
`EDGE0_TORCH_WEIGHT_CACHE_BYTES=<n>` (exact capped dequantized-weight
cache, first-fit, shrunk to the budget headroom; the full cache is now
priced at the measured bf16 total, 1.40 GB for this tier, not 4.1 GB),
and `EDGE0_INT4PACK=1` (torch's built-in int4 kernel: approximate at
bf16 rounding, bounds registered in `tests/test_int4pack.py` before any
Orin run). Design C (custom fused kernel) was scoped to a cross-compile
toolchain record only. Orin acceptance for each knob: the Task 3
reference check with `--test-env <knob>` against the reference path on
the same device, then the three-launch benchmark protocol.

**Finding that reshapes this task's dense half:** traced on the real
model, 400 of 470 dense `QuantizedLinear` calls per step arrive with
float32 activations and only 70 with bfloat16. Both dense knobs require
bf16, so the weight cache fills 35 of the 235 modules it admits
(0.174 GB of 1.402 GB reserved) and the int4 kernel repacks exactly
those same 35. Neither moves decode measurably as a result. The dense
26% is therefore gated by activation dtype, not by the kernel; whether
those call sites can run bf16, and what that costs numerically, is the
question to settle on the device before more kernel effort. The batched
expert gather is unaffected: it covers the routed 37% and engages on
every call.

Use the measured profile to choose expert gather, dense quantized linear work, or no kernel change. For an INT4 prototype, verify the actual checkpoint layout: packed words, group size, signed/negative scale behavior, bias handling, accumulation and output dtype. Pin the CUDA extension/CUTLASS/compiler combination supported by the detected Orin stack and verify runtime loading.

Dispatch only supported dtype/shape/stride/transpose/group-size/device combinations to the new kernel. Keep the current implementation for unsupported cases, including 2/8-bit layouts. Test all existing broadcast and gather shapes, valid index boundaries, quantization extremes and checkpoint dtypes. Compare numerical results against both deterministic MLX fixtures and the current Torch path on Orin. Define tolerances in advance and measure scratch-memory peaks as well as speed.

A fused expert gather does not remove dense `QuantizedLinear` dequantization. If dense work remains dominant, handle it as its own bounded, parity-tested change or document the residual cost. Gate D does not require eliminating all model allocations.

**Status — ACCEPTED ON THE DEVICE (16 September 2026).** Orin Nano 8 GB,
same workload and instrumentation as the Task 3 baseline, budget active,
three launches x two runs per configuration; sanitized tables in
`docs/nvidia.md` (Task 6 Orin acceptance), raw reports in the evidence
set.  Suite 341 passed on-device, 96 under `--require-cuda`.

*Adopted:* the batched expert gather is now the **default** on the CUDA
backend (`EDGE0_QMM_BATCHED=0` selects the reference loop).  It is the
only candidate that both PASSED the registered 32-token reference bound
on the target (32/32 token choices, max |Δlogit| 0.777 vs the bound of
1.0) and delivered a measured benefit there: decode 0.623–0.699,
mean **0.668 tok/s, +20.7%** over the 0.553 baseline, at unchanged CUDA
peak (1.00 GiB) and RSS.  Its transient is bounded and the budget
deducts the cap (verified in the report: `kernel_transient` 268,435,456
with no env var set).  `int4pack.probe` executes on sm_87 — the kernel
runs on hardware the wheel's SASS list does not advertise, decided by
execution as designed.

*Not adopted (kept opt-in):* the capped weight cache (+0.8%) and the
int4 kernel (+3.7%), both because the activation-dtype gate limits them
to 35 of 235 modules, and both exceed the registered bound (1.58 and
2.17) — recorded as that bound's failures, not loosened.

*Gate D's Task 5 re-measurement:* with compute reduced the transfer
knobs are worth much more than before — +7.7% pre-Task-6, **+17.5% on
top of the batched gather** now (0.668 → 0.785 tok/s, expert load wall
72.5 s → 17.0 s over the same six runs).  `EDGE0_MEMORY_BUDGET=auto
EDGE0_CACHE_SLOTS=512 EDGE0_PREDICT_PREFETCH=1` is the recommended Orin
profile and stays opt-in as board-specific tuning that requires the
budget.  End to end the tier now runs at **0.785 tok/s, +42% over the
Task 3 baseline**, inside the resolved budget with no OOM.

*Two workstation claims corrected here, both only visible on the target:*
the capped weight cache is NOT bit-identical on sm_87 — the cached
weights are, but one GEMM versus the reference's per-chunk GEMMs differs
by one bf16 ulp of accumulation order (isolated directly: same weights,
0.03125 on a 30.6 scale), so that exactness was a property of the
workstation's cuBLAS heuristics; and the float32 activation gate is
deliberate MLX promotion parity in the MLA/RoPE path (layer 3 is the
first `BailingMLA`; the residual stream is float32 from there on), so
reaching the remaining dense 26% is a numerics-parity decision for
review, not a kernel increment.

*Open for a later increment:* design C (custom fused kernel) remains
unwritten with its toolchain recorded; the MLA float32 promotion
question; and the deferred pinned-slot transfer pipeline from Task 5,
which is now more attractive than it was at the compute-bound baseline.

## Task 7 — final validation and evidence [C15, C16]

Repeat the identical workload manifest using at least three independent launches. Retain individual prefill/decode results, explain first-versus-repeated-run state, and report sample count, mean, standard deviation and min/max separately. Treat the small sample as descriptive evidence rather than a broad performance guarantee.

Repeat correctness, maximum declared context/generation, memory pressure and repeated-request checks, plus a declared sustained-run test long enough to expose thermal behavior. Capture failures and throttling. State the tested operating envelope: exact board/software/checkpoint/profile, context limits, memory reserve and remaining limitations.

Only claim physical Orin inference once Gate B passes, and qualify the claim to its tested workload. Claim the bounded profile or optimization only after its own gates pass. Commit reviewed code/tests/docs and sanitized result summaries; keep raw benchmarks, telemetry, private paths, identifiers and checkpoints in the separate evidence set. Small reviewed deterministic test fixtures are test assets, not raw benchmark dumps.

**Status — COMPLETE (16 September 2026).**  The tested operating
envelope is stated in `docs/nvidia.md`; raw reports, telemetry logs and
the npz logit captures stay in the untracked evidence set.

*Repeat measurement.*  Final adopted profile, three independent launches
x two runs: decode 0.744–0.786 tok/s, **mean 0.772, sd 0.016** (n=6),
prefill 13.6–15.6 s, CUDA peak 1.58 GiB, RSS 5.57–5.69 GiB.  Reported as
descriptive statistics of a small sample, not a guarantee.

*Correctness repeat.*  Same-device 32-token reference check of the final
profile against the reference loop: **PASS**, 32/32 token choices
identical, max |Δlogit| 0.777 against the registered bound of 1.0.  All
30 requests of the sustained run produced byte-identical tokens.

*Maximum declared context and generation.*  KV growth was MEASURED here
rather than inherited: the marginal cost is 561,320 B/token (548 KiB),
about half the 1,100,000 taken from the README's Apple/MLX figure, so
`kv_bytes_per_token` for this tier is corrected to 600,000 (≈7% margin).
Growth is sublinear below the top of the range because most layers of
this hybrid are KDA with fixed-size state; only the MLA layers' KV grows
with context, which is why the top-of-range marginal slope is the right
constant to extrapolate from.  With it the budget admits a **4096-token
declared context** on this board (it admitted ~2560 before), validated
by a real 3292-token prompt run: prefill 42.4–42.8 tok/s, decode
0.72–0.81 tok/s, CUDA peak 3.14 GiB, RSS 5.8 GiB, no OOM, 39 MB of
resolved headroom at a 3584-token declaration.  Generation validated to
**256 tokens** in one request (0.807 / 0.820 tok/s — no decay with
length).

*Memory pressure, repeated requests, sustained run.*  30 sequential
requests in one process over 25 minutes (960 tokens): decode mean
**0.817 tok/s**, sd 0.028, quartile means 0.791 / 0.834 / 0.816 / 0.824
— no degradation over time.  Junction temperature 54.5–61.8 °C with
**no throttling** (GPU busy in 749 of 774 samples).  RSS moved between
5.45 and 5.70 GiB, rising and falling rather than growing monotonically:
mmapped expert pages reclaimed and re-faulted, not a leak; CUDA
allocator peaks flat at 1.57 GiB.

*Swap, marked explicitly.*  The sustained run DID swap: +1,165 MB during
the first fifth, then flat (+24, 0, +18, −2 MB), with no throughput
loss.  That is a one-time eviction of the resident desktop session's
idle pages, not the model's working set thrashing — but per Gate C a
swapping run cannot establish a resident-memory target, so no swap-free
claim is made for this board with a desktop resident.

*The claim, qualified.*  edge0-8b runs on a physical Jetson Orin Nano
8 GB at **0.77–0.82 tok/s decode**, 42 tok/s prefill at 3.3k tokens,
within a byte budget that rejects what does not fit before loading, with
token choices matching the reference implementation on this device.  The
claim covers exactly the board, build, checkpoint, profile and workloads
above: single request, greedy sampling, ≤ 4096 declared context, ≤ 256
validated generation, 25W mode, desktop resident.

*Remaining, none of it blocking:* design C (custom fused kernel,
toolchain recorded); the MLA float32 promotion question that gates the
dense 26%; the deferred pinned-slot transfer pipeline; concurrent /
batched requests, which were never in scope here; and short-prompt
prefill overhead (2–3 tok/s at 37 tokens against 42 tok/s at 3292),
which no increment targeted.
