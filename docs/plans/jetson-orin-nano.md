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

## Task 4 — memory-budgeted profile [C11]

Implement the pure budget calculation first, then integrate at existing configuration/cache-construction points. Inspect `models/base.py`, the 8B configuration, `streaming/install.py`, `streaming/cache.py` and `streaming/layer.py`; the CUDA backend namespace alone cannot enforce the profile.

Resolve byte limits for completed caches and in-flight builds, prefetch buffers, pinned memory, staged/assembled copies, full-layer prefill, transient dequantization and KV growth. Bound the producer queues as well as their completed outputs. Audit `prefetch_all()`, which can raise buffer capacity beyond its configured starting value.

**Existing zero semantics:** `SharedExpertCache(0)` and `PrefetchBuffer(0)` disable eviction. Never pass a calculated zero allowance to them as if it disabled caching. Initially reject such impossible settings; if cache-off operation is needed, implement an explicit bounded/disabled state with regression tests rather than silently changing shared MLX semantics.

Record resolved policy and rejected overrides. Keep full dequantized weight caching off unless the measured budget permits it. Test exhausted/tiny budgets, missing observations, overflow, invalid overrides, varying tensor sizes, inflight backpressure and repeated-request memory stability.

## Task 5 — bounded asynchronous expert transfers [C12, C13]

The 8B `prod_k8` profile deliberately disables staged decode. Do not enable `staged_k8` or replace routing to activate prefetch. Integrate the new transfer cache with the existing exact expert-consumption path. Predictions may change scheduling, not required expert identity or contribution. Missing required data must wait/load through the reference path rather than use the existing staged zero-row behavior.

Start with bounded reusable pinned buffers and CUDA slots as a candidate design. Both consume the same physical DRAM pool on Jetson; measure their duplication and benefit. Keep alternative memory paths and GPUDirect Storage deferred until support and benefit are established. [NVIDIA Tegra memory architecture](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-for-tegra-appnote/index.html#memory-management).

Define slot ownership from free → filling → ready → in-use → reusable, with generation IDs. Publish readiness only after copy completion. Reuse a slot only after all GPU consumers finish; retain pinned source contents until its copy finishes. Bound queued work; handle stale completion events, failed reads, cancellation, reset, close and partial initialization. Use explicit stream dependencies and allocator lifetime handling as required. [PyTorch stream ownership](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-streams).

Test state transitions with fake events, then real CUDA stress/parity cases: wrong/late predictions, eviction during consumption, repeated requests and shutdown. Define counters and scopes for cache/prediction hits, bytes requested/read/copied, I/O wait and consumer stalls. Distinguish logical file reads from physical NVMe traffic. Demonstrate overlap and end-to-end improvement on the baseline workload before making the path default.

## Task 6 — conditional quantized-kernel work [C14]

Use the measured profile to choose expert gather, dense quantized linear work, or no kernel change. For an INT4 prototype, verify the actual checkpoint layout: packed words, group size, signed/negative scale behavior, bias handling, accumulation and output dtype. Pin the CUDA extension/CUTLASS/compiler combination supported by the detected Orin stack and verify runtime loading.

Dispatch only supported dtype/shape/stride/transpose/group-size/device combinations to the new kernel. Keep the current implementation for unsupported cases, including 2/8-bit layouts. Test all existing broadcast and gather shapes, valid index boundaries, quantization extremes and checkpoint dtypes. Compare numerical results against both deterministic MLX fixtures and the current Torch path on Orin. Define tolerances in advance and measure scratch-memory peaks as well as speed.

A fused expert gather does not remove dense `QuantizedLinear` dequantization. If dense work remains dominant, handle it as its own bounded, parity-tested change or document the residual cost. Gate D does not require eliminating all model allocations.

## Task 7 — final validation and evidence [C15, C16]

Repeat the identical workload manifest using at least three independent launches. Retain individual prefill/decode results, explain first-versus-repeated-run state, and report sample count, mean, standard deviation and min/max separately. Treat the small sample as descriptive evidence rather than a broad performance guarantee.

Repeat correctness, maximum declared context/generation, memory pressure and repeated-request checks, plus a declared sustained-run test long enough to expose thermal behavior. Capture failures and throttling. State the tested operating envelope: exact board/software/checkpoint/profile, context limits, memory reserve and remaining limitations.

Only claim physical Orin inference once Gate B passes, and qualify the claim to its tested workload. Claim the bounded profile or optimization only after its own gates pass. Commit reviewed code/tests/docs and sanitized result summaries; keep raw benchmarks, telemetry, private paths, identifiers and checkpoints in the separate evidence set. Small reviewed deterministic test fixtures are test assets, not raw benchmark dumps.
