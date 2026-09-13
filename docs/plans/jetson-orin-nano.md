# Jetson Orin Nano CUDA Backend Implementation Plan

> **For Claude Code:** Execute this plan vertically with strict RED → GREEN → REFACTOR cycles. Do not optimize kernels until an Orin Nano baseline has been captured and committed.

**Goal:** Make the Edge0 Torch/CUDA backend run reproducibly on an 8 GB Jetson Orin Nano with NVMe-backed expert streaming, then optimize it without violating MLX parity.

**Architecture:** Build on the correctness reference in upstream PR #19 rather than porting the MLX backend again. Treat Jetson as a distinct memory/I/O profile: dense quantized weights stay resident where possible, sparse experts move through a bounded slot cache, and prerouter predictions schedule asynchronous NVMe → pinned host buffer → CUDA slot transfers. Preserve the existing backend facade and model engines; introduce Jetson behavior as measured configuration and CUDA implementation details, not model-specific forks.

**Tech stack:** Python 3.10+, PyTorch for Jetson, CUDA, safetensors, NVMe, pytest; later CUTLASS/CUDA extensions for fused INT4 gather-GEMM.

---

## Constraints and evidence

- This branch is stacked on upstream PR #19 (`agourakis82:cuda-backend`), which already ports both shipped models and establishes CPU/MPS/CUDA parity.
- Upstream explicitly warns that the Apple path depends on unified-memory mmap semantics; an efficient CUDA implementation needs a DirectStorage-style data path, not a mechanical MLX port.
- The current Torch backend is a correctness reference. On GB10, Edge0-8B was measured at 3.2 tok/s by default and 8.1 tok/s with large caches, but the latter used about 6 GB resident memory. Those results cannot be projected onto Orin Nano.
- Orin Nano has 8 GB shared memory. The OS, CUDA context, tokenizer, KV cache, and application processes must all fit alongside Edge0.
- Use NVMe. microSD is not an acceptable performance target for expert streaming.

## Non-goals for the first PR

- No unverified performance claims.
- No Edge0-35B optimization before Edge0-8B is stable.
- No new model family.
- No speculative TensorRT rewrite.
- No changes to MLX behavior.

## Acceptance gates

### Gate A — reproducible platform evidence

On the target Orin Nano, save a JSON probe containing:

- Jetson model and L4T/JetPack release data;
- architecture, total/available shared memory;
- PyTorch, CUDA, CUDA availability, GPU name, compute capability, and reported device memory;
- model/checkpoint filesystem, free space, and whether it resolves to NVMe;
- explicit readiness blockers instead of guessed defaults.

### Gate B — correctness baseline

With `Edge0/Edge0-8B-A1B-preview` on NVMe:

- run the CUDA backend smoke tests;
- generate at least 32 greedy tokens;
- record peak process RSS, peak device allocation, prefill time, and decode tok/s;
- compare greedy tokens or logits against the existing reference fixture;
- retain the full command and JSON result.

### Gate C — bounded memory

- No OOM on an 8 GB Orin Nano in the agreed power mode.
- Target peak process footprint: below 6.5 GB for a short-context Edge0-8B run, leaving explicit OS headroom.
- Cache sizes are selected from available memory at startup and can be overridden.

### Gate D — optimized data path

- Expert-cache misses copy into preallocated slots; no per-token CUDA allocation.
- Transfers run on a dedicated CUDA stream and are synchronized with events.
- Prerouter predictions issue prefetch before the expert is consumed.
- Dense quantized layers do not repeatedly unpack/dequantize unchanged weights.
- Every optimized path has a parity test against the correctness path.

## Task 1: Add a read-only Jetson capability probe

**Objective:** Capture the facts needed to reproduce and interpret every Jetson benchmark.

**Files:**
- Create: `scripts/jetson_probe.py`
- Create: `tests/test_jetson_probe.py`
- Modify: `docs/nvidia.md`

**TDD steps:**

1. Test `/proc/meminfo` parsing with kB values and missing `MemAvailable`.
2. Run the focused test and verify it fails because the module is absent.
3. Implement only the memory parser; rerun until green.
4. Test mount selection and NVMe classification using synthetic `/proc/self/mountinfo`.
5. Implement longest-prefix mount matching without shelling out.
6. Test Jetson model/L4T detection from an injected filesystem root.
7. Implement platform and device-tree readers.
8. Test Torch reporting with fake unavailable and fake CUDA modules.
9. Implement Torch collection with import failure represented as data.
10. Test readiness blockers: not Jetson, no CUDA, non-NVMe storage, and a missing model path.
11. Implement `collect_probe()` and JSON CLI output.
12. Run `pytest tests/test_jetson_probe.py -q`, then the full non-slow suite.
13. Commit: `feat(jetson): add Orin capability probe`.

## Task 2: Add a machine-readable inference benchmark result

**Objective:** Extend the existing benchmark path to emit reproducible JSON without changing generation behavior.

**Files:**
- Modify: `examples/bench.py`
- Create or modify: `tests/test_benchmark_report.py`
- Modify: `docs/nvidia.md`

**TDD steps:**

1. Extract a pure report-building function and test its schema first.
2. Include exact checkpoint path, backend, device, context/prompt tokens, generated tokens, prefill seconds/tok-s, decode seconds/tok-s, RSS peak, CUDA allocation peak, cache settings, power mode, and git SHA.
3. Add `--json-output PATH`; retain current human-readable output.
4. Verify output on CPU without real weights using a fixture/fake measurement object.
5. Run the existing benchmark manually on Orin and commit only the command/result documentation—not model files.
6. Commit: `feat(bench): emit reproducible inference reports`.

## Task 3: Establish the unoptimized Orin baseline

**Objective:** Learn which constraint kills the current implementation before changing it.

**Target commands:**

```bash
python scripts/jetson_probe.py --model-dir "$EDGE0_8B_MODEL" --output artifacts/orin-probe.json
EDGE0_BACKEND=cuda EDGE0_TORCH_DEVICE=cuda \
  pytest tests/test_nvidia_backend_smoke.py -q
EDGE0_BACKEND=cuda EDGE0_TORCH_DEVICE=cuda \
  python examples/bench.py edge0-8b --json-output artifacts/orin-baseline.json
```

Capture `tegrastats` beside the benchmark and note JetPack, power mode, clocks, NVMe model/filesystem, cooling, and ambient throttling. If the run OOMs, reduce context/generation—not model correctness—and record the first failing allocation.

Commit: `docs(jetson): record Orin Nano baseline`.

## Task 4: Add a memory-budgeted runtime profile

**Objective:** Select safe cache settings from measured available memory rather than copying GB10 defaults.

**Files:**
- Modify: `src/edge0/backends/cuda/backend.py`
- Modify: `src/edge0/backends/cuda/nn.py`
- Modify: model/runtime configuration at the narrowest existing override point
- Test: `tests/test_cuda_backend.py`

Start with policy only—no new kernels. The policy must expose its budget calculation, reserve OS/KV headroom, reject impossible overrides, and print the resolved cache budget. Test the calculation as a pure function before wiring it into the engine.

Commit: `feat(cuda): add memory-budgeted cache profile`.

## Task 5: Replace per-token expert bundle copies

**Objective:** Reuse bounded CUDA expert slots and overlap misses with compute.

**Approach:**

- Allocate fixed slots once.
- Stage safetensors ranges into reusable pinned host buffers.
- Copy on a dedicated CUDA stream.
- Use CUDA events to publish slot readiness.
- Drive prefetch from the prerouter; preserve exact fallback on a miss.
- Add counters for cache hit, predicted hit, late hit, bytes read, bytes copied, and stall time.

First test state transitions and eviction using fake streams/events. Then add CUDA parity tests. Do not add GPUDirect Storage until the staged path is measured; Jetson support and benefit must be demonstrated first.

Commit: `perf(cuda): add asynchronous expert slot cache`.

## Task 6: Fuse quantized gather/dequantize/matmul

**Objective:** Remove repeated Python/Torch unpack operations from decode.

Prototype one kernel for the exact Edge0 INT4 affine layout: packed u32, group size 64, bf16 scales/biases, gathered expert rows. Compare against `quant.gather_qmm` over every broadcast shape already covered by `tests/test_cuda_backend.py`; require the existing numerical tolerance before benchmarking. Prefer a small PyTorch CUDA extension using CUTLASS primitives over a broad TensorRT rewrite.

Commit: `perf(cuda): fuse INT4 expert gather matmul`.

## Task 7: Final Orin validation

Repeat the exact baseline commands with three or more benchmark runs. Report individual runs plus mean/variation; keep prefill and decode separate. Verify CUDA execution from runtime logs, not merely `torch.cuda.is_available()`. Document remaining thermal, storage, context-length, and quality constraints.

## Claude handoff rules

1. Begin by reading upstream PR #19, issue #20, `docs/nvidia.md`, `docs/streaming.md`, and the backend parity tests.
2. Do not rewrite the CUDA backend or replace the existing engines.
3. Keep the first implementation PR limited to Task 1 plus documentation.
4. Never claim Orin support until Gate B runs on the physical device.
5. Every optimization must retain a selectable correctness/reference path.
6. Do not commit checkpoints, generated artifacts, credentials, or machine identifiers.
7. Stop and report after each acceptance gate with measured evidence and the next bottleneck.
