# Edge0 Orin plan review and change justifications

Reviewed 14 September 2026.

**Assessment:** retain the staged Torch/CUDA approach, but correct the execution gates and benchmark prerequisites before implementing Task 2. The reviewed benchmark cannot currently run through its first memory reset on the CUDA backend. The baseline smoke command tests MLX, and the proposed memory/prefetch policies omit behaviors that could defeat their correctness or bounds.

The attached file was treated as a document to review, not authorization to execute its embedded Claude prompt. Its referenced full plan, relevant implementation and tests were inspected through a separate public-repository checkout. No Edge0 implementation was changed, no commit/push was made, and no Jetson inference was run.

## Deliverables

- Revised Claude handoff: delivered outside the repository (it names a local checkout path and a private evidence directory).
- [Revised full engineering plan](jetson-orin-nano.md): now the repository plan.
- This review: justification and evidence for every substantive change, identified as C01–C16 in the revised documents.

## Source and verification scope

The original `edge0-orin-claude.md` remains unchanged. SHA-256:

```text
7E2EC53ED180A5BCF91800A61E435B2C18AD57A434AF90C05F59A3F00D8FA455
```

The public [draft PR #1](https://github.com/KarlTaylorKnight/Edge0/pull/1) was inspected on the review date. The code and full plan were read at commit `dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe`; links below pin that snapshot. The Mac checkout path, its working-tree state and historical test outputs in the attachment were not independently inspected. Upstream PR discussion is useful provenance, not physical Orin evidence.

## Changes and reasons

### C01 — Pin the reviewed source and qualify inherited claims

**Change:** add the reviewed commit, distinguish the supplied Mac path from the inspected public checkout, and label prior Mac checks as historical reports. Retain the PR #19 base and separate it from issue #20's MLX experiment.

**Why:** mutable branches and upstream discussions can change. Without a source revision, a later implementer cannot tell whether a finding still applies. The attachment's test claims and another GPU's performance are not fresh Orin results. The [PR](https://github.com/KarlTaylorKnight/Edge0/pull/1) establishes the public branch context; the [reviewed NVIDIA notes](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/docs/nvidia.md) describe the different experiments.

### C02 — Make Task 2's stopping point consistent

**Change:** finish portable reporting in Task 2; put physical benchmark execution in Task 3. Replace “baseline captured and committed” with retained evidence and a reviewed summary. Permit a documented failed baseline attempt to lead to memory-policy correction, while requiring a successful reference run before performance optimization.

**Why:** the handoff says to stop after Task 2, but the [full plan](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/docs/plans/jetson-orin-nano.md) requires an Orin run inside Task 2 and again in Task 3. It also alternates between committing results and forbidding generated artifacts. These contradictions can block useful host work or encourage premature hardware claims. An OOM baseline needs a recovery path rather than a circular prerequisite.

### C03 — Repair the benchmark's missing CUDA measurement interface

**Change:** add a benchmark-owned backend-aware measurement adapter and fake-engine integration tests before serializing results. Keep optional runtime imports out of the pure builder.

**Why:** the [benchmark, lines 95 and 120](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/examples/bench.py#L95) calls peak-memory methods absent from the [CUDA core](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/backends/cuda/core.py). JSON-builder tests would miss the immediate runtime failure. A small benchmark adapter fixes the consumer without expanding the task into a backend rewrite.

### C04 — Define synchronized wall-clock timing

**Change:** synchronize the actual CUDA device at prefill/decode boundaries, including the end of warmup and the final timed step. Keep storage, transfer waits and sampling inside the relevant wall-time interval.

**Why:** CUDA work is asynchronous and [CUDA `core.eval()`](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/backends/cuda/core.py#L292) does not synchronize. Existing incidental synchronizations do not define a reliable measurement contract. Kernel event time alone also omits the host/I/O costs central to this application. [PyTorch documents synchronization requirements for accurate CUDA timing](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution).

### C05 — Specify what the report actually measures

**Change:** version the benchmark schema; record requested and resolved configuration, backend and actual device, per-run results, sampling settings, input/model identity and an actual probe digest/reference. Distinguish prompt, warmup, timed and total generated tokens. Preserve and label the existing fixed-length sample-and-step protocol.

**Why:** the [current loop](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/examples/bench.py#L93) uses greedy warmup despite its comment, executes an additional forward for every emitted timed token, and ignores EOS. Reporting these numbers as user-visible completion latency would be misleading. Also, the [Torch backend can select CPU](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/backends/cuda/core.py#L38); a backend label is not device evidence. A probe version alone cannot identify which machine-state report accompanied a run.

### C06 — Give memory metrics correct units, scope and limitations

**Change:** use integer bytes and measurement methods; separate process-lifetime RSS peaks, sampled peaks, CUDA allocated/reserved peaks and system telemetry. State precisely when CUDA peak statistics reset and exclude earlier loading transients from that per-run claim. Use null/reasons when metrics are unavailable.

**Why:** an endpoint sample is not a peak, and a process high-water mark cannot be reset by subtracting earlier values. PyTorch allocation statistics cover its allocator, not every device allocation. On Tegra, CPU and GPU allocations share physical DRAM; RSS plus CUDA peaks is not a valid total-footprint formula. The combination of scoped metrics and system observations supports an honest fit assessment. [PyTorch allocator documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management), [NVIDIA Tegra memory architecture](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-for-tegra-appnote/index.html#memory-management).

### C07 — Test collection, CLI failure and cleanup as well as fields

**Change:** extend TDD to dispatch, synchronization order, count arithmetic, unavailable/invalid metrics, strict JSON, atomic writing and engine cleanup. Require a new output path so stale success artifacts cannot be mistaken for a failed run's result. Keep mocked and device evidence separate.

**Why:** a schema test cannot detect an incorrect timer, CPU fallback, partial file or leaked engine. The [current CLI](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/examples/bench.py#L134) closes its engine only after successful benchmarking. Python's JSON encoder permits NaN/infinity by default, so a nominal JSON result can fail strict downstream consumers unless validated. [Python JSON documentation](https://docs.python.org/3.10/library/json.html). These are focused behavioral tests for the new reporting feature, not an unrelated testing expansion.

### C08 — Add a reproducible Jetson installation prerequisite

**Change:** resolve the actual board/L4T/JetPack/Python/Torch/CUDA combination, required features and tokenizer imports before inference. Audit backend dependency separation and preserve Apple defaults.

**Why:** “Python 3.10+, PyTorch for Jetson” does not define an installable stack. The reviewed [package metadata](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/pyproject.toml) mandates MLX while [CUDA tokenizer loading](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/backends/cuda/io.py#L141) imports Transformers, and CUDA modules require specific Torch APIs. NVIDIA ties its builds and prerequisites to particular JetPack versions. The plan therefore requires a tested combination rather than inventing a wheel/version from an unknown device. [Installation requirements](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html), [compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).

### C09 — Limit what a successful probe certifies

**Change:** retain the completed probe but explicitly check Nano 8 GB identity, missing memory evidence, environment compatibility and checkpoint completeness separately. Describe Linux-specific test prerequisites.

**Why:** [the probe](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/scripts/jetson_probe.py#L244) matches generic Jetson/Orin model strings, queries CUDA properties and checks storage. Its blockers do not validate the 8 GB Nano variant, model contents, enough RAM or successful kernel execution. This is a limitation of its current contract, not grounds to discard the probe. A passing probe must not silently advance the entire acceptance gate.

### C10 — Replace the incorrect hardware acceptance test

**Change:** add a Torch-only test that requires and exercises CUDA, plus identified reference fixtures for on-device correctness. Retain existing MLX comparisons on supported hosts and define numerical tolerances before the run.

**Why:** [the nominated NVIDIA smoke test](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/tests/test_nvidia_backend_smoke.py#L1) imports MLX directly. Setting `EDGE0_BACKEND=cuda` does not turn it into a Torch test. Nor is [test_cuda_backend.py](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/tests/test_cuda_backend.py#L14) a substitute: it requires MLX and its helper creates CPU tensors. The original “tokens or logits” criterion also allows the metric to be chosen after a mismatch; separate predefined comparisons avoid that ambiguity.

### C11 — Replace the guessed footprint threshold with enforceable budgets

**Change:** derive an explicit byte budget from observed conditions and the declared workload, accounting for load/prefill/KV/transient, staging and inflight allocations. Reject calculated zero capacities unless an explicit disabled-cache mode is implemented. Inspect configuration and cache construction, not only CUDA namespace files.

**Why:** a universal 6.5 GB RSS ceiling does not establish system headroom. More concretely, [SharedExpertCache and PrefetchBuffer](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/streaming/cache.py#L19) evict only for positive capacities: zero is unbounded. The [prefetch implementation](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/streaming/layer.py#L456) also has inflight work and a path that grows capacity. Ignoring those behaviors defeats a nominally bounded LRU policy.

### C12 — Preserve the actual 8B routing profile

**Change:** connect future prefetch/cache work to the existing exact path; retain required expert contributions on misses. Do not enable staged decoding merely to activate prefetch.

**Why:** [the 8B configuration](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/models/edge0_8b/__init__.py#L48) selects `prod_k8`, whose [definition deliberately disables staging](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/streaming/options.py#L112). The [engine](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/engine/ling.py#L70) passes only staged layers to the slot stager, while existing staged misses can use a zero row. A scheduling optimization must not silently become a routing/quality change.

### C13 — Complete the asynchronous ownership contract

**Change:** require both copy-completion and consumer-completion ordering, slot generations, pinned-buffer lifetime, bounded queues, failure/reset/close handling and stress tests. Treat pinned staging as a measured candidate on shared DRAM.

**Why:** a readiness event prevents reading an incomplete copy but does not prevent overwriting a slot still used by a previous kernel. Reusing a pinned source too early can also corrupt a copy. Persistent asynchronous resources need stronger lifecycle handling than the current [reset/close logic](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/streaming/layer.py#L1378). [PyTorch stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-streams) support these ordering requirements; [NVIDIA's Tegra guide](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-for-tegra-appnote/index.html#memory-management) explains why duplicated host/device buffers share the same physical capacity.

### C14 — Make kernel work conditional and align Gate D with its scope

**Change:** select kernels from the measured profile, add guarded dispatch/fallback and target toolchain validation, and distinguish expert gather from dense quantized-linear optimization. Limit the allocation-free promise to expert-transfer payload reuse.

**Why:** an INT4-only implementation cannot replace all the [existing quantization cases](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/backends/cuda/quant.py), and tensor outputs/intermediates still allocate. A fused expert gather also leaves [dense QuantizedLinear dequantization](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/src/edge0/backends/cuda/nn.py#L93) untouched. The revised gate avoids promising a larger kernel/allocator rewrite than the staged plan supplies, while retaining the opportunity to address dense work when measured evidence supports it.

### C15 — Make before/after measurements comparable

**Change:** freeze the workload, collect at least three independent launches on both baseline and optimized versions, retain each internal run, distinguish warm/cold assumptions, and record sustained thermal behavior and failed attempts. A reduced context is a separate experiment.

**Why:** the [benchmark has two internal runs](https://github.com/KarlTaylorKnight/Edge0/blob/dc1d9b3b78f51eb1a5f3705d63ed16c5b2cc9dfe/examples/bench.py#L93), and engine reset does not guarantee cold storage or all caches. Comparing a single cold baseline with a warm optimized mean can attribute caching or thermal differences to the code. Power/thermal/system telemetry supplies the context needed to interpret the result. [NVIDIA tegrastats documentation](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

### C16 — Separate reproducible local evidence from shared artifacts

**Change:** store exact local paths and raw reports in an untracked evidence set; put sanitized summaries and content references in documentation. Hash/reference the specific probe and checkpoint manifest outside timed regions. Permit small reviewed deterministic test fixtures as test assets. Keep the supplied source unchanged and deliver revised copies.

**Why:** exact paths and command strings can reveal usernames/machine details, conflicting with the original no-identifiers/no-generated-artifacts rules when copied into Git. A digest provides identity but must reference retained inputs to support reproduction. Keeping both source and revised copies makes the review auditable. This clarifies the existing artifact restriction rather than introducing a publication requirement.

## Validation performed

The review checked the pinned source, test imports, benchmark calls, cache capacity semantics, profile selection and supporting NVIDIA/PyTorch documentation. The source checkout was kept free of implementation changes. The original attachment's SHA-256 was recorded for preservation verification.

On Windows/Python 3.14, this command ran:

```text
uv run --no-project --with pytest python -m pytest tests/test_jetson_probe.py -q
14 failed, 10 passed in 0.79s
```

Failures include POSIX sysfs fixture names such as `259:1`, Linux mount-path assumptions and unavailable `os.major` on Windows. They are actual failures on that host, not evidence of an Orin regression and not a reproduction of the historical Mac test result.

The same unchanged tests then ran under Ubuntu/WSL with Python 3.12 and pytest 9.1.1, using isolated test packages in the review workspace:

```text
PYTHONPATH=<review-workspace>/work/pytest-linux-packages python3 -m pytest tests/test_jetson_probe.py -q
24 passed in 0.36s
```

The initial WSL interpreter had neither pytest nor working venv bootstrapping. Test packages were installed into a task-local target directory; no system Python packages or operating-system packages were changed. This is a Linux fixture test result, not an on-device readiness result.

No full backend/parity suite or physical CUDA/Orin benchmark was run during the review. The current benchmark failure is established by its calls and the missing functions in the selected backend; a model was not downloaded to reproduce it.
