"""Benchmark-owned measurement: backend peaks, process memory, run identity.

``examples/bench.py`` is the only consumer.  Everything that touches a
backend (``torch.cuda``, ``mlx.core``), the process (``psutil``,
``resource``, ``/proc``) or the machine (``git``, ``nvpmodel``) lives
here, behind lazy imports and injectable hooks so the dispatch logic is
unit-tested on a host with neither CUDA nor MLX.  Nothing here runs
inside a timed interval except the explicit ``synchronize`` /
``reset_peak`` / ``peak_metrics`` calls the benchmark makes at phase
boundaries.

The pure report builder (``benchmark_report``) only sees the dicts these
functions return.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any

try:  # imported as ``examples.benchmark_measure`` (tests)
    from examples import benchmark_report as _report
except ImportError:  # run next to bench.py as a script
    import benchmark_report as _report  # type: ignore[no-redef]

memory_metric = _report.memory_metric

_AUTO = object()

#: What a per-run backend peak covers.  The reset happens immediately
#: before prefill, the read after the synchronized final decode step.
RUN_PEAK_SCOPE = (
    "one benchmark run: statistics reset immediately before prefill and "
    "read after the synchronized final decode step; covers prefill, "
    "warmup and decode and includes allocations already resident at the "
    "reset; an earlier initialization transient is not recovered")

_CUDA_ALLOCATED_SCOPE = (
    RUN_PEAK_SCOPE + "; torch caching allocator only, not driver, context "
    "or non-torch memory")

#: ``max_memory_reserved`` is not reset to zero by
#: ``reset_peak_memory_stats``: its floor is every segment the allocator
#: already holds, so it is close to a lifetime figure.
CUDA_RESERVED_SCOPE = (
    "one benchmark run: statistics reset immediately before prefill and "
    "read after the synchronized final decode step; peak of the segments "
    "held by the torch caching allocator, whose floor is every segment "
    "already cached at the reset (model load, earlier runs, any earlier "
    "initialization transient) because the allocator does not release "
    "segments between runs; not driver, context or non-torch memory")


def _bytes(value):
    """Integer bytes from a backend counter; floats that are whole numbers
    are accepted, anything else is left for ``memory_metric`` to reject."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# ---- backend measurement adapters --------------------------------------------


class NullMeasurement:
    """No backend peak available (torch on cpu/mps, unknown backend)."""

    kind = "null"
    headline = "backend_peak"

    def __init__(self, backend: str, device: str, reason: str):
        self.backend = backend
        self.device = device
        self.reason = reason

    def synchronize(self) -> None:
        return None

    def reset_peak(self) -> None:
        return None

    def peak_metrics(self) -> dict:
        return {self.headline: memory_metric(
            None, method="none", scope=RUN_PEAK_SCOPE,
            unavailable_reason=self.reason)}

    def describe(self) -> dict:
        sync = "none: the benchmark synchronizes only an actual CUDA device"
        if str(self.device).startswith("mps"):
            sync += ("; mps execution is asynchronous, so prefill/decode wall "
                     "times may exclude the final step's in-flight device work")
        return {"kind": self.kind, "backend": self.backend,
                "resolved_device": self.device,
                "synchronization": sync,
                "peak_source": f"none: {self.reason}",
                "headline_peak": self.headline}


class MlxMeasurement:
    """The pre-existing MLX behavior: ``reset_peak_memory`` before
    prefill, ``get_peak_memory`` after decode, no added synchronization
    (the engine's ``core.eval(logits)`` already materializes every step,
    so the wall time is unchanged from the original benchmark)."""

    kind = "mlx"
    headline = "mlx_peak_active"

    def __init__(self, core):
        self._core = core

    def synchronize(self) -> None:
        return None

    def reset_peak(self) -> None:
        self._core.reset_peak_memory()

    def peak_metrics(self) -> dict:
        return {self.headline: memory_metric(
            _bytes(self._core.get_peak_memory()),
            method="mlx.core.get_peak_memory", scope=RUN_PEAK_SCOPE)}

    def describe(self) -> dict:
        default = getattr(self._core, "default_device", None)
        try:
            device = str(default()) if callable(default) else "mlx default"
        except Exception as exc:  # noqa: BLE001 - never fail a report here
            device = f"mlx default (query failed: {type(exc).__name__})"
        return {"kind": self.kind, "backend": "mlx",
                "resolved_device": device,
                "synchronization": (
                    "none added: MLX evaluation is forced by the engine "
                    "(core.eval) at every step; timing behavior unchanged"),
                "peak_source": "mlx.core.get_peak_memory (reset before prefill)",
                "headline_peak": self.headline}


class CudaMeasurement:
    """Torch on an actual CUDA device: synchronize the resolved device at
    phase boundaries and read the caching-allocator peaks.  The headline
    (human output, legacy ``peak_gib``) is ``max_memory_allocated``, the
    analogue of MLX's peak active memory; ``max_memory_reserved`` is
    JSON-only."""

    kind = "cuda"
    headline = "cuda_peak_allocated"

    def __init__(self, device, cuda_api):
        self.device = device
        self._cuda = cuda_api
        self._at_reset: dict[str, Any] = {}

    def synchronize(self) -> None:
        self._cuda.synchronize(self.device)

    def reset_peak(self) -> None:
        self._cuda.reset_peak_memory_stats(self.device)
        # What is resident right now is the floor of both peaks; record it
        # (attribute reads, before the prefill clock starts).
        self._at_reset = {}
        for key, name in (("allocated_at_reset", "memory_allocated"),
                          ("reserved_at_reset", "memory_reserved")):
            reader = getattr(self._cuda, name, None)
            if callable(reader):
                self._at_reset[key] = _bytes(reader(self.device))

    def peak_metrics(self) -> dict:
        allocated = _bytes(self._cuda.max_memory_allocated(self.device))
        reserved = _bytes(self._cuda.max_memory_reserved(self.device))
        alloc_details = {k: v for k, v in self._at_reset.items()
                         if k == "allocated_at_reset"}
        res_details = {k: v for k, v in self._at_reset.items()
                       if k == "reserved_at_reset"}
        return {
            "cuda_peak_allocated": memory_metric(
                allocated, method="torch.cuda.max_memory_allocated",
                scope=_CUDA_ALLOCATED_SCOPE, **alloc_details),
            "cuda_peak_reserved": memory_metric(
                reserved, method="torch.cuda.max_memory_reserved",
                scope=CUDA_RESERVED_SCOPE, **res_details),
        }

    def describe(self) -> dict:
        return {"kind": self.kind, "backend": "cuda",
                "resolved_device": str(self.device),
                "synchronization": (
                    "torch.cuda.synchronize(device) after engine reset, "
                    "at the end of prefill, at the end of warmup and after "
                    "the final timed decode step; no per-token syncs"),
                "peak_source": (
                    "torch.cuda caching allocator statistics, "
                    "reset_peak_memory_stats before prefill"),
                "headline_peak": self.headline}


def _torch_cuda():
    import torch
    return torch.cuda


def make_measurement(backend_name: str, core_module, *, cuda_api=None):
    """Pick the adapter for the ACTIVE backend and its RESOLVED device.

    ``EDGE0_BACKEND=cuda`` alone does not prove the device is CUDA: the
    torch backend runs on cpu/mps when ``EDGE0_TORCH_DEVICE`` says so or
    when no CUDA device exists, and then allocator peaks are simply
    unavailable.  ``resolved_device`` and the unavailable reason are both
    derived from the same ``core_module.DEVICE`` object.
    """
    if backend_name == "mlx":
        if (callable(getattr(core_module, "reset_peak_memory", None))
                and callable(getattr(core_module, "get_peak_memory", None))):
            return MlxMeasurement(core_module)
        return NullMeasurement(
            "mlx", "mlx default",
            "mlx core namespace has no reset_peak_memory/get_peak_memory; "
            "peak memory unavailable")
    if backend_name == "cuda":
        device = getattr(core_module, "DEVICE", None)
        if device is None:
            return NullMeasurement(
                "cuda", "unknown",
                "torch core namespace has no DEVICE; resolved device unknown")
        dev_type = getattr(device, "type", str(device))
        if dev_type == "cuda":
            return CudaMeasurement(device, cuda_api or _torch_cuda())
        return NullMeasurement(
            "cuda", str(device),
            f"torch device is {dev_type}, not cuda: torch allocator peaks "
            f"are unavailable")
    return NullMeasurement(
        backend_name, "unknown",
        f"backend {backend_name!r} exposes no peak-memory interface")


def execution_evidence(logits) -> dict:
    """What the engine actually produced: the array class and, when the
    class carries one, its device (an attribute read, no sync)."""
    cls = type(logits)
    array_type = f"{cls.__module__}.{cls.__name__}"
    evidence: dict[str, Any] = {"array_type": array_type, "device": None,
                                "unavailable_reasons": {}}
    device = getattr(logits, "device", None)
    if device is None:
        evidence["unavailable_reasons"]["device"] = (
            f"{array_type} carries no device attribute (MLX arrays are "
            f"device-agnostic); see runtime.measurement.resolved_device")
    else:
        evidence["device"] = str(device)
    return evidence


# ---- process memory --------------------------------------------------------------


_RSS_COMPOSITION = (
    "resident set size = anonymous memory plus resident file-backed pages "
    "(the mmap'd checkpoint: prod_k8 whole-layer prefill touches every "
    "expert of model.safetensors each run, and those pages are reclaimable "
    "page cache), so it is not the process's anonymous footprint and not "
    "comparable to the 'peak anonymous memory' figure in docs/nvidia.md; "
    "must not be added to device allocator peaks")

_RSS_LIFETIME_SCOPE = (
    "process lifetime high-water mark of " + _RSS_COMPOSITION
    + "; includes model load and every run; cannot be reset per run")


def _rusage_self():
    try:
        import resource
        return getattr(resource, "RUSAGE_SELF", 0)
    except ImportError:
        return 0


def process_peak_rss(*, system=None, getrusage: Any = _AUTO,
                     psutil_module: Any = _AUTO) -> dict:
    """Process-lifetime peak RSS with a platform-specific method and unit
    normalization to bytes."""
    system = system or platform.system()
    if getrusage is _AUTO:
        try:
            import resource
            getrusage = getattr(resource, "getrusage", None)
        except ImportError:
            getrusage = None
    if psutil_module is _AUTO:
        try:
            import psutil as psutil_module
        except ImportError:
            psutil_module = None

    if system == "Linux":
        if getrusage is None:
            return memory_metric(None, method="none",
                                 scope=_RSS_LIFETIME_SCOPE,
                                 unavailable_reason="resource module unavailable")
        return memory_metric(
            int(getrusage(_rusage_self()).ru_maxrss) * 1024,
            method="resource.getrusage(RUSAGE_SELF).ru_maxrss (KiB x 1024)",
            scope=_RSS_LIFETIME_SCOPE)
    if system == "Darwin":
        if getrusage is None:
            return memory_metric(None, method="none",
                                 scope=_RSS_LIFETIME_SCOPE,
                                 unavailable_reason="resource module unavailable")
        return memory_metric(
            int(getrusage(_rusage_self()).ru_maxrss),
            method="resource.getrusage(RUSAGE_SELF).ru_maxrss (bytes)",
            scope=_RSS_LIFETIME_SCOPE)
    if system == "Windows":
        if psutil_module is None:
            return memory_metric(None, method="none",
                                 scope=_RSS_LIFETIME_SCOPE,
                                 unavailable_reason="psutil unavailable")
        return memory_metric(
            int(psutil_module.Process().memory_info().peak_wset),
            method="psutil.Process().memory_info().peak_wset",
            scope=_RSS_LIFETIME_SCOPE)
    return memory_metric(
        None, method="none", scope=_RSS_LIFETIME_SCOPE,
        unavailable_reason=f"no process peak RSS method for platform {system!r}")


_PROC_STATUS_FIELDS = (("VmRSS", "rss"), ("RssAnon", "rss_anon"),
                       ("RssFile", "rss_file"), ("VmSwap", "swap"))


def parse_proc_status(text: str) -> dict:
    """``/proc/self/status`` -> bytes for rss / rss_anon / rss_file / swap
    (``None`` where the kernel does not report a field)."""
    values: dict[str, int | None] = {key: None for _, key in _PROC_STATUS_FIELDS}
    wanted = dict(_PROC_STATUS_FIELDS)
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        if not sep or name not in wanted:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            number = int(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else "bytes"
        values[wanted[name]] = number * (1024 if unit == "kb" else 1)
    return values


def read_process_memory() -> dict | None:
    """Current process memory composition: ``/proc/self/status`` on Linux
    (rss, anonymous, file-backed, swap), else psutil rss only, else
    ``None``.  The ``method`` key names the source used."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            parsed = parse_proc_status(fh.read())
        if parsed["rss"] is not None:
            parsed["method"] = "/proc/self/status (VmRSS, RssAnon, RssFile, VmSwap)"
            return parsed
    except OSError:
        pass
    try:
        import psutil
        rss = int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return {"rss": rss, "rss_anon": None, "rss_file": None, "swap": None,
            "method": "psutil.Process().memory_info().rss"}


_SAMPLED_METRICS = (
    ("rss", "process_sampled_peak_rss",
     "sampled peak of process " + _RSS_COMPOSITION),
    ("rss_anon", "process_sampled_peak_rss_anon",
     ("sampled peak of the process's anonymous (non-file-backed) resident "
      "pages; the closest match to the 'peak anonymous memory' figure in "
      "docs/nvidia.md")),
    ("rss_file", "process_sampled_peak_rss_file",
     ("sampled peak of the process's resident file-backed pages (mmap'd "
      "checkpoint page cache, reclaimable)")),
    ("swap", "process_sampled_peak_swap",
     ("sampled peak of swapped-out process memory; a non-zero value marks "
      "a swapping run, which cannot establish a resident-memory target")),
)


class RssSampler:
    """Lightweight periodic process-memory sampler (one daemon thread).

    Start it BEFORE the model is built so load transients are covered.
    The result is a *sampled* peak: spikes shorter than the interval are
    missed, and the report says so.  On Linux the anonymous, file-backed
    and swap components are tracked too.
    """

    def __init__(self, interval_seconds: float, *, read_rss=None,
                 read_memory=None, join_timeout: float | None = None):
        self.interval_seconds = float(interval_seconds)
        self.join_timeout = (float(join_timeout) if join_timeout is not None
                             else max(5.0, 4 * self.interval_seconds))
        if read_rss is not None:
            self._read = lambda: {"rss": int(read_rss())}
            self._method = "injected read_rss"
        elif read_memory is not None:
            self._read = read_memory
            self._method = "injected read_memory"
        else:
            self._read = None
            self._method = "none"
        self._peaks: dict[str, int | None] = {
            key: None for key, _, _ in _SAMPLED_METRICS}
        self._count = 0
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def peak(self) -> int | None:
        return self._peaks["rss"]

    @property
    def sample_count(self) -> int:
        return self._count

    def _sample(self) -> None:
        read = self._read
        if read is None:
            return
        try:
            mem = read()
        except Exception as exc:  # noqa: BLE001 - sampling must never raise
            self._error = f"{type(exc).__name__}: {exc}"
            return
        if not mem:
            self._error = "memory reader returned nothing"
            return
        method = mem.get("method")
        if method:
            self._method = str(method)
        self._count += 1
        for key in self._peaks:
            value = mem.get(key)
            if value is None:
                continue
            value = int(value)
            current = self._peaks[key]
            self._peaks[key] = value if current is None else max(current, value)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self.interval_seconds <= 0 or self._started:
            return
        if self._read is None:
            probe = read_process_memory()
            if probe is None:
                self._error = "no process memory reader (no /proc, no psutil)"
                return
            self._read = read_process_memory
        self._started = True
        self._sample()
        self._thread = threading.Thread(
            target=self._loop, name="bench-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling (idempotent).  The thread reference is only
        dropped once the thread has actually exited, so ``is_running``
        stays true (and the metric carries a reason) if a reader hangs
        past ``join_timeout``."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=self.join_timeout)
        if thread.is_alive():
            self._error = (f"sampler thread did not stop within "
                           f"{self.join_timeout:g}s (reader blocked?)")
            return
        self._thread = None
        self._sample()

    def metrics(self) -> dict:
        """All sampled peaks as memory metric objects (rss always; the
        composition only where the reader supplied it)."""
        out = {}
        for key, name, scope in _SAMPLED_METRICS:
            details = {"interval_seconds": self.interval_seconds,
                       "sample_count": self._count}
            if self.interval_seconds <= 0:
                out[name] = memory_metric(
                    None, method="none", scope=scope,
                    unavailable_reason="sampler disabled (interval 0)",
                    interval_seconds=self.interval_seconds)
                continue
            value = self._peaks[key]
            if value is not None:
                if self._error and self.is_running:
                    details["warning"] = self._error
                out[name] = memory_metric(value, method=self._method,
                                          scope=scope, **details)
            elif key == "rss":
                out[name] = memory_metric(
                    None, method=self._method, scope=scope,
                    unavailable_reason=self._error or "sampler never ran",
                    **details)
            else:
                out[name] = memory_metric(
                    None, method=self._method, scope=scope,
                    unavailable_reason=(
                        f"{key} needs /proc/self/status (Linux); the "
                        f"memory reader did not supply it"),
                    **details)
        return out

    def metric(self) -> dict:
        return self.metrics()["process_sampled_peak_rss"]


# ---- run identity ------------------------------------------------------------------


def git_identity(repo_dir, *, run=subprocess.run) -> dict:
    """HEAD commit and tracked-file dirty state of ``repo_dir``."""
    out: dict[str, Any] = {"commit": None, "dirty": None,
                           "unavailable_reasons": {}}

    def _git(*args):
        return run(["git", "-C", str(repo_dir), *args], capture_output=True,
                   text=True, timeout=15)

    try:
        head = _git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        reason = f"git unavailable: {type(exc).__name__}: {exc}"
        out["unavailable_reasons"] = {"commit": reason, "dirty": reason}
        return out
    if head.returncode != 0 or not head.stdout.strip():
        reason = (f"git rev-parse failed (exit {head.returncode}): "
                  f"{(head.stderr or '').strip() or 'no output'}")
        out["unavailable_reasons"] = {"commit": reason, "dirty": reason}
        return out
    out["commit"] = head.stdout.strip()
    try:
        status = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError) as exc:
        out["unavailable_reasons"]["dirty"] = (
            f"git status failed: {type(exc).__name__}: {exc}")
        return out
    if status.returncode != 0:
        out["unavailable_reasons"]["dirty"] = (
            f"git status failed (exit {status.returncode})")
        return out
    out["dirty"] = bool(status.stdout.strip())
    return out


_NVPMODEL_STATUS = Path("/var/lib/nvpmodel/status")


def power_mode(*, run=subprocess.run, status_path=_NVPMODEL_STATUS) -> dict:
    """Jetson power mode via ``nvpmodel -q``, else the nvpmodel status
    file, else an explicit unavailable reason."""
    out: dict[str, Any] = {"value": None, "mode_id": None, "method": None,
                           "unavailable_reasons": {}}
    reasons = []
    try:
        proc = run(["nvpmodel", "-q"], capture_output=True, text=True,
                   timeout=10)
        if proc.returncode == 0:
            name = None
            mode_id = None
            for line in (proc.stdout or "").splitlines():
                text = line.strip()
                if text.lower().startswith("nv power mode:"):
                    name = text.split(":", 1)[1].strip()
                elif name is not None and mode_id is None and text.isdigit():
                    mode_id = int(text)
            if name:
                out.update(value=name, mode_id=mode_id, method="nvpmodel -q")
                if mode_id is None:
                    out["unavailable_reasons"]["mode_id"] = (
                        "nvpmodel -q printed no numeric mode id")
                return out
            reasons.append("nvpmodel -q printed no 'NV Power Mode' line")
        else:
            reasons.append(
                f"nvpmodel -q exited {proc.returncode}: "
                f"{(proc.stderr or '').strip() or 'no output'}")
    except (OSError, subprocess.SubprocessError) as exc:
        reasons.append(f"nvpmodel not runnable: {type(exc).__name__}: {exc}")
    try:
        text = Path(status_path).read_text(errors="replace")
        match = re.search(r"pmode:(\d+)", text)
        if match:
            mode_id = int(match.group(1))
            out.update(value=f"pmode {mode_id}", mode_id=mode_id,
                       method=f"{status_path} pmode field")
            return out
        reasons.append(f"{status_path} has no pmode field")
    except OSError as exc:
        reasons.append(f"{status_path}: {type(exc).__name__}")
    reason = ("; ".join(reasons)
              + " (not a Jetson, or nvpmodel is not on PATH)")
    out["unavailable_reasons"] = {"value": reason, "mode_id": reason,
                                  "method": reason}
    return out


# ---- checkpoint identity -----------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path) -> dict:
    """``__metadata__``, tensor count and a hash of the JSON header of a
    safetensors file: an 8-byte length prefix plus the header, no weight
    bytes touched."""
    out: dict[str, Any] = {"metadata": {}, "tensor_count": None,
                           "header_sha256": None, "unavailable_reasons": {}}
    try:
        with open(path, "rb") as fh:
            (length,) = struct.unpack("<Q", fh.read(8))
            if length > (256 << 20):
                raise ValueError(f"header length {length} is implausible")
            raw = fh.read(length)
        header = json.loads(raw.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("header is not a JSON object")
        metadata = header.get("__metadata__")
        out["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
        out["tensor_count"] = len([k for k in header if k != "__metadata__"])
        out["header_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, struct.error) as exc:
        reason = f"safetensors header unreadable: {type(exc).__name__}: {exc}"
        out["unavailable_reasons"] = {"tensor_count": reason,
                                      "header_sha256": reason}
    return out


MANIFEST_ALGORITHM = ("sha256 over 'path\\tsize\\n' lines, one per regular "
                      "file, sorted by relative POSIX path")


def checkpoint_manifest(model_dir, *, hash_limit_bytes: int = 32 << 20) -> dict:
    """Identify a checkpoint directory without hashing weights.

    Every regular file contributes ``(relative path, size)`` to the
    manifest digest and the retained ``files`` list.  Files up to
    ``hash_limit_bytes`` that are not safetensors are content-hashed
    (config, chat template, tokenizer); safetensors files contribute
    their header metadata, tensor count and header hash.
    """
    out: dict[str, Any] = {
        "manifest": {"algorithm": MANIFEST_ALGORITHM, "file_count": None,
                     "total_bytes": None, "sha256": None, "files": [],
                     "unavailable_reasons": {}},
        "small_file_sha256": {},
        "safetensors_headers": {},
        "config": {},
        "config_sha256": None,
    }
    root = Path(model_dir)
    if not root.is_dir():
        reason = f"checkpoint directory {str(root)!r} does not exist"
        out["manifest"]["unavailable_reasons"] = {
            "file_count": reason, "total_bytes": reason, "sha256": reason}
        return out
    entries = []
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        entries.append((rel, size))
        total += size
        if path.suffix == ".safetensors":
            out["safetensors_headers"][rel] = safetensors_header(path)
        elif size <= hash_limit_bytes:
            try:
                out["small_file_sha256"][rel] = _sha256_file(path)
            except OSError:
                pass
    entries.sort()
    digest = hashlib.sha256()
    for rel, size in entries:
        digest.update(f"{rel}\t{size}\n".encode())
    out["manifest"].update(
        file_count=len(entries), total_bytes=total,
        sha256=digest.hexdigest(),
        files=[{"path": rel, "size_bytes": size} for rel, size in entries])
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)
            if isinstance(config, dict):
                out["config"] = config
        except (OSError, ValueError):
            pass
        out["config_sha256"] = out["small_file_sha256"].get("config.json")
    return out


def adapter_identity(path, checkpoint_dir, *,
                     sha256_max_bytes: int = 256 << 20) -> dict:
    """Identity of the adapter file the engine actually applied
    (``engine.cfg.lora`` / ``engine.cfg.prerouter.weights_file``), which
    may live outside the checkpoint directory (``artifacts/`` fallback).
    Adapter files are small enough to hash; the cap keeps a misconfigured
    path from hashing a weight shard."""
    out: dict[str, Any] = {"file": None, "path": None,
                           "inside_checkpoint_dir": None, "size_bytes": None,
                           "sha256": None, "safetensors_header": None,
                           "unavailable_reasons": {}}
    reasons = out["unavailable_reasons"]
    if not path:
        reason = "disabled: no adapter file configured for this tier"
        for key in ("file", "path", "inside_checkpoint_dir", "size_bytes",
                    "sha256", "safetensors_header"):
            reasons[key] = reason
        return out
    p = Path(os.fspath(path))
    resolved = p.resolve()
    out["path"] = str(resolved)
    root = Path(checkpoint_dir).resolve()
    try:
        rel = resolved.relative_to(root).as_posix()
        inside = True
    except ValueError:
        rel = None
        inside = False
    out["inside_checkpoint_dir"] = inside
    out["file"] = rel if inside else str(resolved)
    if not p.is_file():
        reason = f"adapter file not found: {resolved}"
        for key in ("size_bytes", "sha256", "safetensors_header"):
            reasons[key] = reason
        return out
    size = p.stat().st_size
    out["size_bytes"] = size
    if size <= sha256_max_bytes:
        out["sha256"] = _sha256_file(p)
    else:
        reasons["sha256"] = (f"not hashed: {size} bytes is larger than the "
                             f"{sha256_max_bytes}-byte cap")
    if p.suffix == ".safetensors":
        out["safetensors_header"] = safetensors_header(p)
    else:
        reasons["safetensors_header"] = "not a safetensors file"
    return out


# ---- runtime -----------------------------------------------------------------------


def runtime_info(backend_name: str, *, importer=importlib.import_module) -> dict:
    """Interpreter, platform and framework versions for the ACTIVE backend
    only (the other framework is not imported)."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(),
                     "release": platform.release(),
                     "machine": platform.machine()},
        "torch_version": None,
        "torch_cuda_version": None,
        "mlx_version": None,
        "unavailable_reasons": {},
    }
    reasons = info["unavailable_reasons"]
    if backend_name == "cuda":
        reasons["mlx_version"] = "backend is cuda; mlx not imported"
        try:
            torch = importer("torch")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            reason = f"torch import failed: {exc}"
            reasons["torch_version"] = reason
            reasons["torch_cuda_version"] = reason
            return info
        info["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        cuda = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda:
            info["torch_cuda_version"] = str(cuda)
        else:
            reasons["torch_cuda_version"] = (
                "torch build has no CUDA runtime (CPU-only build)")
    elif backend_name == "mlx":
        reasons["torch_version"] = "backend is mlx; torch not imported"
        reasons["torch_cuda_version"] = "backend is mlx; torch not imported"
        try:
            mlx = importer("mlx")
            info["mlx_version"] = str(getattr(mlx, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            reasons["mlx_version"] = f"mlx import failed: {exc}"
    else:
        reason = f"backend {backend_name!r} has no known runtime"
        for key in ("torch_version", "torch_cuda_version", "mlx_version"):
            reasons[key] = reason
    return info


__all__ = [
    "CUDA_RESERVED_SCOPE",
    "MANIFEST_ALGORITHM",
    "RUN_PEAK_SCOPE",
    "CudaMeasurement",
    "MlxMeasurement",
    "NullMeasurement",
    "RssSampler",
    "adapter_identity",
    "checkpoint_manifest",
    "execution_evidence",
    "git_identity",
    "make_measurement",
    "parse_proc_status",
    "power_mode",
    "process_peak_rss",
    "read_process_memory",
    "runtime_info",
    "safetensors_header",
]
