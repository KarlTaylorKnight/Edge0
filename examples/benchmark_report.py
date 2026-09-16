"""Pure benchmark report builder: measured numbers -> JSON contract.

This module is deliberately standard-library only.  It never imports
``edge0``, ``torch`` or ``mlx`` and performs no measurement of its own:
``examples/bench.py`` measures, ``examples/benchmark_measure.py`` collects
backend/process metrics, and the functions here validate and assemble the
result into the versioned schema documented in ``docs/nvidia.md``.

Conventions of the schema:

* ``schema_version`` is the benchmark schema, independent of the probe
  schema in ``scripts/jetson_probe.py``.
* A required measurement (a token count, a wall-clock duration, a byte
  count) that is invalid -- negative, non-finite, wrong type -- raises
  ``ValueError``; it is never written as a plausible-looking number.
* Unavailable *metadata* is ``null`` plus a reason: every group object
  carries an ``unavailable_reasons`` map (``{field: reason}``) and every
  ``null`` field in the group has an entry there (and vice versa).
  Single-valued memory metric objects carry their own
  ``unavailable_reason`` instead.
* Serialization is strict JSON (``allow_nan=False``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from typing import Any

#: Benchmark report schema.  Independent of ``scripts/jetson_probe.py``'s
#: ``schema_version``; bump when a field's meaning or presence changes.
SCHEMA_VERSION = 1

#: What ``examples/bench.py`` measures.  Stated in the report so a number
#: is never mistaken for time-to-first-token or an EOS-aware completion.
PROTOCOL = {
    "name": "fixed-length-sample-and-step",
    "version": 1,
    "description": (
        "Two sequential runs in one process.  Each run: engine reset, "
        "timed chat-templated prefill, untimed warmup steps, then a "
        "timed fixed-length decode window.  Wall time is host monotonic "
        "time and includes storage, transfer and CPU sampling costs; on "
        "CUDA the device is synchronized at phase boundaries only."),
    "warmup": (
        "greedy argmax steps, untimed; excluded from every throughput "
        "figure (expert cold-start and first-step overhead land here)"),
    "timed_phase": (
        "each timed iteration samples one token from the current logits "
        "and runs one engine step (a forward pass); the final step's "
        "logits are computed but never sampled; the loop is fixed-length "
        "and does not stop at EOS"),
    "not_measured": ["TTFT", "EOS-aware completion latency"],
}

#: Required direct fields per group (``unavailable_reasons`` is added or
#: checked separately).
REQUIRED_FIELDS = {
    "identity": ("timestamp_utc", "git", "run_count", "command",
                 "requested"),
    "model": ("checkpoint_path", "tier", "model_type", "architectures",
              "config_sha256", "manifest", "small_file_sha256", "adapters",
              "tokenizer_files"),
    "runtime": ("backend", "backend_version", "resolved_device",
                "measurement", "execution_evidence", "python_version",
                "torch_version", "torch_cuda_version", "mlx_version",
                "platform", "power_mode"),
    "workload": ("prompt_source", "prompt_sha256", "prompt_token_ids_sha256",
                 "prompt_chars", "prompt_tokens", "requested_warmup_tokens",
                 "requested_timed_tokens", "seed", "temperature", "top_k",
                 "top_p", "repetition_penalty", "prefill_chunk", "think"),
    "caches": ("layer_options", "shared_cache_slots_resolved",
               "prefetch_cap_resolved", "weight_cache", "prewarm",
               "threads", "mlx_cache_limit_bytes"),
    "memory": ("process_peak_rss", "process_sampled_peak_rss", "notes"),
}

_RUN_FIELDS = ("index", "prompt_tokens", "requested_warmup_tokens",
               "warmup_tokens", "requested_timed_tokens",
               "timed_decode_tokens", "total_generated_tokens",
               "decode_start_context_tokens", "prefill_seconds",
               "prefill_tokens_per_second", "decode_seconds",
               "decode_tokens_per_second", "memory", "execution_evidence",
               "unavailable_reasons")

_SUMMARY_METRICS = ("decode_tokens_per_second", "prefill_tokens_per_second",
                    "prefill_seconds", "decode_seconds")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_count(name: str, value) -> int:
    if not _is_int(value):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _require_seconds(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def memory_metric(bytes_value, *, method: str, scope: str,
                  unavailable_reason: str | None = None, **details) -> dict:
    """One memory measurement: integer bytes, how it was measured, and
    what interval/process it covers.

    ``bytes_value`` is ``None`` only together with ``unavailable_reason``
    (and a measured value never carries a reason).  Extra keyword details
    (``interval_seconds``, ``sample_count``, ...) are kept verbatim.
    """
    if not method:
        raise ValueError("memory metric needs a non-empty method")
    if not scope:
        raise ValueError("memory metric needs a non-empty scope")
    if bytes_value is None:
        if not unavailable_reason:
            raise ValueError(
                "memory metric without bytes needs an unavailable_reason")
    else:
        if unavailable_reason:
            raise ValueError(
                "memory metric with bytes must not carry an unavailable "
                "reason")
        if not _is_int(bytes_value):
            raise ValueError(
                f"memory metric bytes must be an int, got {bytes_value!r}")
        if bytes_value < 0:
            raise ValueError(
                f"memory metric bytes must be >= 0, got {bytes_value}")
    metric = {
        "bytes": bytes_value,
        "unit": "bytes",
        "method": method,
        "scope": scope,
        "unavailable_reason": unavailable_reason or None,
    }
    metric.update(details)
    return metric


def _rate(tokens: int, seconds: float):
    """tokens/seconds, or ``None`` when the measured duration is zero (an
    unavailable rate, never a fabricated zero)."""
    if seconds == 0.0:
        return None
    return tokens / seconds


def run_result(*, index: int, prompt_tokens: int,
               requested_warmup_tokens: int, warmup_tokens: int,
               requested_timed_tokens: int, timed_decode_tokens: int,
               prefill_seconds: float, decode_seconds: float,
               memory: dict, execution_evidence: dict | None = None) -> dict:
    """Validate one benchmark run and derive its counts and rates.

    For the fixed-length sample-and-step loop in ``examples/bench.py``:
    ``total_generated_tokens = warmup + timed_decode`` and
    ``decode_start_context_tokens = prompt + warmup``.  Throughput uses
    only the timed decode tokens.  Requested counts are kept separately
    from the counts actually executed.
    """
    index = _require_count("index", index)
    prompt_tokens = _require_count("prompt_tokens", prompt_tokens)
    requested_warmup_tokens = _require_count(
        "requested_warmup_tokens", requested_warmup_tokens)
    warmup_tokens = _require_count("warmup_tokens", warmup_tokens)
    requested_timed_tokens = _require_count(
        "requested_timed_tokens", requested_timed_tokens)
    timed_decode_tokens = _require_count(
        "timed_decode_tokens", timed_decode_tokens)
    if timed_decode_tokens <= 0:
        raise ValueError(
            "timed_decode_tokens must be > 0: an empty timed window is "
            "not a successful benchmark")
    prefill_seconds = _require_seconds("prefill_seconds", prefill_seconds)
    decode_seconds = _require_seconds("decode_seconds", decode_seconds)

    unavailable: dict[str, str] = {}
    prefill_tps = _rate(prompt_tokens, prefill_seconds)
    if prefill_tps is None:
        unavailable["prefill_tokens_per_second"] = (
            "measured prefill duration is zero")
    decode_tps = _rate(timed_decode_tokens, decode_seconds)
    if decode_tps is None:
        unavailable["decode_tokens_per_second"] = (
            "measured decode duration is zero")

    return {
        "index": index,
        "prompt_tokens": prompt_tokens,
        "requested_warmup_tokens": requested_warmup_tokens,
        "warmup_tokens": warmup_tokens,
        "requested_timed_tokens": requested_timed_tokens,
        "timed_decode_tokens": timed_decode_tokens,
        "total_generated_tokens": warmup_tokens + timed_decode_tokens,
        "decode_start_context_tokens": prompt_tokens + warmup_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prefill_tps,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tps,
        "memory": dict(memory),
        "execution_evidence": dict(execution_evidence or {}),
        "unavailable_reasons": unavailable,
    }


# ---- validation --------------------------------------------------------------


def _walk(value, path: str, errors: list[str]) -> None:
    """Strict-JSON and null-with-reason walk over a report fragment."""
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                errors.append(f"{path}: non-string key {key!r}")
        reasons = value.get("unavailable_reasons")
        if "unavailable_reasons" in value:
            if not isinstance(reasons, dict):
                errors.append(f"{path}.unavailable_reasons must be a map")
                reasons = {}
            for key, reason in reasons.items():
                if key not in value:
                    errors.append(
                        f"{path}.unavailable_reasons names unknown field "
                        f"{key!r}")
                elif value[key] is not None:
                    errors.append(
                        f"{path}.{key} has an unavailable reason but also a "
                        f"value {value[key]!r}")
                if not isinstance(reason, str) or not reason:
                    errors.append(
                        f"{path}.unavailable_reasons[{key!r}] must be a "
                        f"non-empty string")
        elif "bytes" in value and "unavailable_reason" in value:
            # memory metric object: single value, single reason
            if value["bytes"] is None and not value["unavailable_reason"]:
                errors.append(f"{path}.bytes is null without a reason")
            if value["bytes"] is not None and value["unavailable_reason"]:
                errors.append(
                    f"{path}.bytes has a value and an unavailable reason")
        for key, child in value.items():
            if key == "unavailable_reasons":
                continue
            child_path = f"{path}.{key}"
            if child is None:
                is_metric_null = key == "unavailable_reason" or (
                    key == "bytes" and "unavailable_reason" in value)
                if "unavailable_reasons" in value:
                    if key not in (reasons or {}):
                        errors.append(
                            f"{child_path} is null without an unavailable "
                            f"reason")
                elif not is_metric_null:
                    errors.append(
                        f"{child_path} is null in an object without an "
                        f"unavailable_reasons map")
                continue
            _walk(child, child_path, errors)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if child is None:
                # list entries mirror per-run fields; the reason lives on
                # the run object (see summarize_runs)
                continue
            _walk(child, f"{path}[{i}]", errors)
    elif value is None or isinstance(value, (bool, str, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path}: non-finite number {value!r}")
    else:
        errors.append(
            f"{path}: unsupported type {type(value).__name__} "
            f"(not JSON-serializable)")


def validate_report(report: dict) -> None:
    """Raise ``ValueError`` listing every contract violation."""
    errors: list[str] = []
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")
    for key in ("schema_version", "protocol", "identity", "model",
                "runtime", "workload", "caches", "runs", "summary",
                "memory", "probe"):
        if key not in report:
            errors.append(f"missing top-level field {key!r}")
    if errors:
        raise ValueError("invalid benchmark report:\n  " + "\n  ".join(errors))
    if report["schema_version"] != SCHEMA_VERSION:
        errors.append(
            f"schema_version {report['schema_version']!r} != "
            f"{SCHEMA_VERSION}")
    for group, fields in REQUIRED_FIELDS.items():
        obj = report[group]
        if not isinstance(obj, dict):
            errors.append(f"{group} must be an object")
            continue
        for field in fields:
            if field not in obj:
                errors.append(f"{group}.{field} is required")
        if "unavailable_reasons" not in obj:
            errors.append(f"{group}.unavailable_reasons is required")
    runs = report["runs"]
    if not isinstance(runs, list) or not runs:
        errors.append("runs must be a non-empty list")
    else:
        for i, run in enumerate(runs):
            if not isinstance(run, dict):
                errors.append(f"runs[{i}] must be an object")
                continue
            for field in _RUN_FIELDS:
                if field not in run:
                    errors.append(f"runs[{i}].{field} is required")
            if run.get("index") != i:
                errors.append(
                    f"runs[{i}].index is {run.get('index')!r}, expected {i}")
        run_count = report["identity"].get("run_count") \
            if isinstance(report["identity"], dict) else None
        if run_count != len(runs):
            errors.append(
                f"identity.run_count {run_count!r} != len(runs) "
                f"{len(runs)}")
    _walk(report, "report", errors)
    if errors:
        raise ValueError("invalid benchmark report:\n  " + "\n  ".join(errors))


# ---- summary -----------------------------------------------------------------


def _stat_block(values: list) -> dict:
    """Descriptive statistics over per-run values (``None`` allowed)."""
    values = list(values)
    present = [v for v in values if v is not None]
    block: dict[str, Any] = {
        "n": len(values), "n_valid": len(present), "values": values,
        "mean": None, "min": None, "max": None, "sample_stdev": None,
        "unavailable_reasons": {}}
    if not present:
        reason = "no run has a value for this metric (n_valid = 0)"
        for key in ("mean", "min", "max", "sample_stdev"):
            block["unavailable_reasons"][key] = reason
        return block
    block["mean"] = statistics.fmean(present)
    block["min"] = min(present)
    block["max"] = max(present)
    if len(present) >= 2:
        block["sample_stdev"] = statistics.stdev(present)
    else:
        block["unavailable_reasons"]["sample_stdev"] = (
            f"sample standard deviation needs at least 2 runs with a value "
            f"(n_valid = {len(present)})")
    return block


def summarize_runs(runs: list[dict]) -> dict:
    """Summary over every retained run.  Runs are sequential within one
    process, so this is descriptive, not an estimate from independent
    launches (those are Task 3/7's job)."""
    summary: dict[str, Any] = {"run_count": len(runs)}
    for metric in _SUMMARY_METRICS:
        summary[metric] = _stat_block([run.get(metric) for run in runs])
    summary["method"] = (
        "arithmetic mean, min, max and sample standard deviation (n-1) "
        "over every retained run with a value (n_valid; values keeps the "
        "per-run entries, null included); runs are sequential in one "
        "process (run 0 is the first request after model load) and are "
        "not independent launches")
    return summary


# ---- probe link --------------------------------------------------------------


_PROBE_FIELDS = ("path", "schema_version", "sha256", "ready",
                 "collected_at_utc")


def probe_link(*, path, text) -> dict:
    """Link the report to an actual probe file: its schema version, the
    SHA-256 of the file content as read, and the local reference.

    ``text`` is the raw file content (``bytes`` are hashed as-is; ``str``
    is hashed as UTF-8).  ``None`` means no probe was supplied.
    """
    link: dict[str, Any]
    if text is None:
        reason = "no probe file supplied (--probe-json not given)"
        link = {"supplied": False}
        link.update({field: None for field in _PROBE_FIELDS})
        link["unavailable_reasons"] = {f: reason for f in _PROBE_FIELDS}
        return link
    raw = text if isinstance(text, bytes) else text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"probe file {path!r} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise ValueError(f"probe file {path!r} is not a JSON object")
    schema = data.get("schema_version")
    if not _is_int(schema):
        raise ValueError(
            f"probe file {path!r} has no integer schema_version")
    link = {"supplied": True, "path": os.fspath(path),
            "schema_version": schema, "sha256": digest,
            "ready": None, "collected_at_utc": None,
            "unavailable_reasons": {}}
    ready = data.get("ready")
    if isinstance(ready, bool):
        link["ready"] = ready
    else:
        link["unavailable_reasons"]["ready"] = (
            "probe has no boolean 'ready' field")
    collected = data.get("collected_at_utc")
    if isinstance(collected, str) and collected:
        link["collected_at_utc"] = collected
    else:
        link["unavailable_reasons"]["collected_at_utc"] = (
            "probe has no 'collected_at_utc' field")
    return link


# ---- assembly and serialization ------------------------------------------------


def build_report(*, identity: dict, model: dict, runtime: dict,
                 workload: dict, caches: dict, runs: list[dict],
                 memory: dict, probe: dict | None = None) -> dict:
    """Assemble and validate the full report.  ``runs`` are
    ``run_result`` dicts; ``probe`` is a ``probe_link`` dict (``None`` =
    not supplied)."""
    def _group(obj, name):
        if not isinstance(obj, dict):
            raise ValueError(f"{name} must be an object")
        out = dict(obj)
        out.setdefault("unavailable_reasons", {})
        return out

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": dict(PROTOCOL),
        "identity": _group(identity, "identity"),
        "model": _group(model, "model"),
        "runtime": _group(runtime, "runtime"),
        "workload": _group(workload, "workload"),
        "caches": _group(caches, "caches"),
        "runs": [dict(run) for run in runs],
        "summary": summarize_runs(runs) if runs else {},
        "memory": _group(memory, "memory"),
        "probe": (probe if probe is not None
                  else probe_link(path=None, text=None)),
    }
    validate_report(report)
    return report


def dumps_report(report: dict) -> str:
    """Strict JSON text (``allow_nan=False``, sorted keys, trailing
    newline).  Validates first so a NaN never reaches the encoder."""
    validate_report(report)
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _place_new_file(tmp: str, dest: str) -> None:
    """Move ``tmp`` to ``dest`` without ever overwriting ``dest``.

    A hard link is atomic and fails with EEXIST when the destination
    appeared meanwhile; the temporary name is then unlinked.  Filesystems
    without hard links fall back to a rename (``os.rename`` refuses an
    existing destination on Windows; POSIX ``os.replace`` is the last
    resort after a fresh existence check).
    """
    try:
        os.link(tmp, dest)
    except FileExistsError:
        raise FileExistsError(
            f"benchmark output {dest!r} appeared while the report was being "
            f"written; refusing to overwrite it")
    except (OSError, AttributeError):
        if os.path.lexists(dest):
            raise FileExistsError(
                f"benchmark output {dest!r} appeared while the report was "
                f"being written; refusing to overwrite it")
        if os.name == "nt":
            os.rename(tmp, dest)
        else:
            os.replace(tmp, dest)
        return
    os.unlink(tmp)


def write_report(report: dict, path) -> str:
    """Validate, then write the report atomically to a NEW file.

    The destination must not exist (a stale success report can otherwise
    be mistaken for this run's result).  The text is written to a
    temporary file in the destination directory, fsynced, and linked or
    renamed into place without overwriting; on any failure the temporary
    file is removed and the destination is left as it was.
    """
    text = dumps_report(report)
    path = os.fspath(path)
    if os.path.lexists(path):
        raise FileExistsError(
            f"benchmark output {path!r} already exists; refusing to "
            f"overwrite (choose a new path)")
    dest_dir = os.path.dirname(os.path.abspath(path)) or os.curdir
    if not os.path.isdir(dest_dir):
        raise FileNotFoundError(
            f"benchmark output directory {dest_dir!r} does not exist")
    fd, tmp = tempfile.mkstemp(prefix=".bench-", suffix=".json.tmp",
                               dir=dest_dir)
    try:
        if os.name != "nt":
            # mkstemp creates 0600; give the report the usual umask mode
            mask = os.umask(0)
            os.umask(mask)
            os.chmod(tmp, 0o666 & ~mask)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _place_new_file(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return text
