#!/usr/bin/env python3
"""edge0 speed + memory benchmark (any tier).

Runs the same measurement shape as the deployment's bench: prefill ->
greedy warmup steps (untimed) -> timed sampled decode -> report tok/s and
the backend's peak memory (MLX peak active memory; the torch CUDA
allocator's peak allocated bytes on a CUDA device; n/a for torch on
cpu/mps).  Works for both tiers; the tier is auto-detected from the
checkpoint (or forced via the model name).

What the numbers are (and are not).  Two runs per process; each run
resets the engine, times the chat-templated prefill, runs ``--warmup``
greedy steps untimed, then times ``--ntok`` iterations of "sample one
token, run one engine step".  The loop is fixed-length and does not stop
at EOS; the final step's logits are computed but never sampled.  This is
NOT time-to-first-token and NOT an EOS-aware completion latency.  On a
CUDA device the resolved device is synchronized at those phase
boundaries only (after reset, end of prefill, end of warmup, after the
final timed step) so every wall time includes the device work it
launched; storage, transfer and CPU sampling costs are inside the wall
time by design.  Sampling at temperature 0 is greedy (see
``edge0.sampling``), so ``BENCH_TEMP=0`` output does not depend on the
seed; the seed is still recorded.

Usage:
    python examples/bench.py /path/to/model [--ntok 200] [--warmup 10]
    python examples/bench.py edge0-35b        # via $EDGE0_35B_MODEL
    BENCH_LONG=1 python examples/bench.py edge0-35b   # ~3k-token prefill
    python examples/bench.py edge0-8b --json-output "$RUN/bench.json" \\
        --probe-json "$RUN/orin-probe.json"   # machine-readable report

``--json-output`` must name a NEW file (an existing one is refused before
the model loads); the report schema is described in docs/nvidia.md.
Keep the JSON outside tracked source: it records local paths and the
prompt text.

Env (sampling knobs, defaults follow the tier's GenerationConfig):
    BENCH_TEMP / BENCH_PROMPT / BENCH_SEED / BENCH_NTOK
    BENCH_LONG=1  use a ~3k-token synthetic prompt so the prefill timing
                  is meaningful (short prompts only measure fixed
                  overhead; set BENCH_PROMPT for your own long text)
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # imported as ``examples.bench`` (tests)
    from examples import benchmark_measure as measure
    from examples import benchmark_report as report
except ImportError:  # run as a script: helpers sit next to this file
    import benchmark_measure as measure  # type: ignore[no-redef]
    import benchmark_report as report  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPTS = {
    "edge0-35b": "什么是混合专家模型（MoE）？它和普通 Transformer 有什么区别？",
    "edge0-8b": "9.11 和 9.8 哪个大？请仔细比较。",
}
DEFAULT_PROMPT = "什么是混合专家模型（MoE）？简单介绍一下。"

# Synthetic long-context prompt (~3k tokens after tokenization): two
# non-repetitive paragraphs alternating 40 times, prefixed with a task
# instruction.  Prefill timing on a few dozen tokens only measures fixed
# overhead; a long prompt exercises the real chunked prefill path
# (per-layer expert loading, KV growth).
_PARA_ZH = (
    "流式推理框架的设计要点在于把磁盘带宽、缓存层级与计算核心三者重叠"
    "起来。混合专家模型每一层只在少数专家上激活，路由器给出的稀疏选择"
    "恰好为按需加载提供了天然的批粒度。若能在前一步的隐状态基础上预判"
    "下一步的专家集合，固态硬盘的读取延迟就能被计算完全掩盖，这对边缘"
    "设备上的大模型部署具有直接意义。")
_PARA_EN = (
    "The design of a streaming inference framework hinges on overlapping "
    "disk bandwidth, cache hierarchy, and compute. A mixture-of-experts "
    "layer activates only a handful of experts per token, and the router "
    "sparsity provides a natural granularity for on-demand loading. If "
    "the expert set of the next step can be predicted from the previous "
    "hidden state, SSD read latency hides behind compute entirely, which "
    "matters for large-model deployment on edge hardware.")

#: Environment knobs echoed into the report as REQUESTED (raw strings,
#: snapshotted before the engine is built: ``engine/ling.py`` sets a
#: default ``LING_HIDDEN_CLIP`` the user did not ask for).
ENV_KNOBS = ("BENCH_TEMP", "BENCH_PROMPT", "BENCH_SEED", "BENCH_NTOK",
             "BENCH_LONG", "EDGE0_BACKEND", "EDGE0_TORCH_DEVICE",
             "EDGE0_MEMORY_BUDGET", "EDGE0_BUDGET_CONTEXT",
             "EDGE0_TORCH_WEIGHT_CACHE", "EDGE0_PREWARM", "LING_PREWARM",
             "MLX_CACHE_LIMIT_MB", "LING_HIDDEN_CLIP",
             "PREROUTER_FEATURE_TOPK", "PREROUTER_INTRA")

_TOKENIZER_FILE_PREFIXES = ("tokenizer", "chat_template", "special_tokens",
                            "vocab", "merges", "added_tokens")


def _framework():
    """The active backend, its array namespace and the sampler.

    Imported lazily: ``edge0.backends`` binds to ``EDGE0_BACKEND`` at
    import time, and nothing in argument validation, ``--help`` or the
    pre-load output/probe checks needs a backend.  Tests replace this
    with fakes.
    """
    from edge0.backends import backend, core
    from edge0.sampling import sample
    return backend, core, sample


def _long_prompt() -> str:
    paras = [_PARA_ZH if i % 2 == 0 else _PARA_EN for i in range(40)]
    return "请仔细阅读以下材料并用一句话总结其核心思想：" + " ".join(paras)


def _prompt_source() -> str:
    if os.environ.get("BENCH_PROMPT"):
        return "BENCH_PROMPT"
    if os.environ.get("BENCH_LONG") == "1":
        return "BENCH_LONG"
    return "tier default"


def _prompt_for(engine) -> str:
    if os.environ.get("BENCH_PROMPT"):
        return os.environ["BENCH_PROMPT"]
    if os.environ.get("BENCH_LONG") == "1":
        return _long_prompt()
    return PROMPTS.get(engine.name, DEFAULT_PROMPT)


def _encode_prompt(engine, prompt: str) -> list[int]:
    tok = engine._tok
    if hasattr(engine, "encode_chat"):
        return [int(t) for t in engine.encode_chat(
            [{"role": "user", "content": prompt}], think=False)]
    return [int(t) for t in tok(tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False))["input_ids"]]


def _sampling_knobs(engine) -> dict:
    """Resolved sampling settings: the tier's GenerationConfig, overridden
    by BENCH_TEMP / BENCH_SEED (parsed once; ``BENCH_SEED=0`` is a seed)."""
    gen = getattr(engine.cfg, "gen", None)
    seed_env = os.environ.get("BENCH_SEED")
    return {
        "temperature": float(os.environ.get(
            "BENCH_TEMP", getattr(gen, "temperature", 0.7))),
        "top_k": getattr(gen, "top_k", 64),
        "top_p": getattr(gen, "top_p", 0.95),
        "repetition_penalty": getattr(gen, "repetition_penalty", 1.0),
        "seed": int(seed_env) if seed_env is not None else None,
    }


def _headline_bytes(peaks: dict, measurement):
    """The one per-run peak the human output and the legacy ``peak_gib``
    show (allocated on CUDA, peak active on MLX); ``None`` when that
    metric is unavailable."""
    name = getattr(measurement, "headline", None)
    metric = peaks.get(name) if name else None
    if metric is None:
        for candidate in peaks.values():
            if candidate.get("bytes") is not None:
                return candidate["bytes"]
        return None
    return metric.get("bytes")


def _fmt_gib(value) -> str:
    return "n/a" if value is None else f"{value:.2f} GiB"


def run_bench(engine, ntok: int, warmup: int, *, measurement=None,
              runs: int = 2) -> dict:
    """Run the fixed-length sample-and-step benchmark.

    Returns the legacy dict (``runs`` with prefill_s/decode_s/ntok/
    peak_gib/tok_s, ``mean_tok_s``, ``peak_gib``) plus ``report_runs``
    (validated ``benchmark_report.run_result`` entries), ``workload``,
    ``measurement`` and ``execution_evidence`` for the JSON report.
    ``measurement`` defaults to the adapter for the active backend and
    its resolved device (``benchmark_measure.make_measurement``).
    """
    if not isinstance(ntok, int) or isinstance(ntok, bool) or ntok <= 0:
        raise ValueError(
            "ntok must be a positive integer: an empty timed window is not "
            "a benchmark")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be an integer >= 0")
    backend, core, sample = _framework()
    if measurement is None:
        measurement = measure.make_measurement(backend.name, core)

    prompt = _prompt_for(engine)
    ids = _encode_prompt(engine, prompt)
    knobs = _sampling_knobs(engine)
    temperature = knobs["temperature"]
    top_k, top_p, rp, seed = (knobs["top_k"], knobs["top_p"],
                              knobs["repetition_penalty"], knobs["seed"])

    results = []
    report_runs = []
    evidence = None
    for run in range(runs):
        engine.reset()
        # Finish any device work the reset queued, then reset the peak
        # statistics right before prefill: the per-run peak covers
        # prefill + warmup + decode (plus whatever is resident now).
        measurement.synchronize()
        measurement.reset_peak()
        t0 = time.perf_counter()
        engine.prefill(ids)
        # Prefill's final device work belongs to prefill wall time.
        measurement.synchronize()
        t_prefill = time.perf_counter() - t0
        logits = engine.next_logits()
        if evidence is None:
            evidence = measure.execution_evidence(logits)
        if seed is not None:
            core.random.seed(int(seed))

        history = list(ids)
        # warmup: GREEDY (argmax), untimed -- the first tokens are
        # dominated by expert cold-starts; the timed window below is the
        # steady state.  Only the timed loop samples.
        warmup_done = 0
        for _ in range(warmup):
            tid = int(core.argmax(logits, axis=-1).item())
            history.append(tid)
            logits = engine.step(tid)
            warmup_done += 1
        # Warmup must be complete before the decode clock starts.
        measurement.synchronize()
        out: list[int] = []
        t0 = time.perf_counter()
        # timed: sample from the current logits, then one forward.  The
        # last forward's logits are never sampled; no EOS stop.
        for _ in range(ntok):
            tid = sample(logits, temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=rp,
                         history=history, seed=None)
            history.append(tid)
            out.append(tid)
            logits = engine.step(tid)
        # The final timed step's device work is inside decode wall time.
        measurement.synchronize()
        t_decode = time.perf_counter() - t0
        peaks = measurement.peak_metrics()  # after the synchronized decode
        report_runs.append(report.run_result(
            index=run, prompt_tokens=len(ids),
            requested_warmup_tokens=warmup, warmup_tokens=warmup_done,
            requested_timed_tokens=ntok, timed_decode_tokens=len(out),
            prefill_seconds=t_prefill, decode_seconds=t_decode,
            memory=peaks, execution_evidence=evidence))
        peak_bytes = _headline_bytes(peaks, measurement)
        peak_gib = (peak_bytes / (1024 ** 3)) if peak_bytes is not None else None
        results.append({"prefill_s": t_prefill, "decode_s": t_decode,
                        "ntok": len(out), "peak_gib": peak_gib,
                        "tok_s": len(out) / t_decode if t_decode else 0.0})
        pf_tps = (len(ids) / t_prefill) if t_prefill else 0.0
        print(f"prompt={len(ids)} tok  prefill={t_prefill:.2f}s "
              f"({pf_tps:.0f} tok/s)  "
              f"decode={len(out)}/{t_decode:.2f}s  "
              f"tok/s={results[-1]['tok_s']:.1f}  "
              f"peak_active={_fmt_gib(peak_gib)}", flush=True)
    mean_ts = sum(r["tok_s"] for r in results) / len(results)
    max_peak = max((r["peak_gib"] for r in results
                    if r["peak_gib"] is not None), default=None)
    print(f"[bench] mean tok/s={mean_ts:.1f}  "
          f"peak_active<={_fmt_gib(max_peak)}", flush=True)
    workload = {
        "prompt_source": _prompt_source(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": hashlib.sha256(
            ",".join(str(t) for t in ids).encode()).hexdigest(),
        "prompt_chars": len(prompt),
        "prompt_text": prompt,
        "prompt_tokens": len(ids),
        "requested_warmup_tokens": warmup,
        "requested_timed_tokens": ntok,
        **knobs,
    }
    return {"runs": results, "mean_tok_s": mean_ts, "peak_gib": max_peak,
            "report_runs": report_runs, "workload": workload,
            "measurement": measurement.describe(),
            "execution_evidence": evidence}


# ---- report assembly (outside every timed interval) ---------------------------


def _jsonable(value) -> Any:
    """Config objects -> JSON: enums by name, paths as strings, tuples as
    lists, dataclasses as dicts, anything else by repr."""
    if isinstance(value, bool) or value is None or isinstance(
            value, (str, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    return repr(value)


def _with_reasons(obj: dict, reason: str) -> dict:
    """Give a plain mapping the schema's null convention: every ``None``
    value gets ``reason`` in ``unavailable_reasons``."""
    out = dict(obj)
    reasons = dict(out.get("unavailable_reasons") or {})
    for key, value in out.items():
        if key != "unavailable_reasons" and value is None:
            reasons.setdefault(key, reason)
    out["unavailable_reasons"] = reasons
    return out


def _env_snapshot() -> dict:
    """The requested environment knobs, taken BEFORE the engine is built."""
    return {k: os.environ[k] for k in ENV_KNOBS if k in os.environ}


def _model_env() -> dict:
    """Model-read knobs as the bailing_hybrid code parses them, resolved
    AFTER the engine is built (``engine/ling.py`` fills in a default
    ``LING_HIDDEN_CLIP``).  They change numerics and per-token work, so
    before/after comparisons must hold them equal."""
    out: dict[str, Any] = {
        "ling_hidden_clip": None,
        "prerouter_feature_topk": os.environ.get("PREROUTER_FEATURE_TOPK",
                                                 "teacher"),
        "prerouter_intra": os.environ.get("PREROUTER_INTRA", "0") == "1",
        "note": ("parsed exactly as the bailing_hybrid model code does; "
                 "ling_hidden_clip 0.0 means the hidden-state clip is off"),
        "unavailable_reasons": {},
    }
    raw = os.environ.get("LING_HIDDEN_CLIP", "0")
    try:
        out["ling_hidden_clip"] = float(raw)
    except ValueError:
        out["unavailable_reasons"]["ling_hidden_clip"] = (
            f"LING_HIDDEN_CLIP={raw!r} is not a number")
    return out


def _model_group(engine, model_dir: str) -> dict:
    root = Path(model_dir)
    manifest = measure.checkpoint_manifest(root)
    config = manifest["config"]
    cfg = engine.cfg
    prerouter = getattr(cfg, "prerouter", None)
    adapters = {
        "lora": measure.adapter_identity(getattr(cfg, "lora", None), root),
        "prerouter": measure.adapter_identity(
            getattr(prerouter, "weights_file", None), root),
        "lora_settings": _with_reasons(_jsonable({
            "r": getattr(cfg, "lora_r", None),
            "alpha": getattr(cfg, "lora_alpha", None)}),
            "tier config has no LoRA settings"),
        "prerouter_settings": _with_reasons(_jsonable({
            "start_layer": getattr(prerouter, "start_layer", None),
            "hidden": getattr(prerouter, "hidden", None),
            "dtype": getattr(prerouter, "dtype", None),
            "feature_topk": getattr(prerouter, "feature_topk", None),
            "owners": getattr(prerouter, "owners", None),
            "top_k": getattr(cfg, "prerouter_top_k", None)}),
            "tier config has no prerouter"),
    }
    small = manifest["small_file_sha256"]
    tokenizer_files = []
    for entry in manifest["manifest"].get("files") or []:
        name = entry["path"]
        if not name.startswith(_TOKENIZER_FILE_PREFIXES):
            continue
        item = {"name": name, "size_bytes": entry["size_bytes"],
                "sha256": small.get(name), "unavailable_reasons": {}}
        if item["sha256"] is None:
            item["unavailable_reasons"]["sha256"] = (
                "not hashed: larger than the small-file cap or unreadable")
        tokenizer_files.append(item)
    group = {
        "checkpoint_path": str(root.resolve()),
        "tier": engine.name,
        "model_type": config.get("model_type") or None,
        "architectures": _jsonable(config.get("architectures") or []),
        "config_sha256": manifest["config_sha256"],
        "manifest": manifest["manifest"],
        "small_file_sha256": small,
        "safetensors_headers": manifest["safetensors_headers"],
        "adapters": adapters,
        "tokenizer_files": tokenizer_files,
        "unavailable_reasons": {},
    }
    if group["model_type"] is None:
        group["unavailable_reasons"]["model_type"] = (
            "config.json missing or has no model_type")
    if group["config_sha256"] is None:
        group["unavailable_reasons"]["config_sha256"] = (
            "config.json not found in the checkpoint directory")
    return group


def _runtime_group(bench: dict, backend) -> dict:
    info = measure.runtime_info(backend.name)
    try:
        version = str(backend.version)
    except Exception as exc:  # noqa: BLE001 - never fail the report here
        version = None
        info["unavailable_reasons"]["backend_version"] = (
            f"backend.version failed: {type(exc).__name__}: {exc}")
    return {
        "backend": backend.name,
        "backend_version": version,
        "resolved_device": bench["measurement"]["resolved_device"],
        "measurement": bench["measurement"],
        "execution_evidence": bench["execution_evidence"],
        "power_mode": measure.power_mode(),
        **info,
    }


def _caches_group(engine, backend) -> dict:
    opts = getattr(engine.cfg, "options", None)
    reasons: dict[str, str] = {}
    if dataclasses.is_dataclass(opts) and not isinstance(opts, type):
        layer_options = _with_reasons(
            _jsonable(dataclasses.asdict(opts)),
            "option unset (None): the model config value applies")
    else:
        layer_options = {"unavailable_reasons": {}}
        reasons["layer_options"] = "engine config has no LayerOptions"
    layers = getattr(engine, "_all_stream_layers", {}) or {}
    first = next(iter(layers.values()), None)
    shared_slots = prefetch_cap = None
    if first is None:
        reasons["shared_cache_slots_resolved"] = (
            "engine has no streaming layers")
        reasons["prefetch_cap_resolved"] = "engine has no streaming layers"
    else:
        shared_slots = getattr(getattr(first, "shared_cache", None),
                               "slots", None)
        prefetch_cap = getattr(getattr(first, "_prefetch_buf", None),
                               "cap", None)
        if shared_slots is None:
            reasons["shared_cache_slots_resolved"] = (
                "streaming layer exposes no shared_cache.slots")
        if prefetch_cap is None:
            reasons["prefetch_cap_resolved"] = (
                "streaming layer exposes no _prefetch_buf.cap")
    threads: dict[str, Any] = {
        "load_threads": getattr(opts, "load_threads", None),
        "prefetch_threads": getattr(opts, "prefetch_threads", None),
        "torch_num_threads": None}
    thread_reasons = {}
    weight_cache = None
    mlx_cache_limit = None
    if backend.name == "cuda":
        from edge0.backends.cuda import nn as cuda_nn
        weight_cache = bool(getattr(cuda_nn, "CACHE_DEQUANTIZED", False))
        try:
            import torch
            threads["torch_num_threads"] = int(torch.get_num_threads())
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            thread_reasons["torch_num_threads"] = f"{type(exc).__name__}: {exc}"
        reasons["mlx_cache_limit_bytes"] = "backend is cuda"
    else:
        reasons["weight_cache"] = (
            "EDGE0_TORCH_WEIGHT_CACHE applies to the torch backend only")
        thread_reasons["torch_num_threads"] = "backend is not torch"
        try:
            mlx_cache_limit = int(
                os.environ.get("MLX_CACHE_LIMIT_MB", "256")) * 1024 * 1024
        except ValueError:
            reasons["mlx_cache_limit_bytes"] = "MLX_CACHE_LIMIT_MB not an int"
    for key in ("load_threads", "prefetch_threads"):
        if threads[key] is None:
            thread_reasons[key] = "engine config has no LayerOptions"
    threads["unavailable_reasons"] = thread_reasons
    prewarm = os.environ.get(
        "EDGE0_PREWARM", os.environ.get("LING_PREWARM", "0")) == "1"
    try:
        stats = _jsonable(engine.stats())
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        stats = {"stats": None, "unavailable_reasons": {
            "stats": f"{type(exc).__name__}: {exc}"}}
    resolved_budget = getattr(engine, "memory_budget", None)
    if resolved_budget is None:
        reasons["memory_budget"] = (
            "no memory budget resolved (EDGE0_MEMORY_BUDGET off or "
            "engine predates Task 4)")
    return {
        "layer_options": layer_options,
        "shared_cache_slots_resolved": shared_slots,
        "prefetch_cap_resolved": prefetch_cap,
        "memory_budget": (resolved_budget.as_dict()
                          if resolved_budget is not None else None),
        "weight_cache": weight_cache,
        "prewarm": prewarm,
        "threads": threads,
        "mlx_cache_limit_bytes": mlx_cache_limit,
        "streaming_stats_after_runs": stats,
        "unavailable_reasons": reasons,
    }


def _build_report(engine, bench: dict, args, argv: list[str],
                  started_utc: str, model_dir: str, probe, sampler,
                  env_knobs: dict, model_env: dict, backend) -> dict:
    """Assemble the JSON report.  Called after the timed runs (and after
    the sampler stopped): hashing, git and power queries happen here."""
    requested = {
        "model": args.model,
        "ntok": args.ntok,
        "warmup": args.warmup,
        "rss_sample_interval_seconds": args.rss_sample_interval,
        "json_output": args.json_output,
        "probe_json": args.probe_json,
    }
    identity = {
        "timestamp_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git": measure.git_identity(REPO_ROOT),
        "run_count": len(bench["report_runs"]),
        "command": {"argv": list(argv), "env_knobs": dict(env_knobs)},
        "requested": _with_reasons(requested, "not requested"),
        "unavailable_reasons": {},
    }
    workload = dict(bench["workload"])
    workload["prefill_chunk"] = getattr(engine, "prefill_chunk", None)
    workload["think"] = getattr(engine, "think", None)
    workload["model_env"] = model_env
    workload = _with_reasons(workload, "not set")
    if workload["seed"] is None:
        workload["unavailable_reasons"]["seed"] = (
            "BENCH_SEED not set: sampling is unseeded")
    memory = {
        "process_peak_rss": measure.process_peak_rss(),
        **sampler.metrics(),
        "notes": [
            ("process RSS and device allocator peaks are separate views of "
             "the same shared DRAM on Jetson; never add them"),
            ("RSS includes resident file-backed pages of the mmap'd "
             "checkpoint; the sampled *_rss_anon peak is the anonymous "
             "footprint (Linux only)"),
            ("allocator peaks cover the framework allocator only, not the "
             "CUDA context, driver or non-torch memory"),
            ("cuda_peak_reserved's floor is every segment cached at the "
             "reset, so run 0 and run 1 are expected to be close; use "
             "cuda_peak_allocated for per-run activity"),
            ("process_peak_rss is a lifetime high-water mark (model load "
             "included); per-run peaks are the backend metrics in runs[]"),
        ],
        "unavailable_reasons": {},
    }
    return report.build_report(
        identity=identity,
        model=_model_group(engine, model_dir),
        runtime=_runtime_group(bench, backend),
        workload=workload,
        caches=_caches_group(engine, backend),
        runs=bench["report_runs"],
        memory=memory,
        probe=probe,
    )


# ---- CLI ---------------------------------------------------------------------


def _build_engine(model_dir: str, name):
    from edge0 import AutoEngine
    return AutoEngine.from_pretrained(model_dir, name=name)


#: Copy of ``edge0.cli.TIER_ENV`` so tier resolution needs no backend
#: import (``edge0.cli`` loads the model registry, which loads the
#: backend).  ``tests/test_bench_backend.py::test_tier_env_matches_the_cli``
#: keeps the two in sync.
_TIER_ENV = {
    "edge0-35b": "EDGE0_35B_MODEL",
    "edge0-8b": "EDGE0_8B_MODEL",
}


def _resolve_model(model: str) -> tuple[str, str | None]:
    """tier name -> checkpoint dir via $EDGE0_<TIER>_MODEL (cli parity)."""
    if model.startswith("edge0-"):
        env_name = _TIER_ENV.get(model)
        if env_name is None:
            raise SystemExit(
                f"unknown tier {model!r}; known tiers: "
                f"{', '.join(sorted(_TIER_ENV))} (or pass the checkpoint "
                f"directory itself)")
        env = os.environ.get(env_name)
        if not env:
            raise SystemExit(
                f"set {env_name} to the checkpoint directory for {model}, "
                f"or pass the dir itself")
        return env, None
    return model, None


def _prepare_output(path: str) -> str:
    """Refuse an existing destination and make its directory exist, both
    before the model loads."""
    dest = Path(path).expanduser()
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(
            f"--json-output {dest} already exists; choose a new path so a "
            f"stale report is never mistaken for this run")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return str(dest.resolve())


def _load_probe(path: str) -> dict:
    """Read and validate an explicitly requested probe before inference."""
    return report.probe_link(path=path, text=Path(path).read_bytes())


def _parse_args(ap: argparse.ArgumentParser, argv: list[str]):
    args = ap.parse_args(argv)
    if args.ntok is None:
        raw = os.environ.get("BENCH_NTOK", "200")
        try:
            args.ntok = int(raw)
        except ValueError:
            ap.error(f"BENCH_NTOK must be an integer, got {raw!r}")
    if args.ntok <= 0:
        ap.error("--ntok (or BENCH_NTOK) must be > 0: an empty timed window "
                 "is not a benchmark")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if not (0 <= args.rss_sample_interval < math.inf):
        ap.error("--rss-sample-interval must be a finite number >= 0")
    if args.json_output is not None and not args.json_output.strip():
        ap.error("--json-output needs a path")
    if args.probe_json is not None and not args.probe_json.strip():
        ap.error("--probe-json needs a path")
    seed = os.environ.get("BENCH_SEED")
    if seed is not None:
        try:
            int(seed)
        except ValueError:
            ap.error(f"BENCH_SEED must be an integer, got {seed!r}")
    temp = os.environ.get("BENCH_TEMP")
    if temp is not None:
        try:
            value = float(temp)
        except ValueError:
            ap.error(f"BENCH_TEMP must be a number, got {temp!r}")
        else:
            if not math.isfinite(value):
                ap.error(f"BENCH_TEMP must be finite, got {temp!r}")
    return args


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="checkpoint dir or tier name")
    ap.add_argument("--ntok", type=int, default=None,
                    help="timed decode tokens per run (> 0; default "
                         "$BENCH_NTOK or 200)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="untimed greedy warmup steps per run (>= 0)")
    ap.add_argument("--json-output", metavar="PATH",
                    help="write the machine-readable report to this NEW file")
    ap.add_argument("--probe-json", metavar="PATH",
                    help="link an existing scripts/jetson_probe.py report")
    ap.add_argument("--rss-sample-interval", type=float, default=0.25,
                    metavar="SECONDS",
                    help="process memory sampling interval (0 disables)")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(ap, argv)

    started_utc = datetime.now(timezone.utc).isoformat()
    model_dir, name = _resolve_model(args.model)
    # Everything that can be rejected is rejected before the model loads.
    try:
        json_output = (_prepare_output(args.json_output)
                       if args.json_output is not None else None)
        probe = (_load_probe(args.probe_json)
                 if args.probe_json is not None else None)
    except (OSError, ValueError) as exc:
        ap.exit(2, f"[bench] error: {exc}\n")
    env_knobs = _env_snapshot()  # before the engine can touch the environment

    sampler = measure.RssSampler(args.rss_sample_interval)
    engine = None
    stage = "engine build"
    try:
        sampler.start()  # before the model loads: load transients are sampled
        engine = _build_engine(model_dir, name)
        model_env = _model_env()  # resolved after the build, before the runs
        print(f"[bench] tier={engine.name} ntok={args.ntok} "
              f"warmup={args.warmup}", flush=True)
        stage = "benchmark"
        bench = run_bench(engine, args.ntok, args.warmup)
        if json_output is not None:
            stage = "report"
            sampler.stop()
            backend = _framework()[0]
            doc = _build_report(engine, bench, args, argv, started_utc,
                                model_dir, probe, sampler, env_knobs,
                                model_env, backend)
            stage = "write"
            report.write_report(doc, json_output)
            stage = "output after a successful write"
            print(f"[bench] wrote {json_output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - reported with its stage
        traceback.print_exc()
        print(f"[bench] FAILED during {stage}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            sampler.stop()
        finally:
            if engine is not None:
                engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
