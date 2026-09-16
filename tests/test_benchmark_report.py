"""Pure benchmark report builder (``examples/benchmark_report.py``).

These tests never need MLX, torch or a checkpoint: the builder is a
stdlib-only module that turns already-measured numbers into the JSON
report contract described in ``docs/plans/jetson-orin-nano.md`` (Task 2).
Collection (timers, allocator peaks, RSS) lives in
``examples/benchmark_measure.py`` and is tested separately.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from examples import benchmark_report as br

ROOT = Path(__file__).resolve().parents[1]


# ---- import boundary --------------------------------------------------------


def test_builder_imports_without_torch_mlx_or_edge0():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from examples import benchmark_report; "
        "bad = sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'mlx', 'mlx_lm', 'edge0')); "
        "print(bad)"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "[]", proc.stdout


# ---- per-run counts and timings ----------------------------------------------


def _run(**overrides):
    kw = {"index": 0, "prompt_tokens": 12, "requested_warmup_tokens": 3,
          "warmup_tokens": 3, "requested_timed_tokens": 5,
          "timed_decode_tokens": 5, "prefill_seconds": 0.5,
          "decode_seconds": 2.0, "memory": {}}
    kw.update(overrides)
    return br.run_result(**kw)


def test_run_result_count_arithmetic_and_throughput():
    run = _run()
    assert run["index"] == 0
    assert run["prompt_tokens"] == 12
    assert run["warmup_tokens"] == 3
    assert run["timed_decode_tokens"] == 5
    # total_generated = warmup + timed_decode; decode starts after prompt+warmup
    assert run["total_generated_tokens"] == 8
    assert run["decode_start_context_tokens"] == 15
    # throughput uses ONLY the timed decode tokens
    assert run["decode_tokens_per_second"] == pytest.approx(2.5)
    assert run["prefill_tokens_per_second"] == pytest.approx(24.0)
    assert run["requested_warmup_tokens"] == 3
    assert run["requested_timed_tokens"] == 5
    assert run["unavailable_reasons"] == {}


def test_run_result_keeps_requested_and_actual_counts_separate():
    run = _run(requested_timed_tokens=200, timed_decode_tokens=5)
    assert run["requested_timed_tokens"] == 200
    assert run["timed_decode_tokens"] == 5


@pytest.mark.parametrize("field", [
    "prompt_tokens", "requested_warmup_tokens", "warmup_tokens",
    "requested_timed_tokens", "timed_decode_tokens", "index",
])
def test_run_result_rejects_negative_counts(field):
    with pytest.raises(ValueError, match=field):
        _run(**{field: -1})


@pytest.mark.parametrize("value", [1.0, True, "5", None])
def test_run_result_rejects_non_integer_counts(value):
    with pytest.raises(ValueError, match="prompt_tokens"):
        _run(prompt_tokens=value)


def test_run_result_requires_a_positive_timed_token_count():
    with pytest.raises(ValueError, match="timed_decode_tokens"):
        _run(timed_decode_tokens=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"),
                                   float("-inf"), -0.001, "1.0", None])
def test_run_result_rejects_invalid_timings(value):
    with pytest.raises(ValueError, match="decode_seconds"):
        _run(decode_seconds=value)
    with pytest.raises(ValueError, match="prefill_seconds"):
        _run(prefill_seconds=value)


def test_zero_measured_duration_gives_unavailable_rate_not_zero():
    run = _run(decode_seconds=0.0, prefill_seconds=0)
    assert run["decode_tokens_per_second"] is None
    assert run["prefill_tokens_per_second"] is None
    assert "zero" in run["unavailable_reasons"]["decode_tokens_per_second"]
    assert "zero" in run["unavailable_reasons"]["prefill_tokens_per_second"]


# ---- memory metric objects ----------------------------------------------------


def test_memory_metric_is_integer_bytes_with_method_and_scope():
    m = br.memory_metric(3 * 1024 ** 3, method="torch.cuda.max_memory_allocated",
                         scope="per run after reset")
    assert m == {
        "bytes": 3 * 1024 ** 3,
        "unit": "bytes",
        "method": "torch.cuda.max_memory_allocated",
        "scope": "per run after reset",
        "unavailable_reason": None,
    }


def test_memory_metric_keeps_extra_details():
    m = br.memory_metric(10, method="psutil rss sampler",
                         scope="sampled peak", interval_seconds=0.25,
                         sample_count=40)
    assert m["interval_seconds"] == 0.25
    assert m["sample_count"] == 40


def test_memory_metric_unavailable_requires_a_reason():
    m = br.memory_metric(None, method="none", scope="n/a",
                         unavailable_reason="torch device is cpu, not cuda")
    assert m["bytes"] is None
    assert m["unavailable_reason"] == "torch device is cpu, not cuda"
    with pytest.raises(ValueError, match="reason"):
        br.memory_metric(None, method="none", scope="n/a")
    with pytest.raises(ValueError, match="reason"):
        br.memory_metric(5, method="m", scope="s", unavailable_reason="x")


@pytest.mark.parametrize("value", [-1, 1.5, True, "12", float("nan")])
def test_memory_metric_rejects_non_integer_or_negative_bytes(value):
    with pytest.raises(ValueError, match="bytes"):
        br.memory_metric(value, method="m", scope="s")


def test_memory_metric_requires_method_and_scope():
    with pytest.raises(ValueError, match="method"):
        br.memory_metric(1, method="", scope="s")
    with pytest.raises(ValueError, match="scope"):
        br.memory_metric(1, method="m", scope="")


# ---- report assembly ----------------------------------------------------------


def _metric(n: Any = 1024, **kw):
    kw.setdefault("method", "fake")
    kw.setdefault("scope", "fake scope")
    return br.memory_metric(n, **kw)


def _groups(**overrides):
    g = {
        "identity": {
            "timestamp_utc": "2026-09-14T10:00:00+00:00",
            "git": {"commit": "abc", "dirty": False, "unavailable_reasons": {}},
            "run_count": 2,
            "command": {"argv": ["bench.py", "x"], "env_knobs": {}},
            "requested": {"ntok": 5, "warmup": 3},
        },
        "model": {
            "checkpoint_path": "/tmp/ckpt", "tier": "edge0-8b",
            "model_type": "bailing_hybrid", "architectures": ["X"],
            "config_sha256": "00", "manifest": {"file_count": 1,
                                                "total_bytes": 1,
                                                "sha256": "11"},
            "small_file_sha256": {}, "adapters": {}, "tokenizer_files": [],
        },
        "runtime": {
            "backend": "cuda", "backend_version": "2.14.0",
            "resolved_device": "cpu",
            "measurement": {"kind": "null"},
            "execution_evidence": {"array_type": "torch.Tensor",
                                   "device": "cpu"},
            "python_version": "3.11.9", "torch_version": "2.14.0",
            "torch_cuda_version": None, "mlx_version": None,
            "platform": {"system": "Linux", "release": "x", "machine": "y"},
            "power_mode": {"value": None, "method": None,
                           "unavailable_reasons": {
                               "value": "nvpmodel not found",
                               "method": "nvpmodel not found"}},
            "unavailable_reasons": {"torch_cuda_version": "CPU-only torch",
                                    "mlx_version": "backend is cuda"},
        },
        "workload": {
            "prompt_source": "tier default", "prompt_sha256": "22",
            "prompt_token_ids_sha256": "33",
            "prompt_chars": 10, "prompt_tokens": 12,
            "requested_warmup_tokens": 3, "requested_timed_tokens": 5,
            "seed": None, "temperature": 0.7, "top_k": 64, "top_p": 0.95,
            "repetition_penalty": 1.1, "prefill_chunk": 2048,
            "think": False,
            "unavailable_reasons": {"seed": "BENCH_SEED not set"},
        },
        "caches": {
            "layer_options": {"cache_slots": 64},
            "shared_cache_slots_resolved": 64, "prefetch_cap_resolved": 48,
            "weight_cache": False, "prewarm": False,
            "threads": {"load_threads": 8},
            "mlx_cache_limit_bytes": None,
            "unavailable_reasons": {"mlx_cache_limit_bytes": "backend is cuda"},
        },
        "runs": [_run(index=0, memory={"backend_peak": _metric()}),
                 _run(index=1, decode_seconds=1.0,
                      memory={"backend_peak": _metric(2048)})],
        "memory": {
            "process_peak_rss": _metric(5000),
            "process_sampled_peak_rss": _metric(
                None, unavailable_reason="sampler disabled"),
            "notes": ["RSS and allocator peaks are separate views"],
        },
        "probe": None,
    }
    g.update(overrides)
    return g


def test_schema_version_is_independent_of_the_probe_schema():
    assert isinstance(br.SCHEMA_VERSION, int)
    report = br.build_report(**_groups())
    assert report["schema_version"] == br.SCHEMA_VERSION
    # the probe's own schema number lives only inside the probe link
    assert report["probe"]["supplied"] is False
    assert report["probe"]["schema_version"] is None
    assert "schema_version" in report["probe"]["unavailable_reasons"]


def test_protocol_block_labels_the_workload_honestly():
    report = br.build_report(**_groups())
    proto = report["protocol"]
    assert proto["name"] == "fixed-length-sample-and-step"
    assert isinstance(proto["version"], int)
    assert "greedy" in proto["warmup"].lower()
    assert "eos" in proto["timed_phase"].lower()
    assert "TTFT" in proto["not_measured"]


def test_build_report_contains_every_group_and_summary():
    report = br.build_report(**_groups())
    for key in ("identity", "model", "runtime", "workload", "caches",
                "runs", "summary", "memory", "probe"):
        assert key in report, key
    for group in ("identity", "model", "runtime", "workload", "caches",
                  "memory", "probe"):
        assert "unavailable_reasons" in report[group], group
    assert report["identity"]["run_count"] == 2
    assert len(report["runs"]) == 2


@pytest.mark.parametrize("group,field", [
    ("identity", "timestamp_utc"), ("model", "checkpoint_path"),
    ("runtime", "resolved_device"), ("workload", "prompt_tokens"),
    ("caches", "layer_options"), ("memory", "process_peak_rss"),
])
def test_build_report_rejects_missing_required_fields(group, field):
    groups = _groups()
    del groups[group][field]
    with pytest.raises(ValueError, match=field):
        br.build_report(**groups)


def test_build_report_rejects_null_without_reason_and_reason_without_null():
    groups = _groups()
    groups["workload"]["top_k"] = None
    with pytest.raises(ValueError, match="top_k"):
        br.build_report(**groups)
    groups = _groups()
    groups["workload"]["unavailable_reasons"]["top_k"] = "bogus"
    with pytest.raises(ValueError, match="top_k"):
        br.build_report(**groups)


def test_build_report_checks_nested_reason_maps():
    groups = _groups()
    groups["identity"]["git"]["commit"] = None  # nested null, no reason
    with pytest.raises(ValueError, match="commit"):
        br.build_report(**groups)


def test_build_report_requires_runs_and_consistent_run_count():
    groups = _groups(runs=[])
    with pytest.raises(ValueError, match="runs"):
        br.build_report(**groups)
    groups = _groups()
    groups["identity"]["run_count"] = 3
    with pytest.raises(ValueError, match="run_count"):
        br.build_report(**groups)


def test_build_report_rejects_nonfinite_anywhere():
    groups = _groups()
    groups["caches"]["layer_options"]["hot_decay"] = float("nan")
    with pytest.raises(ValueError, match="hot_decay"):
        br.build_report(**groups)


def test_build_report_rejects_non_json_types():
    groups = _groups()
    groups["model"]["checkpoint_path"] = Path("/tmp/x")
    with pytest.raises(ValueError, match="checkpoint_path"):
        br.build_report(**groups)


# ---- summary -------------------------------------------------------------------


def test_summary_keeps_every_run_and_defines_the_statistics():
    report = br.build_report(**_groups())
    s = report["summary"]
    assert s["run_count"] == 2
    d = s["decode_tokens_per_second"]
    assert d["values"] == pytest.approx([2.5, 5.0])
    assert d["mean"] == pytest.approx(3.75)
    assert d["min"] == pytest.approx(2.5)
    assert d["max"] == pytest.approx(5.0)
    assert d["sample_stdev"] == pytest.approx(1.7677669529663687)
    assert "independent" in s["method"]
    assert set(s) >= {"decode_tokens_per_second", "prefill_tokens_per_second",
                      "prefill_seconds", "decode_seconds"}


def test_summary_single_run_has_no_stdev():
    groups = _groups(runs=[_run(index=0, memory={})])
    groups["identity"]["run_count"] = 1
    s = br.build_report(**groups)["summary"]
    assert s["decode_tokens_per_second"]["sample_stdev"] is None
    assert "sample_stdev" in s["decode_tokens_per_second"]["unavailable_reasons"]


def test_summary_statistics_use_the_runs_that_have_a_value():
    groups = _groups(runs=[_run(index=0, memory={}),
                           _run(index=1, decode_seconds=0.0, memory={})])
    s = br.build_report(**groups)["summary"]
    d = s["decode_tokens_per_second"]
    assert d["values"] == [2.5, None]          # run order, null kept
    assert d["n"] == 2 and d["n_valid"] == 1
    assert d["mean"] == pytest.approx(2.5)
    assert d["min"] == pytest.approx(2.5) and d["max"] == pytest.approx(2.5)
    assert d["sample_stdev"] is None
    assert "n_valid" in d["unavailable_reasons"]["sample_stdev"]
    assert "mean" not in d["unavailable_reasons"]


def test_summary_with_no_valid_value_is_null_with_a_reason():
    groups = _groups(runs=[_run(index=0, decode_seconds=0.0, memory={}),
                           _run(index=1, decode_seconds=0.0, memory={})])
    d = br.build_report(**groups)["summary"]["decode_tokens_per_second"]
    assert d["n_valid"] == 0
    assert d["mean"] is None and d["min"] is None and d["max"] is None
    assert "n_valid = 0" in d["unavailable_reasons"]["mean"]


# ---- probe link ------------------------------------------------------------------


def test_probe_link_hashes_the_actual_file_content():
    import hashlib
    import json
    # deliberately NOT canonical json.dumps output: key order, spacing and
    # a trailing newline differ, so re-serializing would change the hash
    text = ('{\n  "ready": true ,\n  "collected_at_utc":'
            ' "2026-09-14T09:00:00+00:00",\n  "schema_version" : 1\n}\n')
    assert json.dumps(json.loads(text)) != text
    link = br.probe_link(path="/run/orin-probe.json", text=text)
    assert link["supplied"] is True
    assert link["schema_version"] == 1
    assert link["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = text.encode("utf-8")
    assert br.probe_link(path="p", text=raw)["sha256"] == hashlib.sha256(
        raw).hexdigest()
    assert link["ready"] is True
    assert link["collected_at_utc"] == "2026-09-14T09:00:00+00:00"
    assert link["path"] == "/run/orin-probe.json"
    assert link["unavailable_reasons"] == {}


def test_probe_link_without_a_probe():
    link = br.probe_link(path=None, text=None)
    assert link["supplied"] is False
    assert link["sha256"] is None
    assert set(link["unavailable_reasons"]) >= {"path", "schema_version",
                                                "sha256"}


@pytest.mark.parametrize("text", ["{not json", "[]", '{"ready": true}',
                                  '{"schema_version": "1"}'])
def test_probe_link_rejects_malformed_probes(text):
    with pytest.raises(ValueError, match="probe"):
        br.probe_link(path="p.json", text=text)


# ---- strict JSON and atomic writing ---------------------------------------------


def test_dumps_report_is_strict_sorted_json():
    import json
    report = br.build_report(**_groups())
    text = br.dumps_report(report)
    assert text.endswith("\n")
    assert json.loads(text) == report
    report["runs"][0]["decode_seconds"] = float("nan")
    with pytest.raises(ValueError):
        br.dumps_report(report)


def test_write_report_refuses_an_existing_destination(tmp_path):
    dest = tmp_path / "out.json"
    dest.write_text("old")
    with pytest.raises(FileExistsError):
        br.write_report(br.build_report(**_groups()), dest)
    assert dest.read_text() == "old"


def test_write_report_is_atomic_and_leaves_no_temp_files(tmp_path):
    import json
    dest = tmp_path / "out.json"
    br.write_report(br.build_report(**_groups()), dest)
    assert json.loads(dest.read_text(encoding="utf-8"))["schema_version"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_write_report_fails_visibly_when_the_directory_is_missing(tmp_path):
    dest = tmp_path / "missing" / "out.json"
    with pytest.raises(OSError):
        br.write_report(br.build_report(**_groups()), dest)
    assert not dest.exists()


def test_write_report_writes_nothing_for_an_invalid_report(tmp_path):
    dest = tmp_path / "out.json"
    report = br.build_report(**_groups())
    report["runs"][0]["decode_seconds"] = float("inf")
    with pytest.raises(ValueError):
        br.write_report(report, dest)
    assert list(tmp_path.iterdir()) == []


# ---- review follow-ups --------------------------------------------------------------


def test_write_report_removes_the_temp_when_placing_the_file_fails(
        tmp_path, monkeypatch):
    dest = tmp_path / "out.json"

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(br, "_place_new_file", boom)
    with pytest.raises(OSError, match="disk full"):
        br.write_report(br.build_report(**_groups()), dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_report_uses_a_temp_file_in_the_destination_directory(
        tmp_path, monkeypatch):
    dest_dir = tmp_path / "run"
    dest_dir.mkdir()
    seen = []
    real = br.tempfile.mkstemp

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dir"))
        return real(*args, **kwargs)
    monkeypatch.setattr(br.tempfile, "mkstemp", spy)
    br.write_report(br.build_report(**_groups()), dest_dir / "out.json")
    assert [Path(d).resolve() for d in seen] == [dest_dir.resolve()]


def test_write_report_never_overwrites_a_file_that_appears_meanwhile(
        tmp_path, monkeypatch):
    dest = tmp_path / "out.json"
    real_fsync = br.os.fsync

    def racer(fd):
        dest.write_text("someone else")   # appears after the pre-check
        real_fsync(fd)
    monkeypatch.setattr(br.os, "fsync", racer)
    with pytest.raises(FileExistsError):
        br.write_report(br.build_report(**_groups()), dest)
    assert dest.read_text() == "someone else"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_write_report_honors_the_umask(tmp_path):
    import os
    dest = tmp_path / "out.json"
    mask = os.umask(0o027)
    try:
        br.write_report(br.build_report(**_groups()), dest)
    finally:
        os.umask(mask)
    assert dest.stat().st_mode & 0o777 == 0o640


def test_summary_reports_n_and_n_valid():
    groups = _groups(runs=[_run(index=0, memory={}),
                           _run(index=1, decode_seconds=0.0, memory={})])
    s = br.build_report(**groups)["summary"]
    d = s["decode_tokens_per_second"]
    assert d["n"] == 2 and d["n_valid"] == 1
    p = s["prefill_seconds"]
    assert p["n"] == 2 and p["n_valid"] == 2
    assert "n_valid" in s["method"]


def test_workload_requires_the_prompt_token_id_digest():
    groups = _groups()
    del groups["workload"]["prompt_token_ids_sha256"]
    with pytest.raises(ValueError, match="prompt_token_ids_sha256"):
        br.build_report(**groups)
