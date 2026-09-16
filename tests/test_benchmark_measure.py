"""Benchmark-owned measurement adapter (``examples/benchmark_measure.py``).

Everything here runs with fakes: no CUDA, no MLX, no checkpoint.  The
CUDA dispatch path is exercised through an injected ``cuda_api`` object,
the MLX path through a fake ``core`` namespace, RSS/git/power collectors
through injected readers.  Real-device behavior is Task 3's job.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples import benchmark_measure as bm

ROOT = Path(__file__).resolve().parents[1]


# ---- import boundary -----------------------------------------------------------


def test_measure_module_imports_without_torch_or_mlx():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from examples import benchmark_measure; "
        "bad = sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'mlx', 'mlx_lm', 'edge0')); "
        "print(bad)"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "[]", proc.stdout


# ---- fakes -----------------------------------------------------------------------


class FakeCudaApi:
    """Records every call in order, like a torch.cuda namespace would see."""

    def __init__(self, allocated: Any = 3 * 1024 ** 3,
                 reserved: Any = 4 * 1024 ** 3):
        self.calls: list[tuple] = []
        self._allocated = allocated
        self._reserved = reserved

    def synchronize(self, device=None):
        self.calls.append(("synchronize", device))

    def reset_peak_memory_stats(self, device=None):
        self.calls.append(("reset_peak_memory_stats", device))

    def max_memory_allocated(self, device=None):
        self.calls.append(("max_memory_allocated", device))
        return self._allocated

    def max_memory_reserved(self, device=None):
        self.calls.append(("max_memory_reserved", device))
        return self._reserved


class FakeDevice:
    def __init__(self, type_, index=0):
        self.type = type_
        self.index = index

    def __str__(self):
        return f"{self.type}:{self.index}"


class FakeMlxCore:
    def __init__(self, peak=1500):
        self.calls = []
        self._peak = peak

    def reset_peak_memory(self):
        self.calls.append("reset_peak_memory")

    def get_peak_memory(self):
        self.calls.append("get_peak_memory")
        return self._peak

    def default_device(self):
        return "Device(gpu, 0)"


# ---- dispatch -------------------------------------------------------------------


def test_cuda_backend_on_a_cuda_device_uses_torch_cuda_metrics():
    api = FakeCudaApi()
    core = SimpleNamespace(DEVICE=FakeDevice("cuda", 0))
    m = bm.make_measurement("cuda", core, cuda_api=api)
    assert isinstance(m, bm.CudaMeasurement)
    d = m.describe()
    assert d["kind"] == "cuda"
    assert d["backend"] == "cuda"
    assert d["resolved_device"] == "cuda:0"
    assert "synchronize" in d["synchronization"]


def test_cuda_backend_on_a_cpu_device_reports_unavailable_peaks():
    core = SimpleNamespace(DEVICE=FakeDevice("cpu"))
    m = bm.make_measurement("cuda", core, cuda_api=FakeCudaApi())
    assert isinstance(m, bm.NullMeasurement)
    m.synchronize()
    m.reset_peak()
    peaks = m.peak_metrics()
    assert list(peaks) == ["backend_peak"]
    assert peaks["backend_peak"]["bytes"] is None
    assert "cpu" in peaks["backend_peak"]["unavailable_reason"]
    assert "cuda" in peaks["backend_peak"]["unavailable_reason"]
    desc = m.describe()
    assert desc["resolved_device"] == "cpu:0"
    assert desc["synchronization"].startswith("none")
    assert "no device" not in desc["synchronization"]
    assert "mps" not in desc["synchronization"]


def test_cuda_backend_on_mps_reports_unavailable_peaks():
    core = SimpleNamespace(DEVICE=FakeDevice("mps"))
    m = bm.make_measurement("cuda", core, cuda_api=FakeCudaApi())
    assert isinstance(m, bm.NullMeasurement)
    assert "mps" in m.peak_metrics()["backend_peak"]["unavailable_reason"]
    sync = m.describe()["synchronization"]
    assert sync.startswith("none")
    assert "no device" not in sync
    assert "mps" in sync and "asynchronous" in sync


def test_mlx_backend_keeps_the_mlx_peak_memory_behavior():
    core = FakeMlxCore(peak=1500)
    m = bm.make_measurement("mlx", core)
    assert isinstance(m, bm.MlxMeasurement)
    m.synchronize()          # no-op: MLX timing behavior is unchanged
    m.reset_peak()
    peaks = m.peak_metrics()
    assert core.calls == ["reset_peak_memory", "get_peak_memory"]
    assert peaks["mlx_peak_active"]["bytes"] == 1500
    assert peaks["mlx_peak_active"]["method"] == "mlx.core.get_peak_memory"
    assert "reset" in peaks["mlx_peak_active"]["scope"]
    assert m.describe()["kind"] == "mlx"


def test_mlx_core_without_peak_api_is_null_with_a_reason():
    m = bm.make_measurement("mlx", SimpleNamespace())
    assert isinstance(m, bm.NullMeasurement)
    assert "peak" in m.peak_metrics()["backend_peak"]["unavailable_reason"]


def test_unknown_backend_is_null_with_a_reason():
    m = bm.make_measurement("other", SimpleNamespace())
    assert isinstance(m, bm.NullMeasurement)
    assert "other" in m.peak_metrics()["backend_peak"]["unavailable_reason"]


# ---- CUDA call semantics ----------------------------------------------------------


def test_cuda_measurement_calls_are_bound_to_the_resolved_device():
    api = FakeCudaApi(allocated=123, reserved=456)
    dev = FakeDevice("cuda", 1)
    m = bm.CudaMeasurement(dev, api)
    m.synchronize()
    m.reset_peak()
    m.synchronize()
    peaks = m.peak_metrics()
    assert api.calls == [
        ("synchronize", dev),
        ("reset_peak_memory_stats", dev),
        ("synchronize", dev),
        ("max_memory_allocated", dev),
        ("max_memory_reserved", dev),
    ]
    assert peaks["cuda_peak_allocated"]["bytes"] == 123
    assert peaks["cuda_peak_reserved"]["bytes"] == 456
    assert peaks["cuda_peak_allocated"]["method"] == (
        "torch.cuda.max_memory_allocated")
    assert peaks["cuda_peak_reserved"]["method"] == (
        "torch.cuda.max_memory_reserved")
    for metric in peaks.values():
        assert "allocator" in metric["scope"]
        assert "reset" in metric["scope"]


def test_cuda_measurement_rejects_nonfinite_or_negative_peaks():
    m = bm.CudaMeasurement(FakeDevice("cuda"), FakeCudaApi(allocated=-1))
    with pytest.raises(ValueError, match="bytes"):
        m.peak_metrics()
    m = bm.CudaMeasurement(FakeDevice("cuda"),
                           FakeCudaApi(allocated=float("nan")))
    with pytest.raises(ValueError, match="bytes"):
        m.peak_metrics()


# ---- execution evidence ------------------------------------------------------------


def test_execution_evidence_records_array_type_and_device():
    class Tensor:
        device = FakeDevice("cuda", 0)
    Tensor.__module__ = "torch"
    ev = bm.execution_evidence(Tensor())
    assert ev["array_type"] == "torch.Tensor"
    assert ev["device"] == "cuda:0"
    assert ev["unavailable_reasons"] == {}


def test_execution_evidence_without_a_device_attribute():
    class array:  # mimics mlx.core.array (lowercase on purpose)
        pass
    array.__module__ = "mlx.core"
    ev = bm.execution_evidence(array())
    assert ev["array_type"] == "mlx.core.array"
    assert ev["device"] is None
    assert "mlx.core.array" in ev["unavailable_reasons"]["device"]
    assert "device attribute" in ev["unavailable_reasons"]["device"]


# ---- process RSS ----------------------------------------------------------------------


def test_process_peak_rss_linux_converts_kib_to_bytes():
    ru = SimpleNamespace(ru_maxrss=2048)
    m = bm.process_peak_rss(system="Linux", getrusage=lambda who: ru)
    assert m["bytes"] == 2048 * 1024
    assert "ru_maxrss" in m["method"]
    assert "lifetime" in m["scope"]


def test_process_peak_rss_macos_is_already_bytes():
    ru = SimpleNamespace(ru_maxrss=2048)
    m = bm.process_peak_rss(system="Darwin", getrusage=lambda who: ru)
    assert m["bytes"] == 2048


def test_process_peak_rss_windows_uses_peak_working_set():
    fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(peak_wset=777, rss=700)))
    m = bm.process_peak_rss(system="Windows", psutil_module=fake_psutil)
    assert m["bytes"] == 777
    assert "peak_wset" in m["method"]


def test_process_peak_rss_unavailable_has_a_reason():
    m = bm.process_peak_rss(system="Plan9", getrusage=None,
                            psutil_module=None)
    assert m["bytes"] is None
    assert "Plan9" in m["unavailable_reason"]


def test_process_peak_rss_on_this_host_is_a_positive_integer():
    m = bm.process_peak_rss()
    if m["bytes"] is None:
        pytest.skip(m["unavailable_reason"])
    assert isinstance(m["bytes"], int) and m["bytes"] > 0


# ---- RSS sampler ------------------------------------------------------------------------


def test_rss_sampler_reports_a_sampled_peak_with_its_interval():
    values = iter([100, 300, 200, 250])
    sampler = bm.RssSampler(0.01, read_rss=lambda: next(values, 250))
    sampler.start()
    import time
    time.sleep(0.08)
    sampler.stop()
    m = sampler.metric()
    assert m["bytes"] == 300
    assert m["interval_seconds"] == 0.01
    assert m["sample_count"] >= 2
    assert "sampled" in m["scope"]
    assert "psutil" in m["method"] or "read_rss" in m["method"]


def test_rss_sampler_disabled_is_unavailable():
    sampler = bm.RssSampler(0)
    sampler.start()
    sampler.stop()
    m = sampler.metric()
    assert m["bytes"] is None
    assert "disabled" in m["unavailable_reason"]


def test_rss_sampler_stop_is_idempotent_and_thread_exits():
    sampler = bm.RssSampler(0.01, read_rss=lambda: 5)
    sampler.start()
    assert sampler.is_running
    sampler.stop()
    sampler.stop()
    assert not sampler.is_running
    assert sampler.metric()["bytes"] == 5


def test_rss_sampler_reports_a_reader_that_will_not_stop():
    import threading
    import time
    release = threading.Event()
    started = threading.Event()

    def blocked():
        if started.is_set():
            release.wait(5)      # the second sample hangs
        started.set()
        return 7

    sampler = bm.RssSampler(0.01, read_rss=blocked, join_timeout=0.05)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    try:
        assert sampler.is_running          # not hidden: the thread is stuck
        m = sampler.metric()
        assert m["bytes"] == 7
        assert "did not stop" in m["warning"]
    finally:
        release.set()
        sampler.stop()
    assert not sampler.is_running


# ---- git identity --------------------------------------------------------------------------


def _fake_run(responses):
    def run(cmd, **kw):
        key = tuple(cmd[cmd.index("git") + 1:]) if "git" in cmd else tuple(cmd)
        rc, out = responses[key[-1] if key[-1] in responses else key]
        return SimpleNamespace(returncode=rc, stdout=out, stderr="")
    return run


def test_git_identity_reports_commit_and_dirty_state():
    run = _fake_run({"HEAD": (0, "abc123\n"), "--untracked-files=no": (0, " M x.py\n")})
    g = bm.git_identity("/repo", run=run)
    assert g == {"commit": "abc123", "dirty": True, "unavailable_reasons": {}}


def test_git_identity_clean_tree():
    run = _fake_run({"HEAD": (0, "abc123\n"), "--untracked-files=no": (0, "")})
    assert bm.git_identity("/repo", run=run)["dirty"] is False


def test_git_identity_unavailable_when_git_fails():
    def run(cmd, **kw):
        raise FileNotFoundError("git")
    g = bm.git_identity("/repo", run=run)
    assert g["commit"] is None and g["dirty"] is None
    assert "git" in g["unavailable_reasons"]["commit"]


def test_git_identity_on_this_checkout():
    g = bm.git_identity(str(ROOT))
    if g["commit"] is None:
        pytest.skip(g["unavailable_reasons"]["commit"])
    assert len(g["commit"]) == 40


# ---- power mode -------------------------------------------------------------------------------


def test_power_mode_parses_nvpmodel_output():
    def run(cmd, **kw):
        assert cmd[0] == "nvpmodel"
        return SimpleNamespace(returncode=0, stdout="NV Power Mode: 15W\n1\n",
                               stderr="")
    p = bm.power_mode(run=run)
    assert p["value"] == "15W"
    assert p["mode_id"] == 1
    assert p["method"] == "nvpmodel -q"
    assert p["unavailable_reasons"] == {}


def test_power_mode_falls_back_to_the_status_file(tmp_path):
    status = tmp_path / "status"
    status.write_text("pmode:0002 fmode:quiet\n")

    def run(cmd, **kw):
        raise FileNotFoundError("nvpmodel")
    p = bm.power_mode(run=run, status_path=status)
    assert p["mode_id"] == 2
    assert p["value"] == "pmode 2"
    assert "status" in p["method"]


def test_power_mode_unavailable_with_reason(tmp_path):
    def run(cmd, **kw):
        raise FileNotFoundError("nvpmodel")
    p = bm.power_mode(run=run, status_path=tmp_path / "absent")
    assert p["value"] is None and p["method"] is None and p["mode_id"] is None
    assert "nvpmodel" in p["unavailable_reasons"]["value"]


# ---- checkpoint manifest -------------------------------------------------------------------


def _safetensors(path: Path, metadata: dict, n_tensors: int = 2):
    header = {"__metadata__": metadata}
    for i in range(n_tensors):
        header[f"t{i}"] = {"dtype": "F16", "shape": [1],
                           "data_offsets": [2 * i, 2 * i + 2]}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * (2 * n_tensors))


def test_checkpoint_manifest_identifies_files_without_hashing_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "bailing_hybrid"}')
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    _safetensors(tmp_path / "model.safetensors", {"format": "mlx"})
    _safetensors(tmp_path / "lora_edge0_8b.safetensors",
                 {"edge0_adapter": "lora", "commit": "deadbeef"}, 1)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 100)

    man = bm.checkpoint_manifest(tmp_path, hash_limit_bytes=50)
    assert man["manifest"]["file_count"] == 5
    assert man["manifest"]["total_bytes"] == sum(
        p.stat().st_size for p in tmp_path.iterdir())
    assert len(man["manifest"]["sha256"]) == 64
    # small non-weight files are hashed, the "big" file and weights are not
    assert set(man["small_file_sha256"]) == {"chat_template.jinja",
                                             "config.json"}
    header = man["safetensors_headers"]["model.safetensors"]
    assert header["metadata"] == {"format": "mlx"}
    assert header["tensor_count"] == 2
    assert header["unavailable_reasons"] == {}
    assert man["safetensors_headers"]["lora_edge0_8b.safetensors"][
        "metadata"]["commit"] == "deadbeef"
    assert man["config"]["model_type"] == "bailing_hybrid"


def test_checkpoint_manifest_digest_tracks_names_and_sizes(tmp_path):
    (tmp_path / "a").write_bytes(b"12")
    d1 = bm.checkpoint_manifest(tmp_path)["manifest"]["sha256"]
    (tmp_path / "a").write_bytes(b"123")
    d2 = bm.checkpoint_manifest(tmp_path)["manifest"]["sha256"]
    assert d1 != d2


def test_checkpoint_manifest_of_a_missing_directory_is_unavailable(tmp_path):
    man = bm.checkpoint_manifest(tmp_path / "nope")
    assert man["manifest"]["sha256"] is None
    assert "sha256" in man["manifest"]["unavailable_reasons"]
    assert man["config"] == {}


# ---- runtime info --------------------------------------------------------------------------


def test_runtime_info_cuda_backend_reports_torch_versions():
    fake_torch = SimpleNamespace(__version__="2.14.0+cpu",
                                 version=SimpleNamespace(cuda=None))

    def importer(name):
        assert name == "torch"
        return fake_torch
    info = bm.runtime_info("cuda", importer=importer)
    assert info["torch_version"] == "2.14.0+cpu"
    assert info["torch_cuda_version"] is None
    assert "torch_cuda_version" in info["unavailable_reasons"]
    assert info["mlx_version"] is None
    assert "mlx_version" in info["unavailable_reasons"]
    assert info["python_version"] == sys.version.split()[0]
    assert set(info["platform"]) == {"system", "release", "machine"}


def test_runtime_info_mlx_backend_reports_mlx_version():
    def importer(name):
        assert name == "mlx"
        return SimpleNamespace(__version__="0.30.4")
    info = bm.runtime_info("mlx", importer=importer)
    assert info["mlx_version"] == "0.30.4"
    assert info["torch_version"] is None
    assert "torch_version" in info["unavailable_reasons"]


def test_runtime_info_import_failure_is_a_reason_not_a_crash():
    def importer(name):
        raise ImportError("no module")
    info = bm.runtime_info("cuda", importer=importer)
    assert info["torch_version"] is None
    assert "no module" in info["unavailable_reasons"]["torch_version"]


# ---- review follow-ups: headline peaks, reserved scope, at-reset details ------------


class FakeCudaApiWithCurrent(FakeCudaApi):
    def __init__(self, allocated=1 * 1024 ** 3, reserved=2 * 1024 ** 3,
                 current_allocated=100, current_reserved=200):
        super().__init__(allocated, reserved)
        self._cur_alloc = current_allocated
        self._cur_res = current_reserved

    def memory_allocated(self, device=None):
        self.calls.append(("memory_allocated", device))
        return self._cur_alloc

    def memory_reserved(self, device=None):
        self.calls.append(("memory_reserved", device))
        return self._cur_res


def test_cuda_headline_is_allocated_and_reserved_has_its_own_scope():
    api = FakeCudaApiWithCurrent()
    m = bm.CudaMeasurement(FakeDevice("cuda"), api)
    assert m.headline == "cuda_peak_allocated"
    m.reset_peak()
    peaks = m.peak_metrics()
    allocated = peaks["cuda_peak_allocated"]
    reserved = peaks["cuda_peak_reserved"]
    assert allocated["scope"] != reserved["scope"]
    assert "transient" in allocated["scope"]
    assert "cached" in reserved["scope"] or "segments" in reserved["scope"]
    assert "allocator" in reserved["scope"]
    # the values current at the reset are recorded next to each peak
    assert allocated["allocated_at_reset"] == 100
    assert reserved["reserved_at_reset"] == 200
    assert m.describe()["headline_peak"] == "cuda_peak_allocated"


def test_cuda_measurement_without_current_readers_still_reports_peaks():
    m = bm.CudaMeasurement(FakeDevice("cuda"), FakeCudaApi(allocated=5,
                                                         reserved=6))
    m.reset_peak()
    peaks = m.peak_metrics()
    assert peaks["cuda_peak_allocated"]["bytes"] == 5
    assert "allocated_at_reset" not in peaks["cuda_peak_allocated"]


def test_headline_names_for_mlx_and_null():
    assert bm.MlxMeasurement(FakeMlxCore()).headline == "mlx_peak_active"
    assert bm.NullMeasurement("cuda", "cpu", "r").headline == "backend_peak"
    assert bm.NullMeasurement("cuda", "cpu", "r").describe()[
        "headline_peak"] == "backend_peak"


def test_mps_null_measurement_reports_the_resolved_device():
    core = SimpleNamespace(DEVICE=FakeDevice("mps"))
    m = bm.make_measurement("cuda", core, cuda_api=FakeCudaApi())
    assert m.describe()["resolved_device"] == "mps:0"
    assert "cuda" in m.peak_metrics()["backend_peak"]["unavailable_reason"]


# ---- process memory composition ----------------------------------------------------


def test_process_peak_rss_scope_names_file_backed_pages():
    m = bm.process_peak_rss(system="Linux",
                            getrusage=lambda who: SimpleNamespace(ru_maxrss=1))
    assert "file-backed" in m["scope"]
    assert "anonymous" in m["scope"]


def test_parse_proc_status_extracts_rss_composition():
    text = ("Name:\tpython3\nVmRSS:\t   2048 kB\nRssAnon:\t 1024 kB\n"
            "RssFile:\t 1000 kB\nRssShmem:\t 24 kB\nVmSwap:\t 16 kB\n")
    parsed = bm.parse_proc_status(text)
    assert parsed == {"rss": 2048 * 1024, "rss_anon": 1024 * 1024,
                      "rss_file": 1000 * 1024, "swap": 16 * 1024}


def test_parse_proc_status_missing_fields_are_none():
    parsed = bm.parse_proc_status("VmRSS:\t 10 kB\n")
    assert parsed["rss"] == 10 * 1024
    assert parsed["rss_anon"] is None and parsed["swap"] is None


def test_rss_sampler_tracks_anon_file_and_swap_peaks():
    samples = iter([
        {"rss": 100, "rss_anon": 60, "rss_file": 40, "swap": 0},
        {"rss": 300, "rss_anon": 70, "rss_file": 230, "swap": 5},
        {"rss": 200, "rss_anon": 90, "rss_file": 110, "swap": 2},
    ])
    last = {"rss": 200, "rss_anon": 90, "rss_file": 110, "swap": 2}
    sampler = bm.RssSampler(0.01, read_memory=lambda: next(samples, last))
    sampler.start()
    import time
    time.sleep(0.08)
    sampler.stop()
    metrics = sampler.metrics()
    assert metrics["process_sampled_peak_rss"]["bytes"] == 300
    assert metrics["process_sampled_peak_rss_anon"]["bytes"] == 90
    assert metrics["process_sampled_peak_rss_file"]["bytes"] == 230
    assert metrics["process_sampled_peak_swap"]["bytes"] == 5
    assert sampler.metric() == metrics["process_sampled_peak_rss"]
    for m in metrics.values():
        assert m["interval_seconds"] == 0.01
        assert "sampled" in m["scope"]


def test_rss_sampler_reports_unavailable_composition_with_reasons():
    sampler = bm.RssSampler(0.01, read_memory=lambda: {"rss": 7})
    sampler.start()
    sampler.stop()
    metrics = sampler.metrics()
    assert metrics["process_sampled_peak_rss"]["bytes"] == 7
    anon = metrics["process_sampled_peak_rss_anon"]
    assert anon["bytes"] is None
    assert "proc" in anon["unavailable_reason"].lower()


def test_read_process_memory_on_this_host_returns_rss():
    mem = bm.read_process_memory()
    if mem is None:
        pytest.skip("no process memory reader on this host")
    assert isinstance(mem["rss"], int) and mem["rss"] > 0


# ---- checkpoint identity follow-ups ----------------------------------------------------


def test_checkpoint_manifest_retains_the_file_list_that_reproduces_the_digest(tmp_path):
    import hashlib
    (tmp_path / "b").write_bytes(b"12")
    (tmp_path / "a").write_bytes(b"123")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c").write_bytes(b"1")
    man = bm.checkpoint_manifest(tmp_path)["manifest"]
    files = man["files"]
    assert [f["path"] for f in files] == ["a", "b", "sub/c"]
    assert len(files) == man["file_count"]
    assert sum(f["size_bytes"] for f in files) == man["total_bytes"]
    digest = hashlib.sha256("".join(
        f"{f['path']}\t{f['size_bytes']}\n" for f in files).encode()).hexdigest()
    assert digest == man["sha256"]
    assert "sha256" in man["algorithm"] and "size" in man["algorithm"]


def test_safetensors_header_records_a_header_hash(tmp_path):
    import hashlib
    path = tmp_path / "w.safetensors"
    _safetensors(path, {"k": "v"}, 3)
    header = bm.safetensors_header(path)
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    assert header["header_sha256"] == hashlib.sha256(raw[8:8 + n]).hexdigest()
    assert header["tensor_count"] == 3


def test_adapter_identity_hashes_the_actual_adapter_file(tmp_path):
    import hashlib
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    inside = ckpt / "lora.safetensors"
    _safetensors(inside, {"edge0_adapter": "lora"}, 1)
    outside = tmp_path / "artifacts" / "prerouter.safetensors"
    outside.parent.mkdir()
    _safetensors(outside, {}, 2)

    a = bm.adapter_identity(str(inside), ckpt)
    assert a["file"] == "lora.safetensors"
    assert a["path"] == str(inside.resolve())
    assert a["inside_checkpoint_dir"] is True
    assert a["size_bytes"] == inside.stat().st_size
    assert a["sha256"] == hashlib.sha256(inside.read_bytes()).hexdigest()
    assert a["safetensors_header"]["metadata"] == {"edge0_adapter": "lora"}
    assert a["unavailable_reasons"] == {}

    b = bm.adapter_identity(str(outside), ckpt)
    assert b["inside_checkpoint_dir"] is False
    assert b["file"] == str(outside.resolve())

    c = bm.adapter_identity(str(outside), ckpt, sha256_max_bytes=10)
    assert c["sha256"] is None
    assert "sha256" in c["unavailable_reasons"]
    assert "larger" in c["unavailable_reasons"]["sha256"]


def test_adapter_identity_disabled_and_missing(tmp_path):
    d = bm.adapter_identity("", tmp_path)
    assert d["file"] is None and d["sha256"] is None
    assert "disabled" in d["unavailable_reasons"]["file"]
    m = bm.adapter_identity(str(tmp_path / "absent.safetensors"), tmp_path)
    assert m["path"].endswith("absent.safetensors")
    assert m["size_bytes"] is None
    assert "not found" in m["unavailable_reasons"]["size_bytes"]
    assert m["inside_checkpoint_dir"] is True
