"""``examples/bench.py`` end to end with fakes only: no backend, no weights.

``bench.py`` imports the framework lazily (``bench._framework``), so this
module runs on a host with neither torch nor MLX: the engine, the array
namespace, the sampler and the measurement adapter are all fakes that
record the order of every call.  That is what lets the phase
synchronization order, the count arithmetic, the memory units, the
pre-load validation, the failure paths and the cleanup be asserted
anywhere.  ``tests/test_bench_backend.py`` repeats the core cases on a
real backend when one is importable.  Host results are instrumentation
checks, not Orin evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples import bench
from examples import benchmark_measure as bm
from examples import benchmark_report as br

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 8
GIB = 1024 ** 3

HUMAN_LINE = re.compile(
    r"^prompt=(\d+) tok  prefill=([\d.]+)s \((\d+) tok/s\)  "
    r"decode=(\d+)/([\d.]+)s  tok/s=([\d.]+)  "
    r"peak_active=((?:[\d.]+ GiB)|n/a)$")
SUMMARY_LINE = re.compile(
    r"^\[bench\] mean tok/s=([\d.]+)  peak_active<=((?:[\d.]+ GiB)|n/a)$")


# ---- import boundary --------------------------------------------------------------


def test_bench_module_imports_without_torch_mlx_or_edge0():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from examples import bench; "
        "bad = sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'mlx', 'mlx_lm', 'edge0')); "
        "print(bad)"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "[]", proc.stdout


def test_help_and_argument_errors_need_no_backend():
    env = dict(os.environ, EDGE0_BACKEND="no-such-backend")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "bench.py"), "--help"],
        capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "--json-output" in proc.stdout
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "bench.py"), "x", "--ntok", "0"],
        capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 2
    assert "--ntok" in proc.stderr


# ---- fakes ------------------------------------------------------------------------------


def _argmax(logits):
    return max(range(len(logits)), key=logits.__getitem__)


class FakeCore:
    """The two ``core`` calls the benchmark loop makes."""

    def __init__(self):
        self.seeds: list[int] = []
        self.random = SimpleNamespace(seed=self.seeds.append)

    @staticmethod
    def argmax(logits, axis=-1):
        return SimpleNamespace(item=lambda: _argmax(logits))


def fake_sample(logits, temperature=0.7, top_k=None, top_p=None,
                repetition_penalty=1.0, history=(), seed=None):
    """Deliberately NOT the argmax, so a warmup step that sampled (or a
    timed step that took the argmax) is visible in the recorded tokens."""
    return (_argmax(logits) + 1) % VOCAB


class FakeTok:
    def __init__(self, ids):
        self.ids = list(ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False):
        return "templated:" + messages[-1]["content"]

    def __call__(self, text):
        return {"input_ids": list(self.ids)}


@dataclass
class FakeOptions:
    cache_slots: int = 64
    prefetch_cap: int = 48
    load_threads: int = 8
    prefetch_threads: int = 4
    top_k: int | None = None


def _safetensors(path: Path, metadata: dict, n_tensors: int = 1):
    header: dict = {"__metadata__": metadata}
    for i in range(n_tensors):
        header[f"t{i}"] = {"dtype": "F16", "shape": [1],
                           "data_offsets": [2 * i, 2 * i + 2]}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * (2 * n_tensors))


class FakeEngine:
    """Duck-typed Edge0Engine: logits are lists whose argmax is
    ``pos % VOCAB``; every call is recorded in ``events``."""

    name = "fake-tier"

    def __init__(self, prompt_ids=(1, 2, 3), fail_at_step=None,
                 on_step=None, model_dir="fake-dir", lora="",
                 prerouter_file=""):
        self._tok = FakeTok(prompt_ids)
        self.cfg = SimpleNamespace(
            gen=SimpleNamespace(temperature=0.7, top_p=0.95, top_k=4,
                                repetition_penalty=1.0),
            options=FakeOptions(), model_dir=model_dir,
            lora=lora, lora_r=16, lora_alpha=32.0,
            prerouter=SimpleNamespace(
                weights_file=prerouter_file, start_layer=7, hidden=512,
                dtype="fp16", feature_topk="executed", owners=(7, 8)),
            prerouter_top_k=8)
        self.dir = model_dir
        self.prefill_chunk = 2048
        self.think = False
        self.events: list = []
        self.closed = 0
        self.pos = 0
        self.steps = 0
        self._fail_at_step = fail_at_step
        self._on_step = on_step
        self._logits = None
        self._all_stream_layers = {
            1: SimpleNamespace(shared_cache=SimpleNamespace(slots=64),
                               _prefetch_buf=SimpleNamespace(cap=48))}

    def _make_logits(self):
        row = [0.0] * VOCAB
        row[self.pos % VOCAB] = 10.0
        return row

    def reset(self):
        self.events.append("reset")
        self.pos = 0

    def prefill(self, ids):
        self.events.append(("prefill", len(ids)))
        self.pos += len(ids)
        self._logits = self._make_logits()

    def next_logits(self):
        return self._logits

    def step(self, tid):
        self.steps += 1
        self.events.append(("step", int(tid)))
        if self._on_step is not None:
            self._on_step(self.steps)
        if self._fail_at_step is not None and self.steps == self._fail_at_step:
            raise RuntimeError("synthetic decode failure")
        self.pos += 1
        return self._make_logits()

    def stats(self):
        return {1: {"loads": 3, "hits": 5}}

    def close(self):
        self.closed += 1
        self.events.append("close")


class RecordingMeasurement:
    kind = "fake"
    headline = "fake_peak"

    def __init__(self, engine, peak: Any = 3 * GIB, sync_sleep=0.0):
        self.engine = engine
        self.peak = peak
        self.sync_sleep = sync_sleep

    def synchronize(self):
        self.engine.events.append("sync")
        if self.sync_sleep:
            time.sleep(self.sync_sleep)

    def reset_peak(self):
        self.engine.events.append("reset_peak")

    def peak_metrics(self):
        self.engine.events.append("peaks")
        return {"fake_peak": br.memory_metric(self.peak, method="fake",
                                              scope="fake scope")}

    def describe(self):
        return {"kind": "fake", "backend": "fake", "resolved_device": "fake",
                "synchronization": "recorded", "peak_source": "fake",
                "headline_peak": self.headline}


class FakeCudaApi:
    def __init__(self, allocated, reserved):
        self.calls: list = []
        self._a, self._r = allocated, reserved

    def synchronize(self, device=None):
        self.calls.append("synchronize")

    def reset_peak_memory_stats(self, device=None):
        self.calls.append("reset_peak_memory_stats")

    def max_memory_allocated(self, device=None):
        return self._a

    def max_memory_reserved(self, device=None):
        return self._r


def _kinds(events):
    return [e[0] if isinstance(e, tuple) else e for e in events]


def _match(pattern: re.Pattern, line: str) -> re.Match:
    m = pattern.match(line)
    assert m is not None, f"{line!r} does not match {pattern.pattern}"
    return m


def _expected_run(warmup, ntok):
    return (["reset", "sync", "reset_peak", "prefill", "sync"]
            + ["step"] * warmup + ["sync"] + ["step"] * ntok
            + ["sync", "peaks"])


@pytest.fixture
def fake_framework(monkeypatch):
    core = FakeCore()
    backend = SimpleNamespace(name="fake", version="0.0")
    monkeypatch.setattr(bench, "_framework",
                        lambda: (backend, core, fake_sample))
    monkeypatch.delenv("BENCH_PROMPT", raising=False)
    monkeypatch.delenv("BENCH_SEED", raising=False)
    monkeypatch.delenv("BENCH_TEMP", raising=False)
    monkeypatch.delenv("BENCH_LONG", raising=False)
    return core


# ---- run_bench: order, counts, units --------------------------------------------------


def test_run_bench_synchronizes_at_phase_boundaries_only(fake_framework):
    eng = FakeEngine(prompt_ids=(1, 2, 3))
    out = bench.run_bench(eng, ntok=4, warmup=2,
                          measurement=RecordingMeasurement(eng))
    assert _kinds(eng.events) == _expected_run(2, 4) * 2
    assert eng.closed == 0  # run_bench never closes: the CLI's finally does
    assert len(out["report_runs"]) == 2


def test_run_bench_warmup_is_greedy_and_counts_are_separate(fake_framework):
    eng = FakeEngine(prompt_ids=(1, 2, 3))
    out = bench.run_bench(eng, ntok=5, warmup=2,
                          measurement=RecordingMeasurement(eng))
    steps = [e[1] for e in eng.events if isinstance(e, tuple) and e[0] == "step"]
    # after a 3-token prefill the argmax is pos % VOCAB.  Warmup steps take
    # the argmax (3, 4); timed steps go through the sampler, whose fake
    # returns argmax + 1 (5->6, 6->7, 7->0, 0->1, 1->2).
    assert steps[:7] == [3, 4, 6, 7, 0, 1, 2]
    run = out["report_runs"][0]
    assert run["prompt_tokens"] == 3
    assert run["warmup_tokens"] == 2
    assert run["timed_decode_tokens"] == 5
    assert run["total_generated_tokens"] == 7
    assert run["decode_start_context_tokens"] == 5
    assert run["requested_timed_tokens"] == 5
    assert run["requested_warmup_tokens"] == 2
    assert run["decode_tokens_per_second"] == pytest.approx(
        5 / run["decode_seconds"])


def test_run_bench_synchronization_lands_inside_the_timed_phases(fake_framework):
    eng = FakeEngine()
    out = bench.run_bench(eng, ntok=2, warmup=1, runs=1,
                          measurement=RecordingMeasurement(eng, sync_sleep=0.1))
    run = out["report_runs"][0]
    # exactly one boundary sync inside each timed phase; the post-warmup
    # sync is outside both
    assert 0.1 <= run["prefill_seconds"] < 0.2
    assert 0.1 <= run["decode_seconds"] < 0.2


def test_run_bench_seeds_every_run_with_the_resolved_seed(fake_framework,
                                                          monkeypatch):
    core = fake_framework
    monkeypatch.setenv("BENCH_SEED", "0")
    eng = FakeEngine()
    out = bench.run_bench(eng, ntok=1, warmup=0,
                          measurement=RecordingMeasurement(eng))
    assert core.seeds == [0, 0]            # BENCH_SEED=0 seeds both runs
    assert out["workload"]["seed"] == 0
    core.seeds.clear()
    monkeypatch.delenv("BENCH_SEED")
    out = bench.run_bench(eng, ntok=1, warmup=0,
                          measurement=RecordingMeasurement(eng))
    assert core.seeds == []
    assert out["workload"]["seed"] is None


def test_run_bench_memory_units_are_bytes_in_json_and_gib_in_text(
        fake_framework, capsys):
    eng = FakeEngine()
    out = bench.run_bench(eng, ntok=2, warmup=0,
                          measurement=RecordingMeasurement(eng, peak=3 * GIB))
    assert out["report_runs"][0]["memory"]["fake_peak"]["bytes"] == 3 * GIB
    assert out["peak_gib"] == pytest.approx(3.0)
    lines = capsys.readouterr().out.splitlines()
    human = [m for m in (HUMAN_LINE.match(line) for line in lines) if m]
    assert len(human) == 2
    assert all(m.group(7) == "3.00 GiB" for m in human)
    assert _match(SUMMARY_LINE, lines[-1]).group(2) == "3.00 GiB"


def test_run_bench_human_peak_uses_the_allocated_headline_not_reserved(
        fake_framework, capsys):
    eng = FakeEngine()
    dev = SimpleNamespace(type="cuda", index=0, __str__=lambda s: "cuda:0")
    api = FakeCudaApi(allocated=1 * GIB, reserved=2 * GIB)
    out = bench.run_bench(eng, ntok=1, warmup=0, runs=1,
                          measurement=bm.CudaMeasurement(dev, api))
    assert out["peak_gib"] == pytest.approx(1.0)
    run = out["report_runs"][0]
    assert run["memory"]["cuda_peak_allocated"]["bytes"] == GIB
    assert run["memory"]["cuda_peak_reserved"]["bytes"] == 2 * GIB
    lines = capsys.readouterr().out.splitlines()
    assert _match(HUMAN_LINE, lines[0]).group(7) == "1.00 GiB"
    assert api.calls.count("synchronize") == 4
    assert api.calls.count("reset_peak_memory_stats") == 1


def test_run_bench_prints_n_a_when_the_backend_peak_is_unavailable(
        fake_framework, capsys):
    eng = FakeEngine()
    null = bm.NullMeasurement("cuda", "cpu", "torch device is cpu, not cuda")
    out = bench.run_bench(eng, ntok=2, warmup=1, measurement=null)
    assert out["peak_gib"] is None
    assert all(r["peak_gib"] is None for r in out["runs"])
    assert out["report_runs"][0]["memory"]["backend_peak"]["bytes"] is None
    text = capsys.readouterr().out
    lines = text.splitlines()
    assert _match(HUMAN_LINE, lines[0]).group(7) == "n/a"
    assert _match(SUMMARY_LINE, lines[-1]).group(2) == "n/a"
    assert "0.00 GiB" not in text


def test_run_bench_rejects_empty_or_negative_windows(fake_framework):
    eng = FakeEngine()
    with pytest.raises(ValueError, match="ntok"):
        bench.run_bench(eng, ntok=0, warmup=0,
                        measurement=RecordingMeasurement(eng))
    with pytest.raises(ValueError, match="warmup"):
        bench.run_bench(eng, ntok=1, warmup=-1,
                        measurement=RecordingMeasurement(eng))
    assert eng.events == []


def test_run_bench_nonfinite_peak_fails_visibly(fake_framework):
    eng = FakeEngine()
    with pytest.raises(ValueError, match="bytes"):
        bench.run_bench(eng, ntok=1, warmup=0,
                        measurement=RecordingMeasurement(eng, peak=float("nan")))


def test_run_bench_records_the_workload_identity(fake_framework, monkeypatch):
    monkeypatch.setenv("BENCH_PROMPT", "hello there")
    monkeypatch.setenv("BENCH_SEED", "7")
    monkeypatch.setenv("BENCH_TEMP", "0.3")
    eng = FakeEngine(prompt_ids=(4, 5))
    out = bench.run_bench(eng, ntok=1, warmup=0,
                          measurement=RecordingMeasurement(eng))
    w = out["workload"]
    assert w["prompt_source"] == "BENCH_PROMPT"
    assert w["prompt_sha256"] == hashlib.sha256(b"hello there").hexdigest()
    assert w["prompt_token_ids_sha256"] == hashlib.sha256(b"4,5").hexdigest()
    assert w["prompt_chars"] == 11
    assert w["prompt_text"] == "hello there"
    assert w["prompt_tokens"] == 2
    assert w["seed"] == 7
    assert w["temperature"] == pytest.approx(0.3)
    assert w["top_k"] == 4 and w["top_p"] == 0.95
    assert w["repetition_penalty"] == 1.0


# ---- CLI --------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, tmp_path, fake_framework):
    """Patch engine construction and the RSS sampler; return a helper."""
    built: list = []
    samplers: list = []
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "bailing_hybrid", "architectures": ["X"]}')
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    _safetensors(model_dir / "model.safetensors", {"format": "mlx"}, 2)
    # a non-canonical adapter name: only engine.cfg.lora can find it
    _safetensors(model_dir / "my-lora.safetensors", {"edge0_adapter": "lora"})
    _safetensors(model_dir / "lora_edge0_8b.safetensors", {"decoy": "1"})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _safetensors(artifacts / "prerouter_edge0_8b.safetensors", {"heads": "16"})
    running_at_build: list = []

    class Sampler(bm.RssSampler):
        def __init__(self, interval):
            super().__init__(interval, read_memory=lambda: {"rss": 4096})
            self.stop_calls = 0
            samplers.append(self)

        def stop(self):
            self.stop_calls += 1
            super().stop()

    monkeypatch.setattr(bm, "RssSampler", Sampler)
    monkeypatch.delenv("LING_HIDDEN_CLIP", raising=False)

    def use(engine):
        def build(model, name=None):
            built.append((model, name))
            running_at_build.append(samplers[-1].is_running if samplers
                                    else None)
            # mirror engine/ling.py: the engine sets a default the user
            # did not request
            if "LING_HIDDEN_CLIP" not in os.environ:
                monkeypatch.setenv("LING_HIDDEN_CLIP", "1000")
            return engine
        monkeypatch.setattr(bench, "_build_engine", build)
        return engine

    def engine(**kw):
        kw.setdefault("model_dir", str(model_dir))
        kw.setdefault("lora", str(model_dir / "my-lora.safetensors"))
        kw.setdefault("prerouter_file",
                      str(artifacts / "prerouter_edge0_8b.safetensors"))
        return FakeEngine(**kw)

    return SimpleNamespace(model_dir=model_dir, artifacts=artifacts,
                           built=built, samplers=samplers, use=use,
                           engine=engine, tmp=tmp_path,
                           running_at_build=running_at_build)


def _tmp_entries(tmp):
    return sorted(p.name for p in tmp.iterdir())


def test_cli_writes_a_complete_report(cli, capsys):
    eng = cli.use(cli.engine())
    out = cli.tmp / "run" / "bench.json"
    rc = bench.main([str(cli.model_dir), "--ntok", "3", "--warmup", "1",
                     "--json-output", str(out)])
    assert rc == 0
    assert cli.built == [(str(cli.model_dir), None)]
    assert eng.closed == 1
    assert cli.running_at_build == [True]          # sampling covers the load
    assert cli.samplers[0].stop_calls >= 1 and not cli.samplers[0].is_running
    doc = json.loads(out.read_text(encoding="utf-8"))
    br.validate_report(doc)
    assert doc["schema_version"] == br.SCHEMA_VERSION
    assert doc["protocol"]["name"] == "fixed-length-sample-and-step"
    assert len(doc["runs"]) == 2
    ident = doc["identity"]
    assert ident["run_count"] == 2
    assert ident["requested"]["ntok"] == 3
    assert ident["requested"]["warmup"] == 1
    assert ident["command"]["argv"][0] == str(cli.model_dir)
    assert ident["git"]["commit"] is None or len(ident["git"]["commit"]) == 40
    model = doc["model"]
    assert model["checkpoint_path"] == str(cli.model_dir.resolve())
    assert model["tier"] == "fake-tier"
    assert model["model_type"] == "bailing_hybrid"
    assert model["config_sha256"] == hashlib.sha256(
        (cli.model_dir / "config.json").read_bytes()).hexdigest()
    assert [f["path"] for f in model["manifest"]["files"]] == [
        "chat_template.jinja", "config.json", "lora_edge0_8b.safetensors",
        "model.safetensors", "my-lora.safetensors", "tokenizer.json"]
    tok = {f["name"]: f for f in model["tokenizer_files"]}
    assert set(tok) == {"tokenizer.json", "chat_template.jinja"}
    assert tok["tokenizer.json"]["sha256"] == hashlib.sha256(
        (cli.model_dir / "tokenizer.json").read_bytes()).hexdigest()
    lora = model["adapters"]["lora"]
    assert lora["file"] == "my-lora.safetensors"      # from engine.cfg.lora
    assert lora["inside_checkpoint_dir"] is True
    assert lora["sha256"] == hashlib.sha256(
        (cli.model_dir / "my-lora.safetensors").read_bytes()).hexdigest()
    assert lora["safetensors_header"]["metadata"] == {"edge0_adapter": "lora"}
    prerouter = model["adapters"]["prerouter"]
    assert prerouter["inside_checkpoint_dir"] is False
    assert prerouter["safetensors_header"]["metadata"] == {"heads": "16"}
    assert model["adapters"]["prerouter_settings"]["start_layer"] == 7
    assert model["adapters"]["lora_settings"] == {
        "r": 16, "alpha": 32.0, "unavailable_reasons": {}}
    rt = doc["runtime"]
    assert rt["backend"] == "fake"
    assert rt["measurement"]["kind"] == "null"
    assert rt["execution_evidence"]["array_type"] == "builtins.list"
    w = doc["workload"]
    assert w["prompt_tokens"] == 3
    assert w["requested_timed_tokens"] == 3
    assert w["model_env"]["ling_hidden_clip"] == 1000.0
    assert w["model_env"]["prerouter_feature_topk"] == "teacher"
    assert w["model_env"]["prerouter_intra"] is False
    caches = doc["caches"]
    assert caches["layer_options"]["cache_slots"] == 64
    assert caches["shared_cache_slots_resolved"] == 64
    assert caches["prefetch_cap_resolved"] == 48
    assert caches["prewarm"] is False
    assert caches["streaming_stats_after_runs"] == {"1": {"loads": 3, "hits": 5}}
    mem = doc["memory"]
    assert mem["process_sampled_peak_rss"]["bytes"] == 4096
    assert mem["process_sampled_peak_rss"]["interval_seconds"] == 0.25
    assert mem["process_sampled_peak_rss_anon"]["bytes"] is None
    assert doc["probe"]["supplied"] is False
    text = capsys.readouterr().out
    assert sum(1 for line in text.splitlines() if HUMAN_LINE.match(line)) == 2
    assert "[bench] tier=fake-tier ntok=3 warmup=1" in text


def test_cli_env_knobs_are_snapshotted_before_the_engine_runs(cli, monkeypatch):
    # LING_HIDDEN_CLIP is unset by the user; the (fake) engine sets 1000 at
    # build time.  The requested snapshot must not show the engine's value,
    # the resolved model_env must.
    cli.use(cli.engine())
    out = cli.tmp / "a.json"
    assert bench.main([str(cli.model_dir), "--ntok", "1", "--warmup", "0",
                       "--json-output", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "LING_HIDDEN_CLIP" not in doc["identity"]["command"]["env_knobs"]
    assert doc["workload"]["model_env"]["ling_hidden_clip"] == 1000.0
    monkeypatch.setenv("LING_HIDDEN_CLIP", "0")
    monkeypatch.setenv("PREROUTER_FEATURE_TOPK", "executed")
    monkeypatch.setenv("BENCH_SEED", "3")
    out = cli.tmp / "b.json"
    assert bench.main([str(cli.model_dir), "--ntok", "1", "--warmup", "0",
                       "--json-output", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    knobs = doc["identity"]["command"]["env_knobs"]
    assert knobs["LING_HIDDEN_CLIP"] == "0"
    assert knobs["BENCH_SEED"] == "3"
    assert doc["workload"]["model_env"]["ling_hidden_clip"] == 0.0
    assert doc["workload"]["model_env"]["prerouter_feature_topk"] == "executed"
    assert doc["workload"]["seed"] == 3


def test_cli_collects_identity_outside_the_timed_runs(cli, monkeypatch):
    eng = cli.use(cli.engine())
    for name in ("checkpoint_manifest", "adapter_identity", "git_identity",
                 "power_mode", "process_peak_rss", "runtime_info"):
        real = getattr(bm, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            eng.events.append(("collect", _name))
            return _real(*args, **kwargs)
        monkeypatch.setattr(bm, name, spy)
    out = cli.tmp / "bench.json"
    assert bench.main([str(cli.model_dir), "--ntok", "1", "--warmup", "0",
                       "--json-output", str(out)]) == 0
    kinds = [e[0] if isinstance(e, tuple) else e for e in eng.events]
    # every collector runs after the engine's final timed step of the last
    # run (and before close): never inside a run
    last_step = max(i for i, k in enumerate(kinds) if k == "step")
    close = kinds.index("close")
    collects = [i for i, k in enumerate(kinds) if k == "collect"]
    assert collects and last_step < min(collects) and max(collects) < close
    assert {e[1] for e in eng.events if isinstance(e, tuple)
            and e[0] == "collect"} >= {"checkpoint_manifest", "git_identity",
                                       "power_mode", "process_peak_rss"}


def test_cli_without_json_output_keeps_the_old_behavior(cli, capsys):
    eng = cli.use(cli.engine())
    assert bench.main([str(cli.model_dir), "--ntok", "2", "--warmup", "0"]) == 0
    assert eng.closed == 1
    assert _tmp_entries(cli.tmp) == ["artifacts", "ckpt"]
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[bench] tier=fake-tier ntok=2 warmup=0"
    assert HUMAN_LINE.match(lines[1]) and HUMAN_LINE.match(lines[2])
    assert SUMMARY_LINE.match(lines[3])


def test_cli_resolves_tier_names_through_the_environment(cli, monkeypatch):
    cli.use(cli.engine())
    monkeypatch.setenv("EDGE0_8B_MODEL", str(cli.model_dir))
    assert bench.main(["edge0-8b", "--ntok", "1", "--warmup", "0"]) == 0
    assert cli.built == [(str(cli.model_dir), None)]
    monkeypatch.delenv("EDGE0_8B_MODEL")
    with pytest.raises(SystemExit):
        bench.main(["edge0-8b", "--ntok", "1"])
    monkeypatch.setenv("EDGE0_FOO_MODEL", str(cli.model_dir))
    with pytest.raises(SystemExit):           # unknown tier, like the CLI
        bench.main(["edge0-foo", "--ntok", "1"])
    assert len(cli.built) == 1


@pytest.mark.parametrize("argv,env", [
    (["--ntok", "0"], {}), (["--ntok", "-3"], {}), (["--warmup", "-1"], {}),
    (["--rss-sample-interval", "-1"], {}),
    (["--rss-sample-interval", "inf"], {}),
    (["--rss-sample-interval", "nan"], {}),
    (["--json-output", ""], {}), (["--probe-json", " "], {}),
    ([], {"BENCH_SEED": "abc"}), ([], {"BENCH_TEMP": "hot"}),
    ([], {"BENCH_TEMP": "nan"}), ([], {"BENCH_TEMP": "inf"}),
    ([], {"BENCH_NTOK": "0"}), ([], {"BENCH_NTOK": "many"}),
])
def test_cli_rejects_invalid_requests_before_loading(cli, monkeypatch, argv,
                                                     env, capsys):
    cli.use(cli.engine())
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as exc:
        bench.main([str(cli.model_dir), *argv])
    assert exc.value.code == 2
    assert cli.built == []
    err = capsys.readouterr().err
    assert (argv[0] if argv else next(iter(env))) in err


def test_cli_rejects_an_existing_output_before_loading(cli, capsys):
    cli.use(cli.engine())
    out = cli.tmp / "bench.json"
    out.write_text("stale")
    with pytest.raises(SystemExit) as exc:
        bench.main([str(cli.model_dir), "--ntok", "1", "--json-output",
                    str(out)])
    assert exc.value.code == 2
    assert cli.built == []
    assert out.read_text() == "stale"
    assert "already exists" in capsys.readouterr().err


def test_cli_rejects_an_unwritable_output_location_before_loading(cli):
    cli.use(cli.engine())
    blocker = cli.tmp / "file"
    blocker.write_text("x")
    with pytest.raises(SystemExit) as exc:
        bench.main([str(cli.model_dir), "--ntok", "1", "--json-output",
                    str(blocker / "bench.json")])
    assert exc.value.code == 2
    assert cli.built == []


def test_cli_links_the_supplied_probe_by_content_hash(cli):
    cli.use(cli.engine())
    probe = cli.tmp / "probe.json"
    # non-canonical bytes (key order, spacing, trailing newline): the hash
    # must be over the file as read, not over re-serialized JSON
    probe.write_bytes(b'{\n "ready": false ,\n "collected_at_utc":'
                      b' "2026-09-14T09:00:00+00:00",\n'
                      b' "schema_version" : 1\n}\n')
    assert json.dumps(json.loads(probe.read_bytes())).encode() != probe.read_bytes()
    out = cli.tmp / "bench.json"
    assert bench.main([str(cli.model_dir), "--ntok", "1", "--warmup", "0",
                       "--probe-json", str(probe),
                       "--json-output", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["probe"]["supplied"] is True
    assert doc["probe"]["schema_version"] == 1
    assert doc["probe"]["ready"] is False
    assert doc["probe"]["sha256"] == hashlib.sha256(
        probe.read_bytes()).hexdigest()
    assert doc["probe"]["path"] == str(probe)


@pytest.mark.parametrize("content", [b"{oops", b"[]", b'{"ready": true}'])
def test_cli_rejects_a_malformed_probe_before_loading(cli, content, capsys):
    cli.use(cli.engine())
    probe = cli.tmp / "probe.json"
    probe.write_bytes(content)
    with pytest.raises(SystemExit) as exc:
        bench.main([str(cli.model_dir), "--ntok", "1",
                    "--probe-json", str(probe)])
    assert exc.value.code == 2
    assert cli.built == []
    assert "probe" in capsys.readouterr().err


def test_cli_rejects_a_missing_probe_before_loading(cli):
    cli.use(cli.engine())
    with pytest.raises(SystemExit) as exc:
        bench.main([str(cli.model_dir), "--ntok", "1",
                    "--probe-json", str(cli.tmp / "absent.json")])
    assert exc.value.code == 2
    assert cli.built == []


def test_cli_generation_failure_leaves_no_report_and_cleans_up(cli, capsys):
    eng = cli.use(cli.engine(fail_at_step=2))
    out = cli.tmp / "bench.json"
    rc = bench.main([str(cli.model_dir), "--ntok", "3", "--warmup", "0",
                     "--json-output", str(out)])
    assert rc == 1
    assert not out.exists()
    assert _tmp_entries(cli.tmp) == ["artifacts", "ckpt"]   # no temp files
    assert eng.closed == 1
    assert cli.samplers[0].stop_calls >= 1 and not cli.samplers[0].is_running
    err = capsys.readouterr().err
    assert "FAILED" in err and "benchmark" in err
    assert "synthetic decode failure" in err


def test_cli_write_failure_is_nonzero_and_still_closes_the_engine(cli, capsys):
    out = cli.tmp / "bench.json"

    def steal_path(step):
        if step == 1:
            out.write_text("someone else")
    eng = cli.use(cli.engine(on_step=steal_path))
    rc = bench.main([str(cli.model_dir), "--ntok", "2", "--warmup", "0",
                     "--json-output", str(out)])
    assert rc == 1
    assert out.read_text() == "someone else"          # never overwritten
    assert eng.closed == 1
    assert _tmp_entries(cli.tmp) == ["artifacts", "bench.json", "ckpt"]
    err = capsys.readouterr().err
    assert "FAILED" in err and "write" in err


def test_cli_engine_build_failure_is_nonzero_and_stops_the_sampler(
        cli, capsys, monkeypatch):
    def build(model, name=None):
        raise RuntimeError("no weights here")
    monkeypatch.setattr(bench, "_build_engine", build)
    out = cli.tmp / "bench.json"
    rc = bench.main([str(cli.model_dir), "--ntok", "1", "--json-output",
                     str(out)])
    assert rc == 1
    assert not out.exists()
    assert cli.samplers[0].stop_calls >= 1 and not cli.samplers[0].is_running
    assert "engine build" in capsys.readouterr().err


def test_cli_sampler_stop_failure_still_closes_the_engine(cli, monkeypatch):
    eng = cli.use(cli.engine())

    def broken_stop(self):
        raise RuntimeError("sampler exploded")
    monkeypatch.setattr(bm.RssSampler, "stop", broken_stop)
    with pytest.raises(RuntimeError, match="sampler exploded"):
        bench.main([str(cli.model_dir), "--ntok", "1", "--warmup", "0"])
    assert eng.closed == 1


def test_cli_can_disable_the_rss_sampler(cli):
    cli.use(cli.engine())
    out = cli.tmp / "bench.json"
    assert bench.main([str(cli.model_dir), "--ntok", "1", "--warmup", "0",
                       "--rss-sample-interval", "0",
                       "--json-output", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    m = doc["memory"]["process_sampled_peak_rss"]
    assert m["bytes"] is None
    assert "disabled" in m["unavailable_reason"]
    assert cli.running_at_build == [False]
