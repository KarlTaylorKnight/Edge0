"""``examples/bench.py`` on a real backend namespace (still no weights).

``tests/test_bench_cli.py`` covers the benchmark with fakes only; this
module repeats the core cases through the actual ``core`` ops and the
actual sampler of whichever backend is importable, so a real-array
regression (dtype, device, argmax semantics) is caught.  On a host
without MLX and with torch installed the torch backend is selected on the
CPU unless the environment already chose otherwise; with neither, the
module is skipped.  A CPU/MPS result here is an instrumentation check,
not CUDA or Orin evidence.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

if "EDGE0_BACKEND" not in os.environ:
    try:
        import mlx.core  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        os.environ["EDGE0_BACKEND"] = "cuda"
        os.environ.setdefault("EDGE0_TORCH_DEVICE", "cpu")

pytest.importorskip(
    "edge0.backends",
    reason="needs an importable edge0 backend: MLX, or torch (EDGE0_BACKEND=cuda)")

from edge0.backends import backend, core  # noqa: E402
from edge0.config import GenerationConfig  # noqa: E402
from edge0.streaming.options import LayerOptions  # noqa: E402
from examples import bench  # noqa: E402
from examples import benchmark_report as br  # noqa: E402
from tests.test_bench_cli import (  # noqa: E402
    HUMAN_LINE,
    FakeTok,
    RecordingMeasurement,
    _kinds,
    _match,
)

VOCAB = 8


class ArrayFakeEngine:
    """Fake engine whose logits are real backend arrays."""

    name = "fake-tier"

    def __init__(self, prompt_ids=(1, 2, 3)):
        self._tok = FakeTok(prompt_ids)
        self.cfg = SimpleNamespace(
            gen=GenerationConfig(temperature=0.7, top_p=0.95, top_k=4,
                                 repetition_penalty=1.0),
            options=LayerOptions.prod_k8(), model_dir="fake-dir",
            lora="", lora_r=16, lora_alpha=32.0, prerouter=None,
            prerouter_top_k=8)
        self.dir = "fake-dir"
        self.prefill_chunk = 2048
        self.think = False
        self.events: list = []
        self.closed = 0
        self.pos = 0
        self._logits = None
        self._all_stream_layers = {}

    def _make_logits(self):
        row = [0.0] * VOCAB
        row[self.pos % VOCAB] = 10.0
        return core.array(row, dtype=core.float32)

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
        self.events.append(("step", int(tid)))
        self.pos += 1
        return self._make_logits()

    def stats(self):
        return {}

    def close(self):
        self.closed += 1


def test_tier_env_matches_the_cli():
    from edge0.cli import TIER_ENV
    assert bench._TIER_ENV == TIER_ENV


def test_real_core_warmup_is_greedy_and_timed_steps_sample(monkeypatch):
    # At T=0 the real sampler is greedy too, so the token sequence alone
    # cannot tell a sampled warmup from a greedy one: count the calls to
    # the REAL ops (still executed) through the framework seam.
    from edge0.sampling import sample as real_sample
    monkeypatch.setenv("BENCH_TEMP", "0")
    monkeypatch.delenv("BENCH_SEED", raising=False)
    calls = {"argmax": 0, "sample": 0}
    real_argmax = core.argmax

    def counting_argmax(*args, **kwargs):
        calls["argmax"] += 1
        return real_argmax(*args, **kwargs)

    def counting_sample(*args, **kwargs):
        calls["sample"] += 1
        return real_sample(*args, **kwargs)
    monkeypatch.setattr(core, "argmax", counting_argmax)
    monkeypatch.setattr(bench, "_framework",
                        lambda: (backend, core, counting_sample))
    eng = ArrayFakeEngine(prompt_ids=(1, 2, 3))
    out = bench.run_bench(eng, ntok=5, warmup=2,
                          measurement=RecordingMeasurement(eng))
    steps = [e[1] for e in eng.events if isinstance(e, tuple) and e[0] == "step"]
    assert steps[:7] == [3, 4, 5, 6, 7, 0, 1]
    assert calls == {"argmax": 2 * 2, "sample": 5 * 2}   # per run: warmup, ntok
    kinds = _kinds(eng.events)
    assert kinds.count("sync") == 8 and kinds.count("reset_peak") == 2
    run = out["report_runs"][0]
    assert run["total_generated_tokens"] == 7
    assert run["execution_evidence"]["array_type"]


def test_seed_is_applied_through_the_real_core(monkeypatch):
    seen: list = []
    monkeypatch.setattr(core.random, "seed", seen.append)
    monkeypatch.setenv("BENCH_SEED", "0")
    monkeypatch.setenv("BENCH_TEMP", "0")
    eng = ArrayFakeEngine()
    bench.run_bench(eng, ntok=1, warmup=0,
                    measurement=RecordingMeasurement(eng))
    assert seen == [0, 0]


def test_default_measurement_dispatches_on_the_active_backend():
    eng = ArrayFakeEngine()
    out = bench.run_bench(eng, ntok=1, warmup=0)
    desc = out["measurement"]
    assert desc["backend"] == backend.name
    if backend.name == "cuda":
        expected = "cuda" if core.DEVICE.type == "cuda" else "null"
        assert desc["kind"] == expected
        assert desc["resolved_device"] == str(core.DEVICE)
    else:
        assert desc["kind"] == "mlx"


def test_cli_report_states_the_actually_resolved_device(tmp_path, monkeypatch,
                                                        capsys):
    eng = ArrayFakeEngine()
    monkeypatch.setattr(bench, "_build_engine", lambda model, name=None: eng)
    monkeypatch.setenv("BENCH_TEMP", "0")
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    out = tmp_path / "bench.json"
    assert bench.main([str(model_dir), "--ntok", "2", "--warmup", "1",
                       "--rss-sample-interval", "0",
                       "--json-output", str(out)]) == 0
    assert eng.closed == 1
    doc = json.loads(out.read_text(encoding="utf-8"))
    br.validate_report(doc)
    rt = doc["runtime"]
    assert rt["backend"] == backend.name
    if backend.name == "cuda":
        assert rt["resolved_device"] == str(core.DEVICE)
        assert rt["measurement"]["kind"] == (
            "cuda" if core.DEVICE.type == "cuda" else "null")
        assert rt["torch_version"]
    else:
        assert rt["measurement"]["kind"] == "mlx"
        assert rt["mlx_version"]
    assert doc["caches"]["layer_options"]["cache_slots"] == 64
    lines = capsys.readouterr().out.splitlines()
    assert _match(HUMAN_LINE, lines[1])
