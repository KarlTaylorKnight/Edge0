"""Task 5: prediction-driven prefetch plumbing (opt-in, exact path).

Backend-dependent modules are involved (stager, engine), so the suite
skips when no backend is importable; the routing logic itself is tested
with fakes, no weights."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EDGE0_BACKEND",
                      "cuda" if os.environ.get("EDGE0_BACKEND", "") == ""
                      else os.environ["EDGE0_BACKEND"])
pytest.importorskip("edge0.backends",
                    reason="no backend importable on this host")

from edge0.prerouter.stager import LingPrerouterStager  # noqa: E402
from edge0.streaming.options import LayerOptions  # noqa: E402


class _FakeLayer:
    def __init__(self):
        self.staged = []
        self.prefetched = []

    def stage_experts(self, experts):
        self.staged.append(list(experts))

    def prefetch(self, experts):
        self.prefetched.append(list(experts))


def _stager(stream_layers, prefetch_layers):
    # __init__ only stores these; the fakes never touch the model/spec.
    s = LingPrerouterStager.__new__(LingPrerouterStager)
    from edge0.prerouter.stager import PrerouterStager
    PrerouterStager.__init__(
        s, model=None, spec=_FakeSpec(), pspec=None,
        state=_FakeState(), stream_layers=stream_layers, top_k=8,
        prefetch_layers=prefetch_layers)
    s.pg_cache = {}
    return s


class _FakeSpec:
    num_experts = 128


class _FakeState:
    owners = ()


def test_staged_consumer_stages_and_does_not_prefetch():
    staged, plain = _FakeLayer(), _FakeLayer()
    s = _stager({3: staged}, {3: plain, 5: plain})
    s._store(3, "logits", None, None, [1, 2, 3])
    assert staged.staged == [[1, 2, 3]]
    assert plain.prefetched == []
    assert s.pg_cache[3] == "logits"


def test_non_staged_consumer_prefetches():
    plain = _FakeLayer()
    s = _stager({}, {5: plain})
    s._store(5, "logits", None, None, [7, 8])
    assert plain.prefetched == [[7, 8]]
    assert plain.staged == []


def test_consumer_without_layers_only_caches_logits():
    s = _stager({}, {})
    s._store(9, "logits", None, None, [1])
    assert s.pg_cache[9] == "logits"


def test_prefetch_layers_default_is_empty():
    s = _stager({}, None)
    s._store(9, "logits", None, None, [1])  # must not raise


# ---------------------------------------------------------- env profile

def test_apply_env_profile_cache_slots(monkeypatch):
    from edge0.engine.ling import _apply_env_profile
    cfg = _cfg()
    monkeypatch.setenv("EDGE0_CACHE_SLOTS", "512")
    out = _apply_env_profile(cfg, LayerOptions.prod_k8(), cfg.moe_spec)
    assert out.cache_slots == 512


@pytest.mark.parametrize("bad", ["0", "-3", "many"])
def test_apply_env_profile_rejects_bad_slots(monkeypatch, bad):
    from edge0.engine.ling import _apply_env_profile
    cfg = _cfg()
    monkeypatch.setenv("EDGE0_CACHE_SLOTS", bad)
    with pytest.raises(ValueError):
        _apply_env_profile(cfg, LayerOptions.prod_k8(), cfg.moe_spec)


def test_apply_env_profile_predict_prefetch_sizes_buffer(monkeypatch):
    from edge0.engine.ling import _apply_env_profile
    cfg = _cfg()
    monkeypatch.setenv("EDGE0_PREDICT_PREFETCH", "1")
    out = _apply_env_profile(cfg, LayerOptions.prod_k8(), cfg.moe_spec)
    assert out.predict_prefetch
    assert out.max_inflight == 8                # per-layer = K
    assert out.prefetch_cap == 16 * 8           # one full predicted step


def test_apply_env_profile_defaults_unchanged():
    from edge0.engine.ling import _apply_env_profile
    for var in ("EDGE0_CACHE_SLOTS", "EDGE0_PREDICT_PREFETCH"):
        assert os.environ.get(var, "") in ("", "0"), (
            f"{var} leaked into the test environment")
    cfg = _cfg()
    base = LayerOptions.prod_k8()
    assert _apply_env_profile(cfg, base, cfg.moe_spec) == base


def _cfg():
    from edge0.models.edge0_8b import Ling8BConfig
    return Ling8BConfig._defaults("/nonexistent")
