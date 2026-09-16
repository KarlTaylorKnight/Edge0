"""edge0-8b engine: Ling 3.0 hybrid (MLA + MoE, 8B tier).

Port of the deployment's ling ``engine.py`` hybrid-prerouter path:

* the vendored ``bailing_hybrid.py`` model already owns its prerouter
  heads and consumes ``prerouter_cache`` logits inside
  ``BailingSparseMoE``'s ``_select_from_logits`` (sigmoid + group-limited
  top-k, routed-scaling 2.5) — the engine only feeds the cache and
  submits the staged fills.
* step boundary: ONE stacked head batch -> ONE tolist -> per-consumer
  ``stage_experts`` + ``pg_cache[consumer]`` logits; the next forward's
  consuming MoE blocks re-select from those cached logits, so staged
  set == routing set (zero drops) by construction.
* prefill — E3b whole-layer load-drop (``prefill_full_layers``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from edge0.backends import core

from edge0.backends import io
from edge0.engine.base import Edge0Engine, require_backend
from edge0.engine.hooks import (
    make_history_prefetch,
    make_prefill_before_layer,
)
from edge0.prerouter.install import install_prerouter
from edge0.prerouter.stager import LingPrerouterStager
from edge0.streaming import budget as _budget
from edge0.streaming.cache import PrefetchBuffer, SharedExpertCache
from edge0.streaming.install import install_streaming_experts
from edge0.streaming.mmap import SafetensorsMmap


def _apply_env_profile(cfg, opts, spec):
    """Task 5 opt-in profile knobs, applied to the REQUESTED options
    before the budget prices them (the budget may still lower them).

    ``EDGE0_CACHE_SLOTS=<int>``: request a different shared-LRU size.
    The tested default (64) is smaller than the per-step working set on
    this tier (23 MoE layers x K=8 = 184 bundles), so it never hits;
    512 measured a 65% hit rate on the Orin baseline (evidence in the
    plan's Task 5 status).

    ``EDGE0_PREDICT_PREFETCH=1``: enable
    ``LayerOptions.predict_prefetch``; the prefetch buffer is raised to
    hold one full predicted step (owners x K) and the per-layer
    in-flight bound defaults to K, both priced by the budget.
    """
    from dataclasses import replace as _dc_replace
    env_slots = os.environ.get("EDGE0_CACHE_SLOTS", "").strip()
    if env_slots:
        try:
            slots = int(env_slots)
        except ValueError:
            raise ValueError(
                f"EDGE0_CACHE_SLOTS must be an integer, got "
                f"{env_slots!r}") from None
        if slots <= 0:
            raise ValueError(
                f"EDGE0_CACHE_SLOTS must be positive, got {slots} "
                "(0 would disable eviction, not caching)")
        opts = _dc_replace(opts, cache_slots=slots)
    if os.environ.get("EDGE0_PREDICT_PREFETCH", "") == "1":
        opts = _dc_replace(opts, predict_prefetch=True)
    if opts.predict_prefetch:
        owners_n = len(getattr(cfg.prerouter, "owners", ()) or ())
        per_layer = opts.top_k or spec.top_k
        opts = _dc_replace(
            opts,
            prefetch_cap=max(opts.prefetch_cap, owners_n * per_layer),
            max_inflight=opts.max_inflight or per_layer)
    return opts


def _torch_quant_modules():
    """``(nn, quant)`` of the torch backend, ``(None, None)`` on MLX."""
    from edge0.backends import backend
    if backend.name != "cuda":
        return None, None
    from edge0.backends.cuda import nn as cnn
    from edge0.backends.cuda import quant as cq
    return cnn, cq


def _resolve_memory_budget(cfg, opts, shards):
    """Resolve the Task 4 memory budget when ``EDGE0_MEMORY_BUDGET`` asks
    for one; return ``None`` when it is off (the default — behavior is
    then exactly the pre-budget code path).

    ``EDGE0_MEMORY_BUDGET`` accepts ``auto`` (observe MemAvailable and
    any tighter address-space rlimit at this point, right before cache
    construction) or an explicit byte count.  The declared context comes
    from ``EDGE0_BUDGET_CONTEXT`` (tokens, default 1024): contexts
    beyond the declaration are a different workload and must be
    re-declared, not discovered as an OOM.
    """
    req = os.environ.get("EDGE0_MEMORY_BUDGET", "").strip()
    if not req:
        return None
    if req == "auto":
        import psutil
        available = int(psutil.virtual_memory().available)
        limit = None
        try:
            import resource
            soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft not in (-1, resource.RLIM_INFINITY):
                limit = int(soft)
        except (ImportError, OSError, ValueError):
            limit = None
        obs = _budget.Observation(available_bytes=available,
                                  process_limit_bytes=limit)
    else:
        try:
            obs = _budget.Observation(available_bytes=int(req))
        except ValueError:
            raise _budget.BudgetError(
                f"EDGE0_MEMORY_BUDGET must be 'auto' or a byte count, "
                f"got {req!r}") from None
    spec = cfg.moe_spec
    # Layer 0 is dense on this tier; layer 1 carries the expert tensors.
    entries = {}
    for shard in shards:
        entries.update(shard.entries)
    bundle = _budget.bundle_bytes_from_entries(
        entries, spec.key_template.format(layer=1), spec.num_experts)
    if not cfg.kv_bytes_per_token:
        raise _budget.BudgetError(
            f"{cfg.name}: kv_bytes_per_token is not measured for this "
            "tier; a memory budget cannot price the declared context")
    context = int(os.environ.get("EDGE0_BUDGET_CONTEXT", "1024"))
    top_k = opts.top_k or spec.top_k
    # In-flight transient: per-layer bound x layers that can be
    # producing at once.  With predict_prefetch every owner layer may
    # hold a step's predictions in flight; otherwise only the thread
    # pool's builds are outstanding.
    per_layer_inflight = opts.max_inflight or max(1, opts.prefetch_threads)
    if opts.predict_prefetch:
        producing_layers = len(getattr(cfg.prerouter, "owners", ()) or ())
        inflight_total = max(1, per_layer_inflight * producing_layers)
    else:
        inflight_total = per_layer_inflight
    # The dense dequantized-weight cache: the full cache (measured 4.1 GB
    # in float32 on the GB10; the loader now prices the actual bf16 total
    # of this tier's QuantizedLinear modules) is only ever vetoed by the
    # budget, never enabled; the capped cache (Task 6) is shrunk to the
    # headroom.  The batched expert gather's per-call transient cap is
    # deducted once when that path is on.
    weight_cache_requested = (
        os.environ.get("EDGE0_TORCH_WEIGHT_CACHE", "") == "1")
    cnn, cq = _torch_quant_modules()
    candidate_total = cnn.WEIGHT_CACHE.candidate_bytes_total() if cnn else 0
    request = _budget.dense_cache_request(
        full_requested=weight_cache_requested,
        cap_bytes=cnn.WEIGHT_CACHE.cap_bytes if cnn else None,
        candidate_total=candidate_total,
        max_fill_transient_bytes=(
            cnn.WEIGHT_CACHE.max_fill_transient_bytes(admitted_only=False)
            if cnn else 0))
    kernel_transient = int(cq.BATCHED_MAX_BYTES) if (cq and cq.BATCHED) else 0
    resolved = _budget.resolve_budget(
        obs,
        _budget.ExpertFootprint(
            bundle_bytes=bundle, num_experts=spec.num_experts,
            num_moe_layers=23),
        _budget.WorkloadDecl(max_context_tokens=context,
                             kv_bytes_per_token=cfg.kv_bytes_per_token),
        requested_cache_slots=opts.cache_slots,
        requested_prefetch_cap=opts.prefetch_cap,
        full_layer_prefill=opts.full_layer_prefill,
        inflight_builds=inflight_total,
        min_cache_slots=2 * top_k,
        weight_cache_bytes=(request["priced_bytes"]
                            if request["mode"] != "off" else None),
        reserves=_budget.Reserves(kernel_transient_bytes=kernel_transient),
        )
    if request["mode"] == "full" and not resolved.weight_cache_permitted:
        raise _budget.BudgetError(
            "EDGE0_TORCH_WEIGHT_CACHE=1 rejected: the dequantized dense "
            f"weights ({request['resident_bytes']:,} bytes resident plus a "
            f"{request['fill_transient_bytes']:,} byte build transient) do "
            f"not fit the resolved headroom of {resolved.headroom_bytes:,} "
            f"bytes "
            f"(budget: {resolved.as_dict()})")
    if request["mode"] == "capped" and cnn is not None:
        from dataclasses import replace as _dc_replace
        # The policy admits each module with its own build transient, so
        # the effective cap is the headroom itself.
        effective = _budget.effective_capped_bytes(
            cap_bytes=request["cap_bytes"],
            headroom_bytes=resolved.headroom_bytes)
        summary = cnn.WEIGHT_CACHE.finalize(effective)
        resolved = _dc_replace(resolved, notes=resolved.notes + (
            f"capped weight cache: requested {request['cap_bytes']:,} B, "
            f"effective {effective:,} B, admitted "
            f"{summary['admitted_bytes']:,} B in "
            f"{len(summary['admitted'])} modules",))
    return resolved


def _get_model_classes(config):
    """load_model class hook: serve the active backend's port of the bailing
    backbone (imported here so the module itself imports without MLX)."""
    from edge0.backends import backend
    if backend.name == "cuda":
        from edge0.backends.cuda._impl.bailing_hybrid import Model, ModelArgs
    else:
        from edge0.backends.mlx._impl.bailing_hybrid import Model, ModelArgs
    return Model, ModelArgs


def load_installed(model_dir: str, cfg):
    """Load the ling skeleton and install streaming twins, LoRA and the
    hybrid prerouter weights (shared by ``build_model``/``build_engine``).

    Returns ``(model, model_config, shards, installs)``.
    """
    require_backend("edge0-8b", ("mlx", "cuda"))
    model, model_config = io.load_model(
        model_dir, lazy=True, strict=False,
        model_config={"model_type": "bailing_hybrid",
                      "prerouter_enabled": cfg.prerouter is not None,
                      "prerouter_start_layer":
                          getattr(cfg.prerouter, "start_layer", 7),
                      "prerouter_hidden":
                          getattr(cfg.prerouter, "hidden", 512)},
        get_model_classes=_get_model_classes)
    shards = [SafetensorsMmap(os.path.join(
        os.fspath(model_dir), "model.safetensors"))]
    spec = cfg.moe_spec
    opts = cfg.options
    n_layers = model_config["num_hidden_layers"]
    opts = _apply_env_profile(cfg, opts, spec)
    resolved_budget = _resolve_memory_budget(cfg, opts, shards)
    shared_cache = prefetch_buffer = None
    if resolved_budget is not None:
        from dataclasses import replace as _dc_replace
        # cache/prefetch capacities come from the budget; the in-flight
        # bound stays PER-LAYER (the budget priced the total across
        # producing layers, see _resolve_memory_budget).
        opts = _dc_replace(
            opts, cache_slots=resolved_budget.cache_slots,
            prefetch_cap=resolved_budget.prefetch_cap,
            max_inflight=(opts.max_inflight
                          or max(1, opts.prefetch_threads)))
        shared_cache = SharedExpertCache(resolved_budget.cache_slots)
        prefetch_buffer = PrefetchBuffer(
            resolved_budget.prefetch_cap,
            max_cap=resolved_budget.max_prefetch_cap)
        print(f"[edge0-8b] memory budget: usable="
              f"{resolved_budget.usable_bytes:,}B cache_slots="
              f"{resolved_budget.cache_slots} prefetch_cap="
              f"{resolved_budget.prefetch_cap} headroom="
              f"{resolved_budget.headroom_bytes:,}B"
              + (f" notes={list(resolved_budget.notes)}"
                 if resolved_budget.notes else ""), flush=True)
    installed = install_streaming_experts(
        model, shards, spec, options=opts, num_layers=n_layers,
        shared_cache=shared_cache, prefetch_buffer=prefetch_buffer)
    all_stream = {li: t for li, t in enumerate(installed) if t is not None}
    stream_layers = {li: t for li, t in all_stream.items() if t._staged_mode}

    if cfg.lora:
        from edge0.adapters.lora import install_lora
        install_lora(model, cfg.lora, r=cfg.lora_r, alpha=cfg.lora_alpha)

    pg_state = pg_stager = None
    if cfg.prerouter and cfg.prerouter.weights_file:
        pg_state, heads = install_prerouter(
            model=model, spec=spec, pspec=cfg.prerouter, n_layers=n_layers)
        prefetch_layers = None
        if opts.predict_prefetch:
            prefetch_layers = {li: t for li, t in all_stream.items()
                               if not t._staged_mode}
        pg_stager = LingPrerouterStager(
            model=model, spec=spec, pspec=cfg.prerouter,
            state=pg_state, stream_layers=stream_layers,
            top_k=cfg.prerouter_top_k,
            prefetch_layers=prefetch_layers)
        print(f"[edge0-8b] prerouter installed: {len(heads)} heads, "
              f"start={cfg.prerouter.start_layer}, "
              f"K={cfg.prerouter_top_k}", flush=True)
    installs = dict(all_stream_layers=all_stream,
                    stream_layers=stream_layers,
                    pg_state=pg_state, pg_stager=pg_stager,
                    memory_budget=resolved_budget)
    return model, model_config, shards, installs


class Ling8BEngine(Edge0Engine):
    """Streaming Ling-3.0 engine (staged decode + hybrid prerouter)."""

    name = "edge0-8b"

    def __init__(self, model_dir: str, cfg, tokenizer=None,
                 think: bool = False):
        # Deployment parity: the ling server's THINK_MODE knob becomes the
        # engine's ``think`` flag (start_server.sh exports THINK_MODE=0).
        self.think = think
        self._chat_tpl = None
        # NAN_BANG_COLLAPSE_FIX.md parity: hidden clip default 1000 unless
        # the deployer overrides LING_HIDDEN_CLIP explicitly.  One fp16
        # overflow inside a layer otherwise poisons the whole net into
        # all-NaN logits -> argmax fallback token 0 ('!') death spiral.
        if "LING_HIDDEN_CLIP" not in os.environ:
            os.environ["LING_HIDDEN_CLIP"] = "1000"
        super().__init__(model_dir, cfg, tokenizer=tokenizer)

    def _build(self):
        cfg = self.cfg
        (self.model, self.model_config, self.shards,
         inst) = load_installed(cfg.model_dir, cfg)
        self._all_stream_layers = inst["all_stream_layers"]
        self._stream_layers = inst["stream_layers"]
        self._pg_state = inst["pg_state"]
        self._pg_stager = inst["pg_stager"]
        #: Resolved Task 4 memory budget (None when EDGE0_MEMORY_BUDGET
        #: is off); the bench report's caches group records it.
        self.memory_budget = inst["memory_budget"]
        opts = cfg.options
        if self._tok is None:
            try:
                self._tok = io.load_tokenizer(cfg.model_dir)
            except Exception:  # noqa: BLE001 — tokenizer optional for CLI
                pass

        self._prefill_before_layer = make_prefill_before_layer(
            self._all_stream_layers,
            full_n=getattr(opts, "prefill_full_layers", 0),
            hot_n=0, hot_window=1)
        self._history_prefetch = make_history_prefetch(
            self._all_stream_layers, enabled=cfg.prefetch_history)

        self.cache = self.model.make_cache()
        # start_server.sh LING_PREWARM=1 parity (opt-in): warm the OS page
        # cache over the whole checkpoint (madvise + sequential read) and
        # run a tiny dummy prefill+step so the first real request runs at
        # near-steady-state speed (kernel JIT + LRU + hot pins warm).
        if os.environ.get("EDGE0_PREWARM", os.environ.get(
                "LING_PREWARM", "0")) == "1":
            self._prewarm()

    # ---- startup warm-up --------------------------------------------------

    def _prewarm(self):
        """Page-cache + kernel warm-up.

        1. madvise(WILLNEED) + a full sequential read over every shard —
           removes per-expert page-fault cost from the first request.
        2. A dummy prefill + one decode step — materializes attention /
           router / shared weights, warms the LRU and hot pins, and
           pre-compiles the single-token decode kernels.
        KV state is reset afterwards; only page-cache warmth remains.
        """
        import time as _t
        t0 = _t.perf_counter()
        try:
            for shard in self.shards:
                shard.advise_willneed() if hasattr(
                    shard, "advise_willneed") else None
                shard.seq_read() if hasattr(shard, "seq_read") else None
        except Exception:  # noqa: BLE001 — advisory only
            pass
        dummy = [self._tok.bos_token_id or 0] if self._tok else [0]
        try:
            self.reset()
            self.prefill(dummy * 4)
            self.step(dummy[0])
            self.reset()
        except Exception:  # noqa: BLE001 — advisory only
            pass
        print(f"[edge0-8b] prewarm done in "
              f"{_t.perf_counter() - t0:.1f}s", flush=True)

    # ---- forward ----------------------------------------------------------

    def _forward(self, ids, intra_stage: bool = True) -> core.array:
        inputs = core.array(ids)[None, :]
        opts = self.cfg.options
        prefill_multi = len(ids) > 1 and self._prefill_active
        full_layer = bool(
            prefill_multi and self._all_stream_layers
            and opts.full_layer_prefill)
        h = self.model.model(
            inputs, cache=self.cache,
            before_layer_cb=self._prefill_before_layer if prefill_multi
            else None,
            after_layer_cb=None,
            async_eval_per_layer=bool(prefill_multi and full_layer),
            prerouter_cache=(self._pg_stager.pg_cache
                             if self._pg_stager is not None else None))
        logits = self.model.lm_head(h[0, -1])
        core.eval(logits)
        # Step boundary: ONE stacked head batch -> ONE tolist -> fills.
        if (self._pg_stager is not None and not self._prefill_active
                and len(ids) == 1):
            self._pg_stager.stage_all()
        return logits

    # ---- step hooks -------------------------------------------------------

    def _step_pre(self, token_id: int) -> None:
        if self._pg_stager is None and self._history_prefetch is not None:
            self._history_prefetch()

    def _step_post(self, token_id: int) -> None:
        for exp in self._stream_layers.values():
            exp.sync_actuals()
        if self._pg_stager is None:
            # History staging: each layer's next set is its previous
            # actuals (adjacent-token expert locality).
            for exp in self._stream_layers.values():
                if exp.last_used:
                    exp.stage_experts(list(exp.last_used))
                    exp.swap_staged()

    # ---- prefill / reset --------------------------------------------------

    def _prefill_end(self) -> None:
        for exp in self._all_stream_layers.values():
            exp.clear_full_layer()
        if self._pg_stager is not None:
            # Deployment parity: prefill's tail stages the FIRST decode
            # token's expert sets (engine.py:919 calls stage_all right
            # after prefill).  Without this the first decode step has no
            # prerouter logits, falls back to the true gate for one step,
            # and the whole prediction chain shifts by one token
            # (observed: [23982, 2862, ...] vs correct [23982, 4264, ...]).
            self._pg_stager.stage_all()
        if self._stream_layers:
            for exp in self._stream_layers.values():
                exp.stage_from_prefill()

    def _reset_state(self) -> None:
        self.cache = self.model.make_cache()
        if self._pg_stager is not None:
            self._pg_stager.reset()
            self._pg_state.reset()

    def _lm_logits(self, h: core.array) -> core.array:
        return self.model.lm_head(h[0, -1])

    # ---- chat template (deployment parity) -------------------------------

    def _chat_template(self):
        """Render the deployment's ``chat_template.jinja`` (same file and
        renderer as the ling server's ``encode_chat``, engine.py)."""
        from jinja2 import BaseLoader, Environment, StrictUndefined

        if self._chat_tpl is None:
            src = open(
                os.path.join(self.dir, "chat_template.jinja"),
                encoding="utf-8",
            ).read()
            self._chat_tpl = Environment(
                loader=BaseLoader(), undefined=StrictUndefined,
                autoescape=False,
            ).from_string(src)
        return self._chat_tpl

    def encode_chat(self, messages, think=None) -> list:
        """Tokenize chat messages with the deployment chat template.

        ``think`` mirrors THINK_MODE: True renders "detailed thinking on"
        (the model answers with a reasoning preamble), False renders
        "detailed thinking off" (direct answer).  Defaults to the
        engine's ``think`` flag.
        """
        if think is None:
            think = self.think
        msgs = [
            SimpleNamespace(
                role=m.get("role"),
                content=m.get("content") or "",
                reasoning_content=m.get("reasoning_content") or "",
                tool_calls=m.get("tool_calls"),
            )
            for m in messages
        ]
        text = self._chat_template().render(
            messages=msgs,
            add_generation_prompt=True,
            enable_thinking=bool(think),
            tools=None,
        )
        # transformers encode would prepend/append special tokens by
        # default; the template text is already complete.
        return self._tok.encode(text, add_special_tokens=False)
