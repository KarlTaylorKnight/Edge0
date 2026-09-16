#!/usr/bin/env python3
"""On-device Edge0-8B correctness check (Task 3, Gate B).

Generates >= 32 greedy tokens with the torch backend on the TEST device
(default ``cuda``), then teacher-forces the SAME token sequence through
the torch backend on the REFERENCE device (default ``cpu``) and compares
the full per-step logits and token choices.

Why torch-CPU is the reference here: the plan's Gate B fixtures were to
be generated against MLX, which does not exist on a Jetson.  The torch
implementation itself was checked against real MLX layer by layer on a
GB10 (edge0-8b backbone: 5.3e-7 max per layer, 4.3e-7 on the logits,
same argmax — docs/nvidia.md), so torch-on-CPU is the closest identified
reference data available on this device, and the comparison isolates
exactly the variable Task 3 introduces: the CUDA device.  When MLX
fixtures land, point ``--reference-npz`` at them instead.

Tolerances (DEFINED IN ADVANCE, before any Orin result was examined —
do not loosen them to make a run pass):

* ``--logit-tol`` (default 1.0): max absolute per-step logit difference.
  The dense model math runs in bfloat16 (~8 mantissa bits); reduction
  order differs across devices, so logits at a scale of tens can move
  by a few tenths through 24 layers.  1.0 is the a-priori bf16 bound,
  NOT an observed number.
* ``--near-tie-margin`` (default 1.0): a token divergence at step i is
  acceptable only if the reference's own logit gap between its top
  token and the test's chosen token is within this margin (a genuine
  near-tie), AND the logit tolerance holds at that step.  Any other
  divergence fails.

Each phase runs in its own subprocess because the torch backend binds
``EDGE0_TORCH_DEVICE`` at import time.  Exit status: 0 PASS, 1 FAIL,
2 usage/environment error.  The JSON report records checkpoint /
adapter / git identity, prompt digest, per-step statistics and the
verdict; keep it in the untracked evidence set ($EDGE0_RUN_DIR).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROMPT = "9.11 和 9.8 哪个大？请仔细比较。"  # examples/bench.py edge0-8b

#: Knobs recorded per phase as RESOLVED (inherited from the shell or set
#: by --test-env/--reference-env): the comparison is only meaningful when
#: the two phases differ in exactly what the caller intended.
RECORDED_ENV = (
    "EDGE0_BACKEND", "EDGE0_TORCH_DEVICE", "EDGE0_QMM_BATCHED",
    "EDGE0_QMM_BATCHED_MAX_BYTES", "EDGE0_INT4PACK",
    "EDGE0_TORCH_WEIGHT_CACHE", "EDGE0_TORCH_WEIGHT_CACHE_BYTES",
    "EDGE0_MEMORY_BUDGET", "EDGE0_BUDGET_CONTEXT", "EDGE0_CACHE_SLOTS",
    "EDGE0_PREDICT_PREFETCH", "EDGE0_PREWARM", "LING_PREWARM",
    "LING_HIDDEN_CLIP", "PREROUTER_FEATURE_TOPK", "PREROUTER_INTRA",
)


def host_identity() -> dict:
    """Which machine produced this phase.  A report without it reads the
    same whether it came from the Orin or a workstation GPU, and this
    script's results gate an Orin claim."""
    import platform
    out = {
        "hostname_hash": hashlib.sha256(
            platform.node().encode("utf-8")).hexdigest()[:16],
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "gpu_name": None,
        "compute_capability": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "jetson_model": None,
        "unavailable_reasons": {},
    }
    try:
        import torch
        out["torch_version"] = str(torch.__version__)
        out["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out["gpu_name"] = props.name
            out["compute_capability"] = f"{props.major}.{props.minor}"
        else:
            out["unavailable_reasons"]["gpu_name"] = "no CUDA device"
            out["unavailable_reasons"]["compute_capability"] = "no CUDA device"
    except Exception as exc:  # noqa: BLE001 - identity is best-effort
        reason = f"{type(exc).__name__}: {exc}"
        for key in ("torch_version", "torch_cuda_version", "gpu_name",
                    "compute_capability"):
            out["unavailable_reasons"].setdefault(key, reason)
    try:
        out["jetson_model"] = Path(
            "/proc/device-tree/model").read_text(errors="replace").rstrip("\x00\n")
    except OSError:
        out["unavailable_reasons"]["jetson_model"] = (
            "/proc/device-tree/model unreadable: not a Jetson")
    if out["torch_cuda_version"] is None:
        out["unavailable_reasons"].setdefault(
            "torch_cuda_version", "torch build has no CUDA runtime")
    return out


# --------------------------------------------------------------- phases

def _encode_prompt(engine, prompt: str) -> list[int]:
    # Same chat templating as examples/bench.py (think=False).
    tok = engine._tok
    if hasattr(engine, "encode_chat"):
        return [int(t) for t in engine.encode_chat(
            [{"role": "user", "content": prompt}], think=False)]
    return [int(t) for t in tok(tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False))["input_ids"]]


def _logits_to_numpy(logits):
    import numpy as np
    import torch
    if isinstance(logits, torch.Tensor):
        return logits.detach().reshape(-1).to("cpu", torch.float32).numpy()
    return np.asarray(logits, dtype="float32").reshape(-1)


def run_phase(model_dir: str, prompt: str, ntok: int, out_npz: str,
              forced: list[int] | None) -> None:
    """One engine pass on the device the environment selected.

    ``forced is None``: greedy generation, logits captured at every step.
    ``forced``: teacher-forcing — capture logits, then feed forced[i]
    regardless of what this device would have chosen.
    """
    import numpy as np

    from edge0 import AutoEngine

    t0 = time.perf_counter()
    engine = AutoEngine.from_pretrained(model_dir)
    try:
        ids = _encode_prompt(engine, prompt)
        engine.reset()
        engine.prefill(ids)
        logits = engine.next_logits()
        first = _logits_to_numpy(logits)
        steps = [first]
        tokens: list[int] = []
        for i in range(ntok):
            if forced is None:
                tid = int(steps[-1].argmax())
            else:
                tid = int(forced[i])
            tokens.append(tid)
            if i < ntok - 1:
                logits = engine.step(tid)
                steps.append(_logits_to_numpy(logits))
        evidence = {
            "logits_class": type(logits).__name__,
            "device": str(getattr(logits, "device", "unknown")),
            "elapsed_s": time.perf_counter() - t0,
            **host_identity(),
            # every knob as this phase RESOLVED it, not just the CLI
            # overrides: a value inherited from the shell would otherwise
            # turn a same-device comparison into a self-comparison
            "resolved_env": {k: os.environ[k] for k in RECORDED_ENV
                             if k in os.environ},
        }
        np.savez_compressed(
            out_npz, logits=np.stack(steps), tokens=np.asarray(tokens),
            prompt_ids=np.asarray(ids))
        Path(out_npz + ".meta.json").write_text(json.dumps(evidence))
    finally:
        engine.close()


# --------------------------------------------------------------- compare

def compare(test_npz: str, ref_npz: str, logit_tol: float,
            near_tie_margin: float) -> dict:
    import numpy as np

    test = np.load(test_npz)
    ref = np.load(ref_npz)
    if not np.array_equal(test["prompt_ids"], ref["prompt_ids"]):
        return {"pass": False, "error": "prompt ids differ between phases"}
    if not np.array_equal(test["tokens"], ref["tokens"]):
        return {"pass": False,
                "error": "teacher phase did not replay the test tokens"}
    a, b = test["logits"], ref["logits"]
    if a.shape != b.shape:
        return {"pass": False,
                "error": f"logit shapes differ: {a.shape} vs {b.shape}"}
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"pass": False, "error": "non-finite logits"}

    steps = []
    divergences = []
    ok = True
    for i in range(a.shape[0]):
        diff = float(np.abs(a[i] - b[i]).max())
        t_test = int(test["tokens"][i])          # what the test device chose
        r_top = int(b[i].argmax())               # reference greedy choice
        entry = {"step": i, "max_abs_logit_diff": diff,
                 "test_token": t_test, "reference_token": r_top}
        if diff > logit_tol:
            entry["fail"] = f"logit diff {diff:.4g} > tol {logit_tol}"
            ok = False
        if t_test != r_top:
            gap = float(b[i][r_top] - b[i][t_test])
            entry["reference_gap"] = gap
            near_tie = 0.0 <= gap <= near_tie_margin and diff <= logit_tol
            entry["explained_near_tie"] = near_tie
            divergences.append(entry)
            if not near_tie:
                entry.setdefault(
                    "fail", f"token divergence with reference gap "
                            f"{gap:.4g} > margin {near_tie_margin}")
                ok = False
        steps.append(entry)
    diffs = [s["max_abs_logit_diff"] for s in steps]
    return {
        "pass": ok,
        "steps_compared": len(steps),
        "tokens_matched": len(steps) - len(divergences),
        "divergences": divergences,
        "max_abs_logit_diff": max(diffs),
        "mean_abs_logit_diff_of_max": sum(diffs) / len(diffs),
        "per_step": steps,
    }


# ----------------------------------------------------------- orchestrate

def parse_env_overrides(items: list[str] | None) -> dict[str, str]:
    """``KEY=VALUE`` pairs for one phase's environment (Task 6: run the
    reference path and an opt-in path on the SAME device, e.g.
    ``--test-env EDGE0_QMM_BATCHED=1``).  An empty value unsets the key."""
    out: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        out[key.strip()] = value
    return out


def _spawn(device: str, argv: list[str],
           overrides: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    env["EDGE0_BACKEND"] = "cuda"
    env["EDGE0_TORCH_DEVICE"] = device
    for key, value in (overrides or {}).items():
        if value == "":
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run([sys.executable, os.fspath(Path(__file__).resolve()),
                           "--phase-internal", *argv], env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"phase on device {device!r} failed "
                           f"(exit {proc.returncode})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model", help="model directory or tier name")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--test-device", default="cuda")
    ap.add_argument("--reference-device", default="cpu")
    ap.add_argument("--reference-npz", default=None,
                    help="use pre-generated reference logits instead of "
                         "running the reference device phase")
    ap.add_argument("--logit-tol", type=float, default=1.0)
    ap.add_argument("--near-tie-margin", type=float, default=1.0)
    ap.add_argument("--test-env", action="append", metavar="KEY=VALUE",
                    help="environment override for the TEST phase only "
                         "(repeatable; empty value unsets), e.g. "
                         "EDGE0_QMM_BATCHED=1 to check an opt-in path "
                         "against the reference path on the same device")
    ap.add_argument("--reference-env", action="append", metavar="KEY=VALUE",
                    help="environment override for the REFERENCE phase only")
    ap.add_argument("--output", required=True,
                    help="JSON report path (must not exist)")
    ap.add_argument("--phase-internal", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--phase-out", help=argparse.SUPPRESS)
    ap.add_argument("--phase-forced", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.phase_internal:
        forced = (json.loads(Path(args.phase_forced).read_text())
                  if args.phase_forced else None)
        run_phase(args.model, args.prompt, args.tokens, args.phase_out, forced)
        return 0

    if args.tokens < 32:
        print("[reference-check] Gate B requires at least 32 tokens",
              file=sys.stderr)
        return 2
    try:
        test_env = parse_env_overrides(args.test_env)
        reference_env = parse_env_overrides(args.reference_env)
    except ValueError as exc:
        print(f"[reference-check] {exc}", file=sys.stderr)
        return 2
    out = Path(args.output)
    if out.exists():
        print(f"[reference-check] refusing to overwrite {out}",
              file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)

    from examples import benchmark_measure as measure

    model_dir = args.model
    if not Path(model_dir).exists():
        env_name = f"EDGE0_{args.model.split('-')[-1].upper()}_MODEL"
        model_dir = os.environ.get(env_name, model_dir)
    if not Path(model_dir).exists():
        print(f"[reference-check] model not found: {args.model}",
              file=sys.stderr)
        return 2

    work = out.parent / (out.stem + ".phases")
    work.mkdir(exist_ok=True)
    test_npz = str(work / f"test-{args.test_device}.npz")
    ref_npz = args.reference_npz or str(
        work / f"reference-{args.reference_device}.npz")

    common = [model_dir, "--prompt", args.prompt,
              "--tokens", str(args.tokens), "--output", "unused"]
    print(f"[reference-check] phase 1/2: greedy {args.tokens} tokens on "
          f"{args.test_device}"
          + (f" with {test_env}" if test_env else ""))
    _spawn(args.test_device, common + ["--phase-out", test_npz], test_env)

    if not args.reference_npz:
        import numpy as np
        tokens = [int(t) for t in np.load(test_npz)["tokens"]]
        forced_file = work / "forced-tokens.json"
        forced_file.write_text(json.dumps(tokens))
        print(f"[reference-check] phase 2/2: teacher-forcing on "
              f"{args.reference_device}"
              + (f" with {reference_env}" if reference_env else ""))
        _spawn(args.reference_device,
               common + ["--phase-out", ref_npz,
                         "--phase-forced", str(forced_file)],
               reference_env)

    result = compare(test_npz, ref_npz, args.logit_tol, args.near_tie_margin)

    same_device = (not args.reference_npz
                   and args.test_device == args.reference_device)
    phase_env = {
        name: json.loads(Path(p + ".meta.json").read_text()).get(
            "resolved_env", {})
        for name, p in (("test", test_npz), ("reference", ref_npz))
        if Path(p + ".meta.json").exists()}
    env_differences = sorted(
        set(phase_env.get("test", {}).items())
        ^ set(phase_env.get("reference", {}).items()))
    if same_device and not env_differences:
        print("[reference-check] WARNING: same device and identical "
              "resolved knobs in both phases: this compares a "
              "configuration with itself and proves nothing",
              file=sys.stderr)

    report = {
        "schema": "edge0-reference-check/2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": host_identity(),
        "same_device_comparison": same_device,
        "resolved_env_differences": [list(kv) for kv in env_differences],
        "git": measure.git_identity(REPO_ROOT),
        "model_dir": os.path.abspath(model_dir),
        "checkpoint": measure.checkpoint_manifest(model_dir),
        "prompt_sha256": hashlib.sha256(
            args.prompt.encode("utf-8")).hexdigest(),
        "prompt_text": args.prompt,
        "tokens_requested": args.tokens,
        "test_device": args.test_device,
        "test_env": test_env,
        "reference_device": (args.reference_device if not args.reference_npz
                             else f"npz:{args.reference_npz}"),
        "reference_env": reference_env,
        "tolerances": {"logit_tol": args.logit_tol,
                       "near_tie_margin": args.near_tie_margin},
        "phase_evidence": {
            name: json.loads(Path(p + ".meta.json").read_text())
            for name, p in (("test", test_npz), ("reference", ref_npz))
            if Path(p + ".meta.json").exists()},
        "result": {k: v for k, v in result.items() if k != "per_step"},
        "per_step": result.get("per_step"),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, allow_nan=False))
    tmp.rename(out)

    verdict = "PASS" if result.get("pass") else "FAIL"
    # ASCII only: this line is printed on whatever console the target has,
    # and a Windows cp1252 stdout raises UnicodeEncodeError on "Δ" AFTER
    # the report is already written -- turning a PASS into a crash.
    print(f"[reference-check] {verdict}: "
          f"{result.get('tokens_matched')}/{result.get('steps_compared')} "
          f"tokens matched, max abs logit diff = "
          f"{result.get('max_abs_logit_diff', float('nan')):.4g} "
          f"(tol {args.logit_tol}); report: {out}")
    if result.get("error"):
        print(f"[reference-check] error: {result['error']}", file=sys.stderr)
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
