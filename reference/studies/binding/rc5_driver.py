#!/usr/bin/env python3
"""P3-C-T5d (RC5) — SLO-binding configuration existence-proof driver.

Card: docs/papers/p3/reopen_2026-06/cards/P3-C-T5d_slo_binding_existence_proof.md
Decision: P3-D018 (run). Authored P3-D017 (card), A3-pursue P3-D014/NK-017.

WHAT (card §3):
  Stage A — controller-free binding probe (GATE, run first): patch ONLY the live
    compute-worker CPU limit 500m->80m on w0/w1/w2 (source manifests.yaml NOT
    edited; restored in a finally block), then on w2 sweep static replica counts
    {2,4,6,8,10,12,16,20} at constant offered load (workload_gen E --constant-rps)
    and measure e2e p95 from the workload trace. responsive(load) =
    p95@lowest_replicas / p95@highest_replicas; bound iff >= 1.30 at >=1 load
    (the E-V1 DECISIVE rule, analyze.py:31,82-84). If flat at every load -> STOP,
    do NOT run Stage B, verdict = FAIL (binding failed -> axis still inert), §5.
  Stage B — main sub-study (only if Stage A passes): method in
    {confscale-aci, confscale-pid, hpa-qr-monitored} x pattern in {F, G} = 6 cells,
    R=3 reps = 18 cell-runs, 1800 s / 30 s control interval, 3-up wave scheduler on
    w0/w1/w2. Primary metric e2e_p95_ms recovered from each run's
    metrics.json["e2e"]["p95_ms"] (the post-E-V6 e2e block, run_matrix.py:263).

PURELY ADDITIVE: writes a NEW run dir under data/p3_runs/outputs/; does
NOT touch any locked codex-cut table (Table 1-6) or source JSON, and does NOT edit
the source manifests.yaml — the 80m worker limit is applied to the LIVE deployment
for the run window only and restored to 500m at teardown (card §5).

Self-contained, resumable (--start-wave), SELF-RESTORING (worker limit restored to
500m in a finally block even on crash). Verdict is computed here AND re-checked by
the run-session agent against the verbatim §2 criterion before it is declared.

This reuses ev7/T3's tested wave machinery (plan_waves / reset_cluster / preflight /
run_cell) verbatim, with build_cells() swapped for the RC5 F/G x 3-method x R=3
matrix and run_cell() extended to recover e2e_p95 alongside coverage_rate.
"""
from __future__ import annotations

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")


import argparse
import concurrent.futures
import csv
import json
import logging
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- wire src/stage3_scale onto the path ---
REPO = Path("<SOURCE_WORKSPACE>")
STAGE3 = REPO / "src" / "stage3_scale"
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))

from orchestrator.run_matrix import (  # noqa: E402
    execute_single_run, PortForward, _resolve_method_spec,
    PROMETHEUS_NS, PROMETHEUS_SVC, PROMETHEUS_PORT,
)
from orchestrator.methods import (  # noqa: E402
    set_thread_kube_context, clear_thread_kube_context, NAMESPACE,
)
from orchestrator.clusters import make_slots  # noqa: E402

COMPLEXITY = 50000          # matches the locked rig (card §4, HELD)
COMPUTE_NS = "infosys-benchmark"
COMPUTE_DEPLOY = "compute-worker"
COMPUTE_CONTAINER = "compute-worker"
PROM_NS = "monitoring"
PROM_DEPLOY = "prometheus"
DURATION = 1800             # Stage B per-cell (card §3)
REPS = 3                   # card §3 (R=3; verdict written against pooled n=3 std)

# The binding knob (card §4): live-patch only, restore after.
WORKER_LIMIT_BIND = "80m"
WORKER_LIMIT_LOCKED = "500m"
WORKER_CONTEXTS = ["kind-p3-experiments-w0", "kind-p3-experiments-w1",
                   "kind-p3-experiments-w2"]

# Stage A probe (card §3; E-V1 grid analyze.py / sweep_driver.py)
PROBE_CTX = "kind-p3-experiments-w2"
PROBE_TARGET = "http://localhost:31082"     # w2 frontend NodePort (clusters.py:39 + i=2)
PROBE_LOADS = [150, 100]                     # >=1 offered load (card §3); two for robustness
PROBE_REPLICAS = [2, 4, 6, 8, 10, 12, 16, 20]
PROBE_DURATION = 120
PROBE_WARMUP = 30
PROBE_SETTLE = 6
DECISIVE = 1.30                              # responsive-ratio bar (analyze.py:31)
PY = str(REPO / ".venv/bin/python")
WL = str(STAGE3 / "workload_gen.py")

logger = logging.getLogger("rc5")


# ── Worker-limit live patch / restore (additive; source manifest untouched) ──

def kubectl_ctx(ctx: str, args: list[str], timeout: int = 200) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", f"--context={ctx}", *args],
                          capture_output=True, text=True, timeout=timeout)


def get_worker_cpu_limit(ctx: str) -> str | None:
    r = kubectl_ctx(ctx, ["-n", COMPUTE_NS, "get", "deploy", COMPUTE_DEPLOY, "-o",
                          "jsonpath={.spec.template.spec.containers[0].resources.limits.cpu}"],
                    timeout=30)
    return r.stdout.strip() if r.returncode == 0 else None


def set_worker_cpu_limit(ctx: str, cpu: str) -> bool:
    r = kubectl_ctx(ctx, ["-n", COMPUTE_NS, "set", "resources",
                          f"deploy/{COMPUTE_DEPLOY}", f"--limits=cpu={cpu}"], timeout=60)
    if r.returncode != 0:
        logger.error("set worker limit %s on %s FAILED: %s", cpu, ctx, r.stderr.strip()[:200])
        return False
    s = kubectl_ctx(ctx, ["-n", COMPUTE_NS, "rollout", "status",
                          f"deploy/{COMPUTE_DEPLOY}", "--timeout=180s"], timeout=200)
    return s.returncode == 0


def patch_all_workers(cpu: str) -> dict:
    out = {"target": cpu, "per_ctx": {}, "ok": True}
    for ctx in WORKER_CONTEXTS:
        pre = get_worker_cpu_limit(ctx)
        ok = set_worker_cpu_limit(ctx, cpu)
        post = get_worker_cpu_limit(ctx)
        out["per_ctx"][ctx] = {"pre": pre, "post": post, "ok": ok}
        out["ok"] = out["ok"] and ok and (post == cpu)
        logger.info("worker limit %s: %s -> %s (ok=%s)", ctx, pre, post, ok)
    return out


def restore_all_workers() -> dict:
    """Restore 500m on every worker cluster; verify. Card §5 hard requirement."""
    out = {"target": WORKER_LIMIT_LOCKED, "per_ctx": {}, "ok": True}
    for ctx in WORKER_CONTEXTS:
        ok = set_worker_cpu_limit(ctx, WORKER_LIMIT_LOCKED)
        post = get_worker_cpu_limit(ctx)
        verified = (post == WORKER_LIMIT_LOCKED)
        out["per_ctx"][ctx] = {"post": post, "ok": ok, "verified": verified}
        out["ok"] = out["ok"] and ok and verified
        logger.info("RESTORE worker limit %s -> %s (verified=%s)", ctx, post, verified)
    return out


# ── Stage A: controller-free binding probe (reuses sweep_driver.py logic) ─────

def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[0] if len(s) == 1 else s[int(len(s) * p)]


def _parse_probe_trace(csv_path: Path, warmup: int):
    p95s, errs, oks = [], 0, 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["elapsed_s"])
            except (KeyError, ValueError):
                continue
            errs += int(float(row.get("errors", 0) or 0))
            oks += int(float(row.get("ok", 0) or 0))
            if t < warmup:
                continue
            try:
                p95s.append(float(row["p95_ms"]))
            except (KeyError, ValueError):
                continue
    return p95s, errs, oks


def _probe_scale(ctx: str, n: int):
    kubectl_ctx(ctx, ["-n", COMPUTE_NS, "scale", f"deploy/{COMPUTE_DEPLOY}",
                      f"--replicas={n}"], timeout=60)
    kubectl_ctx(ctx, ["-n", COMPUTE_NS, "rollout", "status", f"deploy/{COMPUTE_DEPLOY}",
                      "--timeout=150s"], timeout=170)


def _probe_cell(load: int, n: int, cell_dir: Path):
    cell_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  probe load=%d replicas=%d: scaling...", load, n)
    _probe_scale(PROBE_CTX, n)
    time.sleep(PROBE_SETTLE)
    proc = subprocess.Popen(
        [PY, WL, "E", "--constant-rps", str(load), "--complexity", str(COMPLEXITY),
         "--duration", str(PROBE_DURATION), "--target", PROBE_TARGET,
         "--output-dir", str(cell_dir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        _, stderr = proc.communicate(timeout=PROBE_DURATION + 120)
    except subprocess.TimeoutExpired:
        proc.kill()
        stderr = "killed"
    traces = list(cell_dir.glob("workload_E_*_timeseries.csv"))
    if not traces:
        logger.error("  probe load=%d n=%d: no trace (stderr tail: %s)", load, n,
                     (stderr or "")[-200:])
        return None
    trace = max(traces, key=lambda p: p.stat().st_mtime)
    p95s, errs, oks = _parse_probe_trace(trace, PROBE_WARMUP)
    if not p95s:
        logger.error("  probe load=%d n=%d: no steady ticks", load, n)
        return None
    e2e_p95 = _pct(p95s, 0.95)
    logger.info("  probe load=%d n=%d -> e2e_p95=%.1f ms (errs=%d oks=%d ticks=%d)",
                load, n, e2e_p95, errs, oks, len(p95s))
    return {"load_rps": load, "replicas": n, "e2e_p95_ms": round(e2e_p95, 1),
            "steady_ticks": len(p95s), "errors": errs, "ok": oks, "trace": trace.name}


def stage_a_probe(out_dir: Path) -> dict:
    """Run the binding probe; return {bound, ratios, rows}. analyze.py:82-84 rule."""
    probe_dir = out_dir / "stage_a_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    csv_path = probe_dir / "probe_results.csv"
    fields = ["load_rps", "replicas", "e2e_p95_ms", "steady_ticks", "errors", "ok", "trace"]
    fh = open(csv_path, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    fh.flush()
    for load in PROBE_LOADS:
        for n in PROBE_REPLICAS:
            try:
                row = _probe_cell(load, n, probe_dir / f"load{load}_n{n}")
            except Exception as e:  # noqa: BLE001
                logger.error("  probe EXC load=%d n=%d: %s", load, n, e)
                row = None
            if row:
                rows.append(row)
                w.writerow(row)
                fh.flush()
    fh.close()
    # responsive-ratio per load = p95@lowest_replicas / p95@highest_replicas
    ratios = {}
    for load in PROBE_LOADS:
        curve = sorted([(r["replicas"], r["e2e_p95_ms"]) for r in rows if r["load_rps"] == load])
        if len(curve) < 2:
            continue
        p_lo, p_hi = curve[0][1], curve[-1][1]
        ratio = (p_lo / p_hi) if p_hi else float("nan")
        ratios[load] = {"ratio": round(ratio, 3), "p95_lo": p_lo, "n_lo": curve[0][0],
                        "p95_hi": p_hi, "n_hi": curve[-1][0],
                        "responsive": ratio >= DECISIVE,
                        "floor_ms": min(p for _, p in curve)}
    bound = any(v["responsive"] for v in ratios.values())
    res = {"bound": bound, "bar": DECISIVE, "ratios": ratios, "n_rows": len(rows)}
    (probe_dir / "stage_a_verdict.json").write_text(json.dumps(res, indent=2))
    logger.info("STAGE A: bound=%s ratios=%s", bound,
                {l: v["ratio"] for l, v in ratios.items()})
    return res


# ── Stage B cells + wave machinery (copied verbatim from t3_driver.py) ────────

def build_cells() -> list[dict]:
    """6 cells: {confscale-aci, confscale-pid, hpa-qr-monitored} x {F, G}, R=3."""
    methods = [
        ("confscale-aci", {"base": "confscale-aci", "name": "confscale-aci"}),
        ("confscale-pid", {"base": "confscale-pid", "name": "confscale-pid"}),
        ("hpa-qr-monitored", {"base": "hpa-qr-monitored", "name": "hpa-qr-monitored"}),
    ]
    cells = []
    for mname, mspec in methods:
        for pat in ("F", "G"):
            cells.append({
                "label": f"{mname}/{pat}", "task": "T5d",
                "method_spec": mspec, "pattern": pat,
                "duration_s": DURATION, "n_reps": REPS,
            })
    return cells


def _unit(cell, replicate):
    return {"label": cell["label"], "task": cell["task"], "method_spec": cell["method_spec"],
            "pattern": cell["pattern"], "duration_s": cell["duration_s"], "replicate": replicate}


def _pack_group(units_by_label, n_workers, rng):
    labels = list(units_by_label.keys())
    if len(labels) == 1:
        u = units_by_label[labels[0]][:]
        return [u[i:i + n_workers] for i in range(0, len(u), n_workers)]
    counts = {lbl: len(units_by_label[lbl]) for lbl in labels}
    R = counts[labels[0]]
    assert all(c == R for c in counts.values()), f"unequal counts {counts}"
    ptr = {lbl: 0 for lbl in labels}
    waves = []
    for _ in range(R):
        order = labels[:]
        rng.shuffle(order)
        for i in range(0, len(order), n_workers):
            chunk = order[i:i + n_workers]
            wave = []
            for lbl in chunk:
                wave.append(units_by_label[lbl][ptr[lbl]])
                ptr[lbl] += 1
            waves.append(wave)
    return waves


def _interleave(group_waves):
    tagged = []
    for gi, gkey in enumerate(sorted(group_waves)):
        waves = group_waves[gkey]
        n = len(waves)
        for j, w in enumerate(waves):
            tagged.append(((j + 0.5) / n, gi, j, w))
    tagged.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in tagged]


def plan_waves(cells, n_workers, seed):
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for c in cells:
        for rep in range(1, c["n_reps"] + 1):
            groups[c["task"]][c["label"]].append(_unit(c, rep))
    group_waves = {}
    for gkey in sorted(groups, reverse=True):
        group_waves[gkey] = _pack_group(groups[gkey], n_workers, rng)
    ordered = _interleave(group_waves)
    plan = []
    for wi, wave_units in enumerate(ordered, 1):
        perm = list(range(len(wave_units)))
        rng.shuffle(perm)
        plan.append({"wave": wi, "duration_s": wave_units[0]["duration_s"],
                     "units": wave_units, "worker_perm": perm})
    return plan


def reset_cluster(ctx: str) -> dict:
    out = {"context": ctx, "ok": True, "errors": []}
    for ns, dep in ((COMPUTE_NS, COMPUTE_DEPLOY), (PROM_NS, PROM_DEPLOY)):
        r = kubectl_ctx(ctx, ["rollout", "restart", f"deploy/{dep}", "-n", ns])
        if r.returncode != 0:
            out["ok"] = False
            out["errors"].append(f"restart {dep}: {r.stderr.strip()[:200]}")
    for ns, dep in ((COMPUTE_NS, COMPUTE_DEPLOY), (PROM_NS, PROM_DEPLOY)):
        r = kubectl_ctx(ctx, ["rollout", "status", f"deploy/{dep}", "-n", ns, "--timeout=180s"])
        if r.returncode != 0:
            out["ok"] = False
            out["errors"].append(f"status {dep}: {r.stderr.strip()[:200]}")
    return out


def coverage_from_run_dir(run_dir: Path):
    f = run_dir / "operator_metrics_summary.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except Exception:
        return None
    cov = (data or {}).get("coverage_monitor") or {}
    val = cov.get("coverage_rate")
    return float(val) if isinstance(val, (int, float)) else None


def e2e_p95_from_run_dir(run_dir: Path):
    """Primary metric: metrics.json["e2e"]["p95_ms"] (post-E-V6 block, run_matrix:263)."""
    f = run_dir / "metrics.json"
    if not f.exists():
        return None, None
    try:
        data = json.loads(f.read_text())
    except Exception:
        return None, None
    e2e = (data or {}).get("e2e") or {}
    p95 = e2e.get("p95_ms")
    res = (data or {}).get("resources") or {}
    mean_rep = res.get("mean_replicas") or res.get("avg_replicas")
    return (float(p95) if isinstance(p95, (int, float)) else None,
            float(mean_rep) if isinstance(mean_rep, (int, float)) else None)


def run_cell(slot, unit, output_dir):
    set_thread_kube_context(slot.kube_context)
    pf = PortForward(namespace=PROMETHEUS_NS, service=PROMETHEUS_SVC,
                     local_port=slot.prometheus_port, remote_port=PROMETHEUS_PORT,
                     kube_context=slot.kube_context)
    rec = {"task": unit["task"], "label": unit["label"], "method": "",
           "workload": unit["pattern"], "replicate": unit["replicate"],
           "duration_s": unit["duration_s"], "worker_id": slot.worker_id,
           "cluster": slot.cluster_name, "status": "failed", "run_id": "",
           "e2e_p95_ms": None, "coverage_rate": None, "mean_replicas": None,
           "start_time": "", "end_time": "", "error_message": "", "output_dir": ""}
    try:
        pf.start()
        method = _resolve_method_spec(unit["method_spec"])
        rec["method"] = method.name
        if hasattr(method, "prometheus_port"):
            method.prometheus_port = slot.prometheus_port
        result = execute_single_run(
            method=method, workload_pattern=unit["pattern"], replicate=unit["replicate"],
            duration_s=unit["duration_s"], complexity=COMPLEXITY,
            frontend_url=slot.frontend_url, prometheus_url=slot.prometheus_url,
            output_dir=output_dir)
        rec.update({"status": result.get("status", "unknown"), "run_id": result.get("run_id", ""),
                    "start_time": result.get("start_time", ""), "end_time": result.get("end_time", ""),
                    "error_message": result.get("error_message", ""),
                    "output_dir": result.get("output_dir", "")})
        if rec["output_dir"]:
            rd = Path(rec["output_dir"])
            rec["e2e_p95_ms"], rec["mean_replicas"] = e2e_p95_from_run_dir(rd)
            rec["coverage_rate"] = coverage_from_run_dir(rd)
    except Exception as e:  # noqa: BLE001
        rec["error_message"] = f"{type(e).__name__}: {e}"
        logger.exception("[%s] cell %s rep%d crashed", slot.label(), unit["label"], unit["replicate"])
    finally:
        try:
            pf.stop()
        finally:
            clear_thread_kube_context()
    return rec


def preflight(slots) -> bool:
    import urllib.request
    ok = True
    for s in slots:
        r = kubectl_ctx(s.kube_context, ["get", "ns", NAMESPACE, "-o", "name"], timeout=20)
        if r.returncode != 0:
            logger.error("preflight FAIL %s: namespace %s unreachable", s.label(), NAMESPACE)
            ok = False
            continue
        try:
            with urllib.request.urlopen(f"{s.frontend_url}/health", timeout=6) as resp:
                if resp.status != 200:
                    logger.error("preflight FAIL %s: frontend status %s", s.label(), resp.status)
                    ok = False
                else:
                    logger.info("preflight OK %s (frontend=%s)", s.label(), s.frontend_url)
        except Exception as e:  # noqa: BLE001
            logger.error("preflight FAIL %s: frontend unreachable: %s", s.label(), e)
            ok = False
    return ok


# ── Verdict (card §2 operational decomposition) ───────────────────────────────

def compute_verdict(stage_a: dict, cell_stats: dict) -> dict:
    """cell_stats[(method,pattern)] = {mean, sd, n, vals}. Card §2."""
    import math
    bound = stage_a.get("bound", False)
    recal = ["confscale-aci", "confscale-pid"]
    base = "hpa-qr-monitored"
    contrasts = []
    cleared, nominal_within = [], []
    for pat in ("F", "G"):
        b = cell_stats.get((base, pat))
        if not b:
            continue
        for m in recal:
            c = cell_stats.get((m, pat))
            if not c:
                continue
            pooled = math.sqrt((c["sd"] or 0) ** 2 + (b["sd"] or 0) ** 2)
            margin = b["mean"] - c["mean"]            # >0 means recal is lower (better)
            clears = margin > pooled
            row = {"pattern": pat, "method": m, "mean_recal": round(c["mean"], 1),
                   "sd_recal": round(c["sd"], 1), "mean_base": round(b["mean"], 1),
                   "sd_base": round(b["sd"], 1), "margin_ms": round(margin, 1),
                   "pooled_sd_ms": round(pooled, 1), "clears_pooled_std": clears,
                   "nominally_lower": margin > 0}
            contrasts.append(row)
            if clears:
                cleared.append(row)
            elif margin > 0:
                nominal_within.append(row)
    if not bound:
        verdict = "FAIL"
        why = "binding probe flat (responsive-ratio < 1.30 at every tested load) -> axis still inert"
    elif cleared:
        verdict = "PASS"
        why = (f"bound (ratio>=1.30) AND {len(cleared)} (method,pattern) clear the pooled-std "
               f"margin: " + "; ".join(f"{r['method']}/{r['pattern']} "
               f"{r['mean_recal']}<{r['mean_base']} by {r['margin_ms']}>{r['pooled_sd_ms']}"
               for r in cleared))
    elif nominal_within:
        verdict = "INCONCLUSIVE"
        why = ("bound, some (method,pattern) nominally lower but within pooled std (n=3); "
               "card §2 -> escalate those cells + baseline to R=6 before calling")
    else:
        verdict = "FAIL"
        why = "binding took (ratio>=1.30) but no recalibrated method beats hpa-qr-monitored on e2e p95"
    return {"verdict": verdict, "why": why, "bound": bound,
            "contrasts": contrasts, "cleared": cleared, "nominal_within": nominal_within}


def aggregate_cells(runlog_rows: list[dict]) -> dict:
    import statistics
    by = defaultdict(list)
    for r in runlog_rows:
        if r.get("status") == "ok" and r.get("e2e_p95_ms") is not None:
            method = r.get("method") or r["label"].split("/")[0]
            pat = r["workload"]
            by[(method, pat)].append(float(r["e2e_p95_ms"]))
    stats = {}
    for k, vals in by.items():
        stats[k] = {"mean": statistics.fmean(vals),
                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals), "vals": [round(v, 1) for v in vals]}
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="P3-C-T5d RC5 SLO-binding existence-proof driver")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--start-wave", type=int, default=1)
    ap.add_argument("--skip-stage-a", action="store_true",
                    help="resume: reuse stage_a_probe/stage_a_verdict.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="single short Stage-B cell to validate wiring (no patch/restore loop)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "driver.log")])

    cells = build_cells()
    slots = make_slots(3)
    plan = plan_waves(cells, n_workers=len(slots), seed=args.seed)

    if args.dry_run:
        summary = {"card": "P3-C-T5d", "seed": args.seed, "n_waves": len(plan),
                   "n_cells": sum(len(w["units"]) for w in plan),
                   "worker_limit_bind": WORKER_LIMIT_BIND,
                   "probe_loads": PROBE_LOADS, "probe_replicas": PROBE_REPLICAS,
                   "waves": [{"wave": w["wave"], "cells": [u["label"] for u in w["units"]]}
                             for w in plan]}
        print(json.dumps(summary, indent=2))
        return

    runlog_path = out_dir / "run_log.csv"
    fields = ["stage", "wave", "task", "label", "method", "workload", "replicate", "duration_s",
              "worker_id", "cluster", "status", "e2e_p95_ms", "coverage_rate", "mean_replicas",
              "run_id", "start_time", "end_time", "error_message", "output_dir"]
    new_file = not runlog_path.exists()
    runlog = open(runlog_path, "a", newline="")
    writer = csv.DictWriter(runlog, fieldnames=fields, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        runlog.flush()

    overall_start = time.time()
    final = {"card": "P3-C-T5d", "started": datetime.now(timezone.utc).isoformat()}
    patched = None

    def write_status(state):
        final["state"] = state
        final["elapsed_h"] = round((time.time() - overall_start) / 3600.0, 3)
        (out_dir / "rc5_results.json").write_text(json.dumps(final, indent=2, default=str))

    try:
        # Preflight (abort, no repair — charter §2.8 / card §5).
        if not preflight(slots):
            write_status("preflight_failed")
            logger.error("Preflight FAILED — no repair (charter §2.8). Stopping.")
            return

        # --- Apply the binding knob to the LIVE deployment (additive) ---
        logger.info("Patching worker CPU limit -> %s on %s (live deploy only)",
                    WORKER_LIMIT_BIND, WORKER_CONTEXTS)
        patched = patch_all_workers(WORKER_LIMIT_BIND)
        final["worker_patch"] = patched
        if not patched["ok"]:
            write_status("patch_failed")
            logger.error("Worker-limit patch did not take cleanly; aborting (will restore).")
            return

        if args.smoke:
            # one short cell on w0 to validate the Stage-B run path end-to-end
            logger.info("SMOKE: single confscale-aci/F cell, 180 s, on w0")
            reset_cluster(slots[0].kube_context)
            time.sleep(args.settle)
            u = _unit({"label": "confscale-aci/F", "task": "T5d",
                       "method_spec": {"base": "confscale-aci", "name": "confscale-aci"},
                       "pattern": "F", "duration_s": 180, "n_reps": 1}, 1)
            rec = run_cell(slots[0], u, out_dir)
            rec.update({"stage": "smoke", "wave": 0})
            writer.writerow(rec)
            runlog.flush()
            final["smoke"] = rec
            write_status("smoke_done")
            logger.info("SMOKE result: status=%s e2e_p95=%s coverage=%s",
                        rec["status"], rec["e2e_p95_ms"], rec["coverage_rate"])
            return

        # --- Stage A: binding probe (gate) ---
        if args.skip_stage_a and (out_dir / "stage_a_probe" / "stage_a_verdict.json").exists():
            stage_a = json.loads((out_dir / "stage_a_probe" / "stage_a_verdict.json").read_text())
            logger.info("Stage A reused from disk: bound=%s", stage_a.get("bound"))
        else:
            logger.info("=== STAGE A: controller-free binding probe on %s ===", PROBE_CTX)
            stage_a = stage_a_probe(out_dir)
        final["stage_a"] = stage_a

        if not stage_a.get("bound"):
            # Designed early-out (card §5): binding failed -> FAIL, do NOT run Stage B.
            verdict = compute_verdict(stage_a, {})
            final["verdict"] = verdict
            write_status("done_stage_a_fail")
            logger.info("STAGE A flat -> STOP. Verdict=%s", verdict["verdict"])
            return

        # --- Stage B: main sub-study (3-up waves) ---
        logger.info("=== STAGE B: %d cells in %d waves ===",
                    sum(len(w["units"]) for w in plan), len(plan))
        assign_path = out_dir / "wave_assignments.jsonl"
        stage_b_rows = []
        for w in plan:
            wave = w["wave"]
            if wave < args.start_wave:
                continue
            units, perm = w["units"], w["worker_perm"]
            logger.info("=" * 60)
            logger.info("WAVE %d/%d (%ds): %s", wave, len(plan), w["duration_s"],
                        {units[i]["label"]: slots[perm[i]].cluster_name for i in range(len(units))})
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as ex:
                resets = list(ex.map(reset_cluster, [s.kube_context for s in slots]))
            for r in resets:
                if not r["ok"]:
                    logger.warning("reset issues on %s: %s", r["context"], r["errors"])
            with open(assign_path, "a") as af:
                af.write(json.dumps({"wave": wave,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "map": [{"cell": units[i]["label"], "replicate": units[i]["replicate"],
                             "cluster": slots[perm[i]].cluster_name} for i in range(len(units))],
                    "resets": resets}) + "\n")
            logger.info("WAVE %d: settle %ds...", wave, args.settle)
            time.sleep(args.settle)
            wave_t0 = time.time()
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as ex:
                futs = [ex.submit(run_cell, slots[perm[i]], units[i], out_dir)
                        for i in range(len(units))]
                for f in concurrent.futures.as_completed(futs):
                    results.append(f.result())
            for rec in sorted(results, key=lambda r: r["label"]):
                rec.update({"stage": "B", "wave": wave})
                writer.writerow(rec)
                stage_b_rows.append(rec)
                logger.info("  %s rep%d on %s -> status=%s e2e_p95=%s cov=%s",
                            rec["label"], rec["replicate"], rec["cluster"],
                            rec["status"], rec["e2e_p95_ms"], rec["coverage_rate"])
            runlog.flush()
            logger.info("WAVE %d done in %.1f min (elapsed %.2f h)", wave,
                        (time.time() - wave_t0) / 60.0, (time.time() - overall_start) / 3600.0)
            write_status(f"stage_b_wave_{wave}_done")

        # --- Aggregate + verdict ---
        # re-read full runlog so a --start-wave resume includes earlier waves
        all_rows = []
        with open(runlog_path) as f:
            for r in csv.DictReader(f):
                if r.get("stage") == "B":
                    for k in ("e2e_p95_ms", "mean_replicas", "coverage_rate"):
                        r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
                    all_rows.append(r)
        cell_stats = aggregate_cells(all_rows)
        final["cell_stats"] = {f"{m}/{p}": v for (m, p), v in cell_stats.items()}
        verdict = compute_verdict(stage_a, cell_stats)
        final["verdict"] = verdict
        write_status("done_stage_b")
        logger.info("VERDICT: %s — %s", verdict["verdict"], verdict["why"])

    except Exception as e:  # noqa: BLE001
        final["fatal_error"] = f"{type(e).__name__}: {e}"
        final["traceback"] = traceback.format_exc()
        write_status("fatal_error")
        logger.exception("FATAL in main")
    finally:
        runlog.close()
        # HARD requirement (card §5): restore the worker limit to 500m, verify.
        if patched is not None:
            logger.info("FINALLY: restoring worker CPU limit -> %s on all clusters",
                        WORKER_LIMIT_LOCKED)
            restore = restore_all_workers()
            final["worker_restore"] = restore
            # also restore a sane replica count on the probe cluster (controller-free)
            try:
                _probe_scale(PROBE_CTX, 2)
            except Exception:
                pass
            write_status(final.get("state", "unknown") + ("_restored" if restore["ok"]
                                                           else "_RESTORE_FAILED"))
            if not restore["ok"]:
                logger.error("WORKER LIMIT NOT FULLY RESTORED — manual check required: %s", restore)
        (out_dir / "_RUN_COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat() + "\n" + final.get("state", "unknown") + "\n")
        logger.info("DRIVER COMPLETE (%.2f h). state=%s",
                    (time.time() - overall_start) / 3600.0, final.get("state"))


if __name__ == "__main__":
    main()
