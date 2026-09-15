#!/usr/bin/env python3
"""Run paired HPA cost comparisons for workloads A and B."""
from __future__ import annotations

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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

COMPLEXITY = 50000          # matches the original D cost runs (ev7_driver.py:79)
COMPUTE_NS = "infosys-benchmark"
COMPUTE_DEPLOY = "compute-worker"
PROM_NS = "monitoring"
PROM_DEPLOY = "prometheus"
T4_DURATION = 1800          # ev7 EV7_DURATION; per-hour-normalized cost is duration-invariant

logger = logging.getLogger("t4")


def build_cells(pattern: str, duration_s: int) -> list[dict]:
    cells: list[dict] = []
    # anchor + 2×2 tuned grid (all the v2 behavior path), n=5 — exactly ev7_driver.py:96-109.
    hpa_grid = [
        ("hpa-anchor-u50-s300", 50, 300),   # reproduction anchor
        ("hpa-tuned-u50-s60",   50, 60),
        ("hpa-tuned-u50-s120",  50, 120),
        ("hpa-tuned-u70-s60",   70, 60),
        ("hpa-tuned-u70-s120",  70, 120),
    ]
    for name, util, stab in hpa_grid:
        cells.append({
            "label": f"{name}/{pattern}", "task": "T4",
            "method_spec": {"base": "hpa-reactive", "name": name,
                            "cpu_target": util, "downscale_stabilization_s": stab},
            "pattern": pattern, "duration_s": duration_s, "n_reps": 5,
        })
    # in-batch ConfScale-SCP re-run (matched current-cluster comparator) — ev7_driver.py:111-115.
    cells.append({
        "label": f"confscale-scp/{pattern}", "task": "T4",
        "method_spec": "confscale-scp",
        "pattern": pattern, "duration_s": duration_s, "n_reps": 5,
    })
    return cells


def _unit(cell: dict, replicate: int) -> dict:
    return {
        "label": cell["label"], "task": cell["task"],
        "method_spec": cell["method_spec"], "pattern": cell["pattern"],
        "duration_s": cell["duration_s"], "replicate": replicate,
    }


def _pack_group(units_by_label: dict[str, list[dict]], n_workers: int,
                rng: random.Random) -> list[list[dict]]:
    labels = list(units_by_label.keys())
    if len(labels) == 1:
        u = units_by_label[labels[0]][:]
        return [u[i:i + n_workers] for i in range(0, len(u), n_workers)]
    counts = {lbl: len(units_by_label[lbl]) for lbl in labels}
    R = counts[labels[0]]
    assert all(c == R for c in counts.values()), \
        f"round-based packing needs equal per-label counts, got {counts}"
    ptr = {lbl: 0 for lbl in labels}
    waves: list[list[dict]] = []
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


def _interleave(group_waves: dict[str, list[list[dict]]]) -> list[list[dict]]:
    tagged = []
    for gi, gkey in enumerate(sorted(group_waves)):
        waves = group_waves[gkey]
        n = len(waves)
        for j, w in enumerate(waves):
            pos = (j + 0.5) / n
            tagged.append((pos, gi, j, w))
    tagged.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in tagged]


def plan_waves(cells: list[dict], n_workers: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        for rep in range(1, c["n_reps"] + 1):
            groups[c["task"]][c["label"]].append(_unit(c, rep))
    group_waves: dict[str, list[list[dict]]] = {}
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


def kubectl_ctx(ctx: str, args: list[str], timeout: int = 200) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", f"--context={ctx}", *args],
                          capture_output=True, text=True, timeout=timeout)


def reset_cluster(ctx: str) -> dict:
    out = {"context": ctx, "ok": True, "errors": []}
    for ns, dep in ((COMPUTE_NS, COMPUTE_DEPLOY), (PROM_NS, PROM_DEPLOY)):
        r = kubectl_ctx(ctx, ["rollout", "restart", f"deploy/{dep}", "-n", ns])
        if r.returncode != 0:
            out["ok"] = False
            out["errors"].append(f"restart {dep}: {r.stderr.strip()[:200]}")
    for ns, dep in ((COMPUTE_NS, COMPUTE_DEPLOY), (PROM_NS, PROM_DEPLOY)):
        r = kubectl_ctx(ctx, ["rollout", "status", f"deploy/{dep}", "-n", ns,
                              "--timeout=180s"])
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


def run_cell(slot, unit: dict, output_dir: Path) -> dict:
    """Run ONE cell on ONE worker slot, pinned to that slot's cluster."""
    set_thread_kube_context(slot.kube_context)
    pf = PortForward(namespace=PROMETHEUS_NS, service=PROMETHEUS_SVC,
                     local_port=slot.prometheus_port, remote_port=PROMETHEUS_PORT,
                     kube_context=slot.kube_context)
    rec = {
        "task": unit["task"], "label": unit["label"], "method": "",
        "workload": unit["pattern"], "replicate": unit["replicate"],
        "duration_s": unit["duration_s"], "worker_id": slot.worker_id,
        "cluster": slot.cluster_name, "status": "failed", "run_id": "",
        "coverage_rate": None, "start_time": "", "end_time": "",
        "error_message": "", "output_dir": "",
    }
    try:
        pf.start()
        method = _resolve_method_spec(unit["method_spec"])
        rec["method"] = method.name
        if hasattr(method, "prometheus_port"):
            method.prometheus_port = slot.prometheus_port
        result = execute_single_run(
            method=method, workload_pattern=unit["pattern"],
            replicate=unit["replicate"], duration_s=unit["duration_s"],
            complexity=COMPLEXITY, frontend_url=slot.frontend_url,
            prometheus_url=slot.prometheus_url, output_dir=output_dir,
        )
        rec.update({
            "status": result.get("status", "unknown"),
            "run_id": result.get("run_id", ""),
            "start_time": result.get("start_time", ""),
            "end_time": result.get("end_time", ""),
            "error_message": result.get("error_message", ""),
            "output_dir": result.get("output_dir", ""),
        })
        if rec["output_dir"]:
            rec["coverage_rate"] = coverage_from_run_dir(Path(rec["output_dir"]))
    except Exception as e:  # noqa: BLE001 — must not crash the wave
        rec["error_message"] = f"{type(e).__name__}: {e}"
        logger.exception("[%s] cell %s rep%d crashed",
                         slot.label(), unit["label"], unit["replicate"])
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
                    logger.info("preflight OK %s (frontend=%s, prom_port=%d)",
                                s.label(), s.frontend_url, s.prometheus_port)
        except Exception as e:  # noqa: BLE001
            logger.error("preflight FAIL %s: frontend unreachable: %s", s.label(), e)
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="T4 Pattern A/B/C tuned-HPA cost wave driver")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pattern", required=True, choices=["A", "B", "C"],
                    help="which sub-run grid to execute")
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--settle", type=int, default=30,
                    help="seconds to let Prometheus scrape after reset")
    ap.add_argument("--start-wave", type=int, default=1, help="resume from wave N (1-based)")
    ap.add_argument("--duration-override", type=int, default=None,
                    help="SMOKE ONLY: per-cell duration (s) instead of 1800")
    ap.add_argument("--max-waves", type=int, default=None,
                    help="SMOKE ONLY: run at most this many waves")
    ap.add_argument("--dry-run", action="store_true", help="print the wave plan and exit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = args.duration_override or T4_DURATION
    cells = build_cells(args.pattern, duration)
    slots = make_slots(3)
    plan = plan_waves(cells, n_workers=len(slots), seed=args.seed)
    if args.max_waves is not None:
        plan = plan[:args.max_waves]

    plan_summary = [{
        "wave": w["wave"], "duration_s": w["duration_s"],
        "cells": [u["label"] for u in w["units"]],
    } for w in plan]
    (out_dir / "wave_plan.json").write_text(json.dumps({
        "pattern": args.pattern, "seed": args.seed, "duration_s": duration,
        "n_waves": len(plan), "n_cells": sum(len(w["units"]) for w in plan),
        "waves": plan_summary,
    }, indent=2))

    if args.dry_run:
        total_s = sum(w["duration_s"] + args.settle + 90 for w in plan)
        print(json.dumps({"pattern": args.pattern, "seed": args.seed, "duration_s": duration,
                          "n_waves": len(plan),
                          "n_cells": sum(len(w['units']) for w in plan),
                          "est_wall_h": round(total_s / 3600.0, 1),
                          "waves": plan_summary}, indent=2))
        return

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "driver.log")],
    )
    logger.info("T4 driver: pattern=%s, %d cells in %d waves, dur=%ds, output=%s seed=%d",
                args.pattern, sum(len(w["units"]) for w in plan), len(plan), duration,
                out_dir, args.seed)

    if not preflight(slots):
        logger.error("Preflight failed — cluster unresponsive. Stopping (no repair).")
        sys.exit(2)

    runlog_path = out_dir / "ev7ab_run_log.csv"
    fields = ["wave", "task", "label", "method", "workload", "replicate", "duration_s",
              "worker_id", "cluster", "status", "coverage_rate", "run_id",
              "start_time", "end_time", "error_message", "output_dir"]
    new_file = not runlog_path.exists()
    runlog = open(runlog_path, "a", newline="")
    writer = csv.DictWriter(runlog, fieldnames=fields, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        runlog.flush()

    assign_path = out_dir / "wave_assignments.jsonl"
    overall_start = time.time()

    for w in plan:
        wave = w["wave"]
        if wave < args.start_wave:
            continue
        units, perm = w["units"], w["worker_perm"]
        assignment = {
            "wave": wave, "duration_s": w["duration_s"],
            "time": datetime.now(timezone.utc).isoformat(),
            "map": [{"cell": units[i]["label"], "replicate": units[i]["replicate"],
                     "worker_id": slots[perm[i]].worker_id,
                     "cluster": slots[perm[i]].cluster_name}
                    for i in range(len(units))],
        }
        logger.info("=" * 64)
        logger.info("WAVE %d/%d (%ds): %s", wave, len(plan), w["duration_s"],
                    {a["cell"]: a["cluster"] for a in assignment["map"]})

        logger.info("WAVE %d: resetting deployments + TSDB on all clusters...", wave)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as ex:
            resets = list(ex.map(reset_cluster, [s.kube_context for s in slots]))
        for r in resets:
            if not r["ok"]:
                logger.warning("reset issues on %s: %s", r["context"], r["errors"])
        assignment["resets"] = resets
        with open(assign_path, "a") as af:
            af.write(json.dumps(assignment) + "\n")

        logger.info("WAVE %d: settle %ds...", wave, args.settle)
        time.sleep(args.settle)

        wave_t0 = time.time()
        results: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as ex:
            futs = [ex.submit(run_cell, slots[perm[i]], units[i], out_dir)
                    for i in range(len(units))]
            for f in concurrent.futures.as_completed(futs):
                results.append(f.result())

        for rec in sorted(results, key=lambda r: r["label"]):
            rec["wave"] = wave
            writer.writerow(rec)
            logger.info("  %s rep%d on %s -> status=%s coverage=%s",
                        rec["label"], rec["replicate"], rec["cluster"],
                        rec["status"], rec["coverage_rate"])
        runlog.flush()
        logger.info("WAVE %d done in %.1f min (elapsed %.2f h)",
                    wave, (time.time() - wave_t0) / 60.0,
                    (time.time() - overall_start) / 3600.0)

    runlog.close()
    logger.info("DRIVER COMPLETE: pattern=%s waves %d..%d, total %.2f h",
                args.pattern, args.start_wave, len(plan),
                (time.time() - overall_start) / 3600.0)
    (out_dir / "_DRIVER_DONE").write_text(
        json.dumps({"pattern": args.pattern, "n_waves": len(plan),
                    "elapsed_h": round((time.time() - overall_start) / 3600.0, 2)}) + "\n")


if __name__ == "__main__":
    main()
