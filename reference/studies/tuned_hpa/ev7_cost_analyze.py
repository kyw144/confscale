#!/usr/bin/env python3
"""E-V7 tuned-HPA cost analysis (Pattern D, matched p95 SLO).

Recomputes the ConfScale-SCP replica-seconds saving against a COMPETENTLY-TUNED
HPA-reactive baseline, honestly, whatever it is. Reuses the EXACT cost metric
from analysis/post_reframe.py — overhead_replica_seconds_per_hour = overhead /
(duration_s/3600) — and only swaps the baseline run.

Method (pre-registered, see _EV7_TUNED_BASELINE_STATUS.md)
--------------------------------------------------------
1. Per config, per rep: read metrics.json (resources.overhead_replica_seconds,
   resources.mean/max_replicas; e2e.p95_ms, e2e.slo_violation_rate; slo.p95_ms)
   and run_config.yaml (method, workload, method_config). Compute
   overhead_replica_seconds_per_hour. Aggregate mean±sd (n=5).
2. Reproduction anchor: hpa-anchor-u50-s300 (v2 @ 50%/300s ≈ v1 default) must
   reproduce ≈67,806 overhead_rs/h — validates the v2 path + 3600s duration +
   current cluster vs the historical baseline.
3. Matched-SLO gate: the original 71.9% held at matched e2e operating point
   (both arms ~19.3% violation, ~980-1015 ms e2e p95). A tuned HPA config
   "meets the matched p95 SLO" iff its e2e p95 is NOT statistically worse than
   the in-batch ConfScale-SCP arm (Welch one-sided p>0.05 OR within +5%) AND its
   e2e violation rate is within +3 pp of SCP's. Configs that "save" cost by
   degrading service are disqualified.
4. Comparator = the LOWEST overhead_rs/h tuned HPA config that meets the gate
   (the strongest honest baseline). Saving% = (comparator - SCP)/comparator.
5. Read-band verdict (pre-registered):
     within noise of 71.9% -> ROBUST (keep headline, now defended)
     shrinks               -> restate to the measured number ("competitive while honest")
     vanishes / negative   -> drop the cost headline; lead on calibration honesty.

Originals untouched; corrected numbers in this results dir only.
"""
from __future__ import annotations

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")


import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path("<SOURCE_WORKSPACE>")
STAGE3 = REPO / "src" / "stage3_scale"
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))
from analysis.stats import welch_t_test  # noqa: E402

# Original headline (vs default-tuned HPA), for reproduction + drift checks.
ORIG_HPA_OVERHEAD = 67806.0      # HPA-Reactive/D n=5 (default 50%/300s)
ORIG_SCP_OVERHEAD = 19039.0      # ConfScale-SCP/D n=5
ORIG_SAVING_PCT = 71.9
ANCHOR_NAME = "hpa-anchor-u50-s300"
SCP_NAME = "confscale-scp"
# Pre-registered gate tolerances.
P95_TOL_FRAC = 0.05              # e2e p95 within +5% of SCP counts as "matched"
VIOL_TOL_PP = 0.03              # e2e violation within +3 pp of SCP counts as "matched"
ROBUST_BAND_PP = 4.0             # saving within ±4 pp of 71.9 -> "robust"
VANISH_PCT = 10.0                # saving <=10% -> "vanishes"


def _f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def extract_cell(run_dir: Path) -> dict | None:
    cfg_p, met_p = run_dir / "run_config.yaml", run_dir / "metrics.json"
    if not met_p.exists():
        return None
    cfg = yaml.safe_load(cfg_p.read_text()) if cfg_p.exists() else {}
    met = json.loads(met_p.read_text())
    res, slo, e2e = met.get("resources", {}), met.get("slo", {}), met.get("e2e", {})
    dur = _f(cfg.get("duration_s")) or _f(met.get("duration_s")) or 0.0
    overhead = _f(res.get("overhead_replica_seconds"))
    per_hour = (overhead / (dur / 3600.0)) if (overhead is not None and dur) else None
    return {
        "run_id": run_dir.name,
        "method": cfg.get("method") or met.get("method"),
        "workload": cfg.get("workload") or met.get("workload"),
        "replicate": cfg.get("replicate"),
        "duration_s": dur,
        "method_config": cfg.get("method_config", {}),
        "overhead_replica_seconds": overhead,
        "overhead_per_hour": per_hour,
        "mean_replicas": _f(res.get("mean_replicas")),
        "max_replicas": _f(res.get("max_replicas")),
        "e2e_p95_ms": _f(e2e.get("p95_ms")),
        "e2e_violation": _f(e2e.get("slo_violation_rate")),
        "proxy_p95_ms": _f(slo.get("p95_ms")),
    }


def agg(vals):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": a.tolist()}


def collect(batch_dir: Path, extra_dirs: list[Path] | None = None) -> dict:
    """Group Pattern-D cells by config (method name)."""
    cells = []
    roots = [batch_dir] + (extra_dirs or [])
    for root in roots:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            c = extract_cell(d)
            if c and c.get("workload", "").upper() == "D":
                cells.append(c)
    configs: dict[str, list[dict]] = {}
    for c in cells:
        configs.setdefault(c["method"], []).append(c)
    return configs


def config_stats(reps: list[dict]) -> dict:
    mc = reps[0].get("method_config", {}) or {}
    return {
        "n": len(reps),
        "cpu_target": mc.get("cpu_target_pct"),
        "downscale_stab_s": mc.get("downscale_stabilization_s"),
        "hpa_api": mc.get("hpa_api_version"),
        "overhead_per_hour": agg([r["overhead_per_hour"] for r in reps]),
        "overhead_rs": agg([r["overhead_replica_seconds"] for r in reps]),
        "mean_replicas": agg([r["mean_replicas"] for r in reps]),
        "e2e_p95_ms": agg([r["e2e_p95_ms"] for r in reps]),
        "e2e_violation": agg([r["e2e_violation"] for r in reps]),
        "proxy_p95_ms": agg([r["proxy_p95_ms"] for r in reps]),
    }


def meets_slo(cfg: dict, scp: dict) -> dict:
    """Matched-p95-SLO gate vs the in-batch SCP arm."""
    cp95, sp95 = cfg["e2e_p95_ms"], scp["e2e_p95_ms"]
    cv, sv = cfg["e2e_violation"], scp["e2e_violation"]
    detail = {"reason": []}
    if cp95["n"] == 0 or sp95["n"] == 0:
        return {"meets": False, "reason": ["missing e2e p95"]}
    # p95 not statistically worse (one-sided) OR within +5%.
    w = welch_t_test(np.array(cp95["values"]), np.array(sp95["values"]))
    p_two = w.get("p_value", 1.0)
    higher = cp95["mean"] > sp95["mean"]
    p95_within = cp95["mean"] <= sp95["mean"] * (1 + P95_TOL_FRAC)
    p95_not_sig_worse = (not higher) or (p_two > 0.05) or p95_within
    viol_ok = (cv["mean"] is None or sv["mean"] is None or
               cv["mean"] <= sv["mean"] + VIOL_TOL_PP)
    detail.update({
        "e2e_p95_cfg": cp95["mean"], "e2e_p95_scp": sp95["mean"],
        "p95_welch_p": p_two, "p95_within_5pct": p95_within,
        "p95_not_sig_worse": bool(p95_not_sig_worse),
        "e2e_viol_cfg": cv["mean"], "e2e_viol_scp": sv["mean"], "viol_ok": bool(viol_ok),
    })
    meets = bool(p95_not_sig_worse and viol_ok)
    if not p95_not_sig_worse:
        detail["reason"].append("e2e p95 significantly worse than SCP")
    if not viol_ok:
        detail["reason"].append("e2e violation > SCP + 3pp")
    detail["meets"] = meets
    return detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True, help="outputs batch dir")
    ap.add_argument("--out", type=Path, required=True, help="results dir for the JSON")
    ap.add_argument("--extra-dir", type=Path, action="append", default=[],
                    help="extra run-dir roots (e.g. for testing against original data)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    configs = collect(args.batch, args.extra_dir)
    stats = {name: config_stats(reps) for name, reps in configs.items()}

    if SCP_NAME not in stats:
        raise SystemExit(f"ConfScale-SCP ('{SCP_NAME}') not found in batch — cannot compare.")
    scp = stats[SCP_NAME]

    # Reproduction anchor.
    anchor = stats.get(ANCHOR_NAME)
    anchor_report = None
    if anchor:
        am = anchor["overhead_per_hour"]["mean"]
        anchor_report = {
            "overhead_per_hour_mean": am, "sd": anchor["overhead_per_hour"]["sd"],
            "n": anchor["overhead_per_hour"]["n"],
            "mean_replicas": anchor["mean_replicas"]["mean"],
            "original": ORIG_HPA_OVERHEAD,
            "ratio_to_original": (am / ORIG_HPA_OVERHEAD) if am else None,
            "reproduces": bool(am and 0.90 <= am / ORIG_HPA_OVERHEAD <= 1.10),
        }

    # SCP drift check.
    sm = scp["overhead_per_hour"]["mean"]
    scp_report = {
        "overhead_per_hour_mean": sm, "sd": scp["overhead_per_hour"]["sd"],
        "n": scp["overhead_per_hour"]["n"], "mean_replicas": scp["mean_replicas"]["mean"],
        "e2e_p95_ms": scp["e2e_p95_ms"]["mean"], "e2e_violation": scp["e2e_violation"]["mean"],
        "original": ORIG_SCP_OVERHEAD,
        "ratio_to_original": (sm / ORIG_SCP_OVERHEAD) if sm else None,
    }

    # Per-config table + SLO gate (all HPA configs incl. anchor).
    table, gate = [], {}
    for name in sorted(stats):
        if name == SCP_NAME:
            continue
        cfg = stats[name]
        g = meets_slo(cfg, scp)
        gate[name] = g
        table.append({
            "config": name, "cpu_target": cfg["cpu_target"],
            "downscale_stab_s": cfg["downscale_stab_s"], "hpa_api": cfg["hpa_api"],
            "overhead_per_hour_mean": cfg["overhead_per_hour"]["mean"],
            "overhead_per_hour_sd": cfg["overhead_per_hour"]["sd"], "n": cfg["overhead_per_hour"]["n"],
            "mean_replicas": cfg["mean_replicas"]["mean"],
            "e2e_p95_ms": cfg["e2e_p95_ms"]["mean"], "e2e_violation": cfg["e2e_violation"]["mean"],
            "meets_matched_slo": g["meets"],
        })

    # Comparator = lowest-overhead config that meets the matched SLO.
    eligible = [t for t in table if t["meets_matched_slo"] and t["overhead_per_hour_mean"] is not None]
    eligible.sort(key=lambda t: t["overhead_per_hour_mean"])
    comparator = eligible[0] if eligible else None

    saving = None
    if comparator and sm is not None:
        comp = comparator["overhead_per_hour_mean"]
        saving_pct = (comp - sm) / comp * 100.0
        saving = {
            "comparator_config": comparator["config"],
            "comparator_overhead_per_hour": comp,
            "scp_overhead_per_hour": sm,
            "saving_pct": saving_pct,
            "delta_vs_original_pp": saving_pct - ORIG_SAVING_PCT,
        }
        # Read-band verdict.
        if saving_pct <= VANISH_PCT:
            verdict = ("VANISHES — drop the cost-savings headline; lead on the "
                       "calibration-honesty thesis")
        elif abs(saving_pct - ORIG_SAVING_PCT) <= ROBUST_BAND_PP:
            verdict = (f"ROBUST — saving {saving_pct:.1f}% within noise of {ORIG_SAVING_PCT}%; "
                       "keep the headline, now defended against a competently-tuned baseline")
        else:
            verdict = (f"SHRINKS — restate the magnitude to {saving_pct:.1f}% (vs {ORIG_SAVING_PCT}% "
                       "default-tuned); the defensible claim is 'competitive while honest'")
        saving["verdict"] = verdict

    report = {
        "batch": str(args.batch),
        "anchor_reproduction": anchor_report,
        "scp": scp_report,
        "config_table": sorted(table, key=lambda t: (t["overhead_per_hour_mean"] is None,
                                                      t["overhead_per_hour_mean"] or 0)),
        "slo_gate_detail": gate,
        "comparator_and_saving": saving,
    }
    (args.out / "ev7_cost_analysis.json").write_text(json.dumps(report, indent=2, default=str))

    # Printed summary.
    print("\n================ E-V7 TUNED-HPA COST ANALYSIS ================")
    if anchor_report:
        print(f"Reproduction anchor ({ANCHOR_NAME}): {anchor_report['overhead_per_hour_mean']:.0f} rs/h "
              f"(orig {ORIG_HPA_OVERHEAD:.0f}; ratio {anchor_report['ratio_to_original']:.3f}; "
              f"reproduces={anchor_report['reproduces']}; mean_repl={anchor_report['mean_replicas']:.2f})")
    print(f"ConfScale-SCP (re-run):       {sm:.0f} rs/h (orig {ORIG_SCP_OVERHEAD:.0f}; "
          f"ratio {scp_report['ratio_to_original']:.3f}; e2e p95 {scp_report['e2e_p95_ms']:.0f} ms, "
          f"viol {scp_report['e2e_violation']:.3f})")
    print("\n-- Per-config (sorted by overhead/h); matched-SLO gate vs SCP --")
    print(f"{'config':<24}{'cpu':>5}{'stab':>6}{'overhead/h':>14}{'e2e_p95':>9}{'viol':>7}{'SLO?':>7}")
    for t in report["config_table"]:
        oh = "—" if t["overhead_per_hour_mean"] is None else f"{t['overhead_per_hour_mean']:.0f}±{t['overhead_per_hour_sd']:.0f}"
        print(f"{t['config']:<24}{str(t['cpu_target']):>5}{str(t['downscale_stab_s']):>6}"
              f"{oh:>14}{(t['e2e_p95_ms'] or 0):>9.0f}{(t['e2e_violation'] or 0):>7.3f}"
              f"{'MEETS' if t['meets_matched_slo'] else 'fails':>7}")
    if saving:
        print(f"\nComparator (lowest-overhead SLO-meeting HPA): {saving['comparator_config']} "
              f"@ {saving['comparator_overhead_per_hour']:.0f} rs/h")
        print(f"ConfScale-SCP saving vs comparator: {saving['saving_pct']:.1f}% "
              f"(was {ORIG_SAVING_PCT}% vs default; Δ {saving['delta_vs_original_pp']:+.1f} pp)")
        print(f"\nVERDICT: {saving['verdict']}")
    else:
        print("\nNo HPA config met the matched SLO — comparator undefined (investigate).")
    print("==============================================================\n")


if __name__ == "__main__":
    main()
