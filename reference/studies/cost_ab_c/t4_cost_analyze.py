#!/usr/bin/env python3
"""Compare replica cost against tuned HPA for workloads A and B."""
from __future__ import annotations

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
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

ANCHOR_NAME = "hpa-anchor-u50-s300"
SCP_NAME = "confscale-scp"
# Pre-registered gate tolerances (verbatim ev7).
P95_TOL_FRAC = 0.05              # e2e p95 within +5% of SCP counts as "matched"
VIOL_TOL_PP = 0.03              # e2e violation within +3 pp of SCP counts as "matched"
# Card §2 thresholds.
SAVING_MIN_PCT = 10.0           # "genuine A/B saving" requires Saving% >= 10%
ANCHOR_RAILS_REPL = 18.0        # anchor "rails" (pins HPA near ceiling 20) iff mean_replicas >= this


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
        return {"mean": None, "sd": None, "n": 0, "values": []}
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": a.tolist()}


def collect(batch_dir: Path, pattern: str) -> dict:
    """Group cells of the given pattern by config (method name)."""
    cells = []
    for d in sorted(batch_dir.iterdir()):
        if not d.is_dir():
            continue
        c = extract_cell(d)
        if c and (c.get("workload") or "").upper() == pattern.upper():
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
        "max_replicas": agg([r["max_replicas"] for r in reps]),
        "e2e_p95_ms": agg([r["e2e_p95_ms"] for r in reps]),
        "e2e_violation": agg([r["e2e_violation"] for r in reps]),
        "proxy_p95_ms": agg([r["proxy_p95_ms"] for r in reps]),
    }


def meets_slo(cfg: dict, scp: dict) -> dict:
    """Matched-p95-SLO gate vs the in-batch SCP arm (verbatim ev7_cost_analyze.meets_slo)."""
    cp95, sp95 = cfg["e2e_p95_ms"], scp["e2e_p95_ms"]
    cv, sv = cfg["e2e_violation"], scp["e2e_violation"]
    detail = {"reason": []}
    if cp95["n"] == 0 or sp95["n"] == 0:
        return {"meets": False, "reason": ["missing e2e p95"]}
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


def verdict_card_s2(comparator, scp_mean, scp_vals, anchor_rails: bool | None) -> dict:
    """Card P3-C-T4 §2 per-pattern LOCKED verdict."""
    if comparator is None or scp_mean is None:
        return {"verdict": "FAIL-INCONCLUSIVE",
                "detail": "no tuned config passed the matched-SLO gate (or SCP missing) — "
                          "cannot site a comparator; no defensible A/B claim",
                "saving_pct": None, "cost_welch_p": None, "anchor_rails": anchor_rails}
    comp_mean = comparator["overhead_per_hour_mean"]
    comp_vals = comparator.get("overhead_per_hour_values", [])
    saving_pct = (comp_mean - scp_mean) / comp_mean * 100.0 if comp_mean else None
    p_cost = None
    if comp_vals and scp_vals:
        p_cost = welch_t_test(np.array(comp_vals), np.array(scp_vals)).get("p_value", 1.0)
    sig = (p_cost is not None and p_cost < 0.05)

    if saving_pct is not None and saving_pct >= SAVING_MIN_PCT and sig and scp_mean < comp_mean:
        v = "PASS-saving"
        detail = (f"genuine A/B saving: SCP {scp_mean:.0f} < cheapest gate-passing tuned HPA "
                  f"{comp_mean:.0f} rs/h; Saving {saving_pct:.1f}% (≥10%), cost Welch p={p_cost:.3g} (<0.05)")
    elif (comp_mean <= scp_mean) or (not sig):
        v = "PASS-mirror"
        detail = (f"A/B mirror D — SCP shows NO positive saving vs the tuned comparator "
                  f"({comparator['config']} @ {comp_mean:.0f} vs SCP {scp_mean:.0f} rs/h; "
                  f"Saving {saving_pct:.1f}%, cost Welch p={'n/a' if p_cost is None else format(p_cost,'.3g')}). "
                  f"The earlier A/B 'savings' were a default-baseline artifact"
                  + ("" if anchor_rails else
                     " — NOTE: anchor did NOT rail like D, so the artifact arises by a DIFFERENT "
                     "mechanism than D's 300s downscale-window resonance (interpret in the record)."))
    else:
        v = "INCONCLUSIVE"
        detail = (f"saving sign CI-ambiguous at n=5: Saving {saving_pct:.1f}% "
                  f"(0<…<10% or CI includes 0), cost Welch p={p_cost:.3g}; re-run at n=8 before prose")
    return {"verdict": v, "detail": detail, "saving_pct": saving_pct,
            "cost_welch_p": p_cost, "anchor_rails": anchor_rails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True, help="A-grid/B-grid run dir")
    ap.add_argument("--pattern", required=True, choices=["A", "B", "C"])
    ap.add_argument("--out", type=Path, required=True, help="results dir for the JSON")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    configs = collect(args.batch, args.pattern)
    stats = {name: config_stats(reps) for name, reps in configs.items()}
    if SCP_NAME not in stats:
        raise SystemExit(f"ConfScale-SCP ('{SCP_NAME}') not found for pattern {args.pattern} — cannot compare.")
    scp = stats[SCP_NAME]
    sm = scp["overhead_per_hour"]["mean"]
    scp_vals = scp["overhead_per_hour"]["values"]

    # Anchor sanity (report-only; NOT the 67806 D assertion).
    anchor = stats.get(ANCHOR_NAME)
    anchor_report, anchor_rails = None, None
    if anchor:
        amr = anchor["mean_replicas"]["mean"]
        anchor_rails = (amr is not None and amr >= ANCHOR_RAILS_REPL)
        anchor_report = {
            "overhead_per_hour_mean": anchor["overhead_per_hour"]["mean"],
            "mean_replicas": amr, "max_replicas": anchor["max_replicas"]["mean"],
            "rails_like_D": bool(anchor_rails),
            "note": ("anchor pins HPA near ceiling (rails like D)" if anchor_rails else
                     "anchor does NOT rail — default HPA is not wasteful on this pattern; "
                     "any 'mirror-D' verdict holds by a different mechanism than D's resonance"),
        }

    scp_report = {
        "overhead_per_hour_mean": sm, "sd": scp["overhead_per_hour"]["sd"],
        "n": scp["overhead_per_hour"]["n"], "mean_replicas": scp["mean_replicas"]["mean"],
        "e2e_p95_ms": scp["e2e_p95_ms"]["mean"], "e2e_violation": scp["e2e_violation"]["mean"],
        "empirical_coverage_note": "coverage reported separately (operator_metrics_summary)",
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
            "overhead_per_hour_sd": cfg["overhead_per_hour"]["sd"],
            "overhead_per_hour_values": cfg["overhead_per_hour"]["values"],
            "n": cfg["overhead_per_hour"]["n"], "mean_replicas": cfg["mean_replicas"]["mean"],
            "e2e_p95_ms": cfg["e2e_p95_ms"]["mean"], "e2e_violation": cfg["e2e_violation"]["mean"],
            "meets_matched_slo": g["meets"],
        })

    eligible = [t for t in table if t["meets_matched_slo"] and t["overhead_per_hour_mean"] is not None]
    eligible.sort(key=lambda t: t["overhead_per_hour_mean"])
    comparator = eligible[0] if eligible else None

    saving = None
    if comparator and sm is not None:
        comp = comparator["overhead_per_hour_mean"]
        saving = {"comparator_config": comparator["config"],
                  "comparator_overhead_per_hour": comp, "scp_overhead_per_hour": sm,
                  "saving_pct": (comp - sm) / comp * 100.0 if comp else None}

    final = verdict_card_s2(comparator, sm, scp_vals, anchor_rails)

    report = {
        "pattern": args.pattern, "batch": str(args.batch),
        "anchor_sanity": anchor_report, "scp": scp_report,
        "config_table": sorted(table, key=lambda t: (t["overhead_per_hour_mean"] is None,
                                                      t["overhead_per_hour_mean"] or 0)),
        "slo_gate_detail": gate, "comparator_and_saving": saving,
        "card_s2_verdict": final,
    }
    (args.out / f"t4_cost_analysis_{args.pattern}.json").write_text(
        json.dumps(report, indent=2, default=str))

    print(f"\n========== T4 PATTERN {args.pattern} TUNED-HPA COST ANALYSIS ==========")
    if anchor_report:
        print(f"Anchor ({ANCHOR_NAME}): {anchor_report['mean_replicas']:.1f} mean repl "
              f"(max {anchor_report['max_replicas']:.1f}) -> rails_like_D={anchor_report['rails_like_D']}")
    print(f"ConfScale-SCP/{args.pattern}: {sm:.0f} rs/h "
          f"(e2e p95 {scp_report['e2e_p95_ms']:.0f} ms, viol {scp_report['e2e_violation']:.3f})")
    print(f"\n{'config':<24}{'cpu':>5}{'stab':>6}{'overhead/h':>15}{'e2e_p95':>9}{'viol':>7}{'SLO?':>7}")
    for t in report["config_table"]:
        oh = "—" if t["overhead_per_hour_mean"] is None else f"{t['overhead_per_hour_mean']:.0f}±{t['overhead_per_hour_sd']:.0f}"
        print(f"{t['config']:<24}{str(t['cpu_target']):>5}{str(t['downscale_stab_s']):>6}"
              f"{oh:>15}{(t['e2e_p95_ms'] or 0):>9.0f}{(t['e2e_violation'] or 0):>7.3f}"
              f"{'MEETS' if t['meets_matched_slo'] else 'fails':>7}")
    if saving:
        print(f"\nComparator (lowest-overhead SLO-meeting HPA): {saving['comparator_config']} "
              f"@ {saving['comparator_overhead_per_hour']:.0f} rs/h; SCP saving {saving['saving_pct']:.1f}%")
    print(f"\nCARD §2 VERDICT [{args.pattern}]: {final['verdict']}")
    print(f"  {final['detail']}")
    print("==============================================================\n")


if __name__ == "__main__":
    main()
