#!/usr/bin/env python
"""Replay per-horizon recalibration across the fixed volatility candidates."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")

import sys, json, importlib.util, os
import numpy as np
import pandas as pd

POOL_INPUTS = "inputs/studies/expanded_pool"
OUTDIR = "generated/studies/per_horizon_pool"
OUT = f"{OUTDIR}/recal_pool_h1.json"
AGG = f"{OUTDIR}/recal_pool_h1_aggregate.json"

T1SCRIPT = "reference/studies/per_horizon/recal_perservice_h1.py"
spec = importlib.util.spec_from_file_location("t1h1", T1SCRIPT)
t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)
R, NOMINAL = t1.R, t1.NOMINAL

CAND = pd.read_csv(f"{POOL_INPUTS}/candidates.csv")
GATE = f"{POOL_INPUTS}/sanity_gate.json"
METHODS = ['frozen', 'aci', 'pid']   # Non-anchor protocol (ladder is anchor-only; verdict is on plain ACI)


def select_cells():
    """The 16 disjoint volatility candidates (set==vol_band_disjoint AND sanity_pass), windows from CSV."""
    with open(GATE) as f:
        gate = json.load(f)
    sel = [g for g in gate if g.get('set') == 'vol_band_disjoint' and g.get('sanity_pass')]
    cells = []
    for i, g in enumerate(sel):
        svc = g['service']
        row = CAND[(CAND['service'] == svc) & (CAND['channel'] == 'volatility')].iloc[0]
        cells.append({
            'service': svc,
            'train':  (int(row['train_lo']), int(row['train_hi'])),
            'ref':    (int(row['cal_lo']),   int(row['cal_hi'])),
            'deploy': (int(row['dep_lo']),   int(row['dep_hi'])),
            'anchor': False,
            'src': 'expanded_pool: disjoint volatility candidates passing the sanity gate',
            'idx': i, 'ratio': float(g.get('ratio')),
            'gate_frozen_cov_h0': g.get('deploy_cov_h0'), 'gate_frozen_cov_h1': g.get('deploy_cov_h1'),
        })
    return cells


def load_out():
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as f:
            return {r['service']: r for r in json.load(f)}
    return {}


def run_range(start, end):
    cells = select_cells()
    assert len(cells) == 16, f"expected 16 vol_band_disjoint sanity-passing cells, got {len(cells)}"
    os.makedirs(OUTDIR, exist_ok=True)
    done = load_out()
    print(f"Per-horizon pool: {len(cells)} cells | R={R} | methods={METHODS} | nominal {NOMINAL}% | range [{start},{end})",
          flush=True)
    for i in range(start, min(end, len(cells))):
        c = cells[i]
        if c['service'] in done:
            print(f"[{i}] {c['service']} already done — skip", flush=True); continue
        r = t1.run_cell(c, METHODS)
        r['idx'] = c['idx']; r['ratio'] = c['ratio']
        r['gate_frozen_cov_h0'] = c['gate_frozen_cov_h0']; r['gate_frozen_cov_h1'] = c['gate_frozen_cov_h1']
        done[c['service']] = r
        with open(OUT, 'w') as f:
            json.dump(sorted(done.values(), key=lambda x: x['idx']), f, indent=2)
        fh = r['methods']['aci']['frozen_h1']; ph = r['methods']['aci']['per_horizon']
        print(f"[{i}] {c['service']:10s} ({c['ratio']}x) gate_frozen_cov_h1={c['gate_frozen_cov_h1']} | "
              f"ACI frozen_h1 cov_h1={fh['cov_h1'][0]:5.1f} gap={fh['gap_h1']:+5.1f}  ->  "
              f"per_horizon cov_h1={ph['cov_h1'][0]:5.1f} gap={ph['gap_h1']:+5.1f} "
              f"w_h1x{ph['width_h1_x_frozen']} reach={ph['actuator_reach_pct'][0]:.0f}%", flush=True)
    print(f"wrote {OUT}: {len(done)}/16 cells", flush=True)


def aggregate():
    with open(OUT) as f:
        results = json.load(f)
    assert len(results) == 16, f"aggregate needs all 16 cells, have {len(results)}"
    agg = {}
    for method in ['aci', 'pid']:
        for mode in ['frozen_h1', 'per_horizon']:
            gaps = [r['methods'][method][mode]['gap_h1'] for r in results]
            covs = [r['methods'][method][mode]['cov_h1'][0] for r in results]
            wmult = [r['methods'][method][mode]['width_h1_x_frozen']
                     for r in results if r['methods'][method][mode]['width_h1_x_frozen'] is not None]
            reach = [r['methods'][method][mode]['actuator_reach_pct'][0] for r in results]
            agg[f'{method}_{mode}'] = dict(
                mean_gap_h1=round(float(np.mean(gaps)), 2),
                median_gap_h1=round(float(np.median(gaps)), 2),
                worst_gap_h1=round(float(np.max(gaps)), 2),
                mean_cov_h1=round(float(np.mean(covs)), 2),
                median_width_h1_x_frozen=round(float(np.median(wmult)), 2) if wmult else None,
                min_width_h1_x_frozen=round(float(np.min(wmult)), 2) if wmult else None,
                max_width_h1_x_frozen=round(float(np.max(wmult)), 2) if wmult else None,
                mean_actuator_reach_pct=round(float(np.mean(reach)), 1),
            )
    primary = agg['aci_per_horizon']['mean_gap_h1']                 # mean ACI per-horizon h1 gap (90 - cov_h1)
    secondary = agg['aci_per_horizon']['median_width_h1_x_frozen']  # median ACI h1 width mult vs frozen-SCP
    if primary <= 5.0 and secondary <= 6.0:
        verdict = 'GENERALIZES (PASS)'
    elif primary <= 10.0 and secondary <= 10.0:
        verdict = 'GENERALIZES-WITH-RESIDUAL (PARTIAL)'
    else:
        verdict = 'DOES NOT GENERALIZE (FAIL)'
    residual = sorted(
        [{'service': r['service'], 'ratio': r['ratio'],
          'aci_ph_cov_h1': r['methods']['aci']['per_horizon']['cov_h1'][0],
          'aci_ph_gap_h1': r['methods']['aci']['per_horizon']['gap_h1'],
          'aci_ph_width_h1_x_frozen': r['methods']['aci']['per_horizon']['width_h1_x_frozen']}
         for r in results if r['methods']['aci']['per_horizon']['gap_h1'] > 5.0],
        key=lambda x: -x['aci_ph_gap_h1'])
    out = dict(
        task='per-horizon h1 recalibration generalized to 16 disjoint volatility candidates (OFFLINE, R=6)',
        n_cells=len(results), R=R, nominal_pct=NOMINAL,
        verdict_criterion='PASS: mean ACI per-horizon h1 gap <= 5.0pp AND median h1 width mult <= 6x; '
                          'PARTIAL: gap <= 10pp AND width <= 10x; FAIL: gap > 10pp OR width > 10x',
        primary_mean_ACI_per_horizon_gap_h1=primary,
        secondary_median_ACI_per_horizon_width_h1_x_frozen=secondary,
        VERDICT=verdict,
        aci_baseline_to_perhorizon=dict(
            frozen_h1_mean_gap=agg['aci_frozen_h1']['mean_gap_h1'],
            per_horizon_mean_gap=agg['aci_per_horizon']['mean_gap_h1'],
            frozen_h1_mean_cov_h1=agg['aci_frozen_h1']['mean_cov_h1'],
            per_horizon_mean_cov_h1=agg['aci_per_horizon']['mean_cov_h1']),
        aggregates=agg,
        residual_cells_aci_ph_gap_gt5=residual)
    with open(AGG, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'aggregate':
        aggregate()
    else:
        run_range(int(sys.argv[1]), int(sys.argv[2]))
