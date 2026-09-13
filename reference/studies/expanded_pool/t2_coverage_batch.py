#!/usr/bin/env python
"""T2 Step 3 — reduced coverage batch: frozen SCP + ACI/PID(+ladder) recovery at R=3.

ADDITIVE: imports the LOCKED `pass2_perservice.py` for series/window helpers and
COPIES the three recal-walk helpers (`forecasts`, `make_recal`, `walk`) VERBATIM from the locked
`recal_perservice.py` (that script runs the MS_7129 analysis at module scope, so it cannot be
imported without side effects). Edits no locked code.

Per batch service (selected from sanity-passing T2 candidates, 3 volatility + 3 floor-aware level),
reproduces frozen-SCP vs adaptive-recal coverage over the deploy window at R=3 replicates,
h0-only recalibration with warm-started buffer, matching the controller protocol exactly
(`recal_perservice.py:59-93`). Reports per-service frozen vs ACI/PID/laddered coverage h0/h1 and
width multipliers vs frozen.

Usage:  .venv/bin/python data/p3_runs/reopen_2026-06/T2/t2_coverage_batch.py <start_idx> <end_idx>
        # half-open [start,end) into the selected batch list (0-based).
Appends/updates t2_coverage_batch.json after EACH service (crash-safe; resume-safe).
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import sys, json, importlib.util, os
from collections import deque
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/stage3_scale')
import torch
from predictor.data import NormalizationParams
from uq.conformal import SplitConformal
from uq.conformal_pid import ConformalPID
from uq.aci import ACI
from baselines.escalation_ladder import EscalationLadder

T2 = "data/p3_runs/reopen_2026-06/T2"
LOCK = "data/p3_runs/results/ev8b_perservice_20260601_024537"
spec = importlib.util.spec_from_file_location("p2", f"{LOCK}/pass2_perservice.py")
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
H, K, ALPHA = p2.H, p2.K, p2.ALPHA
NOMINAL = 90.0
R = 3
COVWIN = 30
METHODS = ['frozen', 'aci', 'pid', 'aci-lad', 'pid-lad']

CAND = pd.read_csv(f"{T2}/t2_candidates.csv")
GATE = f"{T2}/t2_sanity_gate.json"
OUT = f"{T2}/t2_coverage_batch.json"


# ----- helpers copied VERBATIM from recal_perservice.py:40-93 (locked logic) -----
def forecasts(u, seg_arr, mu, sigma):
    """Per-window raw-RPS point forecasts (h0,h1) + actuals (h0,h1) for a segment."""
    X, Y = p2.windows(seg_arr, mu, sigma)
    fc, ac = [], []
    for i in range(len(X)):
        p = u.predict_with_uncertainty(u.norm_params.denormalize(X[i].flatten()))
        fc.append([float(p['point_forecast'][0]), float(p['point_forecast'][1])])
        yd = u.norm_params.denormalize(Y[i])
        ac.append([float(yd[0]), float(yd[1])])
    return np.array(fc), np.array(ac)


def make_recal(kind):
    if kind == 'aci':
        return ACI(target_alpha=ALPHA, eta=0.1, residual_buffer_size=200, alpha_clip=(1e-4, 0.5))
    return ConformalPID(target_alpha=ALPHA, k_p=0.1, k_i=0.01, k_d=0.05,
                        residual_buffer_size=200, alpha_clip=(1e-4, 0.5))


def walk(method, fc, ac, sigma, q0_frozen, q1_frozen, ref_resids):
    """Offline online-recalibration walk over the deploy windows. Returns coverage h0/h1 + width + ladder hist."""
    laddered = method.endswith('-lad')
    kind = 'aci' if method.startswith('aci') else 'pid'
    recal = None
    if method != 'frozen':
        recal = make_recal(kind)
        for r in ref_resids[-200:]:
            recal.residuals.append(abs(float(r)))   # warm-start buffer -> quantile()==frozen q̂
    ladder = EscalationLadder(target_coverage=0.9, escalation_band=0.05, escalation_persistence=5,
                              recovery_persistence=10, widening_factor=1.5, conservative_factor=3.0) if laddered else None
    mon = deque(maxlen=COVWIN)
    cov0, cov1, w0 = [], [], []
    prev = None
    levels = []
    for i in range(len(fc)):
        # 1. update recalibrator on PRIOR step, set q̂[0]
        if recal is not None and prev is not None:
            recal.update(prev[0], prev[1])
        q0 = float(recal.quantile()) if recal is not None else q0_frozen
        # 2. ladder on trailing coverage (prior window)
        if ladder is not None:
            tc = (sum(mon) / len(mon)) if mon else 1.0
            ladder.step(tc); q0 = float(ladder.apply(q0)); levels.append(ladder.level)
        # 3. form intervals (h0 adapted; h1 frozen for all — recalibration is h0-only)
        hw0 = sigma * q0
        lo0, hi0 = max(fc[i, 0] - hw0, 0.0), max(fc[i, 0] + hw0, 0.0)
        c0 = bool(lo0 <= ac[i, 0] <= hi0)
        hw1 = sigma * q1_frozen
        c1 = bool(max(fc[i, 1] - hw1, 0.0) <= ac[i, 1] <= max(fc[i, 1] + hw1, 0.0))
        cov0.append(c0); cov1.append(c1); w0.append(2 * hw0)
        mon.append(c0)
        prev = (abs(ac[i, 0] - fc[i, 0]) / sigma, not c0)
    return (100 * np.mean(cov0), 100 * np.mean(cov1), float(np.mean(w0)),
            max(levels) if levels else 0)
# ----- end verbatim copy -----


def windows_for(service, channel):
    row = CAND[(CAND['service'] == service) & (CAND['channel'] == channel)].iloc[0]
    return ((int(row['train_lo']), int(row['train_hi'])),
            (int(row['cal_lo']), int(row['cal_hi'])),
            (int(row['dep_lo']), int(row['dep_hi'])))


def select_batch():
    """Deterministic batch: first 3 sanity-passing volatility (ratio desc) + first 3 sanity-passing
    floor-aware level (resid_inflation asc) candidates, in CSV row order."""
    with open(GATE) as f:
        gate = json.load(f)
    passing = [g for g in gate if g.get('sanity_pass')]
    vol = [g for g in passing if g['channel'] == 'volatility']
    lvl_fa = [g for g in passing if g.get('set') == 'level_floor_aware']
    batch = vol[:3] + lvl_fa[:3]
    return [(g['service'], g['channel'], g.get('set'), g.get('ratio')) for g in batch]


def run_service(service, channel):
    TRAIN, REF, DEPLOY = windows_for(service, channel)
    acc = {m: {'h0': [], 'h1': [], 'w': [], 'maxlvl': []} for m in METHODS}
    bias_acc, q0s, q1s, sig = [], [], [], []
    for rep in range(R):
        torch.manual_seed(1000 + rep); np.random.seed(1000 + rep)
        s = p2.load_series(service)
        tr, rf, dp = p2.seg(s.values, *TRAIN), p2.seg(s.values, *REF), p2.seg(s.values, *DEPLOY)
        mu, sigma = float(tr.mean()), float(tr.std()) or 1.0
        norm = NormalizationParams(mu, sigma)
        Xtr, Ytr = p2.windows(tr, mu, sigma)
        Xrf, Yrf = p2.windows(rf, mu, sigma)
        u = SplitConformal(alpha=ALPHA, h=H, k=K, device='cpu'); u.norm_params = norm
        u.fit((Xtr, Ytr), calibration_data=(Xrf, Yrf))
        q0_frozen, q1_frozen = float(u.q_hat[0]), float(u.q_hat[1])
        q0s.append(q0_frozen); q1s.append(q1_frozen); sig.append(sigma)
        fc_rf, ac_rf = forecasts(u, rf, mu, sigma)
        ref_resids = np.abs(ac_rf[:, 0] - fc_rf[:, 0]) / sigma
        fc_dp, ac_dp = forecasts(u, dp, mu, sigma)
        bias_acc.append(100 * (fc_dp[:, 0].mean() - ac_dp[:, 0].mean()) / (ac_dp[:, 0].mean() + 1e-9))
        for m in METHODS:
            h0, h1, w, lvl = walk(m, fc_dp, ac_dp, sigma, q0_frozen, q1_frozen, ref_resids)
            acc[m]['h0'].append(h0); acc[m]['h1'].append(h1); acc[m]['w'].append(w); acc[m]['maxlvl'].append(lvl)
    frozen_w = float(np.mean(acc['frozen']['w']))
    res = dict(service=service, channel=channel,
               windows=dict(train=TRAIN, ref=REF, deploy=DEPLOY), R=R,
               sanity_bias_pct=round(float(np.mean(bias_acc)), 1), n_deploy=len(fc_dp),
               sigma=round(float(np.mean(sig)), 4),
               q0_frozen=round(float(np.mean(q0s)), 4), q1_frozen=round(float(np.mean(q1s)), 4),
               frozen_mean_width_h0=round(frozen_w, 4), methods={})
    for m in METHODS:
        a = acc[m]
        mw = float(np.mean(a['w']))
        res['methods'][m] = dict(
            cov_h0=[round(float(np.mean(a['h0'])), 1), round(float(np.std(a['h0'])), 1)],
            gap_h0=round(NOMINAL - float(np.mean(a['h0'])), 1),
            cov_h1_frozen=round(float(np.mean(a['h1'])), 1),
            mean_width_h0=round(mw, 4),
            width_mult_vs_frozen=round(mw / (frozen_w + 1e-12), 2),
            max_ladder_level=int(np.max(a['maxlvl'])) if a['maxlvl'] else 0)
    return res


def load_out():
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        with open(OUT) as f:
            return {f"{r['service']}|{r['channel']}": r for r in json.load(f)}
    return {}


def main():
    start, end = int(sys.argv[1]), int(sys.argv[2])
    batch = select_batch()
    print(f"batch ({len(batch)}): " + ", ".join(f"{s}|{c}" for s, c, _, _ in batch), flush=True)
    done = load_out()
    for i in range(start, min(end, len(batch))):
        service, channel, setname, ratio = batch[i]
        key = f"{service}|{channel}"
        if key in done:
            print(f"[{i}] {key} already done — skip", flush=True); continue
        res = run_service(service, channel)
        res['idx'] = i; res['set'] = setname; res['ratio'] = ratio
        done[key] = res
        with open(OUT, 'w') as f:
            json.dump(sorted(done.values(), key=lambda r: r['idx']), f, indent=2)
        fz = res['methods']['frozen']; ac = res['methods']['aci']; pd_ = res['methods']['pid']
        print(f"[{i}] {key:26s} ({setname}, {ratio}x) bias={res['sanity_bias_pct']}% n={res['n_deploy']}\n"
              f"      frozen h0={fz['cov_h0'][0]:.1f}% gap={fz['gap_h0']:+.1f}  "
              f"aci h0={ac['cov_h0'][0]:.1f}% (x{ac['width_mult_vs_frozen']})  "
              f"pid h0={pd_['cov_h0'][0]:.1f}% (x{pd_['width_mult_vs_frozen']})", flush=True)
    print(f"\nwrote {OUT}: {len(done)} services", flush=True)


if __name__ == '__main__':
    main()
