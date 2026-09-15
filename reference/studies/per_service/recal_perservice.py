#!/usr/bin/env python
"""Evaluate recalibration on the fixed MS_7129 volatility window."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")

from pathlib import Path

import sys, json, importlib.util
from collections import deque
import numpy as np
sys.path.insert(0, 'reference/stage3_scale')
import torch
from predictor.data import NormalizationParams
from uq.conformal import SplitConformal
from uq.conformal_pid import ConformalPID
from uq.aci import ACI
from baselines.escalation_ladder import EscalationLadder

DIR = "generated/studies/per_service"
Path(DIR).mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("p2", "reference/studies/per_service/pass2_perservice.py")
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
H, K, ALPHA = p2.H, p2.K, p2.ALPHA
NOMINAL = 90.0
R = 6
# LOCKED MS_7129 volatility windows (fixed service windows)
TRAIN, REF, DEPLOY = (120, 840), (480, 720), (840, 960)
COVWIN = 30


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
    """Offline online-recalibration walk over the deploy windows."""
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


METHODS = ['frozen', 'aci', 'pid', 'aci-lad', 'pid-lad']
acc = {m: {'h0': [], 'h1': [], 'w': [], 'maxlvl': []} for m in METHODS}
bias_acc = []
for rep in range(R):
    torch.manual_seed(1000 + rep); np.random.seed(1000 + rep)
    s = p2.load_series('MS_7129')
    tr, rf, dp = p2.seg(s.values, *TRAIN), p2.seg(s.values, *REF), p2.seg(s.values, *DEPLOY)
    mu, sigma = float(tr.mean()), float(tr.std()) or 1.0
    norm = NormalizationParams(mu, sigma)
    Xtr, Ytr = p2.windows(tr, mu, sigma)
    Xrf, Yrf = p2.windows(rf, mu, sigma)
    u = SplitConformal(alpha=ALPHA, h=H, k=K, device='cpu'); u.norm_params = norm
    u.fit((Xtr, Ytr), calibration_data=(Xrf, Yrf))
    q0_frozen, q1_frozen = float(u.q_hat[0]), float(u.q_hat[1])
    fc_rf, ac_rf = forecasts(u, rf, mu, sigma)
    ref_resids = np.abs(ac_rf[:, 0] - fc_rf[:, 0]) / sigma          # normalized abs residuals (warm-start)
    fc_dp, ac_dp = forecasts(u, dp, mu, sigma)
    bias_acc.append(100 * (fc_dp[:, 0].mean() - ac_dp[:, 0].mean()) / (ac_dp[:, 0].mean() + 1e-9))
    for m in METHODS:
        h0, h1, w, lvl = walk(m, fc_dp, ac_dp, sigma, q0_frozen, q1_frozen, ref_resids)
        acc[m]['h0'].append(h0); acc[m]['h1'].append(h1); acc[m]['w'].append(w); acc[m]['maxlvl'].append(lvl)

out = {'cell': 'MS_7129 volatility (locked)', 'windows': {'train': TRAIN, 'ref': REF, 'deploy': DEPLOY},
       'R': R, 'sanity_bias_pct': round(float(np.mean(bias_acc)), 1), 'n_deploy': len(fc_dp),
       'frozen_scp_pass2_ref': '63.0±0.6% (+27.0pp)', 'qr_pass2_ref': '81.4% (+8.6pp)',
       'synthetic_F_ref': {'pid_raw': 0.90, 'laddered': 0.88}, 'methods': {}}
for m in METHODS:
    a = acc[m]
    out['methods'][m] = {
        'cov_h0': [round(float(np.mean(a['h0'])), 1), round(float(np.std(a['h0'])), 1)],
        'gap_h0': round(NOMINAL - float(np.mean(a['h0'])), 1),
        'cov_h1_frozen': round(float(np.mean(a['h1'])), 1),
        'mean_width_h0': round(float(np.mean(a['w'])), 4),
        'max_ladder_level': int(np.max(a['maxlvl'])) if a['maxlvl'] else 0}
    mm = out['methods'][m]
    print(f"{m:8s} h0={mm['cov_h0'][0]:5.1f}±{mm['cov_h0'][1]:4.1f}%  gap={mm['gap_h0']:+5.1f}  "
          f"width={mm['mean_width_h0']:.4f}  maxL={mm['max_ladder_level']}  (vs frozen +27, synthF pid .90/lad .88)")
with open(f"{DIR}/recal_volatility.json", 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nsanity bias {out['sanity_bias_pct']}% | n_deploy={out['n_deploy']} | wrote recal_volatility.json")
