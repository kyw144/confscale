#!/usr/bin/env python
"""Measure when the frozen second horizon determines the replica target."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")

import sys, json, importlib.util
from collections import deque
import numpy as np
sys.path.insert(0, 'src/stage3_scale')
import torch
from predictor.data import NormalizationParams
from uq.conformal import SplitConformal
from uq.conformal_pid import ConformalPID
from uq.aci import ACI
from baselines.escalation_ladder import EscalationLadder

DIR = "data/p3_runs/results/ev8b_perservice_20260601_024537"
spec = importlib.util.spec_from_file_location("p2", f"{DIR}/pass2_perservice.py")
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
H, K, ALPHA = p2.H, p2.K, p2.ALPHA
NOMINAL = 90.0
R = 6
# LOCKED MS_7129 volatility windows (ev8b_LOCK_2026-06-01.md) — verbatim from recal_perservice.py
TRAIN, REF, DEPLOY = (120, 840), (480, 720), (840, 960)
COVWIN = 30


def forecasts(u, seg_arr, mu, sigma):
    """Per-window raw-RPS point forecasts (h0,h1) + actuals (h0,h1)."""
    X, Y = p2.windows(seg_arr, mu, sigma)
    fc, ac = [], []
    for i in range(len(X)):
        p = u.predict_with_uncertainty(u.norm_params.denormalize(X[i].flatten()))
        fc.append([float(p['point_forecast'][0]), float(p['point_forecast'][1])])
        yd = u.norm_params.denormalize(Y[i])
        ac.append([float(yd[0]), float(yd[1])])
    return np.array(fc), np.array(ac)


def make_recal(kind):
    """Verbatim from recal_perservice.py (methods.py defaults)."""
    if kind == 'aci':
        return ACI(target_alpha=ALPHA, eta=0.1, residual_buffer_size=200, alpha_clip=(1e-4, 0.5))
    return ConformalPID(target_alpha=ALPHA, k_p=0.1, k_i=0.01, k_d=0.05,
                        residual_buffer_size=200, alpha_clip=(1e-4, 0.5))


def walk(method, fc, ac, sigma, q0_frozen, q1_frozen, ref_resids):
    """Offline online-recalibration walk over the deploy window."""
    laddered = method.endswith('-lad')
    kind = 'aci' if method.startswith('aci') else 'pid'
    recal = None
    if method != 'frozen':
        recal = make_recal(kind)
        for r in ref_resids[-200:]:
            recal.residuals.append(abs(float(r)))   # warm-start -> quantile()==frozen q̂ at cycle 0
    ladder = EscalationLadder(target_coverage=0.9, escalation_band=0.05, escalation_persistence=5,
                              recovery_persistence=10, widening_factor=1.5, conservative_factor=3.0) if laddered else None
    mon = deque(maxlen=COVWIN)
    cov0, cov1, w0 = [], [], []
    prev = None
    levels = []
    up0_recal, up1_frozen, up0_frozen, q0_used, fc0_rec, fc1_rec = [], [], [], [], [], []
    hw1 = sigma * q1_frozen                              # h1 half-width is frozen for ALL methods
    hw0_frozen = sigma * q0_frozen                       # what h0 would be WITHOUT recalibration
    for i in range(len(fc)):
        if recal is not None and prev is not None:
            recal.update(prev[0], prev[1])
        q0 = float(recal.quantile()) if recal is not None else q0_frozen
        # 2. ladder on trailing coverage (prior window) — q_hat[0] only (controller §1d)
        if ladder is not None:
            tc = (sum(mon) / len(mon)) if mon else 1.0
            ladder.step(tc); q0 = float(ladder.apply(q0)); levels.append(ladder.level)
        # 3. form intervals (h0 adapted; h1 frozen)
        hw0 = sigma * q0
        lo0, hi0 = max(fc[i, 0] - hw0, 0.0), max(fc[i, 0] + hw0, 0.0)
        c0 = bool(lo0 <= ac[i, 0] <= hi0)
        ci_up1 = max(fc[i, 1] + hw1, 0.0)
        c1 = bool(max(fc[i, 1] - hw1, 0.0) <= ac[i, 1] <= ci_up1)
        cov0.append(c0); cov1.append(c1); w0.append(2 * hw0)
        mon.append(c0)
        prev = (abs(ac[i, 0] - fc[i, 0]) / sigma, not c0)
        # NEW: the three upper bounds the actuator's max() arbitrates between
        up0_recal.append(hi0)                                       # ci_upper[0] actually used (recalibrated)
        up1_frozen.append(ci_up1)                                   # ci_upper[1] (frozen-SCP, never recalibrated)
        up0_frozen.append(max(fc[i, 0] + hw0_frozen, 0.0))          # ci_upper[0] had h0 NOT been recalibrated
        q0_used.append(q0); fc0_rec.append(float(fc[i, 0])); fc1_rec.append(float(fc[i, 1]))
    return dict(
        cov_h0=100 * float(np.mean(cov0)), cov_h1=100 * float(np.mean(cov1)),
        width_h0=float(np.mean(w0)), max_level=max(levels) if levels else 0,
        up0_recal=np.array(up0_recal), up1_frozen=np.array(up1_frozen),
        up0_frozen=np.array(up0_frozen), q0_used=np.array(q0_used),
        fc0=np.array(fc0_rec), fc1=np.array(fc1_rec))


METHODS = ['frozen', 'aci', 'pid', 'aci-lad', 'pid-lad']   # task centers aci/pid; frozen=baseline, -lad optional
per = {m: {'h1binds': [], 'h0binds_frac': [], 'recal_gt_frozen_frac': [],
           'h0_uplift_rps': [], 'h0_uplift_x': [], 'sig_uplift_rps': [],
           'cov_h0': [], 'cov_h1': [], 'width_h0': [], 'maxlvl': [],
           'mean_up0_recal': [], 'mean_up1_frozen': [], 'mean_fc0': [], 'mean_fc1': [],
           'mean_q0': []}
       for m in METHODS}
q_frozen_acc = {'q0': [], 'q1': []}
bias_acc = []

for rep in range(R):
    torch.manual_seed(1000 + rep); np.random.seed(1000 + rep)      # SAME seeds as recal_perservice.py
    s = p2.load_series('MS_7129')
    tr, rf, dp = p2.seg(s.values, *TRAIN), p2.seg(s.values, *REF), p2.seg(s.values, *DEPLOY)
    mu, sigma = float(tr.mean()), float(tr.std()) or 1.0
    norm = NormalizationParams(mu, sigma)
    Xtr, Ytr = p2.windows(tr, mu, sigma)
    Xrf, Yrf = p2.windows(rf, mu, sigma)
    u = SplitConformal(alpha=ALPHA, h=H, k=K, device='cpu'); u.norm_params = norm
    u.fit((Xtr, Ytr), calibration_data=(Xrf, Yrf))
    q0_frozen, q1_frozen = float(u.q_hat[0]), float(u.q_hat[1])
    q_frozen_acc['q0'].append(q0_frozen); q_frozen_acc['q1'].append(q1_frozen)
    fc_rf, ac_rf = forecasts(u, rf, mu, sigma)
    ref_resids = np.abs(ac_rf[:, 0] - fc_rf[:, 0]) / sigma
    fc_dp, ac_dp = forecasts(u, dp, mu, sigma)
    bias_acc.append(100 * (fc_dp[:, 0].mean() - ac_dp[:, 0].mean()) / (ac_dp[:, 0].mean() + 1e-9))
    for m in METHODS:
        w = walk(m, fc_dp, ac_dp, sigma, q0_frozen, q1_frozen, ref_resids)
        u0r, u1f, u0f = w['up0_recal'], w['up1_frozen'], w['up0_frozen']
        # PRIMARY: fraction of deploy cycles where max-over-horizon picks the FROZEN h1 bound
        h1_binds = (u1f >= u0r)
        per[m]['h1binds'].append(100 * float(np.mean(h1_binds)))
        # COMPLEMENT: on h0-binding cycles, does recalibrated h0 exceed the frozen h0?
        h0_mask = ~h1_binds                                         # ci_upper[0]_recal > ci_upper[1]_frozen
        n_h0 = int(h0_mask.sum())
        per[m]['h0binds_frac'].append(100 * float(np.mean(h0_mask)))
        if n_h0 > 0 and m != 'frozen':
            recal_gt = (u0r[h0_mask] > u0f[h0_mask])
            per[m]['recal_gt_frozen_frac'].append(100 * float(np.mean(recal_gt)))
            per[m]['h0_uplift_rps'].append(float(np.mean(u0r[h0_mask] - u0f[h0_mask])))
            per[m]['h0_uplift_x'].append(float(np.mean(u0r[h0_mask] / (u0f[h0_mask] + 1e-12))))
            # net change in the ACTUAL scaling signal on these cycles: recal max vs frozen-h0 max
            sig_recal = u0r[h0_mask]                                # h0 binds -> signal == ci_upper[0]_recal
            sig_frozen = np.maximum(u0f[h0_mask], u1f[h0_mask])     # signal had h0 stayed frozen
            per[m]['sig_uplift_rps'].append(float(np.mean(sig_recal - sig_frozen)))
        per[m]['cov_h0'].append(w['cov_h0']); per[m]['cov_h1'].append(w['cov_h1'])
        per[m]['width_h0'].append(w['width_h0']); per[m]['maxlvl'].append(w['max_level'])
        per[m]['mean_up0_recal'].append(float(u0r.mean())); per[m]['mean_up1_frozen'].append(float(u1f.mean()))
        per[m]['mean_fc0'].append(float(w['fc0'].mean())); per[m]['mean_fc1'].append(float(w['fc1'].mean()))
        per[m]['mean_q0'].append(float(w['q0_used'].mean()))


def band(frac):
    if frac < 15: return "FOOTNOTE (<15%): recalibration reaches the actuator on most cycles"
    if frac <= 40: return "MATERIAL (15-40%): state plainly in limitations"
    return "PROMINENT (>40%): mechanism largely invisible to scaling on this cell; per-horizon q̂[1] recal is the named fix"


def ms(xs): return [round(float(np.mean(xs)), 1), round(float(np.std(xs)), 1)] if xs else [None, None]
def m1(xs, nd=4): return round(float(np.mean(xs)), nd) if xs else None

out = {
    'analysis': 'h1-binds: how often the max-over-horizon scaling decision is driven by the frozen h1 bound',
    'item': 'paper3_claims_update item 9',
    'cell': 'MS_7129 volatility (locked)', 'windows': {'train': TRAIN, 'ref': REF, 'deploy': DEPLOY},
    'R': R, 'n_deploy': len(fc_dp), 'sanity_bias_pct': round(float(np.mean(bias_acc)), 1),
    'actuator': ("compute_target_replicas: effective_rate = float(np.max(ci_upper)) "
                 "(ci-upper / risk-quantile / tier-3 policy; exported gauge 'max over forecast horizon'). "
                 "K=2 -> arbitrates max(ci_upper[0]_recalibrated, ci_upper[1]_frozen). "
                 "Recalibration is h0-only by code (controller.py §1c/§1d); h1 stays frozen-SCP."),
    'metric_def': ("PRIMARY h1-binds = fraction of deploy cycles with ci_upper[1]_frozen >= ci_upper[0]_recal "
                   "(max picks frozen h1 -> h0 recalibration invisible). COMPLEMENT on h0-binding cycles: "
                   "does ci_upper[0]_recal exceed ci_upper[0]_frozen, and by how much."),
    'read_bands': {'footnote': '<15%', 'material': '15-40%', 'prominent': '>40%'},
    'frozen_q_hat': {'q0': m1(np.mean(q_frozen_acc['q0']), 6), 'q1': m1(np.mean(q_frozen_acc['q1']), 6),
                     'note': 'normalized residual space; ci_upper[h]=fc[h]+sigma*q_hat[h]'},
    'methods': {},
}
for m in METHODS:
    d = per[m]
    rec = {
        'h1_binds_pct': {'mean': ms(d['h1binds'])[0], 'sd': ms(d['h1binds'])[1], 'per_rep': [round(x, 1) for x in d['h1binds']]},
        'read_band': band(float(np.mean(d['h1binds']))),
        'h0_binds_pct_mean': ms(d['h0binds_frac'])[0],
        'crosscheck_vs_recal_volatility_json': {
            'cov_h0': ms(d['cov_h0']), 'cov_h1_mean': round(float(np.mean(d['cov_h1'])), 1),
            'width_h0_mean': m1(d['width_h0']), 'max_ladder_level': int(np.max(d['maxlvl'])) if d['maxlvl'] else 0},
        'decomposition_mean_rps': {
            'fc_h0': m1(d['mean_fc0']), 'fc_h1': m1(d['mean_fc1']),
            'ci_upper0_recal': m1(d['mean_up0_recal']), 'ci_upper1_frozen': m1(d['mean_up1_frozen']),
            'mean_q0_used_norm': m1(d['mean_q0'], 6)},
    }
    if m != 'frozen':
        rec['complement_on_h0_binding_cycles'] = {
            'recal_exceeds_frozen_h0_pct': ms(d['recal_gt_frozen_frac'])[0],
            'mean_h0_uplift_rps': m1(d['h0_uplift_rps']),
            'mean_h0_uplift_x': m1(d['h0_uplift_x'], 2),
            'mean_actuator_signal_uplift_rps': m1(d['sig_uplift_rps']),
            'note': 'uplift = ci_upper[0]_recal - ci_upper[0]_frozen; signal_uplift = recal max - frozen-h0 max'}
    out['methods'][m] = rec

with open(f"{DIR}/h1_binds.json", 'w') as f:
    json.dump(out, f, indent=2)

print(f"\nMS_7129 volatility (locked) | R={R} | n_deploy={out['n_deploy']} | sanity bias {out['sanity_bias_pct']}%")
print(f"frozen q_hat: q0={out['frozen_q_hat']['q0']} q1={out['frozen_q_hat']['q1']} (norm)  "
      f"-> sigma*q0={out['frozen_q_hat']['q0']*np.mean([np.mean(q_frozen_acc['q0'])]):.4g} (ref only)")
print(f"\n{'method':9s} {'h1-binds %':>14s} {'h0-binds %':>11s} | crosscheck: {'cov_h0':>8s} {'cov_h1':>7s} {'w_h0':>7s}")
for m in METHODS:
    r = out['methods'][m]; hb = r['h1_binds_pct']; cc = r['crosscheck_vs_recal_volatility_json']
    print(f"{m:9s} {hb['mean']:6.1f} ± {hb['sd']:4.1f}  {r['h0_binds_pct_mean']:9.1f}  | "
          f"{cc['cov_h0'][0]:6.1f}±{cc['cov_h0'][1]:.1f} {cc['cov_h1_mean']:6.1f} {cc['width_h0_mean']:7.4f}")
print("\nPRIMARY read-band per method (aci/pid are the task targets):")
for m in ['aci', 'pid', 'aci-lad', 'pid-lad', 'frozen']:
    print(f"  {m:9s} {out['methods'][m]['h1_binds_pct']['mean']:5.1f}% -> {out['methods'][m]['read_band']}")
print("\nCOMPLEMENT (h0-binding cycles, recalibrated vs frozen h0):")
for m in ['aci', 'pid', 'aci-lad', 'pid-lad']:
    c = out['methods'][m].get('complement_on_h0_binding_cycles')
    if c:
        print(f"  {m:9s} recal>frozen on {c['recal_exceeds_frozen_h0_pct']}% of h0-binding cycles | "
              f"h0 uplift {c['mean_h0_uplift_rps']} RPS ({c['mean_h0_uplift_x']}x) | "
              f"actuator-signal uplift {c['mean_actuator_signal_uplift_rps']} RPS")
print(f"\nwrote {DIR}/h1_binds.json")
