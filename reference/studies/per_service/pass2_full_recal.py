#!/usr/bin/env python
"""E-V8b Pass 2 (FULL) — recalibrator closeout on the volatility-channel services with a gap.

For each feasible volatility-channel service (≠ anchor MS_7129, already done) that shows a frozen-SCP
deploy gap ≥10 pp AND passes the sanity gate, drive raw ACI (η=0.1) and PID (kp.1/ki.01/kd.05),
h0-only, R=6 — and report whether the gap recovers to ≈ target (as on MS_7129: ACI 89.8 / PID 90.4)
and the width ×frozen.

REPLICATES recal_perservice.py's machinery VERBATIM (does not import it — that module writes
recal_volatility.json at import). Same protocol: normalized-abs residuals, h0-only, warm-start q̂ =
frozen SCP on the ref/cal window, R=6. Only generalization: per-service windows from the lock
(train/cal/deploy) instead of the hardcoded MS_7129 windows. Heads/recalibrators unchanged as code.
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
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

DIR = open('/tmp/ev8b_pass2_full_dir.txt').read().strip()
P2DIR = "data/p3_runs/results/ev8b_perservice_20260601_024537"
spec = importlib.util.spec_from_file_location("p2", f"{P2DIR}/pass2_perservice.py")
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
H, K, ALPHA = p2.H, p2.K, p2.ALPHA
NOMINAL, R, COVWIN = 90.0, 6, 30

LOCK = json.load(open(f"{DIR}/pass2_full_windows.json"))
COV = json.load(open(f"{DIR}/pass2_full_coverage.json"))
GAP_THRESHOLD = 10.0   # frozen-SCP gap_h0 to trigger a recal closeout


def forecasts(u, seg_arr, mu, sigma):
    X, Y = p2.windows(seg_arr, mu, sigma)
    fc, ac = [], []
    for i in range(len(X)):
        pr = u.predict_with_uncertainty(u.norm_params.denormalize(X[i].flatten()))
        fc.append([float(pr['point_forecast'][0]), float(pr['point_forecast'][1])])
        yd = u.norm_params.denormalize(Y[i]); ac.append([float(yd[0]), float(yd[1])])
    return np.array(fc), np.array(ac)


def make_recal(kind):
    if kind == 'aci':
        return ACI(target_alpha=ALPHA, eta=0.1, residual_buffer_size=200, alpha_clip=(1e-4, 0.5))
    return ConformalPID(target_alpha=ALPHA, k_p=0.1, k_i=0.01, k_d=0.05,
                        residual_buffer_size=200, alpha_clip=(1e-4, 0.5))


def walk(method, fc, ac, sigma, q0_frozen, q1_frozen, ref_resids):
    kind = 'aci' if method == 'aci' else 'pid'
    recal = None
    if method != 'frozen':
        recal = make_recal(kind)
        for r in ref_resids[-200:]:
            recal.residuals.append(abs(float(r)))
    cov0, cov1, w0 = [], [], []
    prev = None
    for i in range(len(fc)):
        if recal is not None and prev is not None:
            recal.update(prev[0], prev[1])
        q0 = float(recal.quantile()) if recal is not None else q0_frozen
        hw0 = sigma * q0
        lo0, hi0 = max(fc[i, 0] - hw0, 0.0), max(fc[i, 0] + hw0, 0.0)
        c0 = bool(lo0 <= ac[i, 0] <= hi0)
        hw1 = sigma * q1_frozen
        c1 = bool(max(fc[i, 1] - hw1, 0.0) <= ac[i, 1] <= max(fc[i, 1] + hw1, 0.0))
        cov0.append(c0); cov1.append(c1); w0.append(2 * hw0)
        prev = (abs(ac[i, 0] - fc[i, 0]) / sigma, not c0)
    return 100 * np.mean(cov0), 100 * np.mean(cov1), float(np.mean(w0))


def recal_service(ms, train, ref, deploy):
    METHODS = ['frozen', 'aci', 'pid']
    acc = {m: {'h0': [], 'h1': [], 'w': []} for m in METHODS}
    bias_acc = []
    for rep in range(R):
        torch.manual_seed(1000 + rep); np.random.seed(1000 + rep)
        s = p2.load_series(ms)
        tr, rf, dp = p2.seg(s.values, *train), p2.seg(s.values, *ref), p2.seg(s.values, *deploy)
        mu, sigma = float(tr.mean()), float(tr.std()) or 1.0
        norm = NormalizationParams(mu, sigma)
        Xtr, Ytr = p2.windows(tr, mu, sigma); Xrf, Yrf = p2.windows(rf, mu, sigma)
        u = SplitConformal(alpha=ALPHA, h=H, k=K, device='cpu'); u.norm_params = norm
        u.fit((Xtr, Ytr), calibration_data=(Xrf, Yrf))
        q0_frozen, q1_frozen = float(u.q_hat[0]), float(u.q_hat[1])
        fc_rf, ac_rf = forecasts(u, rf, mu, sigma)
        ref_resids = np.abs(ac_rf[:, 0] - fc_rf[:, 0]) / sigma
        fc_dp, ac_dp = forecasts(u, dp, mu, sigma)
        bias_acc.append(100 * (fc_dp[:, 0].mean() - ac_dp[:, 0].mean()) / (ac_dp[:, 0].mean() + 1e-9))
        for m in METHODS:
            h0, h1, w = walk(m, fc_dp, ac_dp, sigma, q0_frozen, q1_frozen, ref_resids)
            acc[m]['h0'].append(h0); acc[m]['h1'].append(h1); acc[m]['w'].append(w)
    fw = float(np.mean(acc['frozen']['w']))
    out = dict(service=ms, windows=dict(train=train, ref=ref, deploy=deploy),
               sanity_bias_pct=round(float(np.mean(bias_acc)), 1), n_deploy=len(fc_dp), methods={})
    for m in METHODS:
        a = acc[m]
        out['methods'][m] = dict(cov_h0=[round(float(np.mean(a['h0'])), 1), round(float(np.std(a['h0'])), 1)],
                                 gap_h0=round(NOMINAL - float(np.mean(a['h0'])), 1),
                                 cov_h1_frozen=round(float(np.mean(a['h1'])), 1),
                                 mean_width_h0=round(float(np.mean(a['w'])), 4),
                                 width_x_frozen=round(float(np.mean(a['w'])) / fw, 1) if fw > 0 else None)
    return out


# pick volatility-channel services with a frozen-SCP gap, excluding the done anchor
covmap = {f"{c['service']}|{c['channel']}": c for c in COV['cells']}
targets = []
for c in LOCK['cells']:
    if c['channel'] != 'volatility' or c['service'] == 'MS_7129':
        continue
    sc = covmap.get(f"{c['service']}|volatility", {}).get('heads', {}).get('scp', {})
    if 'error' in sc or not sc:
        print(f"skip {c['service']}: no SCP coverage"); continue
    if not sc.get('sanity_pass'):
        print(f"skip {c['service']}: sanity FAIL (bias {sc.get('mean_abs_bias')}%)"); continue
    if sc.get('gap_h0', 0) < GAP_THRESHOLD:
        print(f"skip {c['service']}: frozen-SCP gap {sc.get('gap_h0')}pp < {GAP_THRESHOLD} (no gap to recover)"); continue
    targets.append(c)

print(f"\nrecal targets (vol gap ≥{GAP_THRESHOLD}pp, sanity-pass): {[c['service'] for c in targets]}")
results = []
for c in targets:
    print(f"\n=== recal {c['service']} vol (volR={c['vol_ratio']}x, train={c['train']} cal={c['cal']} dep={c['deploy']}) ===", flush=True)
    r = recal_service(c['service'], tuple(c['train']), tuple(c['cal']), tuple(c['deploy']))
    results.append(r)
    for m in ['frozen', 'aci', 'pid']:
        mm = r['methods'][m]
        print(f"  {m:6s} h0={mm['cov_h0'][0]:5.1f}±{mm['cov_h0'][1]:.1f}% gap={mm['gap_h0']:+5.1f} width×frozen={mm['width_x_frozen']}")

out = dict(anchor_ref=dict(service='MS_7129', frozen='63.0/70.1%', aci='89.8%', pid='90.4%',
                           note='from recal_volatility.json; the done n=1'),
           gap_threshold_pp=GAP_THRESHOLD, R=R, n_targets=len(targets),
           skipped='vol services with no frozen gap / sanity-fail (see console)', services=results)
with open(f"{DIR}/pass2_full_recal.json", 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {DIR}/pass2_full_recal.json ({len(results)} services recalibrated)")
