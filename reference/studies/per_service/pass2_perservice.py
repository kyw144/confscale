#!/usr/bin/env python
"""E-V8b Pass 2 — per-service offline coverage on the LOCKED config (ev8b_LOCK_2026-06-01.md).

Reuses the E-V8 v2 designB machinery (heads BE/SCP/QR + GRU UNMODIFIED). The only changes the
lock allows: (1) per-service / per-bucket series instead of cluster-aggregate; (2) explicit
calibration window for SCP (q̂ on the low plateau, not the internal tail); (3) report h0 AND h1;
(4) sanity gate FIRST per service.

Two distinct analyses (do not conflate):
  A. MATCHED drift test (verdict-bearing) — headline services, locked windows, severity-matched.
  B. coverage-vs-N aggregation ladder (characterization) — natural diurnal signal, no injected contrast.

Run ONCE against the lock. No window/service iteration after coverage is seen.
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/stage3_scale')
import torch
torch.manual_seed(42); np.random.seed(42)
from predictor.data import NormalizationParams
from uq.bootstrap import BootstrapEnsemble
from uq.conformal import SplitConformal
from uq.quantile import QuantileRegressor
from uq.evaluate import evaluate_uq_method

PASS1 = "data/p3_runs/results/ev8b_pass1_20260601"
EV8 = "data/p3_runs/results/ev8_etl_build_20260531_215632"
H, K, ALPHA = 60, 2, 0.1
NOMINAL = (1 - ALPHA) * 100
GAP = (1080, 1140)   # global 1 h data gap — all windows must avoid it

_wide = pd.read_csv(f"{PASS1}/perservice_pool_series.csv").set_index('timestamp_min')


def load_series(ms):
    """Per-service series on the 1440 grid; interpolate SHORT internal gaps, leave the big gap NaN."""
    s = _wide[ms].copy()
    s = s.interpolate(method='linear', limit=8, limit_area='inside')  # small gaps only
    return s


def seg(arr, lo, hi):
    v = np.asarray(arr[lo:hi], dtype=np.float32)
    assert np.isfinite(v).all(), f"NaN/gap in [{lo},{hi}) — pick a clean window"
    return v


def windows(x, mu, sigma):
    xn = (x - mu) / sigma
    n = len(xn) - H - K + 1
    X = np.array([xn[i:i + H] for i in range(n)], dtype=np.float32).reshape(-1, H, 1)
    Y = np.array([xn[i + H:i + H + K] for i in range(n)], dtype=np.float32)
    return X, Y


def make(name, norm, Xtr, Ytr, cal=None):
    if name == 'scp':
        u = SplitConformal(alpha=ALPHA, h=H, k=K, device='cpu')
    elif name == 'be':
        u = BootstrapEnsemble(B=10, alpha=ALPHA, h=H, k=K, device='cpu', seed=42)
        u.epochs, u.patience = 200, 20
    elif name == 'qr':
        u = QuantileRegressor(alpha=ALPHA, h=H, k=K, device='cpu', seed=42)
        u.epochs, u.patience = 200, 20
    u.norm_params = norm
    if name == 'scp' and cal is not None:
        u.fit((Xtr, Ytr), calibration_data=cal)   # q̂ on the explicit low plateau
    else:
        u.fit((Xtr, Ytr))
    return u


def diag(u, X, Y):
    """h0 forecast/actual (sanity gate) + interval half-width (raw RPS)."""
    yr = np.array([u.norm_params.denormalize(Y[i]) for i in range(len(Y))])[:, 0]
    fc, lo, hi = [], [], []
    for i in range(len(X)):
        p = u.predict_with_uncertainty(u.norm_params.denormalize(X[i].flatten()))
        fc.append(p['point_forecast'][0]); lo.append(p['ci_lower'][0]); hi.append(p['ci_upper'][0])
    fc, yr, lo, hi = map(np.array, (fc, yr, lo, hi))
    bias = float((fc.mean() - yr.mean()) / (yr.mean() + 1e-9))
    return dict(forecast=round(float(fc.mean()), 4), actual=round(float(yr.mean()), 4),
                bias_pct=round(100 * bias, 1), halfwidth=round(float(((hi - lo) / 2).mean()), 4))


def run_test(ms, label, train_win, cal_win, deploy_win, heads=('be', 'scp', 'qr')):
    """One (service, drift) test. Returns per-head sanity + h0/h1 coverage on deploy + cal-window ref."""
    s = load_series(ms)
    tr = seg(s.values, *train_win)
    cl = seg(s.values, *cal_win)
    dp = seg(s.values, *deploy_win)
    mu, sigma = float(tr.mean()), float(tr.std()) or 1.0
    norm = NormalizationParams(mu, sigma)
    Xtr, Ytr = windows(tr, mu, sigma)
    Xcl, Ycl = windows(cl, mu, sigma)
    Xdp, Ydp = windows(dp, mu, sigma)
    lvl_ratio = float(dp.mean() / (cl.mean() + 1e-9))
    vol_ratio = float(np.diff(dp).std() / (np.diff(cl).std() + 1e-9))
    out = dict(service=ms, test=label, train=train_win, cal=cal_win, deploy=deploy_win,
               cal_mean=round(float(cl.mean()), 4), deploy_mean=round(float(dp.mean()), 4),
               level_ratio=round(lvl_ratio, 2), vol_ratio_sigdiff=round(vol_ratio, 2),
               n_train=len(Xtr), n_cal=len(Xcl), n_deploy=len(Xdp), heads={})
    for name in heads:
        t0 = time.time()
        u = make(name, norm, Xtr, Ytr, cal=(Xcl, Ycl))
        rdep = evaluate_uq_method(u, (Xdp, Ydp))
        rcal = evaluate_uq_method(u, (Xcl, Ycl))
        d = diag(u, Xdp, Ydp)
        ph = rdep['per_horizon']
        out['heads'][name] = dict(
            deploy_cov_h0=round(ph['horizon_0']['coverage_pct'], 1),
            deploy_cov_h1=round(ph['horizon_1']['coverage_pct'], 1),
            deploy_gap_h0=round(NOMINAL - ph['horizon_0']['coverage_pct'], 1),
            deploy_gap_h1=round(NOMINAL - ph['horizon_1']['coverage_pct'], 1),
            cal_cov_h0=round(rcal['per_horizon']['horizon_0']['coverage_pct'], 1),
            sanity_bias_pct=d['bias_pct'], sanity_pass=bool(abs(d['bias_pct']) < 10.0),
            deploy_halfwidth=d['halfwidth'], forecast=d['forecast'], actual=d['actual'],
            fit_s=round(time.time() - t0, 1))
    return out


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else 'headline'
    DIR = open('/tmp/ev8b_pass2_dir.txt').read().strip()
    results = []
    if only in ('headline', 'all'):
        # MS_7129 — primary dual exemplar (lock windows verbatim)
        results.append(run_test('MS_7129', 'level_3.06x',
                                (120, 840), (165, 285), (674, 794)))
        results.append(run_test('MS_7129', 'volatility_~10x',
                                (120, 840), (480, 720), (840, 960)))
        # MS_69588 — conditional (level only; sanity gate decides headline vs exploratory)
        results.append(run_test('MS_69588', 'level_5.62x_conditional',
                                (120, 960), (304, 424), (842, 962)))
    for r in results:
        print(f"\n=== {r['service']} {r['test']} | level={r['level_ratio']}x vol={r['vol_ratio_sigdiff']}x "
              f"cal_mean={r['cal_mean']} deploy_mean={r['deploy_mean']} (n_dep={r['n_deploy']}) ===")
        for nm, h in r['heads'].items():
            print(f"  {nm.upper():3s} deploy h0={h['deploy_cov_h0']:5.1f}% (gap {h['deploy_gap_h0']:+5.1f}) "
                  f"h1={h['deploy_cov_h1']:5.1f}% | cal h0={h['cal_cov_h0']:5.1f}% | "
                  f"sanity bias={h['sanity_bias_pct']:+5.1f}% {'PASS' if h['sanity_pass'] else 'FAIL'} | "
                  f"hw={h['deploy_halfwidth']} | {h['fit_s']}s")
    with open(f"{DIR}/pass2_{only}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {DIR}/pass2_{only}.json")
