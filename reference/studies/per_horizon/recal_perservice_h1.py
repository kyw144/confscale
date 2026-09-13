#!/usr/bin/env python
"""T1 (P3 re-open) — PER-HORIZON h1 recalibration, OFFLINE additive replay.

Converts the Table-6 / §4.5 limitation ("recalibration is h0-only; the actuator's
max-over-horizon decision is driven by the FROZEN h1 bound on a large fraction of
cycles") into a measured result, additively.

WHAT THIS DOES (and does NOT do):
  - COPIES + extends the LOCKED offline replay machinery of
        data/p3_runs/results/ev8b_perservice_20260601_024537/recal_perservice.py
        data/p3_runs/results/ev8b_perservice_20260601_024537/h1_binds_analysis.py
        data/p3_runs/results/ev8b_pass2_full_20260601_210028/pass2_full_recal.py
    verbatim (same windows, R=6 seeds, warm-start, h0-only order) and ADDS a second
    recalibrator instance per method that recalibrates the h1 quantile too.
  - Per-horizon design = a LIST of k unchanged scalar ConformalPID/ACI objects
    (Option B of the A1 memo). The classes are reused verbatim — no edit to
    conformal_pid.py / aci.py / controller.py. h1 instance is warm-started on the
    REF-window h1 residuals (so q̂1(cycle0)==frozen SCP q1, mirroring the locked h0
    warm-start) and updated on a 2-tick-lagged h1 residual stream (h1 validates two
    cycles later vs h0's one).
  - Runs BOTH modes in one pass per method:
        frozen_h1  : h0 recalibrated, h1 frozen  -> reproduces the locked baseline
        per_horizon: h0 recalibrated, h1 recalibrated  -> the T1 result
    The frozen_h1 reproduction is the faithfulness cross-check: cov_h0/width_h0 and
    the frozen-h1-binds % must match the locked recal_volatility.json / h1_binds.json
    / pass2_full_recal.json before the per_horizon numbers are trusted.

METRICS (additive; locked Table 6 preserved verbatim):
  - cov_h0, cov_h1 (cov_h1 ADAPTED under per_horizon vs frozen 72.9% on anchor)
  - width_h0, width_h1 (raw RPS full widths) + width-x-frozen multipliers
  - h1_binds_pct  = fraction of deploy cycles where ci_upper[1] >= ci_upper[0]
                    (which horizon wins the actuator's np.max). May NOT fall to zero
                    under per_horizon — recalibrating h1 WIDENS it.
  - actuator_reach_pct = fraction of cycles where the WINNING (max) bound is a
                    *recalibrated* bound. frozen_h1: = 100 - h1_binds (only h0-wins
                    cycles carry a recalibrated bound). per_horizon: = 100% by
                    construction (no frozen quantity left in the max()).

OFFLINE ONLY — torch CPU, numpy, pandas, the uq.* heads, the pre-extracted Alibaba
per-service CSV. No kubectl/kind/Prometheus. Run from repo root:
    .venv/bin/python data/p3_runs/reopen_2026-06/T1/recal_perservice_h1.py

Timescale note (A1): the Alibaba replay is on a 60s grid, so h0=60s / h1=120s here
(vs the paper's 30s/60s testbed). Horizon INDEX (k=2, one-/two-step) is identical;
do not conflate the wall-clock.
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
from baselines.escalation_ladder import EscalationLadder

# ── reuse the locked helpers verbatim (same import pattern as the originals) ──────────────────────
P2DIR = "data/p3_runs/results/ev8b_perservice_20260601_024537"
spec = importlib.util.spec_from_file_location("p2", f"{P2DIR}/pass2_perservice.py")
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
H, K, ALPHA = p2.H, p2.K, p2.ALPHA
NOMINAL, R, COVWIN = 90.0, 6, 30
OUTDIR = "data/p3_runs/reopen_2026-06/T1"

# LOCKED windows (ev8b_LOCK_2026-06-01.md) — verbatim from the three source scripts.
# n=4 is the structural [1080,1140) trace-gap cap; do not invent windows.
CELLS = [
    {"service": "MS_7129",  "train": (120, 840), "ref": (480, 720), "deploy": (840, 960),
     "anchor": True,  "src": "recal_volatility.json / h1_binds.json"},
    {"service": "MS_21558", "train": (0, 163),   "ref": (397, 517), "deploy": (163, 283),
     "anchor": False, "src": "pass2_full_recal.json"},
    {"service": "MS_7420",  "train": (119, 681), "ref": (180, 300), "deploy": (681, 801),
     "anchor": False, "src": "pass2_full_recal.json"},
    {"service": "MS_41763", "train": (92, 812),  "ref": (424, 544), "deploy": (812, 932),
     "anchor": False, "src": "pass2_full_recal.json"},
]


def forecasts(u, seg_arr, mu, sigma):
    """Per-window raw-RPS point forecasts (h0,h1) + actuals (h0,h1). Verbatim from the originals."""
    X, Y = p2.windows(seg_arr, mu, sigma)
    fc, ac = [], []
    for i in range(len(X)):
        p = u.predict_with_uncertainty(u.norm_params.denormalize(X[i].flatten()))
        fc.append([float(p['point_forecast'][0]), float(p['point_forecast'][1])])
        yd = u.norm_params.denormalize(Y[i])
        ac.append([float(yd[0]), float(yd[1])])
    return np.array(fc), np.array(ac)


def make_recal(kind):
    """methods.py defaults — verbatim from the originals."""
    if kind == 'aci':
        return ACI(target_alpha=ALPHA, eta=0.1, residual_buffer_size=200, alpha_clip=(1e-4, 0.5))
    return ConformalPID(target_alpha=ALPHA, k_p=0.1, k_i=0.01, k_d=0.05,
                        residual_buffer_size=200, alpha_clip=(1e-4, 0.5))


def warm_start(recal, ref_resids):
    """Prime the buffer with REF-window residuals so quantile()==frozen SCP q̂ at cycle 0.
    Identical mechanism to the locked h0 warm-start (recal_perservice.py:66-67)."""
    for r in ref_resids[-200:]:
        recal.residuals.append(abs(float(r)))


def walk(method, per_horizon, fc, ac, sigma, q0_frozen, q1_frozen, ref_resids0, ref_resids1):
    """Offline recalibration walk over the deploy windows.

    method      : 'frozen' | 'aci' | 'pid' | 'aci-lad' | 'pid-lad'
    per_horizon : False -> h1 frozen (reproduces the locked baseline)
                  True  -> h1 also recalibrated (the T1 result)

    h0 path is BIT-IDENTICAL to the locked walk (recal0, 1-lag, warm-start, ladder on
    h0 trailing coverage). The h1 path adds recal1 (2-lag, warm-start) when per_horizon.
    For the laddered variant the ladder multiplier is applied to q1 too ONLY under
    per_horizon (A1 §5.2 symmetry); the ladder TRIGGER stays the h0 coverage monitor.
    """
    laddered = method.endswith('-lad')
    kind = 'aci' if method.startswith('aci') else 'pid'
    is_recal = (method != 'frozen')

    recal0 = None
    recal1 = None
    if is_recal:
        recal0 = make_recal(kind); warm_start(recal0, ref_resids0)
        if per_horizon:
            recal1 = make_recal(kind); warm_start(recal1, ref_resids1)
    ladder = EscalationLadder(target_coverage=0.9, escalation_band=0.05, escalation_persistence=5,
                              recovery_persistence=10, widening_factor=1.5, conservative_factor=3.0) if laddered else None
    mon = deque(maxlen=COVWIN)        # h0 trailing-coverage monitor (controller is h0-only)
    prev0 = None                      # (residual, miss) for h0 from the PRIOR cycle (1-lag)
    prev1_buf = deque(maxlen=2)       # (residual, miss) queue for h1 (2-lag: validates 2 cycles later)

    cov0, cov1, w0, w1 = [], [], [], []
    up0, up1 = [], []                 # ci_upper[0], ci_upper[1] the actuator's np.max arbitrates
    q0_used, q1_used = [], []
    levels = []
    for i in range(len(fc)):
        # ── h0: update recalibrator on PRIOR step, then set q̂0 (controller §1c) ──
        if recal0 is not None and prev0 is not None:
            recal0.update(prev0[0], prev0[1])
        q0 = float(recal0.quantile()) if recal0 is not None else q0_frozen
        # ── h1: update on the 2-tick-lagged step, then set q̂1 ──
        if recal1 is not None and len(prev1_buf) == 2:
            r1, m1 = prev1_buf[0]                       # residual/miss from cycle i-2
            recal1.update(r1, m1)
        if recal1 is not None:
            q1 = float(recal1.quantile())
        else:
            q1 = q1_frozen
        # ── ladder on the h0 trailing coverage (controller §1d) ──
        if ladder is not None:
            tc = (sum(mon) / len(mon)) if mon else 1.0
            ladder.step(tc); levels.append(ladder.level)
            q0 = float(ladder.apply(q0))
            if per_horizon:                              # A1 §5.2: widen every horizon symmetrically
                q1 = float(ladder.apply(q1))
        # ── form intervals ──
        hw0 = sigma * q0
        hw1 = sigma * q1
        lo0, hi0 = max(fc[i, 0] - hw0, 0.0), max(fc[i, 0] + hw0, 0.0)
        lo1, hi1 = max(fc[i, 1] - hw1, 0.0), max(fc[i, 1] + hw1, 0.0)
        c0 = bool(lo0 <= ac[i, 0] <= hi0)
        c1 = bool(lo1 <= ac[i, 1] <= hi1)
        cov0.append(c0); cov1.append(c1); w0.append(2 * hw0); w1.append(2 * hw1)
        up0.append(hi0); up1.append(hi1); q0_used.append(q0); q1_used.append(q1)
        # ── feedback bookkeeping (miss computed against the FINAL interval, as the originals do) ──
        mon.append(c0)
        prev0 = (abs(ac[i, 0] - fc[i, 0]) / sigma, not c0)
        prev1_buf.append((abs(ac[i, 1] - fc[i, 1]) / sigma, not c1))

    up0, up1 = np.array(up0), np.array(up1)
    h1_binds = (up1 >= up0)                              # which horizon wins the actuator's np.max
    h0_is_recal = is_recal
    h1_is_recal = is_recal and per_horizon
    # winning bound is recalibrated iff: (h1 wins & h1 recal) or (h0 wins & h0 recal)
    reach = (h1_binds & h1_is_recal) | (~h1_binds & h0_is_recal)
    return dict(
        cov_h0=100 * float(np.mean(cov0)), cov_h1=100 * float(np.mean(cov1)),
        width_h0=float(np.mean(w0)), width_h1=float(np.mean(w1)),
        h1_binds_pct=100 * float(np.mean(h1_binds)),
        reach_pct=100 * float(np.mean(reach)),
        max_level=max(levels) if levels else 0,
        mean_q0=float(np.mean(q0_used)), mean_q1=float(np.mean(q1_used)),
        mean_up0=float(up0.mean()), mean_up1=float(up1.mean()))


def run_cell(cell, methods):
    """R=6 replication for one locked cell; returns per-method frozen_h1 vs per_horizon summary."""
    TRAIN, REF, DEPLOY = cell['train'], cell['ref'], cell['deploy']
    modes = ['frozen_h1', 'per_horizon']
    acc = {m: {md: {k: [] for k in ('cov_h0', 'cov_h1', 'width_h0', 'width_h1',
                                     'h1_binds', 'reach', 'maxlvl', 'q0', 'q1', 'up0', 'up1')}
               for md in modes} for m in methods}
    frozen_w0, frozen_w1 = [], []      # frozen-method widths per rep (for x-frozen multipliers)
    bias_acc = []
    for rep in range(R):
        torch.manual_seed(1000 + rep); np.random.seed(1000 + rep)     # SAME seeds as the originals
        s = p2.load_series(cell['service'])
        tr, rf, dp = p2.seg(s.values, *TRAIN), p2.seg(s.values, *REF), p2.seg(s.values, *DEPLOY)
        mu, sigma = float(tr.mean()), float(tr.std()) or 1.0
        norm = NormalizationParams(mu, sigma)
        Xtr, Ytr = p2.windows(tr, mu, sigma)
        Xrf, Yrf = p2.windows(rf, mu, sigma)
        u = SplitConformal(alpha=ALPHA, h=H, k=K, device='cpu'); u.norm_params = norm
        u.fit((Xtr, Ytr), calibration_data=(Xrf, Yrf))
        q0_frozen, q1_frozen = float(u.q_hat[0]), float(u.q_hat[1])
        fc_rf, ac_rf = forecasts(u, rf, mu, sigma)
        ref_resids0 = np.abs(ac_rf[:, 0] - fc_rf[:, 0]) / sigma       # h0 warm-start (locked)
        ref_resids1 = np.abs(ac_rf[:, 1] - fc_rf[:, 1]) / sigma       # h1 warm-start (NEW)
        fc_dp, ac_dp = forecasts(u, dp, mu, sigma)
        bias_acc.append(100 * (fc_dp[:, 0].mean() - ac_dp[:, 0].mean()) / (ac_dp[:, 0].mean() + 1e-9))
        # frozen-method widths (mode-invariant) for x-frozen multipliers
        fz = walk('frozen', False, fc_dp, ac_dp, sigma, q0_frozen, q1_frozen, ref_resids0, ref_resids1)
        frozen_w0.append(fz['width_h0']); frozen_w1.append(fz['width_h1'])
        for m in methods:
            for md in modes:
                w = walk(m, md == 'per_horizon', fc_dp, ac_dp, sigma,
                         q0_frozen, q1_frozen, ref_resids0, ref_resids1)
                a = acc[m][md]
                a['cov_h0'].append(w['cov_h0']); a['cov_h1'].append(w['cov_h1'])
                a['width_h0'].append(w['width_h0']); a['width_h1'].append(w['width_h1'])
                a['h1_binds'].append(w['h1_binds_pct']); a['reach'].append(w['reach_pct'])
                a['maxlvl'].append(w['max_level'])
                a['q0'].append(w['mean_q0']); a['q1'].append(w['mean_q1'])
                a['up0'].append(w['mean_up0']); a['up1'].append(w['mean_up1'])
    fw0, fw1 = float(np.mean(frozen_w0)), float(np.mean(frozen_w1))

    def ms(xs, nd=1):
        return [round(float(np.mean(xs)), nd), round(float(np.std(xs)), nd)]

    out = {'service': cell['service'], 'anchor': cell['anchor'], 'baseline_src': cell['src'],
           'windows': {'train': list(TRAIN), 'ref': list(REF), 'deploy': list(DEPLOY)},
           'R': R, 'n_deploy': len(fc_dp), 'sanity_bias_pct': round(float(np.mean(bias_acc)), 1),
           'frozen_width_h0': round(fw0, 4), 'frozen_width_h1': round(fw1, 4),
           'methods': {}}
    for m in methods:
        rec = {}
        for md in ['frozen_h1', 'per_horizon']:
            a = acc[m][md]
            rec[md] = {
                'cov_h0': ms(a['cov_h0']), 'gap_h0': round(NOMINAL - float(np.mean(a['cov_h0'])), 1),
                'cov_h1': ms(a['cov_h1']), 'gap_h1': round(NOMINAL - float(np.mean(a['cov_h1'])), 1),
                'width_h0': round(float(np.mean(a['width_h0'])), 4),
                'width_h1': round(float(np.mean(a['width_h1'])), 4),
                'width_h0_x_frozen': round(float(np.mean(a['width_h0'])) / fw0, 1) if fw0 > 0 else None,
                'width_h1_x_frozen': round(float(np.mean(a['width_h1'])) / fw1, 1) if fw1 > 0 else None,
                'h1_binds_pct': ms(a['h1_binds']),
                'h1_binds_per_rep': [round(x, 1) for x in a['h1_binds']],
                'actuator_reach_pct': ms(a['reach']),
                'mean_q0_norm': round(float(np.mean(a['q0'])), 4),
                'mean_q1_norm': round(float(np.mean(a['q1'])), 4),
                'mean_ci_upper0': round(float(np.mean(a['up0'])), 4),
                'mean_ci_upper1': round(float(np.mean(a['up1'])), 4),
                'max_ladder_level': int(np.max(a['maxlvl'])) if a['maxlvl'] else 0,
            }
        out['methods'][m] = rec
    return out


if __name__ == '__main__':
    print(f"T1 per-horizon h1 recalibration | R={R} | nominal {NOMINAL}% | h0=60s h1=120s grid\n")
    results = []
    for cell in CELLS:
        methods = ['frozen', 'aci', 'pid', 'aci-lad', 'pid-lad'] if cell['anchor'] else ['frozen', 'aci', 'pid']
        print(f"=== {cell['service']} {'(ANCHOR)' if cell['anchor'] else ''} "
              f"train={cell['train']} ref={cell['ref']} deploy={cell['deploy']} ===", flush=True)
        r = run_cell(cell, methods)
        results.append(r)
        print(f"  sanity bias {r['sanity_bias_pct']}% | n_deploy={r['n_deploy']} | "
              f"frozen widths h0={r['frozen_width_h0']} h1={r['frozen_width_h1']}")
        for m in methods:
            fz = r['methods'][m]['frozen_h1']; ph = r['methods'][m]['per_horizon']
            print(f"  {m:8s} FROZEN-h1 : cov_h0={fz['cov_h0'][0]:5.1f} cov_h1={fz['cov_h1'][0]:5.1f} "
                  f"w_h1={fz['width_h1']:.4f}(x{fz['width_h1_x_frozen']}) "
                  f"h1binds={fz['h1_binds_pct'][0]:5.1f}% reach={fz['actuator_reach_pct'][0]:5.1f}%")
            print(f"  {m:8s} PER-HORIZ : cov_h0={ph['cov_h0'][0]:5.1f} cov_h1={ph['cov_h1'][0]:5.1f} "
                  f"w_h1={ph['width_h1']:.4f}(x{ph['width_h1_x_frozen']}) "
                  f"h1binds={ph['h1_binds_pct'][0]:5.1f}% reach={ph['actuator_reach_pct'][0]:5.1f}%")
        print(flush=True)

    payload = {
        'task': 'T1 per-horizon h1 recalibration (P3 re-open, additive offline replay)',
        'design': 'per-horizon recalibrator instances (list of k scalar ConformalPID/ACI); classes reused verbatim',
        'modes': {'frozen_h1': 'h0 recal, h1 frozen (reproduces locked baseline)',
                  'per_horizon': 'h0 + h1 recal (warm-started, 2-lag h1; ladder applied per-horizon)'},
        'metric_defs': {
            'h1_binds_pct': 'fraction of deploy cycles where ci_upper[1] >= ci_upper[0] (which horizon wins np.max)',
            'actuator_reach_pct': 'fraction of cycles where the winning (max) bound is a RECALIBRATED bound; '
                                  'frozen_h1 = 100 - h1_binds; per_horizon = 100 by construction',
            'width_x_frozen': 'mean interval full-width / frozen-SCP full-width (per horizon)'},
        'timescale_note': 'Alibaba 60s grid -> h0=60s h1=120s (vs paper 30s/60s testbed); horizon index identical',
        'baselines_reproduced': 'frozen_h1 mode cross-checks locked h1_binds.json / recal_volatility.json / pass2_full_recal.json',
        'R': R, 'nominal_pct': NOMINAL, 'cells': results,
    }
    with open(f"{OUTDIR}/recal_perservice_h1.json", 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUTDIR}/recal_perservice_h1.json")
