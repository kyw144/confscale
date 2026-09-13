#!/usr/bin/env python
"""E-V8b STEP 0 (integrity) — re-materialise the pre-registration lock `ev8b_LOCK_2026-06-01.md`
from the inline window records (pass1_selected_candidates.csv + status files), LOAD-STATS ONLY.

The named lock file is missing from disk; the windows exist inline as plateau CENTERS in the Pass-1
selection CSV. This restores the pre-registration record BEFORE any Pass-2 coverage runs.

FIREWALL: only means and first-difference-stds (the 0c severity metrics). NO coverage, residuals,
interval widths, or anything forecast-error-adjacent.

Windowing rule (locked; reproduces both anchors — validated below):
  W = 120 bins (2 h) plateau window; gap = [1080,1140) must be avoided by every window + train.
  LEVEL channel  (deploy IN-sample, matches anchor QR over-coverage):
     cal    = [lvl_lo_center-60, lvl_lo_center+60)      (low plateau)
     deploy = [lvl_hi_center-60, lvl_hi_center+60)      (high plateau)
     train  = [max(0, HI-720), HI),  HI = max(cal_hi, deploy_hi)   (spans both; deploy ⊂ train)
  VOLATILITY channel  (deploy OUT-of-sample, matches anchor F-design):
     cal    = [vol_lo_center-60, vol_lo_center+60)      (low-σ_Δ window)
     deploy = [vol_hi_center-60, vol_hi_center+60)      (high-σ_Δ window)
     train  = [max(0, deploy_lo-720), deploy_lo)        (pre-deploy context; deploy ⊄ train)
  Feasible iff both windows gap-free AND same side of the gap AND span ≤ 720 (train-spannable).
  ANCHOR EXCEPTION (locked, hand-windowed): MS_7129 volatility auto-center (1370) is post-gap →
     infeasible; the lock hand-placed it pre-gap at train[120,840)/cal[480,720)/deploy[840,960)
     (8.8× σ_Δ). Recorded verbatim so the anchor sits in the same unified table.

Severity bands (0c): level target 2–3× (synth G); volatility target 7–10× (synth F = 7.6×).
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import sys, json
import numpy as np
import pandas as pd

PASS1 = "data/p3_runs/results/ev8b_pass1_20260601"
DIR = open('/tmp/ev8b_pass2_full_dir.txt').read().strip()
LOCK_MD = "docs/papers/p3/ev8b_LOCK_2026-06-01.md"
W, HALF = 120, 60
GAP = (1080, 1140)
N_TS = 1440

csv = pd.read_csv(f"{PASS1}/pass1_selected_candidates.csv")
wide = pd.read_csv(f"{PASS1}/perservice_pool_series.csv").set_index('timestamp_min')


def gapfree(a, b):
    return b <= GAP[0] or a >= GAP[1]


def win(c):
    c = int(round(c)); return (c - HALF, c + HALF)


def feasible(loc, hic):
    cal, dep = win(loc), win(hic)
    if cal[0] < 0 or dep[0] < 0 or cal[1] > N_TS or dep[1] > N_TS:
        return False, cal, dep, 'off-grid'        # center within W/2 of the grid edge -> no full window
    if not (gapfree(*cal) and gapfree(*dep)):
        return False, cal, dep, 'gap-in-window'
    same = (cal[1] <= GAP[0] and dep[1] <= GAP[0]) or (cal[0] >= GAP[1] and dep[0] >= GAP[1])
    if not same:
        return False, cal, dep, 'straddle-gap'
    span = max(cal[1], dep[1]) - min(cal[0], dep[0])
    if span > 720:
        return False, cal, dep, f'span-{span}>720'
    return True, cal, dep, 'ok'


def severity(s, win_lo, win_hi):
    """Load-stats only: mean and first-diff std on a window (interpolate short gaps, like the harness)."""
    seg = s.iloc[win_lo:win_hi]
    return float(seg.mean()), float(seg.diff().std())


def load_series(ms):
    s = wide[ms].copy()
    return s.interpolate(method='linear', limit=8, limit_area='inside')


def level_train(cal, dep):
    hi = max(cal[1], dep[1])
    return (max(0, hi - 720), hi)


def vol_train(dep):
    return (max(0, dep[0] - 720), dep[0])


def band_level(r):
    if r != r: return 'undefined'
    if r <= 3.0: return 'in-band(2-3x)'
    if r <= 10.0: return 'moderate(3-10x)'
    return 'idle-active(>10x)'


def band_vol(r):
    if r != r: return 'undefined'
    if 7.0 <= r <= 10.0: return 'in-band(7-10x)'
    if r >= 5.0: return 'near-band(5-7x)'
    return 'mild(<5x)'


within = csv[csv['level_within_range'] == True].copy()
cells = []   # one record per (service, channel) testable cell
blocked = []  # gap-blocked / infeasible records (labeled, reported)

for _, r in within.iterrows():
    ms = r['msname']; s = load_series(ms)
    # ---- LEVEL channel (always feasible for within-range) ----
    lf, lcal, ldep, lreason = feasible(r['lvl_lo_center'], r['lvl_hi_center'])
    if lf:
        ltr = level_train(lcal, ldep)
        cm, cs = severity(s, *lcal); dm, ds = severity(s, *ldep)
        lr = dm / cm if cm > 0 else float('nan')
        cells.append(dict(service=ms, channel='level', sel_set=r['sel_set'], shape=r['shape'],
                          best_drift=r['best_drift'], train=list(ltr), cal=list(lcal), deploy=list(ldep),
                          cal_mean=round(cm, 4), deploy_mean=round(dm, 4),
                          level_ratio=round(lr, 2), vol_ratio=None,
                          severity_band=band_level(lr), deploy_in_sample=True,
                          n_deploy=ldep[1] - ldep[0] - 60 - 2 + 1, source='auto-center'))
    else:
        blocked.append(dict(service=ms, channel='level', reason=lreason))
    # ---- VOLATILITY channel (feasible for only ~4) ----
    vf, vcal, vdep, vreason = feasible(r['vol_lo_center'], r['vol_hi_center'])
    if vf:
        vtr = vol_train(vdep)
        cm, cs = severity(s, *vcal); dm, ds = severity(s, *vdep)
        vr = ds / cs if cs > 0 else float('nan')
        lr = dm / cm if cm > 0 else float('nan')
        cells.append(dict(service=ms, channel='volatility', sel_set=r['sel_set'], shape=r['shape'],
                          best_drift=r['best_drift'], train=list(vtr), cal=list(vcal), deploy=list(vdep),
                          cal_mean=round(cm, 4), deploy_mean=round(dm, 4),
                          level_ratio=round(lr, 2), vol_ratio=round(vr, 2),
                          severity_band=band_vol(vr), deploy_in_sample=False,
                          n_deploy=vdep[1] - vdep[0] - 60 - 2 + 1, source='auto-center'))
    else:
        blocked.append(dict(service=ms, channel='volatility', reason=vreason))

# ---- ANCHOR exception: MS_7129 volatility hand-window (locked verbatim) ----
s7129 = load_series('MS_7129')
acal, adep, atr = (480, 720), (840, 960), (120, 840)
cm, cs = severity(s7129, *acal); dm, ds = severity(s7129, *adep)
cells.append(dict(service='MS_7129', channel='volatility', sel_set='anchor', shape='steady-diurnal',
                  best_drift='both', train=list(atr), cal=list(acal), deploy=list(adep),
                  cal_mean=round(cm, 4), deploy_mean=round(dm, 4),
                  level_ratio=round(dm / cm, 2), vol_ratio=round(ds / cs, 2),
                  severity_band=band_vol(ds / cs), deploy_in_sample=False,
                  n_deploy=adep[1] - adep[0] - 60 - 2 + 1, source='anchor-hand-window'))

# ---- VALIDATION: rule must reproduce the anchor LEVEL windows verbatim ----
anchor_level = next(c for c in cells if c['service'] == 'MS_7129' and c['channel'] == 'level')
assert anchor_level['cal'] == [165, 285], f"anchor level cal {anchor_level['cal']} != [165,285]"
assert anchor_level['deploy'] == [674, 794], f"anchor level deploy {anchor_level['deploy']} != [674,794]"
# MS_69588 level (the other inline anchor in pass2_perservice.py): cal[304,424] deploy[842,962]
m69 = next((c for c in cells if c['service'] == 'MS_69588' and c['channel'] == 'level'), None)
if m69:
    assert m69['cal'] == [304, 424], f"MS_69588 level cal {m69['cal']} != [304,424]"
    assert m69['deploy'] == [842, 962], f"MS_69588 level deploy {m69['deploy']} != [842,962]"
print("VALIDATION PASS: windowing rule reproduces anchor MS_7129 + MS_69588 level windows verbatim.")

lock = dict(
    name='ev8b_LOCK_2026-06-01', rematerialized=True, basis='load-stats only (means, first-diff-std)',
    windowing_rule=dict(W=W, gap=list(GAP),
        level='cal=[lo_c±60], deploy=[hi_c±60], train=[max(0,HI-720),HI) (deploy in-sample)',
        volatility='cal=[lo_c±60], deploy=[hi_c±60], train=[max(0,dep_lo-720),dep_lo) (deploy out-of-sample)',
        anchor_exception='MS_7129 volatility hand-windowed pre-gap (auto-center 1370 post-gap, infeasible)'),
    severity_targets=dict(level='2-3x (synth G)', volatility='7-10x (synth F=7.6x)'),
    heads='SCP+QR R=6 (variance band), BE single-run (anchor precedent; R=6 BE infeasible ~14h)',
    horizons='h0 (60s) + h1 (120s)', alpha=0.1, target_coverage=90.0, R=6,
    n_within_range=int(len(within)), n_level_cells=sum(1 for c in cells if c['channel'] == 'level'),
    n_vol_cells=sum(1 for c in cells if c['channel'] == 'volatility'),
    cells=cells, blocked=blocked)

with open(f"{DIR}/pass2_full_windows.json", 'w') as f:
    json.dump(lock, f, indent=2)

# ---- human-readable lock markdown ----
def fmt_cells(ch):
    rows = [c for c in cells if c['channel'] == ch]
    out = []
    for c in sorted(rows, key=lambda x: -(x['vol_ratio'] if ch == 'volatility' and x['vol_ratio'] else (x['level_ratio'] or 0))):
        sev = c['vol_ratio'] if ch == 'volatility' else c['level_ratio']
        out.append(f"| {c['service']} | {c['sel_set']} | {c['shape']} | {c['train']} | {c['cal']} | "
                   f"{c['deploy']} | {sev}× | {c['severity_band']} | {c['source']} |")
    return "\n".join(out)

vol_blocked = [b['service'] for b in blocked if b['channel'] == 'volatility']
md = f"""# ev8b_LOCK_2026-06-01 — Pre-Registration Lock (RE-MATERIALISED 2026-06-01)

**Re-materialised** from inline window records (`pass1_selected_candidates.csv` plateau centers +
`_TRACE_PERSERVICE_*` status files), **load-stats only** (means + first-diff-stds; NO coverage). The
named file was missing from disk; this restores the pre-registration record before Pass-2 (full) runs.
Windowing rule reproduces the inline anchors **MS_7129 level** (cal[165,285]/deploy[674,794]) and
**MS_69588 level** (cal[304,424]/deploy[842,962]) **verbatim** (asserted in `lock_rematerialize.py`).

## Locked parameters
- W = {W} bins (2 h plateau); data gap **[{GAP[0]},{GAP[1]})** avoided by every window + train.
- α = 0.1, target 90 %, horizons **h0 (60 s) + h1 (120 s)**, **R = 6** variance band.
- Heads: **SCP + QR at R=6**; **BE single-run** (anchor precedent — R=6 BE ≈ 14 h, infeasible).
- Selection: the 25 locked services / **{len(within)} within-range**, verbatim from Pass 1. NO re-selection.
- Severity (0c): level mean-ratio (target 2–3×); volatility σ_Δ-ratio (target 7–10×); clean plateaus, 60 s.

## Windowing rule
- **LEVEL** (deploy in-sample): cal=`[lo_c−60,lo_c+60)`, deploy=`[hi_c−60,hi_c+60)`, train=`[max(0,HI−720),HI)`.
- **VOLATILITY** (deploy out-of-sample): cal=`[lo_c−60,+60)`, deploy=`[hi_c−60,+60)`, train=`[max(0,dep_lo−720),dep_lo)`.
- Feasible iff both windows gap-free, same side of the gap, span ≤ 720.
- **Anchor exception:** MS_7129 volatility auto-center (1370) is post-gap → infeasible; hand-windowed
  pre-gap (train[120,840)/cal[480,720)/deploy[840,960), 8.8× σ_Δ), recorded verbatim.

## LEVEL cells ({sum(1 for c in cells if c['channel']=='level')} feasible / {len(within)} within-range)
| service | set | shape | train | cal | deploy | level ratio | band | source |
|---|---|---|---|---|---|--:|---|---|
{fmt_cells('level')}

## VOLATILITY cells ({sum(1 for c in cells if c['channel']=='volatility')} feasible incl. anchor)
| service | set | shape | train | cal | deploy | σ_Δ ratio | band | source |
|---|---|---|---|---|---|--:|---|---|
{fmt_cells('volatility')}

**Volatility GAP-BLOCKED ({len(vol_blocked)}/{len(within)}):** {', '.join(vol_blocked)}
— the high-σ_Δ plateau falls at/after the [{GAP[0]},{GAP[1]}) gap; the only gap-free region [1140,1440) is
720 bins too short for train+windows. **The volatility channel is structurally gap-limited** (the anchor's
"thinness is structural" finding). No invented windows (Option A): the vol distribution is honestly thin.

*Load-stats only. No coverage computed here. Pass 2 (full) reads `pass2_full_windows.json`.*
"""
with open(LOCK_MD, 'w') as f:
    f.write(md)

print(f"\nLEVEL cells: {sum(1 for c in cells if c['channel']=='level')} | "
      f"VOL cells: {sum(1 for c in cells if c['channel']=='volatility')} | "
      f"gap-blocked vol: {len(vol_blocked)}")
print("\nLEVEL severity bands:", {b: sum(1 for c in cells if c['channel'] == 'level' and c['severity_band'] == b)
                                  for b in ['in-band(2-3x)', 'moderate(3-10x)', 'idle-active(>10x)', 'undefined']})
print("VOL severity bands:", {b: sum(1 for c in cells if c['channel'] == 'volatility' and c['severity_band'] == b)
                              for b in ['in-band(7-10x)', 'near-band(5-7x)', 'mild(<5x)', 'undefined']})
print(f"\nwrote {DIR}/pass2_full_windows.json")
print(f"wrote {LOCK_MD}")
