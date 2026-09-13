#!/usr/bin/env python
"""E-V8b Pass 2 (FULL) — per-service coverage across the LOCKED within-range set.

Converts the +27 pp MS_7129 anchor (n=1) into a per-service gap DISTRIBUTION. Reuses the LOCKED
harness `pass2_perservice.run_test` verbatim (heads BE/SCP/QR + GRU unmodified; SCP q̂ on the explicit
cal window; h0 & h1; per-service sanity gate first). Reads the re-materialised windows from
`pass2_full_windows.json`. NO re-selection, NO invented windows (Option A).

Heads (Option A, anchor precedent): SCP + QR at R=6 (FP-nondeterminism variance band, no per-rep
re-seed — matches pass2_variance.py); BE single-run diagnostic (R=6 BE ≈ 14 h, infeasible). Phases
ordered SCP → QR → BE so the verdict-bearing SCP distribution is written first. Incremental writes
(per cell) for resumability; per-cell try/except so a data-gap in one window can't abort the sweep.

Distribution deliverable: per-channel (level / volatility, separately) min/median/max of the SCP
deploy gap + fraction under-covering by ≥10 pp, over SANITY-PASSING cells, broken out by severity band.
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import sys, json, importlib.util, time, traceback
import numpy as np
sys.path.insert(0, 'src/stage3_scale')

DIR = open('/tmp/ev8b_pass2_full_dir.txt').read().strip()
P2DIR = "data/p3_runs/results/ev8b_perservice_20260601_024537"   # locked harness lives here (orig)
spec = importlib.util.spec_from_file_location("p2", f"{P2DIR}/pass2_perservice.py")
p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)

LOCK = json.load(open(f"{DIR}/pass2_full_windows.json"))
CELLS = LOCK['cells']
R = 6
NOMINAL = 90.0
OUT = f"{DIR}/pass2_full_coverage.json"
LOG = f"{DIR}/pass2_full_coverage.log"


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + "\n")


def agg(vals):
    a = np.array(vals, dtype=float)
    return dict(mean=round(float(a.mean()), 1), sd=round(float(a.std()), 1),
                min=round(float(a.min()), 1), max=round(float(a.max()), 1), per_rep=[round(float(v), 1) for v in a])


# results keyed by "service|channel"; seed with cell metadata
results = {}
for c in CELLS:
    key = f"{c['service']}|{c['channel']}"
    results[key] = {k: c[k] for k in ('service', 'channel', 'sel_set', 'shape', 'best_drift',
                                      'train', 'cal', 'deploy', 'cal_mean', 'deploy_mean',
                                      'level_ratio', 'vol_ratio', 'severity_band', 'deploy_in_sample',
                                      'n_deploy', 'source')}
    results[key]['heads'] = {}


def write():
    with open(OUT, 'w') as f:
        json.dump(dict(lock='ev8b_LOCK_2026-06-01', R=R, nominal=NOMINAL,
                       n_level_cells=LOCK['n_level_cells'], n_vol_cells=LOCK['n_vol_cells'],
                       windowing=LOCK['windowing_rule'], cells=list(results.values()),
                       blocked=LOCK['blocked'], distribution=DISTRIB), f, indent=2)


DISTRIB = {}


def run_phase(head, reps):
    logln(f"=== PHASE {head.upper()} (R={reps}) over {len(CELLS)} cells ===")
    for i, c in enumerate(CELLS, 1):
        key = f"{c['service']}|{c['channel']}"
        acc = dict(dep_h0=[], dep_h1=[], cal_h0=[], bias=[], hw=[])
        fits = 0.0; err = None
        for r in range(reps):
            try:
                res = p2.run_test(c['service'], c['channel'], tuple(c['train']),
                                  tuple(c['cal']), tuple(c['deploy']), heads=(head,))
                hd = res['heads'][head]
                acc['dep_h0'].append(hd['deploy_cov_h0']); acc['dep_h1'].append(hd['deploy_cov_h1'])
                acc['cal_h0'].append(hd['cal_cov_h0']); acc['bias'].append(hd['sanity_bias_pct'])
                acc['hw'].append(hd['deploy_halfwidth']); fits += hd['fit_s']
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                logln(f"  ! {key} {head} rep{r} ERROR: {err}")
                break
        if err is not None or not acc['dep_h0']:
            results[key]['heads'][head] = dict(error=err or 'no-data')
            write(); continue
        rec = {k: agg(v) for k, v in acc.items() if v}
        mean_bias = float(np.mean(np.abs(acc['bias'])))
        rec['sanity_pass'] = bool(mean_bias < 10.0)
        rec['mean_abs_bias'] = round(mean_bias, 1)
        rec['gap_h0'] = round(NOMINAL - rec['dep_h0']['mean'], 1)
        rec['gap_h1'] = round(NOMINAL - rec['dep_h1']['mean'], 1)
        rec['fit_s_total'] = round(fits, 1)
        results[key]['heads'][head] = rec
        write()
        logln(f"  [{i:2d}/{len(CELLS)}] {key:26s} {head} dep_h0={rec['dep_h0']['mean']:5.1f}±{rec['dep_h0']['sd']:4.1f} "
              f"gap={rec['gap_h0']:+5.1f} cal_h0={rec['cal_h0']['mean']:5.1f} bias={rec['mean_abs_bias']:4.1f}% "
              f"{'PASS' if rec['sanity_pass'] else 'FAIL'} ({fits:.0f}s)")


def distribution():
    """Per-channel SCP gap distribution over sanity-PASSING cells + by severity band."""
    out = {}
    for ch in ['level', 'volatility']:
        cells = [r for r in results.values() if r['channel'] == ch and 'scp' in r['heads']
                 and 'error' not in r['heads']['scp']]
        passing = [r for r in cells if r['heads']['scp']['sanity_pass']]
        failed = [r for r in cells if not r['heads']['scp']['sanity_pass']]
        def gapdist(rs, hz):
            g = [r['heads']['scp'][f'gap_{hz}'] for r in rs]
            if not g: return None
            g = sorted(g)
            return dict(n=len(g), min=round(min(g), 1), median=round(float(np.median(g)), 1),
                        max=round(max(g), 1), frac_undercover_ge10pp=round(float(np.mean([x >= 10 for x in g])), 2),
                        services=[(r['service'], r['heads']['scp'][f'gap_{hz}']) for r in sorted(rs, key=lambda r: -r['heads']['scp'][f'gap_{hz}'])])
        by_band = {}
        for band in sorted(set(r['severity_band'] for r in passing)):
            bcells = [r for r in passing if r['severity_band'] == band]
            by_band[band] = gapdist(bcells, 'h0')
        out[ch] = dict(n_cells=len(cells), n_sanity_pass=len(passing), n_sanity_fail=len(failed),
                       sanity_failed_services=[(r['service'], r['heads']['scp']['mean_abs_bias']) for r in failed],
                       gap_h0_distribution=gapdist(passing, 'h0'),
                       gap_h1_distribution=gapdist(passing, 'h1'),
                       by_severity_band=by_band)
    return out


t0 = time.time()
open(LOG, 'w').close()
logln(f"Pass 2 FULL coverage start | {len(CELLS)} cells ({LOCK['n_level_cells']} level + {LOCK['n_vol_cells']} vol) | R={R}")

run_phase('scp', R)
DISTRIB = distribution(); write()
logln(f"--- SCP distribution written ({time.time()-t0:.0f}s elapsed) ---")
for ch, d in DISTRIB.items():
    gd = d['gap_h0_distribution']
    if gd:
        logln(f"  {ch:10s} SCP gap_h0 over {gd['n']} sanity-pass: min={gd['min']} median={gd['median']} "
              f"max={gd['max']} frac≥10pp={gd['frac_undercover_ge10pp']} (sanity-fail {d['n_sanity_fail']})")

run_phase('qr', R)
DISTRIB = distribution(); write()
logln(f"--- QR done ({time.time()-t0:.0f}s) ---")

run_phase('be', 1)
DISTRIB = distribution(); write()
logln(f"=== DONE ({time.time()-t0:.0f}s total). wrote {OUT} ===")
