#!/usr/bin/env python
"""E-V8b Pass 1 — re-aggregate the EXISTING MSRTMCR ETL to PER-SERVICE granularity
and compute per-service LOAD descriptors. NO new download. NO coverage/forecasting.

FIREWALL (Pass 1): only load statistics. No coverage, interval widths, residuals, or
anything forecast-error-adjacent. Selection happens on descriptors only (separate script).

Dedup: WITHIN-service only. df.drop_duplicates() (full-row; msname is in every row so it
is inherently within-service) collapses the 81.5% exact-duplicate rows, THEN
groupby(msname,timestamp).sum() aggregates the service's distinct instances/nodes -> the
service's request rate. Never dedup across services.

Pool: to keep the wide pivot tractable and operationally meaningful (autoscaling a
0.0006-RPS service is meaningless), descriptors are computed for the TOP-K services by mean
level among >=80%-covered services. K is a structural pool size (locked on load stats), not
a coverage choice. The CoV quartile etc. are taken WITHIN this pool.

Outputs (in this dir):
  perservice_pool_series.csv     wide: timestamp_min + one col per pool service (60s grid, NaN=gap)
  perservice_descriptors.csv     one row/service: mean, range, cov, ac1, sigma_diff, suff, shape tag
  pass1_etl_summary.json
"""

# Local artifact reference entrypoint; cluster behavior is unverified.
if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read docs/MAC_VERIFICATION.md; "
                         "local demo: python -m confscale demo")

import sys, time, json, glob, tarfile
from multiprocessing import Pool
import numpy as np
import pandas as pd

DIR = "data/p3_runs/results/ev8b_pass1_20260601"
RAW = "data/alibaba_2022/raw/MSRTMCR"
TOTALS = "data/p3_runs/results/ev8_etl_build_20260531_215632/service_totals.csv"
COLS = ['timestamp', 'msname', 'msinstanceid', 'nodeid', 'providerrpc_mcr']
DTYPES = {'timestamp': 'int64', 'msname': 'object', 'msinstanceid': 'object',
          'nodeid': 'object', 'providerrpc_mcr': 'float64'}
NWORK = 20
STEP_MS = 60000
N_TS = 1440
SUFF_MIN = int(0.80 * N_TS)   # 1152 bins
POOL_K = 500                  # top-K by mean level (structural pool size)

# --- pool selection from the immutable totals (mean level = total / n_ts) ---
st = pd.read_csv(TOTALS)
st = st[st['n_ts'] >= SUFF_MIN].copy()
st['mean_level'] = st['providerrpc_mcr'] / st['n_ts']
pool = set(st.sort_values('mean_level', ascending=False).head(POOL_K)['msname'])
POOL_LEVEL_FLOOR = float(st.sort_values('mean_level', ascending=False).head(POOL_K)['mean_level'].min())
print(f"pool = top {POOL_K} by mean level among >=80%-covered; implied mean-level floor = {POOL_LEVEL_FLOOR:.4f} RPS/bin", flush=True)


def process_file(path):
    try:
        tf = tarfile.open(path, 'r:gz')
        m = tf.getmembers()[0]
        df = pd.read_csv(tf.extractfile(m), usecols=COLS, dtype=DTYPES)
        df = df.drop_duplicates()                       # within-service exact-dup collapse
        df = df[df['msname'].isin(pool)]                # keep only pool services
        g = df.groupby(['msname', 'timestamp'], sort=False)['providerrpc_mcr'].sum().reset_index()
        return g
    except Exception as e:
        return ('ERR', repr(e), path)


def main():
    files = sorted(glob.glob(f"{RAW}/MCRRTUpdate_*.tar.gz"),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    print(f"{len(files)} files, {NWORK} workers, pool={len(pool)} services", flush=True)
    t0 = time.time()
    parts, fails = [], []
    with Pool(NWORK) as pool_:
        for i, r in enumerate(pool_.imap_unordered(process_file, files), 1):
            if isinstance(r, tuple) and r and r[0] == 'ERR':
                fails.append(r[1:]); continue
            parts.append(r)
            if i % 80 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  elapsed={time.time()-t0:.0f}s", flush=True)

    big = pd.concat(parts, ignore_index=True)
    big['timestamp_min'] = big['timestamp'] // STEP_MS
    grid = pd.Index(np.arange(N_TS), name='timestamp_min')
    wide = (big.pivot_table(index='timestamp_min', columns='msname',
                            values='providerrpc_mcr', aggfunc='sum')
              .reindex(grid))
    print(f"wide pivot shape={wide.shape} elapsed={time.time()-t0:.0f}s", flush=True)
    wide.reset_index().to_csv(f"{DIR}/perservice_pool_series.csv", index=False)

    # --- per-service load descriptors (NO coverage) ---
    rows = []
    for ms in wide.columns:
        s = wide[ms]
        present = s.dropna()
        n = int(present.shape[0])
        if n < 5:
            continue
        mean = float(present.mean()); sd = float(present.std())
        mn, mx = float(present.min()), float(present.max())
        cov = sd / mean if mean > 0 else np.nan
        ac1 = float(s.autocorr(lag=1))                  # grid series, pairwise-complete
        sigdiff = float(s.diff().std())                 # THE first-diff-std metric (60s, full series)
        suff = n / N_TS
        # idle/active structure (for bimodal tag + within-range level-shift screen)
        thr = 0.10 * mx if mx > 0 else 0.0
        frac_idle = float((present <= thr).mean())
        # coarse within-day level swing: best 2h-mean window vs worst 2h-mean window (>=80% filled)
        roll = s.rolling(120, min_periods=96).mean()
        lvl_hi, lvl_lo = float(roll.max()), float(roll.min())
        lvl_ratio = lvl_hi / lvl_lo if lvl_lo and lvl_lo > 0 else np.nan
        rows.append(dict(msname=ms, mean_level=round(mean, 4), min=round(mn, 4), max=round(mx, 4),
                         dyn_range=round(mx - mn, 4), cov=round(cov, 3) if cov == cov else None,
                         ac1=round(ac1, 4) if ac1 == ac1 else None, sigma_diff=round(sigdiff, 4),
                         data_suff=round(suff, 3), frac_idle=round(frac_idle, 3),
                         lvl2h_hi=round(lvl_hi, 4), lvl2h_lo=round(lvl_lo, 4),
                         lvl2h_ratio=round(lvl_ratio, 2) if lvl_ratio == lvl_ratio else None))
    desc = pd.DataFrame(rows).sort_values('mean_level', ascending=False)

    # coarse shape tag (load-stat rules only)
    def tag(r):
        if r['cov'] is None:
            return 'na'
        if r['frac_idle'] >= 0.30 and r['cov'] >= 0.8:
            return 'bimodal-idle-active'
        if r['cov'] >= 0.8 and (r['ac1'] is None or r['ac1'] < 0.85):
            return 'bursty-spiky'
        if r['ac1'] is not None and r['ac1'] >= 0.9 and r['cov'] < 0.6:
            return 'steady-diurnal'
        return 'mixed'
    desc['shape'] = desc.apply(tag, axis=1)
    desc.to_csv(f"{DIR}/perservice_descriptors.csv", index=False)

    summary = dict(n_files=len(files), failures=fails, pool_k=POOL_K,
                   pool_level_floor=round(POOL_LEVEL_FLOOR, 4), suff_min_bins=SUFF_MIN,
                   n_descriptors=int(len(desc)), elapsed_s=round(time.time() - t0, 1),
                   shape_counts=desc['shape'].value_counts().to_dict(),
                   cov_pctiles={p: round(float(desc['cov'].dropna().quantile(p)), 3)
                                for p in [.25, .5, .75, .9]},
                   ac1_pctiles={p: round(float(desc['ac1'].dropna().quantile(p)), 3)
                                for p in [.1, .25, .5, .75]})
    with open(f"{DIR}/pass1_etl_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("DONE", json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
