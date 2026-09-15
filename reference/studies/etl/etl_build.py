#!/usr/bin/env python
"""Deduplicate microservice traces and aggregate request rates."""

if __name__ == "__main__":
    import os as _artifact_os
    if _artifact_os.environ.get("CONFSCALE_ENABLE_REFERENCE_RUNTIME") != "1":
        raise SystemExit("Reference runtime disabled. Read README.md#cluster-runs; "
                         "local demo: python -m confscale demo")

from pathlib import Path

import sys, time, json, glob, tarfile
from multiprocessing import Pool
import pandas as pd
import numpy as np

DIR = "generated/studies/etl"
Path(DIR).mkdir(parents=True, exist_ok=True)
RAW = "inputs/traces/MSRTMCR"
COLS = ['timestamp', 'msname', 'msinstanceid', 'nodeid', 'providerrpc_mcr', 'http_mcr']
MCR = ['providerrpc_mcr', 'http_mcr']
DTYPES = {'timestamp': 'int64', 'msname': 'object', 'msinstanceid': 'object',
          'nodeid': 'object', 'providerrpc_mcr': 'float64', 'http_mcr': 'float64'}
NWORK = 20
STEP_MS = 60000
N_TS = 1440  # 24h / 60s


def process_file(path):
    try:
        tf = tarfile.open(path, 'r:gz')
        m = tf.getmembers()[0]
        df = pd.read_csv(tf.extractfile(m), usecols=COLS, dtype=DTYPES)
        n_raw = len(df)
        df = df.drop_duplicates()          # collapse ~89% exact-duplicate rows
        n_dedup = len(df)
        g = df.groupby(['msname', 'timestamp'], sort=False)[MCR].sum().reset_index()
        return (path, n_raw, n_dedup, g)
    except Exception as e:
        return (path, -1, -1, repr(e))


def main():
    files = sorted(glob.glob(f"{RAW}/MCRRTUpdate_*.tar.gz"),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    print(f"{len(files)} files, {NWORK} workers", flush=True)
    t0 = time.time()
    parts, failures = [], []
    n_raw_tot = n_dedup_tot = 0
    done = 0
    with Pool(NWORK) as pool:
        for path, n_raw, n_dedup, g in pool.imap_unordered(process_file, files):
            done += 1
            if n_raw == -1:
                failures.append((path, g))
                print(f"FAIL {path}: {g}", flush=True)
                continue
            parts.append(g)
            n_raw_tot += n_raw; n_dedup_tot += n_dedup
            if done % 40 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files  elapsed={time.time()-t0:.0f}s", flush=True)

    big = pd.concat(parts, ignore_index=True)
    print(f"concat rows={len(big):,}  elapsed={time.time()-t0:.0f}s", flush=True)

    ca = big.groupby('timestamp')[MCR].sum().sort_index()
    full_grid = pd.Index(np.arange(0, N_TS * STEP_MS, STEP_MS), name='timestamp')
    ca = ca.reindex(full_grid)
    missing = int(ca['providerrpc_mcr'].isna().sum())
    ca_out = ca.reset_index().rename(columns={'timestamp': 'timestamp_ms'})
    ca_out.insert(1, 'timestamp_min', ca_out['timestamp_ms'] // STEP_MS)
    ca_out.to_csv(f"{DIR}/cluster_agg_series.csv", index=False)

    st = big.groupby('msname')[MCR].sum()
    st['n_ts'] = big.groupby('msname')['timestamp'].nunique()
    st = st.sort_values('providerrpc_mcr', ascending=False)
    st.reset_index().to_csv(f"{DIR}/service_totals.csv", index=False)
    topN = st.head(20).index.tolist()

    piv = (big[big.msname.isin(topN)]
           .pivot_table(index='timestamp', columns='msname',
                        values='providerrpc_mcr', aggfunc='sum')
           .reindex(full_grid))
    piv = piv[topN]  # order by total volume
    piv.reset_index().rename(columns={'timestamp': 'timestamp_ms'}).to_csv(
        f"{DIR}/topN_providerrpc_series.csv", index=False)

    summary = {
        'n_files': len(files), 'failures': failures,
        'n_raw_rows_total': int(n_raw_tot), 'n_dedup_rows_total': int(n_dedup_tot),
        'dedup_drop_frac': round(1 - n_dedup_tot / max(n_raw_tot, 1), 4),
        'n_services': int(big['msname'].nunique()),
        'grid_points': N_TS, 'missing_timestamps': missing,
        'cluster_agg_providerrpc_describe': {
            k: float(v) for k, v in ca['providerrpc_mcr'].describe().items()},
        'cluster_agg_http_describe': {
            k: float(v) for k, v in ca['http_mcr'].describe().items()},
        'top5_services': st.head(5).reset_index().to_dict('records'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(f"{DIR}/etl_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("DONE", json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
