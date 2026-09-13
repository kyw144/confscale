#!/usr/bin/env python3
"""
_STATUS.md generator for the hpa-qr-monitored × {F,G,H} drift batch.

Pulls hpa-qr-monitored cells from the new drift output dir and
confscale-pid cells from the 2026-05-23 overnight drift batch, joins
per pattern, and produces the side-by-side coverage + cost tables
plus G4 verdicts the D2 brief asks for.

Run:
  python hpa_qr_drift_status.py \
      --new-dir data/p3_runs/outputs/hpa_qr_drift_20260525 \
      --pid-dir data/p3_runs/outputs/drift_injection_e1_20260523_020148 \
      --out data/p3_runs/outputs/hpa_qr_drift_20260525/_STATUS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


def _load_cell(cell_dir: Path) -> dict | None:
    """Return per-cell metrics dict or None if the cell is unparseable."""
    metrics_path = cell_dir / 'metrics.json'
    op_summary_path = cell_dir / 'operator_metrics_summary.json'
    if not metrics_path.exists() and not op_summary_path.exists():
        return None
    metrics = {}
    op_summary = {}
    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
    except Exception:
        pass
    try:
        if op_summary_path.exists():
            op_summary = json.loads(op_summary_path.read_text())
    except Exception:
        pass

    resources = metrics.get('resources', {}) if isinstance(metrics, dict) else {}
    cov = op_summary.get('coverage_monitor', {}) if isinstance(op_summary, dict) else {}
    return {
        'run_dir': cell_dir.name,
        'overhead_replica_seconds': resources.get('overhead_replica_seconds'),
        'mean_replicas': resources.get('mean_replicas'),
        'max_replicas': resources.get('max_replicas'),
        'coverage_rate': cov.get('coverage_rate') if isinstance(cov, dict) else None,
        'total_validated': cov.get('total_validated') if isinstance(cov, dict) else None,
        'final_alert_state': cov.get('final_alert_state') if isinstance(cov, dict) else None,
    }


def _scan_method_pattern(root: Path, method: str, pattern: str) -> list[dict]:
    """Find all cells for (method, pattern) under root (recursive 2 levels).

    Matches cell dirs named like '<method>_<patternlower>_rep<n>_*'.
    Handles both flat layouts (root/<cell>) and pattern-bucketed layouts
    (root/<pattern_bucket>/<cell>) — e.g. the 2026-05-23 drift batch
    has F_phase1/F_phase2/G/H subdirs.
    """
    needle = f"{method}_{pattern.lower()}_rep"
    cells: list[dict] = []
    if not root.is_dir():
        return cells

    subdirs = [root] + [d for d in root.iterdir() if d.is_dir()]
    for sub in subdirs:
        for cell in sub.iterdir():
            if cell.is_dir() and cell.name.startswith(needle):
                row = _load_cell(cell)
                if row is not None:
                    row['method'] = method
                    row['pattern'] = pattern
                    cells.append(row)
    # De-dup by run_dir (a cell could match under both root and a subdir)
    seen = set()
    deduped = []
    for c in cells:
        if c['run_dir'] in seen:
            continue
        seen.add(c['run_dir'])
        deduped.append(c)
    deduped.sort(key=lambda c: c['run_dir'])
    return deduped


def _fmt(v, fmt='{:.4f}'):
    if v is None:
        return 'n/a'
    try:
        if isinstance(v, float):
            return fmt.format(v)
        return str(v)
    except Exception:
        return str(v)


def _mean_std(values: list[float]) -> tuple[float | None, float | None, int]:
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return None, None, 0
    m = mean(clean)
    s = stdev(clean) if n >= 2 else 0.0
    return m, s, n


def _coverage_table(cells_by_pattern: dict[str, list[dict]]) -> str:
    """Per-pattern coverage table for one method."""
    lines = ['| pattern | rep1 cov | rep2 | rep3 | mean ± std (n) |',
             '|---------|----------|------|------|-----------------|']
    for pat in ['F', 'G', 'H']:
        cells = cells_by_pattern.get(pat, [])
        covs = [c['coverage_rate'] for c in cells]
        # pad up to 3
        padded = covs + [None] * (3 - len(covs))
        m, s, n = _mean_std(covs)
        mean_str = f'{m:.4f} ± {s:.4f} (n={n})' if m is not None else 'n/a'
        lines.append(f'| {pat} | {_fmt(padded[0])} | {_fmt(padded[1])} | {_fmt(padded[2])} | {mean_str} |')
    return '\n'.join(lines)


def _cost_table(cells_by_pattern: dict[str, list[dict]]) -> str:
    """Per-pattern overhead-replica-seconds table for one method."""
    lines = ['| pattern | rep1 overhead_rs | rep2 | rep3 | mean ± std (n) |',
             '|---------|------------------|------|------|-----------------|']
    for pat in ['F', 'G', 'H']:
        cells = cells_by_pattern.get(pat, [])
        costs = [c['overhead_replica_seconds'] for c in cells]
        padded = costs + [None] * (3 - len(costs))
        m, s, n = _mean_std(costs)
        mean_str = f'{m:.1f} ± {s:.1f} (n={n})' if m is not None else 'n/a'
        lines.append(f'| {pat} | {_fmt(padded[0], "{:.1f}")} | {_fmt(padded[1], "{:.1f}")} | {_fmt(padded[2], "{:.1f}")} | {mean_str} |')
    return '\n'.join(lines)


def _side_by_side(qr_by_pattern, pid_by_pattern) -> str:
    lines = ['| pattern | qr coverage (mean ± std, n) | pid coverage (mean ± std, n) | qr−pid | qr cost rs (mean) | pid cost rs (mean) |',
             '|---------|----------------------------|------------------------------|--------|-------------------|---------------------|']
    for pat in ['F', 'G', 'H']:
        qr = qr_by_pattern.get(pat, [])
        pid = pid_by_pattern.get(pat, [])
        qm, qs, qn = _mean_std([c['coverage_rate'] for c in qr])
        pm, ps, pn = _mean_std([c['coverage_rate'] for c in pid])
        qcm, _, _ = _mean_std([c['overhead_replica_seconds'] for c in qr])
        pcm, _, _ = _mean_std([c['overhead_replica_seconds'] for c in pid])
        qstr = f'{qm:.4f} ± {qs:.4f} (n={qn})' if qm is not None else 'n/a'
        pstr = f'{pm:.4f} ± {ps:.4f} (n={pn})' if pm is not None else 'n/a'
        diff = f'{qm - pm:+.4f}' if (qm is not None and pm is not None) else 'n/a'
        lines.append(f'| {pat} | {qstr} | {pstr} | {diff} | {_fmt(qcm, "{:.1f}")} | {_fmt(pcm, "{:.1f}")} |')
    return '\n'.join(lines)


def _g4_verdict(qr_by_pattern, pid_by_pattern) -> tuple[str, dict]:
    """G4: qr coverage on ≥2 of F/G/H is ≥0.10 below pid coverage."""
    deltas = {}
    for pat in ['F', 'G', 'H']:
        qm, _, qn = _mean_std([c['coverage_rate'] for c in qr_by_pattern.get(pat, [])])
        pm, _, pn = _mean_std([c['coverage_rate'] for c in pid_by_pattern.get(pat, [])])
        deltas[pat] = {
            'qr_mean': qm,
            'pid_mean': pm,
            'qr_minus_pid': (qm - pm) if (qm is not None and pm is not None) else None,
            'qr_n': qn,
            'pid_n': pn,
        }
    n_passing = sum(
        1 for pat, d in deltas.items()
        if d['qr_minus_pid'] is not None and d['qr_minus_pid'] <= -0.10
    )
    verdict = 'PASS' if n_passing >= 2 else 'FAIL'
    return verdict, deltas


def _g3_verdict(qr_by_pattern) -> tuple[str, dict]:
    """G3: mean_replicas > 1 on every cell."""
    per_cell = {}
    all_pass = True
    for pat in ['F', 'G', 'H']:
        per_cell[pat] = []
        for c in qr_by_pattern.get(pat, []):
            mr = c['mean_replicas']
            per_cell[pat].append({'run_dir': c['run_dir'], 'mean_replicas': mr})
            if mr is None or mr <= 1.0:
                all_pass = False
    return ('PASS' if all_pass else 'FAIL'), per_cell


def _g2_verdict(qr_by_pattern) -> tuple[str, dict]:
    """G2: each cell emits coverage_monitor.coverage_rate."""
    per_cell = {}
    all_pass = True
    for pat in ['F', 'G', 'H']:
        per_cell[pat] = []
        for c in qr_by_pattern.get(pat, []):
            cr = c['coverage_rate']
            per_cell[pat].append({'run_dir': c['run_dir'], 'coverage_rate': cr})
            if cr is None:
                all_pass = False
    return ('PASS' if all_pass else 'FAIL'), per_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-dir', type=Path, required=True,
                    help='hpa-qr-monitored drift output dir (flat or with pattern subdirs)')
    ap.add_argument('--pid-dir', type=Path, required=True,
                    help='Existing confscale-pid drift batch root '
                         '(e.g. drift_injection_e1_20260523_020148)')
    ap.add_argument('--out', type=Path, required=True, help='Path to _STATUS.md')
    args = ap.parse_args()

    qr_by_pattern = {p: _scan_method_pattern(args.new_dir, 'hpa-qr-monitored', p)
                     for p in ['F', 'G', 'H']}
    pid_by_pattern = {p: _scan_method_pattern(args.pid_dir, 'confscale-pid', p)
                      for p in ['F', 'G', 'H']}

    total_qr_cells = sum(len(v) for v in qr_by_pattern.values())
    qr_with_cov = sum(1 for v in qr_by_pattern.values() for c in v
                      if c['coverage_rate'] is not None)
    qr_with_mr = sum(1 for v in qr_by_pattern.values() for c in v
                     if c['mean_replicas'] is not None)

    g2, g2_detail = _g2_verdict(qr_by_pattern)
    g3, g3_detail = _g3_verdict(qr_by_pattern)
    g4, g4_detail = _g4_verdict(qr_by_pattern, pid_by_pattern)

    lines = [
        '# hpa-qr-monitored × {F,G,H} drift — _STATUS.md',
        '',
        f'**Run dir:** `{args.new_dir}`',
        f'**Comparator:** `{args.pid_dir}` (confscale-pid F/G/H, n=3 per pattern from 2026-05-23 overnight batch)',
        '',
        '## 1. Cell counts',
        '',
        f'- hpa-qr-monitored cells found: **{total_qr_cells}/9**',
        f'  - F: {len(qr_by_pattern["F"])}, G: {len(qr_by_pattern["G"])}, H: {len(qr_by_pattern["H"])}',
        f'- Cells with `coverage_monitor.coverage_rate`: {qr_with_cov}/{total_qr_cells}',
        f'- Cells with `resources.mean_replicas`: {qr_with_mr}/{total_qr_cells}',
        '',
        '## 2. Coverage table (hpa-qr-monitored)',
        '',
        _coverage_table(qr_by_pattern),
        '',
        '## 3. Cost table (hpa-qr-monitored, overhead_replica_seconds)',
        '',
        _cost_table(qr_by_pattern),
        '',
        '## 4. Side-by-side: hpa-qr-monitored vs confscale-pid',
        '',
        _side_by_side(qr_by_pattern, pid_by_pattern),
        '',
        '## 5. Pre-registered acceptance criteria',
        '',
        f'- **G1** (9/9 cells complete): **{"PASS" if total_qr_cells == 9 else "FAIL"}** ({total_qr_cells}/9 cells)',
        f'- **G2** (every cell has coverage_rate): **{g2}** ({qr_with_cov}/{total_qr_cells})',
        f'- **G3** (mean_replicas > 1 on every cell): **{g3}**',
        f'- **G4** (qr coverage ≤ pid − 0.10 on ≥2 of F/G/H): **{g4}**',
        '',
        '### G4 per-pattern detail',
        '',
        '| pattern | qr mean | pid mean | qr − pid | ≤ -0.10? |',
        '|---------|---------|----------|----------|----------|',
    ]
    for pat in ['F', 'G', 'H']:
        d = g4_detail[pat]
        delta = d['qr_minus_pid']
        passes = 'yes' if (delta is not None and delta <= -0.10) else 'no'
        lines.append(
            f'| {pat} | {_fmt(d["qr_mean"])} | {_fmt(d["pid_mean"])} | '
            f'{_fmt(delta, "{:+.4f}")} | {passes} |'
        )

    if g3 == 'FAIL':
        lines += ['', '### G3 cells failing (mean_replicas ≤ 1)']
        for pat, cells in g3_detail.items():
            for c in cells:
                if c['mean_replicas'] is None or c['mean_replicas'] <= 1.0:
                    lines.append(f'- {c["run_dir"]}: mean_replicas={c["mean_replicas"]}')

    out = '\n'.join(lines) + '\n'
    args.out.write_text(out)
    print(out)


if __name__ == '__main__':
    main()
