"""Regenerate the five camera-ready tables from frozen, shipped evidence."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from . import entities_legacy as legacy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def rows_from_markdown(text):
    return [[c.strip().strip('`*') for c in l.strip().strip('|').split('|')]
            for l in text.splitlines() if l.startswith('|') and not re.match(r'^\|[ :\-]+\|', l)]

def numbers(text):
    text = text.replace(',', '').replace('±', ' ').replace('+/-', ' ').replace('−', '-')
    return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', text)]

def check_table_cells(table_id, rows, expected):
    if len(rows) != len(expected):
        raise ValueError(f'{table_id}: row count differs: {len(rows)} vs {len(expected)}')
    count = 0
    aliases = {'confscale-be':'Bootstrap', 'confscale-qr':'QR', 'confscale-scp':'SCP',
               'rolling-origin':'RO', 'risk-quantile-scp':'RQ-SCP', 'confscale-pid':'PID',
               'confscale-pid-laddered':'PID+L', 'confscale-aci':'ACI', 'confscale-aci-laddered':'ACI+L',
               'confscale-rolling-origin-laddered':'RO+L', 'hpa-qr-monitored':'HPA-QR', 'hpa-reactive':'HPA',
               'ConfScale-SCP':'ConfScale-SCP (ref)', 'hpa-anchor-u50-s300':'Anchor'}
    for i, (actual_row, expected_row) in enumerate(zip(rows[1:], expected[1:]), 1):
        label = 'Tuned HPA' if actual_row[0].startswith('hpa-tuned-') else aliases.get(actual_row[0], actual_row[0])
        if label != expected_row[0]: raise ValueError(f'{table_id} row {i}: method/service label mismatch')
        if len(actual_row) != len(expected_row): raise ValueError(f'{table_id} row {i}: column count differs')
        for j, (a, b) in enumerate(zip(actual_row[1:], expected_row[1:]), 1):
            if numbers(a) != numbers(b):
                raise ValueError(f'{table_id} row {i} column {j}: {a!r} != paper {b!r}')
            count += 1
    return count

def build_binding_table(evidence):
    rel = 'data/p3_runs/outputs/slo_binding_existence_proof_20260621_134302/rc5_results.json'
    data = json.loads((evidence/rel).read_text(encoding='utf-8'))
    text = ['# Table 5. Coverage repair and end-to-end p95 on the worker-binding testbed', '',
            'Frozen measurements; mean +/- sample s.d. in ms, R=3; 18/18 cells completed.',
            'The worker CPU limit was reduced from 500m to 80m. This is an engineered existence proof.', '',
            '| drift | ConfScale ACI | ConfScale PID | HPA-QR monitored | best recal. - baseline |',
            '|---|---:|---:|---:|---:|']
    for p, label in [('F', 'F (volatility)'), ('G', 'G (level)')]:
        stats = []
        for m in ['confscale-aci', 'confscale-pid', 'hpa-qr-monitored']:
            cell = data['cell_stats'][m+'/'+p]
            vals = cell['vals']
            if len(vals) != cell['n'] or len(vals) != 3: raise ValueError('binding evidence R mismatch')
            mean, sd = statistics.mean(vals), statistics.stdev(vals)
            if not math.isclose(mean, cell['mean'], abs_tol=1e-10) or not math.isclose(sd, cell['sd'], abs_tol=1e-10):
                raise ValueError('binding evidence summary does not match its retained per-run values')
            stats.append((mean, sd))
        best = min(stats[:2]); baseline = stats[2]
        delta = best[0]-baseline[0]
        rss = math.sqrt(best[1]**2+baseline[1]**2)
        text.append('| '+label+' | '+' | '.join(f'{a:,.1f} +/- {b:.1f}' for a,b in stats)+f' | {delta:.1f} ms (RSS SD {rss:.0f}) |')
    text += ['', 'RSS SD is the root-sum-square of the two run-to-run sample deviations; it is descriptive, not a significance test.',
             f'Source: `evidence/{rel}`.']
    return '\n'.join(text)+'\n'

def reproduce(output, evidence=None):
    output = Path(output)
    evidence = Path(evidence) if evidence else PACKAGE_ROOT/'evidence'
    output.mkdir(parents=True, exist_ok=True)
    legacy.ROOT, legacy.OUT = evidence, output/'supporting_entities'
    legacy.TABLES, legacy.FIGURES = legacy.OUT/'tables', legacy.OUT/'figures'
    legacy.main()
    mapping = [(1, 'table_1_uq_calibration.md'), (2, 'table_2_tuned_hpa_pattern_d.md'),
               (3, 'table_3_drift_coverage_cost.md'), (4, 'table_5_perservice_volatility.md')]
    tables = {}
    for number, name in mapping:
        text = (legacy.TABLES/name).read_text(encoding='utf-8')
        text = re.sub(r'^# Table \d+ -', f'# Table {number}.', text, count=1)
        if number == 1:
            text = text.replace('which is the calibration-as-frugality pitfall', 'illustrating why low cost alone cannot establish reliable intervals')
        if number == 3:
            text = text.replace('Drift Coverage and Cost Geometry', 'Coverage and Replica Cost under Drift')
            text = text.replace('Raw G/H correction notes are in Table 4.',
                                'Raw G/H comparisons are retained in `supporting_entities/tables/table_4_k8_ladder_rescue.md` (older supporting-table numbering).')
        text = text.replace('`data/p3_runs/', '`evidence/data/p3_runs/')
        tables[str(number)] = text
    tables['5'] = build_binding_table(evidence)
    expected = json.loads((PACKAGE_ROOT/'provenance/paper_tables_expected.json').read_text(encoding='utf-8'))
    receipt = {'kind':'REGENERATION_FROM_FROZEN_AGGREGATES_NOT_NEW_EXPERIMENTS',
               'paper_sha256': expected['paper_sha256'], 'tables':{},
               'coverage': 'All five main paper tables. Tables 1 and 3 begin at archived aggregate tables; no raw episodes are shipped.'}
    for key, text in tables.items():
        rows = rows_from_markdown(text)
        checked = check_table_cells(key, rows, expected['tables'][key])
        dest = output/f'table_{key}.md'
        dest.write_text(text, encoding='utf-8')
        with (output/f'table_{key}.csv').open('w', newline='', encoding='utf-8') as f:
            csv.writer(f, lineterminator='\n').writerows(rows)
        receipt['tables'][key] = {'data_cells_checked_against_paper':checked,
                                 'numeric_values_checked':sum(len(numbers(c)) for row in rows[1:] for c in row[1:]),
                                 'row_labels_checked':len(rows)-1, 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt
