"""Validate and render separately retained comparison results, offline."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = 'comparisons/evidence/20261026'
BATCHES = {
    'point-vs-tuned': (5, ('confscale-scp-frozen', 'confscale-point-corrected', 'hpa-cpu-tuned')),
    'point-vs-stock': (3, ('hpa-cpu-stock', 'hpa-cpu-memory', 'confscale-point-corrected')),
}
METRICS = ('overhead_replica_seconds_per_hour', 'e2e_run_p95_ms',
           'tick_slo_violation_fraction', 'desired_count_changes')
UNITS = ('replica-seconds/hour', 'ms', 'fraction', 'sampled desired-count changes/run')
KINDS = {'saved_score', 'metrics', 'run_configuration', 'controller_configuration', 'model_manifest', 'model_input',
         'definition_reference', 'workload_trace'}


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'Invalid {label} fields')


def _number(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid {label} number')
    if minimum is not None and value < minimum:
        raise ValueError(f'Invalid {label} range')


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256')


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Non-finite JSON number')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=invalid)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_private_locations(value):
    if isinstance(value, str):
        boundary = r"""(?:^|[\s=:'"(\[{,;<])"""
        for match in re.finditer(boundary + r"(?P<location>/+|[a-z]:[\\/])", value, re.I):
            # Skip only a URL's opening slashes, not later path-like text.
            if (match['location'] == '//' and
                    re.search(boundary + r'[a-z][a-z0-9+.-]*:\Z', value[:match.start('location')], re.I) and
                    re.match(r"""[^\s/\\?'"#<>]""", value[match.end('location'):])):
                continue
            raise ValueError('Absolute private location in comparison metadata')
    elif isinstance(value, dict):
        for key, item in value.items():
            _no_private_locations(key)
            _no_private_locations(item)
    elif isinstance(value, list):
        for item in value:
            _no_private_locations(item)


def validate_batch(data, sources, artifact_input_hashes):
    """Check the fixed admitted cohort and scalar/aggregate correspondence."""
    _keys(data, ('schema_version', 'kind', 'batch', 'cohort', 'metric_definitions',
                 'method_configurations', 'checkpoint_provenance', 'qualifications',
                 'recorded_status', 'cells', 'summaries', 'recorded_comparisons'), 'comparison')
    if data['schema_version'] != 1 or data['kind'] != 'STORED_RESULT_COMPARISON' or data['batch'] not in BATCHES:
        raise ValueError('Unsupported comparison schema')
    _no_private_locations(data)
    reps, methods = BATCHES[data['batch']]
    cohort = data['cohort']
    _keys(cohort, ('patterns', 'methods', 'replicates_per_method_pattern', 'admitted_cells',
                   'duration_s', 'complexity', 'pool_with_other_batches'), 'cohort')
    if cohort != {'patterns': list('ABCD'), 'methods': list(methods), 'replicates_per_method_pattern': reps,
                  'admitted_cells': 12 * reps, 'duration_s': 1800, 'complexity': 50000, 'pool_with_other_batches': False}:
        raise ValueError('Comparison cohort differs from the retained batch')
    _keys(data['metric_definitions'], METRICS, 'metric definitions')
    for metric, unit in zip(METRICS, UNITS):
        definition = data['metric_definitions'][metric]
        fields = ('unit', 'definition') + (('threshold_ms',) if metric == 'tick_slo_violation_fraction' else ())
        _keys(definition, fields, 'metric definition')
        if definition['unit'] != unit or not isinstance(definition['definition'], str) or not definition['definition']:
            raise ValueError('Invalid metric definition or unit')
        if metric == 'tick_slo_violation_fraction' and definition['threshold_ms'] != 200:
            raise ValueError('Invalid retained tick-violation threshold')
    _keys(data['method_configurations'], methods, 'method configurations')
    settings = {'min_replicas', 'max_replicas', 'coverage_monitor', 'online_recal', 'recalibrator', 'ladder',
                'slo_capacity', 'target_util', 'safety_factor', 'uq_method', 'policy', 'h0_only',
                'point_bias_window', 'cpu_target_pct', 'memory_target_pct', 'behavior_present',
                'upscale_stabilization_s', 'downscale_stabilization_s', 'hpa_api_version'}
    for method, config in data['method_configurations'].items():
        _keys(config, ('recorded_settings', 'runtime_support'), 'method configuration')
        expected = 'stored-only' if method == 'hpa-cpu-memory' else 'supported-method'
        if config['runtime_support'] != expected or not isinstance(config['recorded_settings'], dict):
            raise ValueError('Invalid method support classification')
        if not set(config['recorded_settings']) <= settings:
            raise ValueError('Unexpected configuration field')
    checkpoint = data['checkpoint_provenance']
    _keys(checkpoint, ('status', 'files'), 'checkpoint provenance')
    if checkpoint['status'] not in ('configuration-only', 'recorded-hashes'):
        raise ValueError('Invalid checkpoint provenance status')
    models = set()
    for item in checkpoint['files']:
        _keys(item, ('model', 'sha256', 'artifact_input_sha256', 'artifact_input_relation'), 'checkpoint file')
        if not re.fullmatch(r'(diurnal|bursty|batch_ramp|signaling)/scp/(base_model\.pt|scp_config\.yaml)', item['model']):
            raise ValueError('Invalid model identifier')
        _hash(item['sha256'])
        _hash(item['artifact_input_sha256'])
        if item['model'] in models or artifact_input_hashes.get(item['model']) != item['artifact_input_sha256']:
            raise ValueError('Checkpoint differs from artifact input manifest')
        models.add(item['model'])
        if item['artifact_input_relation'] not in ('identical', 'LF-to-CRLF-only'):
            raise ValueError('Invalid checkpoint normalization')
        if (item['sha256'] == item['artifact_input_sha256']) != (item['artifact_input_relation'] == 'identical'):
            raise ValueError('Checkpoint hash relation contradicts recorded identity')
    if len(checkpoint['files']) != (8 if checkpoint['status'] == 'recorded-hashes' else 0):
        raise ValueError('Incomplete checkpoint provenance')
    if not isinstance(data['qualifications'], list) or not all(isinstance(x, str) and x for x in data['qualifications']):
        raise ValueError('Missing comparison qualifications')
    status = data['recorded_status']
    _keys(status, ('cohort_complete', 'hpa_event_log_complete', 'proposal_observation_complete',
                   'overall_rule_verdict', 'rule_patterns_passed'), 'recorded status')
    expected_status = {'cohort_complete': True, 'hpa_event_log_complete': None, 'proposal_observation_complete': None,
                       'overall_rule_verdict': 'PASS', 'rule_patterns_passed': 3} if reps == 5 else {
                       'cohort_complete': True, 'hpa_event_log_complete': False, 'proposal_observation_complete': False,
                       'overall_rule_verdict': None, 'rule_patterns_passed': None}
    if status != expected_status:
        raise ValueError('Retained qualification/status changed')
    cells = {}
    for cell in data['cells']:
        _keys(cell, ('id', 'pattern', 'method', 'replicate', 'zero_success', 'values', 'sources'), 'cell')
        key = (cell['pattern'], cell['method'], cell['replicate'])
        if (cell['pattern'] not in 'ABCD' or len(cell['pattern']) != 1 or cell['method'] not in methods
                or type(cell['replicate']) is not int or cell['replicate'] not in range(1, reps + 1) or key in cells):
            raise ValueError('Invalid or duplicate admitted cell')
        if cell['id'] != f'{key[0].lower()}-{key[1]}-rep-{key[2]}' or type(cell['zero_success']) is not bool:
            raise ValueError('Invalid cell identity or success flag')
        _keys(cell['values'], METRICS, 'cell values')
        for metric, value in cell['values'].items():
            _number(value, metric, 0)
        if cell['values']['tick_slo_violation_fraction'] > 1:
            raise ValueError('Invalid fraction')
        expected_refs = {'score', 'metrics', 'run_configuration', 'workload_trace'}
        if cell['method'].startswith('confscale-'):
            expected_refs.add('controller_configuration')
        _keys(cell['sources'], expected_refs, 'cell source references')
        for key_name, source_id in cell['sources'].items():
            kind = 'saved_score' if key_name == 'score' else key_name
            if source_id not in sources or sources[source_id]['kind'] != kind:
                raise ValueError('Unresolved comparison source reference')
        cells[key] = cell
    if len(cells) != 12 * reps:
        raise ValueError('Incomplete admitted cohort')
    seen, summary_values = set(), {}
    for row in data['summaries']:
        _keys(row, ('pattern', 'method', 'n', 'statistics'), 'summary')
        key = (row['pattern'], row['method'])
        if key in seen or key not in {(p, m) for p in 'ABCD' for m in methods} or row['n'] != reps:
            raise ValueError('Invalid summary group')
        seen.add(key)
        summary_values[key] = row['statistics']
        _keys(row['statistics'], METRICS, 'summary statistics')
        for metric, stat in row['statistics'].items():
            keys = ('mean', 'n', 'values', 'aggregation') + (() if metric == 'desired_count_changes' else ('sd',))
            _keys(stat, keys, 'summary statistic')
            values = [cells[(*key, rep)]['values'][metric] for rep in range(1, reps + 1)]
            aggregation = 'arithmetic_mean_of_exported_cells' if metric == 'desired_count_changes' else 'saved_summary'
            if stat['n'] != reps or stat['values'] != values or stat['aggregation'] != aggregation:
                raise ValueError('Summary does not match retained cells')
            _number(stat['mean'], 'mean', 0)
            if not math.isclose(stat['mean'], sum(values) / reps, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Summary mean does not match retained values')
            if 'sd' in stat:
                _number(stat['sd'], 'standard deviation', 0)
    if len(seen) != 12:
        raise ValueError('Incomplete summaries')
    comparisons, primary_passes = set(), 0
    for row in data['recorded_comparisons']:
        _keys(row, ('pattern', 'candidate', 'reference', 'differences', 'recorded_gate'), 'recorded comparison')
        key = (row['pattern'], row['candidate'], row['reference'])
        if key in comparisons or row['pattern'] not in 'ABCD' or len(row['pattern']) != 1:
            raise ValueError('Invalid comparison group')
        if row['candidate'] not in methods or row['reference'] not in methods or row['candidate'] == row['reference']:
            raise ValueError('Invalid comparison methods')
        comparisons.add(key)
        _keys(row['differences'], METRICS[:3], 'recorded differences')
        for metric, value in row['differences'].items():
            _keys(value, ('difference', 'relative_percent', 'welch_95'), 'recorded difference')
            _number(value['difference'], 'difference')
            _number(value['relative_percent'], 'relative percent')
            candidate_mean = summary_values[(row['pattern'], row['candidate'])][metric]['mean']
            reference_mean = summary_values[(row['pattern'], row['reference'])][metric]['mean']
            expected_difference = candidate_mean - reference_mean
            if not math.isclose(value['difference'], expected_difference, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError('Recorded difference contradicts saved means')
            if not reference_mean or not math.isclose(value['relative_percent'],
                    100 * expected_difference / reference_mean, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError('Recorded relative difference contradicts saved means')
            if value['welch_95'] is not None:
                interval = value['welch_95']
                _keys(interval, ('low', 'high', 'df'), 'recorded interval')
                for v in interval.values():
                    _number(v, 'interval')
                if interval['low'] > interval['high'] or interval['df'] <= 0:
                    raise ValueError('Invalid interval ordering or degrees of freedom')
        gate = row['recorded_gate']
        if reps == 5:
            _keys(gate, ('cost', 'service', 'pattern_pass', 'strictly_cheaper'), 'recorded gate')
            if not all(type(v) is bool for v in gate.values()):
                raise ValueError('Invalid recorded gate')
            candidate_cost = summary_values[(row['pattern'], row['candidate'])][METRICS[0]]['mean']
            reference_cost = summary_values[(row['pattern'], row['reference'])][METRICS[0]]['mean']
            if gate['cost'] != (candidate_cost <= 1.05 * reference_cost):
                raise ValueError('Recorded cost gate contradicts saved means')
            if gate['strictly_cheaper'] != (candidate_cost < reference_cost):
                raise ValueError('Recorded strictly-cheaper gate contradicts saved means')
            if gate['pattern_pass'] != (gate['cost'] and gate['service']):
                raise ValueError('Recorded pattern gate contradicts component gates')
            if (row['candidate'], row['reference']) == ('confscale-point-corrected', 'hpa-cpu-tuned'):
                primary_passes += int(gate['pattern_pass'])
        elif gate is not None:
            raise ValueError('Unexpected gate for descriptive comparison')
    expected_pairs = [('confscale-point-corrected', 'hpa-cpu-tuned'), ('confscale-point-corrected', 'confscale-scp-frozen'),
                      ('confscale-scp-frozen', 'hpa-cpu-tuned')] if reps == 5 else [('hpa-cpu-memory', 'hpa-cpu-stock')]
    if comparisons != {(p, a, b) for p in 'ABCD' for a, b in expected_pairs}:
        raise ValueError('Incomplete recorded comparisons')
    if reps == 5 and (primary_passes != status['rule_patterns_passed'] or
                     (primary_passes >= 3) != (status['overall_rule_verdict'] == 'PASS')):
        raise ValueError('Overall recorded rule contradicts pattern gates')
    return data


def load_comparisons(root=ROOT):
    root = Path(root).resolve()
    manifest = _json(root / 'provenance/comparison_manifest.json')
    _keys(manifest, ('schema_version', 'kind', 'scope', 'records'), 'comparison manifest')
    if manifest['schema_version'] != 1 or manifest['kind'] != 'SANITIZED_STORED_COMPARISON_PROVENANCE':
        raise ValueError('Unsupported comparison manifest')
    _no_private_locations(manifest)
    omitted = _json(root / 'provenance/omitted_inputs.json')
    artifact_inputs = {row['path'].removeprefix('models/uq/'): row['sha256']
                       for row in omitted['inputs'] if row['path'].startswith('models/uq/')}
    loaded = {}
    for row in manifest['records']:
        _keys(row, ('batch', 'destination', 'destination_sha256', 'sources'), 'manifest record')
        batch = row['batch']
        if batch not in BATCHES or batch in loaded or row['destination'] != f'{DIRECTORY}/{batch}.json':
            raise ValueError('Invalid comparison destination')
        path = (root / row['destination']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Comparison path escapes artifact')
        _hash(row['destination_sha256'])
        if _digest(path) != row['destination_sha256']:
            raise ValueError(f'Comparison hash mismatch: {batch}')
        sources = {}
        for source in row['sources']:
            _keys(source, ('id', 'kind', 'sha256'), 'source identity')
            if not re.fullmatch(r'[a-z0-9-]+', source['id']) or source['id'] in sources or source['kind'] not in KINDS:
                raise ValueError('Invalid comparison source identity')
            _hash(source['sha256'])
            sources[source['id']] = source
        data = validate_batch(_json(path), sources, artifact_inputs)
        if data['batch'] != batch:
            raise ValueError('Manifest and batch identity differ')
        loaded[batch] = data
    if set(loaded) != set(BATCHES):
        raise ValueError('Missing comparison batch')
    return loaded, manifest


def verify_comparisons(root=ROOT):
    data, manifest = load_comparisons(root)
    return {'comparison_files_verified': len(data),
            'comparison_cells_verified': sum(len(d['cells']) for d in data.values()),
            'comparison_source_hashes_recorded': sum(len(r['sources']) for r in manifest['records'])}


def _markdown(data):
    cohort = data['cohort']
    lines = [f'# {data["batch"]}', '', 'Stored-result reproduction; no independent rescore or scientific replication.', '',
             f'{cohort["admitted_cells"]} admitted cells; {cohort["replicates_per_method_pattern"]} runs per method and pattern.', '']
    status = data['recorded_status']
    if status['overall_rule_verdict'] is not None:
        lines += [f'Recorded overall rule: **{status["overall_rule_verdict"]}**, {status["rule_patterns_passed"]}/4 patterns pass. Pattern D fails the cost criterion.', '']
    lines.extend('- ' + text for text in data['qualifications'])
    lines += ['', 'All entries below are means across runs. CSV files retain full numeric precision.', '',
              '| Pattern | Method | n | Overhead (replica-s/hour) | Run p95 (ms) | Tick violations (%) | Desired-count changes/run |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in data['summaries']:
        s = row['statistics']
        values = [s[m]['mean'] for m in METRICS]
        lines.append(f'| {row["pattern"]} | {row["method"]} | {row["n"]} | {values[0]:.2f} | {values[1]:.2f} | {100 * values[2]:.2f} | {values[3]:.2f} |')
    lines += ['', 'Recorded comparisons use candidate minus reference. Intervals and criteria below are copied from saved results.', '',
              '| Pattern | Candidate | Reference | Overhead difference | Recorded 95% interval | p95 difference (ms) | Tick-share difference (pp) | Cost criterion | Service criterion |',
              '|---|---|---|---:|---|---:|---:|---|---|']
    for row in data['recorded_comparisons']:
        d = row['differences']
        overhead = d[METRICS[0]]
        interval = overhead['welch_95']
        ci = f'[{interval["low"]:.2f}, {interval["high"]:.2f}]' if interval else 'not recorded'
        gate = row['recorded_gate']
        cost, service = [('PASS' if gate[k] else '**FAIL**') if gate else 'not defined' for k in ('cost', 'service')]
        lines.append(f'| {row["pattern"]} | {row["candidate"]} | {row["reference"]} | {overhead["difference"]:.2f} | {ci} | {d[METRICS[1]]["difference"]:.2f} | {100 * d[METRICS[2]]["difference"]:.2f} | {cost} | {service} |')
    lines += ['', 'Metric definitions:', '']
    lines.extend(f'- **{metric}** ({definition["unit"]}): {definition["definition"]}' for metric, definition in data['metric_definitions'].items())
    return '\n'.join(lines) + '\n'


def reproduce_comparisons(output, root=ROOT):
    data, manifest = load_comparisons(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files, batches = {}, {}
    for batch in BATCHES:
        value = data[batch]
        arm_file = output / f'{batch}.csv'
        with arm_file.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['pattern', 'method', 'n', *[metric + '_mean' for metric in METRICS]])
            for row in value['summaries']:
                writer.writerow([row['pattern'], row['method'], row['n'], *[row['statistics'][m]['mean'] for m in METRICS]])
        cell_file = output / f'{batch}-cells.csv'
        with cell_file.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['cell_id', 'pattern', 'method', 'replicate', 'zero_success', *METRICS])
            for cell in value['cells']:
                writer.writerow([cell['id'], cell['pattern'], cell['method'], cell['replicate'], cell['zero_success'], *[cell['values'][m] for m in METRICS]])
        markdown = output / f'{batch}.md'
        markdown.write_text(_markdown(value), encoding='utf-8')
        for path in (arm_file, cell_file, markdown):
            files[path.name] = _digest(path)
        batches[batch] = {k: value[k] for k in ('cohort', 'metric_definitions', 'recorded_status', 'qualifications', 'recorded_comparisons')}
    result = {'kind': 'STORED_RESULT_REPRODUCTION',
              'scope': 'Render retained values; not an independent raw-run rescore, new experiment, or scientific replication.',
              'batches': batches, 'output_sha256': files,
              'input_sha256': {r['destination']: r['destination_sha256'] for r in manifest['records']}}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'kind': result['kind'], 'scope': result['scope'],
            'batches': {batch: {'admitted_cells': value['cohort']['admitted_cells'],
                                'summaries': len(value['summaries']), 'recorded_status': value['recorded_status']}
                        for batch, value in data.items()},
            'output_sha256': {**files, 'summary.json': _digest(output / 'summary.json')},
            'input_sha256': result['input_sha256']}
