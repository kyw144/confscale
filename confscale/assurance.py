"""Offline checks for experiment completeness and consistency."""
import csv
from collections import deque
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

from .capabilities import (COMPARISON_METHODS, capabilities, capability_record,
                           preset_configuration)


def finite(value, minimum=0):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def _close(actual, expected):
    return finite(actual, -1e100) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9)


def _routing_matches(record, actual):
    return (isinstance(actual.get('context'), str) and bool(actual['context']) and
            re.fullmatch(r'[0-9a-f]{64}', actual.get('kubeconfig_path_sha256', '')) is not None and
            all(record.get(key) == actual[key] for key in ('context', 'kubeconfig_path_sha256')))


def _command_matches(record, command, actual, success=True):
    if not isinstance(record, dict) or record.get('command') != command or not _routing_matches(record, actual):
        return False
    start, end = (datetime.fromisoformat(record[key]) for key in ('time', 'end_time'))
    if start.tzinfo is None or end.tzinfo is None or end < start:
        return False
    if success:
        return (record.get('state') == 'completed' and type(record.get('returncode')) is int and record['returncode'] == 0 and
                isinstance(record.get('stdout'), str) and isinstance(record.get('stderr'), str) and
                'error' not in record and not record.get('interrupted'))
    if (record.get('state') != 'failed' or 'interrupted' in record or
            not isinstance(record.get('stdout'), str) or not isinstance(record.get('stderr'), str)):
        return False
    has_error = isinstance(record.get('error'), str) and bool(record['error'])
    if 'error' in record and not has_error:
        return False
    if 'returncode' in record:
        return type(record['returncode']) is int and record['returncode'] != 0
    return has_error


def _inventory_items(record):
    items = json.loads(record['stdout'])['items']
    if not isinstance(items, list):
        raise ValueError('HPA inventory must be a list')
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get('metadata'), dict) or
                not isinstance(item['metadata'].get('name'), str) or not item['metadata']['name'] or
                not isinstance(item.get('spec'), dict) or
                not isinstance(item['spec'].get('scaleTargetRef'), dict)):
            raise ValueError('Malformed HPA inventory object')
    return items


def _targets_worker(item):
    target = item['spec']['scaleTargetRef']
    return target.get('kind') == 'Deployment' and target.get('name') == 'compute-worker'


def _configured_native_hpa(run_dir, name, actual, expected_context):
    from .native_hpa import hpa_manifest, stored_object_errors
    namespace = 'infosys-benchmark'
    expected = hpa_manifest(actual['hpa_name'], name, namespace)
    if expected_context is not None and actual.get('context') != expected_context:
        return False
    submitted = json.loads((run_dir / 'hpa_v2.yaml').read_text())
    initial = json.loads((run_dir / 'hpa_initial_inventory.json').read_text())
    journal = json.loads((run_dir / 'hpa_configure_check.json').read_text())
    stored = json.loads((run_dir / 'hpa_stored.json').read_text())
    reads = json.loads((run_dir / 'hpa_stored_readback.json').read_text())
    if submitted != expected or not (journal.get('ok') is True and _routing_matches(journal, actual) and
            journal.get('hpa_name') == actual['hpa_name'] and journal.get('hpa_uid') == actual['hpa_uid'] and
            journal.get('namespace') == namespace and 'error' not in journal and not journal.get('interrupted')):
        return False
    inventory_command = ['get', 'hpa', '-n', namespace, '-o', 'json']
    if not _command_matches(initial, inventory_command, actual) or any(_targets_worker(i) for i in _inventory_items(initial)):
        return False
    commands = journal['commands']
    if not isinstance(reads, list) or not 1 <= len(reads) <= 2 or not isinstance(commands, list) or len(commands) != 4 + len(reads):
        return False
    if commands[0] != initial or commands[4:] != reads:
        return False
    prefix = [inventory_command,
              ['scale', 'deployment', 'compute-worker', '--replicas=1', '-n', namespace],
              ['rollout', 'status', 'deployment', 'compute-worker', '-n', namespace, '--timeout=120s']]
    if any(not _command_matches(row, command, actual) for row, command in zip(commands, prefix)):
        return False
    apply = commands[3]['command']
    if (not isinstance(apply, list) or len(apply) != 3 or apply[:2] != ['apply', '-f'] or
            not isinstance(apply[2], str) or Path(apply[2]).name != 'hpa_v2.yaml' or
            apply[2] != journal.get('manifest_path') or
            journal.get('manifest_sha256') != hashlib.sha256((run_dir / 'hpa_v2.yaml').read_bytes()).hexdigest() or
            not _command_matches(commands[3], apply, actual)):
        return False
    for index, row in enumerate(reads):
        command = ['get', 'hpa', actual['hpa_name'], '-n', namespace, '-o', 'json']
        if _command_matches(row, command, actual):
            obj = json.loads(row['stdout'])
            if stored_object_errors(obj, actual['hpa_name'], name, namespace, uid=actual['hpa_uid']) or row.get('errors') != []:
                return False
        elif index == len(reads) - 1 or not _command_matches(row, command, actual, success=False):
            return False
    return (all(datetime.fromisoformat(a['end_time']) <= datetime.fromisoformat(b['time']) for a, b in zip(commands, commands[1:])) and
            json.loads(reads[-1]['stdout']) == stored and
            not stored_object_errors(stored, actual['hpa_name'], name, namespace, uid=actual['hpa_uid']))


def _restored_native_hpa(reset, actual):
    from .native_hpa import stored_object_errors
    if not (reset.get('ok') is True and reset.get('hpa_name') == actual['hpa_name'] and
            reset.get('hpa_uid') == actual['hpa_uid'] and reset.get('namespace') == 'infosys-benchmark' and
            _routing_matches(reset, actual)):
        return False
    if 'error' in reset or reset.get('interrupted'):
        return False
    namespace = 'infosys-benchmark'
    inventory_command = ['get', 'hpa', '-n', namespace, '-o', 'json']
    if not _command_matches(reset['inventory'], inventory_command, actual):
        return False
    own = [i for i in _inventory_items(reset['inventory']) if i['metadata']['name'] == actual['hpa_name']]
    if len(own) != 1 or stored_object_errors(own[0], actual['hpa_name'], actual['method'], namespace, uid=actual['hpa_uid']):
        return False
    deletion = reset['delete']
    if deletion['ok'] is not True or not isinstance(deletion['attempts'], list) or not 1 <= len(deletion['attempts']) <= 2:
        return False
    last = deletion['attempts'][-1]
    namespace = 'infosys-benchmark'
    expected_commands = [
        (last['delete'], ['delete', 'hpa', actual['hpa_name'], '-n', namespace, '--ignore-not-found=true']),
        (last['readback'], ['get', 'hpa', '-n', namespace, '-o', 'json']),
        (reset['scale'], ['scale', 'deployment', 'compute-worker', '--replicas=1', '-n', namespace]),
        (reset['rollout'], ['rollout', 'status', 'deployment', 'compute-worker', '-n', namespace, '--timeout=120s']),
        (reset['deployment'], ['get', 'deployment', 'compute-worker', '-n', namespace, '-o', 'json']),
    ]
    if any(not _command_matches(record, expected, actual) for record, expected in expected_commands):
        return False
    for attempt in deletion['attempts'][:-1]:
        for record, command in ((attempt['delete'], expected_commands[0][1]), (attempt['readback'], inventory_command)):
            if not (_command_matches(record, command, actual) or _command_matches(record, command, actual, success=False)):
                return False
    rows = [reset['inventory']]
    for attempt in deletion['attempts']:
        rows.extend((attempt['delete'], attempt['readback']))
    rows.extend((reset['scale'], reset['rollout'], reset['deployment']))
    if reset.get('commands') != rows or not all(
            datetime.fromisoformat(a['end_time']) <= datetime.fromisoformat(b['time']) for a, b in zip(rows, rows[1:])):
        return False
    remaining = _inventory_items(last['readback'])
    if any(item['metadata']['name'] == actual['hpa_name'] or
           (item['spec']['scaleTargetRef'].get('kind') == 'Deployment' and
            item['spec']['scaleTargetRef'].get('name') == 'compute-worker') for item in remaining):
        return False
    deployment = json.loads(reset['deployment']['stdout'])
    return (deployment['spec'].get('replicas') == deployment['status'].get('readyReplicas') ==
            deployment['status'].get('replicas') == 1)


def _float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def _forecast_dtypes(row):
    dtypes = [row[key] for key in ('point_forecast_dtype', 'ci_upper_dtype')]
    if any(dtype not in ('float32', 'float64') for dtype in dtypes):
        raise ValueError('Unsupported forecast dtype')
    for values, dtype in zip((row['point_forecast'], row['ci_upper']), dtypes):
        if dtype == 'float32' and any(_float32(value) != value for value in values):
            raise ValueError('Forecast values do not match recorded float32 dtype')
    return dtypes


def _tier2_rate(row, safety_factor):
    dtypes = _forecast_dtypes(row)
    round_value = _float32 if dtypes == ['float32', 'float32'] else float
    # Match array promotion and rounding after each NumPy operation, without NumPy.
    return max(round_value(p + round_value(round_value(safety_factor) *
               round_value(round_value(u - p) / round_value(1.645))))
               for p, u in zip(row['point_forecast'], row['ci_upper']))


def _target_evidence(entries, configuration, corrected):
    """Independently recompute raw targets and causal correction from retained rows."""
    residuals, previous = deque(maxlen=10), None
    matches, targets = True, True
    matched_count = 0
    for row in entries:
        if previous is not None and finite(row.get('rps'), -1e100):
            residuals.append(previous - row['rps'])
            matched_count += 1
        predicted = row.get('prediction_made') is True and bool(row.get('point_forecast'))
        h0 = row['point_forecast'][0] if predicted else None
        previous = h0 if finite(h0, -1e100) else None
        bias = statistics.mean(residuals) if residuals else 0.0
        if corrected:
            correction = row.get('point_correction', {})
            matches &= (correction.get('raw_h0') == h0 and
                        _close(correction.get('bias_rps'), bias) and
                        correction.get('matched_errors') == len(residuals))
            if h0 is None:
                matches &= correction.get('corrected_rps') is None
            else:
                matches &= _close(correction.get('corrected_rps'), max(0, h0 - bias))
        if not predicted:
            continue
        confidence = row['confidence_score']
        tier = 1 if confidence < .15 else 2 if confidence < .4 else 3
        point, upper = row['point_forecast'], row['ci_upper']
        if not corrected:
            _forecast_dtypes(row)
        if corrected:
            rate = max(0, h0 - bias)
        elif tier == 1:
            rate = max(point)
        elif tier == 2:
            rate = _tier2_rate(row, configuration['safety_factor'])
        else:
            rate = max(upper)
        raw = math.ceil(rate / (configuration['slo_capacity'] * configuration['target_util']))
        raw = max(configuration['min_replicas'], min(raw, configuration['max_replicas']))
        targets &= row.get('raw_target') == raw and row.get('raw_tier') == tier
        targets &= (finite(row.get('target_replicas'), configuration['min_replicas']) and
                    row['target_replicas'] <= configuration['max_replicas'])
    return bool(matches), bool(targets), matched_count


def assess(run_dir, predictive, criteria, method_name=None, expected_context=None):
    """Technical validity only."""
    run_dir = Path(run_dir)
    checks = {}
    details = {}
    try:
        result = json.loads((run_dir / 'execution.json').read_text())
        name = method_name or result.get('method')
        cap = capabilities(name) if name is not None else None
        if method_name is not None and result.get('method') is not None:
            checks['method_matches_plan'] = result['method'] == method_name
        if cap is not None:
            predictive = cap.forecasts
        comparison = name in COMPARISON_METHODS
        if comparison:
            effective = json.loads((run_dir / 'effective_method.json').read_text())
            checks['effective_method_recorded'] = (effective.get('method') == name and
                effective.get('capabilities') == capability_record(name) and
                effective.get('expected_config') == preset_configuration(name) and
                effective.get('actual_config', {}).get('method') == name)
        checks['execution_succeeded'] = result.get('status') == 'ok' and result.get('workload_returncode') == 0
        checks['method_reset_succeeded'] = result.get('reset_ok') is True
        if (cap is not None and cap.native_hpa) or result.get('method') == 'hpa-reactive':
            state = json.loads((run_dir / 'before_reset_status.json').read_text())
            hpas = [o for o in state['items'] if o['kind'] == 'HorizontalPodAutoscaler'
                    and o['spec']['scaleTargetRef']['name'] == 'compute-worker']
            conditions = {c['type']: c['status'] for h in hpas for c in h.get('status', {}).get('conditions', [])}
            checks['hpa_active'] = len(hpas) == 1 and all(conditions.get(k) == 'True' for k in ('AbleToScale', 'ScalingActive'))
            if comparison:
                from .native_hpa import stored_object_errors
                actual = effective['actual_config']
                checks['hpa_configuration_evidence'] = _configured_native_hpa(run_dir, name, actual, expected_context)
                checks['hpa_identity_and_spec'] = (len(hpas) == 1 and bool(actual.get('hpa_uid')) and
                    not stored_object_errors(hpas[0], actual['hpa_name'], name, uid=actual['hpa_uid']))
                reset = json.loads((run_dir / 'hpa_reset_check.json').read_text())
                checks['hpa_teardown_and_restoration'] = _restored_native_hpa(reset, actual)
        summary = json.loads((run_dir / 'collection_summary.json').read_text())
        checks['collection_succeeded'] = summary.get('status') == 'ok'
        for key, minimum in [('total_requests', 1), ('mean_replicas', 1),
                             ('p95_ms', 0), ('e2e_p95_ms', 0), ('timeseries_datapoints', 1)]:
            checks[key] = finite(summary.get(key), minimum)
        traces = list(run_dir.glob('workload_*_timeseries.csv'))
        checks['one_workload_trace'] = len(traces) == 1
        if len(traces) == 1:
            with traces[0].open() as stream:
                rows = list(csv.DictReader(stream))
            numeric = [[float(r[k]) for k in ('elapsed_s', 'target_rps', 'actual_rps',
                                              'ok', 'errors', 'p50_ms', 'p95_ms', 'p99_ms')] for r in rows]
            checks['trace_finite'] = all(all(math.isfinite(v) and v >= 0 for v in row) for row in numeric)
            checks['trace_ordered'] = all(a[0] < b[0] for a, b in zip(numeric, numeric[1:]))
            checks['trace_has_traffic'] = sum(row[3] for row in numeric) > 0
            checks['trace_ticks'] = len(rows) >= criteria['min_trace_ticks']
            checks['workload_duration_observed'] = bool(numeric) and finite(result.get('duration_s'), 1) and numeric[-1][0] >= .8 * result['duration_s']
            # Closed-loop dispatch may miss its nominal RPS; retain achieved
            # traffic and errors. A smoke PASS does not certify offered load.
            details['trace_ticks'] = len(rows)
            details['successful_requests'] = sum(row[3] for row in numeric)
            details['failed_requests'] = sum(row[4] for row in numeric)
            details['last_trace_second'] = numeric[-1][0] if numeric else None
        if predictive:
            entries = json.loads((run_dir / 'controller_scale_log.json').read_text())
            predictions = [e for e in entries if e.get('prediction_made') is True and e.get('point_forecast')]
            checks['predictions_issued'] = len(predictions) >= criteria['min_predictions']
            checks['predictions_finite'] = bool(predictions) and all(
                all(finite(v, -1e100) for v in e[k]) and len(e[k]) == len(e['point_forecast'])
                for e in predictions for k in ('point_forecast', 'ci_lower', 'ci_upper'))
            checks['interval_bounds_ordered'] = all(
                all(lo <= hi for lo, hi in zip(e['ci_lower'], e['ci_upper'])) for e in predictions)
            checks['observed_replicas_logged'] = bool(entries) and all(
                finite(e.get('observed_replicas')) and finite(e.get('last_requested_replicas'), 1)
                for e in entries)
            times = [e['elapsed_s'] for e in entries]
            checks['controller_time_ordered'] = all(finite(t) for t in times) and all(a < b for a, b in zip(times, times[1:]))
            if cap is not None and not cap.interval_monitor:
                controller_config = json.loads((run_dir / 'controller_config.json').read_text())
                expected = preset_configuration(name)
                defaults = {'h0_only': False, 'point_bias_window': 0}
                actual = effective['actual_config']
                checks['preset_configuration_matches'] = all(actual.get(key) == value for key, value in expected.items())
                checks['controller_configuration_matches'] = all(
                    controller_config.get('method' if key == 'uq_method' else key, defaults.get(key)) == value
                    for key, value in expected.items())
                checks['confidence_finite'] = all(finite(e.get('confidence_score')) for e in predictions)
                corrected = cap.target_rule == 'corrected-h0'
                matched, targets, matched_count = _target_evidence(entries, expected, corrected)
                checks['causal_correction_recomputed'] = matched
                checks['raw_targets_recomputed'] = targets
                if corrected:
                    checks['point_errors_observed_later'] = matched_count >= criteria['min_validated_intervals']
                    details['matched_point_errors'] = matched_count
                checks['monitoring_disabled'] = all('coverage' not in e and 'recal' not in e and
                                                   'recalibrator' not in e and 'ladder' not in e for e in entries)
                details.update(predictions=len(predictions), target_rule=cap.target_rule)
                return {'kind': 'TECHNICAL_ASSURANCE_NOT_SCIENTIFIC_REPLICATION',
                        'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks, 'details': details}
            operator = json.loads((run_dir / 'operator_metrics_summary.json').read_text())
            validated = operator.get('coverage_monitor', {}).get('total_validated', 0)
            checks['intervals_observed_later'] = finite(validated, criteria['min_validated_intervals'])
            # Recompute FIFO matches independently of the controller's summary.
            pending, scored = deque(), []
            times = [e['elapsed_s'] for e in entries]
            checks['controller_time_ordered'] = all(finite(t) for t in times) and all(a < b for a, b in zip(times, times[1:]))
            for entry in entries:
                if finite(entry.get('rps')) and pending:
                    low, high = pending.popleft()
                    scored.append(low <= entry['rps'] <= high)
                if entry.get('prediction_made') and entry.get('ci_lower') and entry.get('ci_upper'):
                    pending.append((entry['ci_lower'][0], entry['ci_upper'][0]))
            checks['coverage_counts_recomputed'] = (len(scored) == validated and
                sum(scored) == operator.get('coverage_monitor', {}).get('total_covered'))
            details.update(predictions=len(predictions), validated_intervals=validated)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        checks['readable_complete_evidence'] = False
        details['error'] = str(error)
    return {'kind': 'TECHNICAL_ASSURANCE_NOT_SCIENTIFIC_REPLICATION',
            'status': 'pass' if checks and all(checks.values()) else 'fail',
            'checks': checks, 'details': details}
