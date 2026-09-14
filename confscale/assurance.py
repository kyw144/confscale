"""Fail-closed technical checks on retained experiment outputs, offline."""
import csv
from collections import deque
import json
import math
from pathlib import Path


def finite(value, minimum=0):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def assess(run_dir, predictive, criteria):
    """Technical validity only. Never compare newly observed values to a paper claim."""
    run_dir = Path(run_dir)
    checks = {}
    details = {}
    try:
        result = json.loads((run_dir / 'execution.json').read_text())
        checks['execution_succeeded'] = result.get('status') == 'ok' and result.get('workload_returncode') == 0
        checks['method_reset_succeeded'] = result.get('reset_ok') is True
        if result.get('method') == 'hpa-reactive':
            state = json.loads((run_dir / 'before_reset_status.json').read_text())
            hpas = [o for o in state['items'] if o['kind'] == 'HorizontalPodAutoscaler'
                    and o['spec']['scaleTargetRef']['name'] == 'compute-worker']
            conditions = {c['type']: c['status'] for h in hpas for c in h.get('status', {}).get('conditions', [])}
            checks['hpa_active'] = len(hpas) == 1 and all(conditions.get(k) == 'True' for k in ('AbleToScale', 'ScalingActive'))
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
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        checks['readable_complete_evidence'] = False
        details['error'] = str(error)
    return {'kind': 'TECHNICAL_ASSURANCE_NOT_SCIENTIFIC_REPLICATION',
            'status': 'pass' if checks and all(checks.values()) else 'fail',
            'checks': checks, 'details': details}
