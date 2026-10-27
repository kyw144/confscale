"""Cross-check native execution artifacts and exact numerical replay offline."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from confscale.assurance import assess, _command_matches, _target_evidence, _tier2_rate
from confscale.capabilities import capability_record, preset_configuration
from confscale.native_hpa import make_native_hpa
from confscale.run_support import ROOT, write_json


CRITERIA = {'min_trace_ticks': 2, 'min_predictions': 2, 'min_validated_intervals': 1}
CONTEXT = 'kind-assurance-test'


@dataclass
class BaseMethod:
    name: str
    description: str = ''
    run_dir: Path | None = None


def native_fixture(path, name='hpa-cpu-stock', retry=False):
    """Create real adapter artifacts using an in-memory API, never a subprocess."""
    items = []
    attempts = 0
    def kubectl(command, timeout=30):
        nonlocal attempts
        output = ''
        if command[0] == 'apply':
            obj = json.loads(Path(command[2]).read_text())
            obj['metadata']['uid'] = 'api-issued-uid'
            obj['status'] = {'conditions': [{'type': key, 'status': 'True'}
                                            for key in ('AbleToScale', 'ScalingActive')]}
            items.append(obj)
        elif command[:2] == ['get', 'hpa']:
            if command[2] == '-n':
                output = json.dumps({'items': items})
            else:
                attempts += 1
                if retry and attempts == 1:
                    return subprocess.CompletedProcess(command, 1, '', 'temporary error')
                output = json.dumps(items[0])
        elif command[0] == 'delete':
            items.clear()
        elif command[:2] == ['get', 'deployment']:
            output = json.dumps({'spec': {'replicas': 1}, 'status': {'replicas': 1, 'readyReplicas': 1}})
        return subprocess.CompletedProcess(command, 0, output, '')
    methods = SimpleNamespace(MethodConfig=BaseMethod, explicit_thread_kube_context=lambda: CONTEXT, kubectl=kubectl)
    method = make_native_hpa(name, methods)
    method.run_dir = path
    with patch.dict(os.environ, {'KUBECONFIG': str(path / 'private-config')}):
        assert method.configure()
        hpa = deepcopy(items[0])
        assert method.reset()
    actual = method.get_state()
    write_json(path / 'execution.json', {'status': 'ok', 'workload_returncode': 0, 'reset_ok': True,
                                        'duration_s': 60, 'method': name})
    write_json(path / 'effective_method.json', {'method': name, 'capabilities': capability_record(name),
        'expected_config': preset_configuration(name), 'actual_config': actual})
    write_json(path / 'collection_summary.json', {'status': 'ok', 'total_requests': 20, 'mean_replicas': 2,
        'p95_ms': 20, 'e2e_p95_ms': 23, 'timeseries_datapoints': 20})
    (path / 'workload_B_timeseries.csv').write_text('elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\n'
        '0,10,10,10,0,12,23,30\n60,10,10,10,0,12,23,30\n')
    write_json(path / 'before_reset_status.json', {'items': [hpa]})
    return hpa, deepcopy(method.reset_evidence), actual


@pytest.mark.parametrize('name', ['hpa-cpu-stock', 'hpa-cpu-tuned'])
@pytest.mark.parametrize('retry', [False, True])
def test_complete_native_artifacts_pass_and_are_dependency_free(tmp_path, name, retry):
    native_fixture(tmp_path, name, retry)
    verdict = assess(tmp_path, False, CRITERIA, method_name=name, expected_context=CONTEXT)
    assert verdict['status'] == 'pass', verdict
    command = [sys.executable, '-S', '-c',
        'import json,sys; from confscale.assurance import assess; '
        'r=assess(sys.argv[1],False,json.loads(sys.argv[3]),method_name=sys.argv[2],expected_context=sys.argv[4]); '
        'print(json.dumps(r)); sys.exit(r["status"]!="pass")', str(tmp_path), name, json.dumps(CRITERIA), CONTEXT]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('filename', ['hpa_v2.yaml', 'hpa_initial_inventory.json', 'hpa_stored.json',
    'hpa_stored_readback.json', 'hpa_configure_check.json', 'hpa_reset_check.json'])
@pytest.mark.parametrize('invalid', [False, True])
def test_missing_or_malformed_native_proof_fails(tmp_path, filename, invalid):
    native_fixture(tmp_path)
    if invalid:
        (tmp_path / filename).write_text('{invalid')
    else:
        (tmp_path / filename).unlink()
    assert assess(tmp_path, False, CRITERIA, expected_context=CONTEXT)['status'] == 'fail'


@pytest.mark.parametrize('mutation', ['behavior', 'initial_target', 'initial_command', 'initial_returncode',
    'journal_context', 'journal_path_hash', 'journal_uid', 'journal_interrupted', 'journal_order',
    'apply_path', 'manifest_hash', 'stored_uid', 'stored_spec', 'readback_object', 'readback_context',
    'readback_namespace', 'reset_context', 'reset_path_hash', 'reset_inventory_uid', 'reset_journal', 'later_uid'])
def test_contradictory_native_proof_fails(tmp_path, mutation):
    hpa, reset, _ = native_fixture(tmp_path)
    def edit(filename, change):
        record = json.loads((tmp_path / filename).read_text())
        change(record)
        write_json(tmp_path / filename, record)
    if mutation == 'behavior':
        edit('hpa_v2.yaml', lambda obj: obj['spec'].update(behavior={}))
    elif mutation == 'initial_target':
        edit('hpa_initial_inventory.json', lambda obj: obj.update(stdout=json.dumps({'items': [hpa]})))
    elif mutation == 'initial_command':
        edit('hpa_initial_inventory.json', lambda obj: obj['command'].__setitem__(3, 'other'))
    elif mutation == 'initial_returncode':
        edit('hpa_initial_inventory.json', lambda obj: obj.update(returncode=1))
    elif mutation.startswith('journal_'):
        key = mutation.removeprefix('journal_')
        def change(obj):
            if key == 'order': obj['commands'][1:3] = reversed(obj['commands'][1:3])
            else: obj[{'context': 'context', 'path_hash': 'kubeconfig_path_sha256', 'uid': 'hpa_uid',
                       'interrupted': 'interrupted'}[key]] = True if key == 'interrupted' else 'different'
        edit('hpa_configure_check.json', change)
    elif mutation == 'apply_path':
        edit('hpa_configure_check.json', lambda obj: obj['commands'][3]['command'].__setitem__(2, 'other/hpa_v2.yaml'))
    elif mutation == 'manifest_hash':
        edit('hpa_configure_check.json', lambda obj: obj.update(manifest_sha256='0' * 64))
    elif mutation == 'stored_uid':
        edit('hpa_stored.json', lambda obj: obj['metadata'].update(uid='different'))
    elif mutation == 'stored_spec':
        edit('hpa_stored.json', lambda obj: obj['spec'].update(maxReplicas=21))
    elif mutation == 'readback_object':
        edit('hpa_stored_readback.json', lambda rows: rows[-1].update(stdout='{}'))
    elif mutation == 'readback_context':
        edit('hpa_stored_readback.json', lambda rows: rows[-1].update(context='kind-other'))
    elif mutation == 'readback_namespace':
        edit('hpa_stored_readback.json', lambda rows: rows[-1]['command'].__setitem__(4, 'other'))
    elif mutation == 'reset_context': reset['context'] = 'kind-other'
    elif mutation == 'reset_path_hash': reset['kubeconfig_path_sha256'] = '0' * 64
    elif mutation == 'reset_inventory_uid':
        obj = json.loads(reset['inventory']['stdout']); obj['items'][0]['metadata']['uid'] = 'different'
        reset['inventory']['stdout'] = json.dumps(obj)
    elif mutation == 'reset_journal': reset['commands'] = []
    else:
        hpa['metadata']['uid'] = 'different'
        write_json(tmp_path / 'before_reset_status.json', {'items': [hpa]})
    if mutation.startswith('reset_'):
        write_json(tmp_path / 'hpa_reset_check.json', reset)
    assert assess(tmp_path, False, CRITERIA, expected_context=CONTEXT)['status'] == 'fail'


def test_plan_context_must_match_native_routing(tmp_path):
    native_fixture(tmp_path)
    assert assess(tmp_path, False, CRITERIA, expected_context='kind-other')['status'] == 'fail'


def test_a_successful_but_contradictory_first_readback_cannot_be_hidden_by_retry(tmp_path):
    hpa, _, _ = native_fixture(tmp_path, retry=True)
    journal = json.loads((tmp_path / 'hpa_configure_check.json').read_text())
    reads = json.loads((tmp_path / 'hpa_stored_readback.json').read_text())
    hpa['metadata']['uid'] = 'replaced'
    reads[0].update(state='completed', returncode=0, stdout=json.dumps(hpa), stderr='', errors=[])
    journal['commands'][4:] = reads
    write_json(tmp_path / 'hpa_configure_check.json', journal)
    write_json(tmp_path / 'hpa_stored_readback.json', reads)
    assert assess(tmp_path, False, CRITERIA, expected_context=CONTEXT)['status'] == 'fail'


@pytest.mark.parametrize('mutation', ['zero_returncode', 'interrupted', 'zero_and_interrupted', 'boolean_returncode',
                                    'null_returncode', 'empty_error', 'no_outcome'])
def test_contradictory_failure_record_cannot_hide_wrong_uid(tmp_path, mutation):
    hpa, _, _ = native_fixture(tmp_path, retry=True)
    journal = json.loads((tmp_path / 'hpa_configure_check.json').read_text())
    reads = json.loads((tmp_path / 'hpa_stored_readback.json').read_text())
    hpa['metadata']['uid'] = 'wrong-uid'
    row = reads[0]
    row.update(state='failed', error='claimed client error', stdout=json.dumps(hpa))
    if mutation == 'zero_returncode': row['returncode'] = 0
    elif mutation == 'interrupted': row['interrupted'] = True
    elif mutation == 'zero_and_interrupted': row.update(returncode=0, interrupted=True)
    elif mutation == 'boolean_returncode': row['returncode'] = True
    elif mutation == 'null_returncode': row['returncode'] = None
    elif mutation == 'empty_error': row.update(error='')
    else:
        row.pop('returncode'); row.pop('error')
    journal['commands'][4:] = reads
    write_json(tmp_path / 'hpa_configure_check.json', journal)
    write_json(tmp_path / 'hpa_stored_readback.json', reads)
    assert assess(tmp_path, False, CRITERIA, expected_context=CONTEXT)['status'] == 'fail'


def test_timeout_failure_record_with_partial_output_can_be_retried(tmp_path):
    _, _, actual = native_fixture(tmp_path, retry=True)
    journal = json.loads((tmp_path / 'hpa_configure_check.json').read_text())
    reads = json.loads((tmp_path / 'hpa_stored_readback.json').read_text())
    row = reads[0]
    row.pop('returncode')
    row.update(error='TimeoutExpired: read timed out', stdout='partial', stderr='partial diagnostics')
    assert _command_matches(row, row['command'], actual, success=False)
    journal['commands'][4:] = reads
    write_json(tmp_path / 'hpa_configure_check.json', journal)
    write_json(tmp_path / 'hpa_stored_readback.json', reads)
    verdict = assess(tmp_path, False, CRITERIA, expected_context=CONTEXT)
    assert verdict['status'] == 'pass', verdict


def test_float32_tier2_boundary_replays_exactly_and_rejects_adjacent_target():
    upper = float(np.float32(28.7875))
    row = {'prediction_made': True, 'rps': 0, 'point_forecast': [0.0], 'ci_upper': [upper],
           'point_forecast_dtype': 'float32', 'ci_upper_dtype': 'float32', 'confidence_score': .2,
           'raw_tier': 2, 'raw_target': 5, 'target_replicas': 5}
    assert math.ceil((2 * upper / 1.645) / 7) == 6
    assert _tier2_rate(row, 2) == 35.0
    assert _target_evidence([row], preset_configuration('confscale-scp-frozen'), False)[1]
    row['raw_target'] = 6
    assert not _target_evidence([row], preset_configuration('confscale-scp-frozen'), False)[1]


@pytest.mark.parametrize('point_dtype,upper_dtype', [('float32', 'float32'), ('float32', 'float64'), ('float64', 'float64')])
def test_tier2_replay_matches_numpy_operation_rounding(point_dtype, upper_dtype):
    for p, u in ((0., 28.7875), (12.3, 16.4), (11.5, 34.66), (-1.2, 1.1)):
        point = np.array([p], dtype=point_dtype); upper = np.array([u], dtype=upper_dtype)
        row = {'point_forecast': point.tolist(), 'ci_upper': upper.tolist(),
               'point_forecast_dtype': point_dtype, 'ci_upper_dtype': upper_dtype}
        assert _tier2_rate(row, 2) == float(np.max(point + 2 * ((upper - point) / 1.645)))


def test_corrected_point_requires_actual_cumulative_matches(tmp_path):
    from test_comparison_methods import comparison_fixture
    name = 'confscale-point-corrected'
    rows = comparison_fixture(tmp_path, name)
    for row in rows:
        row['rps'] = None
        row['point_correction'].update(bias_rps=0, matched_errors=0, corrected_rps=row['point_forecast'][0])
        row['raw_target'] = math.ceil(row['point_forecast'][0] / 7)
    write_json(tmp_path / 'controller_scale_log.json', rows)
    result = assess(tmp_path, True, CRITERIA, method_name=name)
    assert result['checks']['causal_correction_recomputed']
    assert not result['checks']['point_errors_observed_later']
    assert result['status'] == 'fail'


def test_valid_pair_count_can_exceed_the_ten_error_buffer():
    from confscale.planning_numpy import PointBiasCorrector
    corrector = PointBiasCorrector(10)
    rows = []
    for index in range(14):
        prediction = {'point_forecast': np.array([14.]), 'ci_upper': np.array([15.]), 'confidence_score': .1}
        correction = corrector.step(14., prediction)
        rows.append({'rps': 14., 'prediction_made': True, 'point_forecast': [14.], 'ci_upper': [15.],
                     'confidence_score': .1, 'point_correction': correction,
                     'raw_tier': 1, 'raw_target': 2, 'target_replicas': 2})
    matched, targets, count = _target_evidence(rows, preset_configuration('confscale-point-corrected'), True)
    assert matched and targets and count == 13
    assert rows[-1]['point_correction']['matched_errors'] == 10


def test_cumulative_match_requirement_can_exceed_buffer_size(tmp_path):
    from test_comparison_methods import comparison_fixture
    from confscale.planning_numpy import PointBiasCorrector
    name = 'confscale-point-corrected'
    comparison_fixture(tmp_path, name)
    corrector = PointBiasCorrector(10)
    rows = []
    for index in range(14):
        correction = corrector.step(14., {'point_forecast': np.array([14.])})
        rows.append({'elapsed_s': index * 30, 'rps': 14., 'prediction_made': True,
            'point_forecast': [14.], 'ci_lower': [13.], 'ci_upper': [15.], 'confidence_score': .1,
            'point_correction': correction, 'raw_tier': 1, 'raw_target': 2, 'target_replicas': 2,
            'observed_replicas': 2, 'last_requested_replicas': 2})
    write_json(tmp_path / 'controller_scale_log.json', rows)
    verdict = assess(tmp_path, True, {**CRITERIA, 'min_validated_intervals': 12}, method_name=name)
    assert verdict['status'] == 'pass', verdict
    assert verdict['details']['matched_point_errors'] == 13


@pytest.mark.parametrize('mutation', ['missing', 'unsupported', 'inconsistent_values'])
def test_tier2_dtype_evidence_is_required_and_exact(mutation):
    row = {'point_forecast': [0.0], 'ci_upper': [float(np.float32(28.7875))],
           'point_forecast_dtype': 'float32', 'ci_upper_dtype': 'float32'}
    if mutation == 'missing': row.pop('ci_upper_dtype')
    elif mutation == 'unsupported': row['ci_upper_dtype'] = 'float16'
    else: row['ci_upper'] = [28.7875]
    with pytest.raises((KeyError, ValueError)):
        _tier2_rate(row, 2)


def test_signed_error_replay_matches_statistics_mean_under_cancellation():
    from confscale.planning_numpy import PointBiasCorrector
    corrector = PointBiasCorrector(10)
    rows = []
    for h0 in (1e16, 1., -1e16, 7.1):
        correction = corrector.step(0., {'point_forecast': np.array([h0])})
        raw = min(20, max(1, math.ceil(correction['corrected_rps'] / 7)))
        rows.append({'rps': 0., 'prediction_made': True, 'point_forecast': [h0], 'ci_upper': [h0 + 1],
                     'confidence_score': .1, 'point_correction': correction,
                     'raw_tier': 1, 'raw_target': raw, 'target_replicas': raw})
    assert rows[-1]['point_correction']['bias_rps'] == 1 / 3
    assert rows[-1]['raw_target'] == 1
    assert _target_evidence(rows, preset_configuration('confscale-point-corrected'), True) == (True, True, 3)
