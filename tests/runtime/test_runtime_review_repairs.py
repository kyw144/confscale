"""Offline integration checks for cleanup, sealed audits and invalid forecasts."""
from contextlib import contextmanager
import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from confscale import experiment
from confscale.planning_numpy import PointBiasCorrector, compute_target_replicas
from confscale.run_support import ROOT, sha256, write_json

sys.path.insert(0, str(ROOT / 'reference/stage3_scale'))
methods = importlib.import_module('orchestrator.methods')
controller = importlib.import_module('orchestrator.controller')
runner = importlib.import_module('orchestrator.run_matrix')


def hpa(name='foreign-worker-hpa', target='compute-worker'):
    return {'kind': 'HorizontalPodAutoscaler', 'metadata': {'name': name},
            'spec': {'scaleTargetRef': {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'name': target}}}


def baseline_kube(inventory, calls):
    def call(profile, *args, **kwargs):
        calls.append(args)
        if args[2:4] == ('get', 'hpa'):
            if isinstance(inventory, Exception):
                raise inventory
            return inventory if isinstance(inventory, str) else json.dumps(inventory)
        if args[2:4] == ('get', 'deployment/compute-worker'):
            return '{"spec": {"replicas": 1}, "status": {"readyReplicas": 1}}'
        return ''
    return call


@pytest.mark.parametrize('inventory', [
    {'items': [hpa()]}, {'items': [hpa('confscale-cpu-stock-' + 'a' * 32)]},
    {'items': {}}, {'items': [None]}, {'items': [{'spec': None}]}, {}, 'not json',
    subprocess.CalledProcessError(1, ['kubectl'], stderr='inventory unavailable'),
])
def test_outer_cleanup_never_scales_without_proven_hpa_absence(inventory):
    calls = []
    with patch.object(experiment, 'kube', side_effect=baseline_kube(inventory, calls)):
        with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
            experiment.restore_baseline({})
    assert not any(args[2] == 'scale' for args in calls)
    assert all('foreign-worker-hpa' not in args for args in calls)


@pytest.mark.parametrize('inventory', [{'items': []}, {'items': [hpa(target='another-deployment')]}])
def test_outer_cleanup_preserves_successful_legacy_restoration(inventory):
    calls = []
    with patch.object(experiment, 'kube', side_effect=baseline_kube(inventory, calls)):
        experiment.restore_baseline({})
    inventory_index = next(i for i, args in enumerate(calls) if args[2:4] == ('get', 'hpa'))
    scale_index = next(i for i, args in enumerate(calls) if args[2] == 'scale')
    assert inventory_index < scale_index
    assert calls[0][2:6] == ('delete', 'hpa', 'compute-worker', 'compute-worker-hpa')


def test_failed_native_reset_remains_failed_through_outer_run_cleanup(tmp_path):
    config, profile = experiment.load_config(ROOT / 'configs/comparison-smoke.json')
    config['methods'] = ['hpa-cpu-stock']
    cluster = tmp_path / 'cluster'
    cluster.mkdir()
    owner = {'name': profile['name'], 'cluster_uid': 'test-cluster'}
    write_json(cluster / 'deployment.json', {'profile': profile, 'ownership': owner, 'rendered_sha256': {}})
    calls, surviving = [], []
    method = SimpleNamespace(name='hpa-cpu-stock', get_state=lambda: {
        'method': 'hpa-cpu-stock', 'hpa_name': 'surviving-uuid', 'hpa_uid': 'issued-uid'})

    def kube(profile, *args, **kwargs):
        return baseline_kube({'items': list(surviving)}, calls)(profile, *args, **kwargs)

    def failed_cell(*args, **kwargs):
        directory = Path(args[7]) / 'failed-cell'
        directory.mkdir()
        surviving.append(hpa('surviving-uuid'))
        write_json(directory / 'hpa_reset_check.json', {'ok': False, 'hpa_uid': 'issued-uid'})
        return {'output_dir': str(directory), 'method': 'hpa-cpu-stock', 'status': 'failed',
                'reset_ok': False, 'workload_returncode': 0}

    @contextmanager
    def forward(*args):
        yield 'http://unused'

    response = MagicMock()
    response.__enter__.return_value.status = 200
    verdict = {'status': 'fail', 'checks': {'method_reset_succeeded': False}}
    with patch.object(experiment, 'verify_models', return_value={}), \
         patch.object(experiment, 'ownership', return_value=owner), \
         patch.object(experiment, 'location', return_value=cluster), \
         patch.object(experiment.socket, 'socket', return_value=MagicMock()), \
         patch.object(experiment, 'environment', return_value={}), \
         patch.object(experiment, 'snapshot', return_value={'items': []}), \
         patch.object(experiment, 'forward', forward), \
         patch.object(experiment, 'metrics_preflight', return_value={}), \
         patch.object(experiment, 'urlopen', return_value=response), \
         patch.object(experiment, 'kube', side_effect=kube), \
         patch.object(experiment, 'assess', return_value=verdict), \
         patch.object(methods, 'set_thread_kube_context'), \
         patch.object(methods, 'get_method', return_value=method), \
         patch.object(runner, 'execute_single_run', side_effect=failed_cell):
        receipt = experiment.run(config, profile, tmp_path / 'output')
    assert receipt['status'] == 'failed'
    assert 'HPA still targets' in receipt['restoration']
    assert receipt['runs'][0]['assurance'] == verdict
    saved = json.loads((tmp_path / 'output/failed-cell/assurance.json').read_text())
    assert saved['checks']['method_reset_succeeded'] is False
    assert sum(args[2] == 'scale' for args in calls) == 1  # initial clean baseline only
    assert not any(args[2] == 'delete' and 'surviving-uuid' in args for args in calls)
    assert not (cluster / 'experiment.lock').exists()


def audit_fixture(output, legacy=False, methods_list=None):
    config, profile = experiment.load_config(ROOT / 'configs/smoke.json')
    config['methods'] = methods_list or ['static-2']
    config['duration_s'] = 30
    config['assurance'] = {'min_trace_ticks': 2, 'min_predictions': 2, 'min_validated_intervals': 1}
    planned = experiment.plan(config, profile)
    if legacy:
        for cell in planned['cells']:
            cell.pop('capabilities')
            cell.pop('preset_configuration')
    runs = []
    for i, cell in enumerate(planned['cells']):
        directory = output / f'cell-{i}'
        directory.mkdir(parents=True)
        write_json(directory / 'execution.json', {'method': cell['method'], 'status': 'ok',
            'workload_returncode': 0, 'reset_ok': True, 'duration_s': 30})
        write_json(directory / 'collection_summary.json', {'status': 'ok', 'total_requests': 20,
            'mean_replicas': 2, 'p95_ms': 20, 'e2e_p95_ms': 23, 'timeseries_datapoints': 20})
        (directory / 'workload_B_timeseries.csv').write_text(
            'elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\n'
            '0,10,10,10,0,12,23,30\n30,10,10,10,0,12,23,30\n')
        runs.append({'cell': cell, 'directory': directory.name})
    write_json(output / 'plan.json', planned)
    receipt = {'status': 'pass', 'restoration': 'pass', 'runs': runs}
    reseal(output, receipt)
    return planned, receipt


def reseal(output, receipt):
    receipt['output_sha256'] = {p.relative_to(output).as_posix(): sha256(p)
        for p in output.rglob('*') if p.is_file() and p != output / 'receipt.json'}
    write_json(output / 'receipt.json', receipt)


@pytest.mark.parametrize('legacy', [False, True])
def test_full_valid_current_and_legacy_plans_audit_offline(tmp_path, legacy):
    audit_fixture(tmp_path, legacy)
    with patch('subprocess.run', side_effect=AssertionError('process during audit')):
        assert experiment.audit(tmp_path)['cells_verified'] == 1


@pytest.mark.parametrize('mutation', ['omit_hash', 'extra_file', 'extra_hash', 'nested_receipt', 'symlink'])
def test_audit_requires_the_complete_file_inventory(tmp_path, mutation):
    _, receipt = audit_fixture(tmp_path)
    if mutation == 'omit_hash':
        receipt['output_sha256'].pop('cell-0/collection_summary.json')
        write_json(tmp_path / 'cell-0/collection_summary.json', {'status': 'ok', 'total_requests': 999,
            'mean_replicas': 2, 'p95_ms': 20, 'e2e_p95_ms': 23, 'timeseries_datapoints': 20})
    elif mutation == 'extra_file':
        (tmp_path / 'unsealed.log').write_text('extra')
    elif mutation == 'extra_hash':
        receipt['output_sha256']['missing.log'] = '0' * 64
    elif mutation == 'nested_receipt':
        write_json(tmp_path / 'cell-0/receipt.json', {'must': 'also be hashed'})
    else:
        (tmp_path / 'alias').symlink_to(tmp_path / 'cell-0/collection_summary.json')
    write_json(tmp_path / 'receipt.json', receipt)
    with pytest.raises(ValueError, match='integrity failure'):
        experiment.audit(tmp_path)


def test_nested_receipt_is_included_when_correctly_sealed(tmp_path):
    _, receipt = audit_fixture(tmp_path)
    write_json(tmp_path / 'cell-0/receipt.json', {'nested': True})
    reseal(tmp_path, receipt)
    assert experiment.audit(tmp_path)['status'] == 'pass'


def sealed_native_fixture(output, name):
    """Reuse complete native evidence produced solely by the in-memory adapter."""
    import test_assurance_artifacts as artifacts
    config, profile = experiment.load_config(ROOT / 'configs/comparison-smoke.json')
    config['methods'] = [name]
    config['duration_s'] = 60
    config['assurance'] = artifacts.CRITERIA
    planned = experiment.plan(config, profile)
    cell_dir = output / 'native-cell'
    cell_dir.mkdir(parents=True)
    with patch.object(artifacts, 'CONTEXT', planned['context']), \
         patch('subprocess.run', side_effect=AssertionError('external process in native fixture')):
        artifacts.native_fixture(cell_dir, name)
    write_json(output / 'plan.json', planned)
    receipt = {'status': 'pass', 'restoration': 'pass', 'runs': [
        {'cell': planned['cells'][0], 'directory': cell_dir.name}]}
    reseal(output, receipt)
    return planned, receipt


@pytest.mark.parametrize('name', ['hpa-cpu-stock', 'hpa-cpu-tuned'])
def test_native_audit_rejects_a_wrong_expected_context(tmp_path, name):
    planned, receipt = sealed_native_fixture(tmp_path, name)
    assert experiment.audit(tmp_path)['status'] == 'pass'
    # Keep plan metadata internally consistent so assurance must reject the
    # mismatch with the context actually recorded in every native command.
    planned['cluster_profile']['name'] = 'confscale-other'
    planned['context'] = 'kind-confscale-other'
    write_json(tmp_path / 'plan.json', planned)
    reseal(tmp_path, receipt)
    with patch.object(experiment, 'assess', wraps=experiment.assess) as assess, \
         patch('subprocess.run', side_effect=AssertionError('process during offline audit')):
        with pytest.raises(ValueError, match='no longer passes assurance'):
            experiment.audit(tmp_path)
    assert assess.call_args.kwargs['expected_context'] == 'kind-confscale-other'


@pytest.mark.parametrize('name', ['hpa-cpu-stock', 'hpa-cpu-tuned'])
@pytest.mark.parametrize('absolute', [False, True])
def test_native_audit_rejects_a_cell_directory_outside_the_sealed_run(tmp_path, name, absolute):
    output = tmp_path / 'sealed-run'
    _, receipt = sealed_native_fixture(output, name)
    outside = tmp_path / 'outside-cell'
    shutil.copytree(output / 'native-cell', outside)
    receipt['runs'][0]['directory'] = str(outside) if absolute else '../outside-cell'
    write_json(output / 'receipt.json', receipt)
    with patch.object(experiment, 'assess', side_effect=AssertionError('read escaped cell')), \
         patch('subprocess.run', side_effect=AssertionError('process during offline audit')):
        with pytest.raises(ValueError, match='Cell directory escapes run'):
            experiment.audit(output)


@pytest.mark.parametrize('name', ['hpa-cpu-stock', 'hpa-cpu-tuned'])
def test_relocated_native_run_audits_without_original_paths_or_dependencies(tmp_path, name):
    original = tmp_path / 'original-run'
    sealed_native_fixture(original, name)
    moved = tmp_path / 'archive' / 'relocated-run'
    shutil.copytree(original, moved)
    original_manifest = json.loads((moved / 'native-cell/hpa_configure_check.json').read_text())['manifest_path']
    shutil.rmtree(original)
    assert not Path(original_manifest).exists()
    with patch('subprocess.run', side_effect=AssertionError('process during offline audit')):
        assert experiment.audit(moved)['cells_verified'] == 1
    result = subprocess.run([sys.executable, '-S', '-m', 'confscale.experiment', 'audit',
                             '--output', str(moved)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'pass'


@pytest.mark.parametrize('mutation', ['empty', 'missing_config', 'missing_cell', 'seed', 'order', 'directory'])
def test_audit_rebuilds_the_nonempty_matrix_from_configuration(tmp_path, mutation):
    planned, receipt = audit_fixture(tmp_path, methods_list=['static-2', 'static-4'])
    if mutation == 'empty':
        planned['cells'] = []
        receipt['runs'] = []
    elif mutation == 'missing_config':
        planned['config'] = {'assurance': planned['config']['assurance']}
    elif mutation == 'missing_cell':
        planned['cells'].pop()
        receipt['runs'].pop()
    elif mutation == 'seed':
        planned['config']['seeds'] = [123]
    elif mutation == 'order':
        planned['cells'].reverse()
        receipt['runs'].reverse()
    else:
        receipt['runs'][1]['directory'] = receipt['runs'][0]['directory']
    write_json(tmp_path / 'plan.json', planned)
    reseal(tmp_path, receipt)
    with pytest.raises(ValueError):
        experiment.audit(tmp_path)


@pytest.mark.parametrize('h0,bias', [(float('nan'), 0), (float('inf'), 0), (-float('inf'), 0),
                                  (28, float('nan')), (28, float('inf')), (28, -float('inf'))])
@pytest.mark.parametrize('planner', [compute_target_replicas, controller.compute_target_replicas])
def test_corrected_direct_planners_reject_nonfinite_inputs(planner, h0, bias):
    with pytest.raises(ValueError, match='finite h0 and bias'):
        planner(np.array([h0, 100.]), .5, np.array([100., 200.]), h0_only=True, point_bias=bias)


@pytest.mark.parametrize('corrector', [PointBiasCorrector, controller.PointBiasCorrector])
def test_nonfinite_forecast_is_missing_in_the_bias_state(corrector):
    state = corrector(10).step(20, {'point_forecast': np.array([float('nan'), 100.])})
    assert state == {'raw_h0': None, 'bias_rps': 0, 'matched_errors': 0, 'corrected_rps': None}


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_controller_holds_prior_target_when_corrected_h0_is_invalid(tmp_path, invalid):
    def pred(h0):
        return {'point_forecast': np.array([h0, 100.], dtype=np.float32),
                'ci_lower': np.array([0., 90.], dtype=np.float32),
                'ci_upper': np.array([40., 110.], dtype=np.float32), 'confidence_score': .5}
    uq = SimpleNamespace(h=2, k=2, alpha=.1,
                         predict_with_uncertainty=Mock(side_effect=[pred(28), pred(invalid)]))
    clock = SimpleNamespace(now=0.0)
    def sleep(seconds):
        clock.now += seconds
    argv = ['controller', '--method', 'scp', '--model-dir', 'unused', '--h0-only', '--point-bias-window', '10',
            '--duration', '31', '--output-dir', str(tmp_path)]
    with patch.object(sys, 'argv', argv), patch.object(controller, 'start_http_server'), \
         patch.object(controller.SplitConformal, 'load', return_value=uq), \
         patch.object(controller, 'query_prometheus', return_value=21), \
         patch.object(controller, 'query_prometheus_range', return_value=[21, 21]), \
         patch.object(controller, 'kubectl_get_replicas', return_value=4), \
         patch.object(controller, 'kubectl_scale') as scale, \
         patch.object(controller, '_update_prometheus_metrics'), \
         patch.object(controller.signal, 'signal'), \
         patch.object(controller.time, 'time', side_effect=lambda: clock.now), \
         patch.object(controller.time, 'sleep', side_effect=sleep):
        controller.main()
    rows = json.loads((tmp_path / 'controller_scale_log.json').read_text())
    assert len(rows) == 2
    assert rows[0]['prediction_made'] is True and rows[0]['raw_target'] == 4
    assert rows[0]['point_forecast_dtype'] == rows[0]['ci_upper_dtype'] == 'float32'
    assert rows[1]['prediction_made'] is False and rows[1]['point_forecast'] == []
    assert rows[1]['point_correction']['raw_h0'] is None
    assert rows[1]['raw_target'] == rows[1]['target_replicas'] == 4
    assert [call.args[-1] for call in scale.call_args_list] == [1, 4]


def test_explicit_context_accessor_never_falls_back_to_default():
    previous = methods.explicit_thread_kube_context()
    try:
        methods.clear_thread_kube_context()
        assert methods.explicit_thread_kube_context() is None
        assert methods.current_kube_context() == methods.KUBE_CONTEXT
        methods.set_thread_kube_context('kind-explicit-test')
        assert methods.explicit_thread_kube_context() == 'kind-explicit-test'
    finally:
        methods.clear_thread_kube_context()
        if previous is not None:
            methods.set_thread_kube_context(previous)
