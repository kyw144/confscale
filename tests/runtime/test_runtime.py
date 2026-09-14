"""Failure injection; every subprocess/cluster operation is mocked."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reference/stage3_scale'))
runner = importlib.import_module('orchestrator.run_matrix')


@pytest.mark.parametrize('failure', ['workload', 'collection', 'reset', 'missing_trace', 'controller'])
def test_cell_failure_never_reports_success(tmp_path, failure):
    method = Mock(name='method')
    method.name = 'confscale-pid'
    method.configure.return_value = True
    method.reset.return_value = failure != 'reset'
    method.get_state.return_value = {}
    method.controller_process = Mock() if failure == 'controller' else None
    if method.controller_process:
        method.controller_process.poll.return_value = 1
        method.controller_process.returncode = 1

    def workload(command, **kwargs):
        if command[0] == 'kubectl':
            return subprocess.CompletedProcess(command, 0, '{"items": []}', '')
        if failure != 'missing_trace':
            directory = Path(kwargs['cwd'])
            (directory / 'workload_B_timeseries.csv').write_text('elapsed_s,ok\n0,1\n')
        return subprocess.CompletedProcess(command, 1 if failure == 'workload' else 0, 'stdout', 'stderr')

    summary = {'status': 'degraded' if failure == 'collection' else 'ok'}
    with patch.object(runner.subprocess, 'run', side_effect=workload), \
         patch.object(runner, 'collect_metrics', return_value=summary), \
         patch.object(runner.time, 'sleep'):
        result = runner.execute_single_run(method, 'B', 1, 30, 50000,
                    'http://localhost:32080', 'http://localhost:9290', tmp_path)
    assert result['status'] != 'ok'
    method.reset.assert_called_once()
    saved = yaml.safe_load((Path(result['output_dir']) / 'run_config.yaml').read_text())
    assert saved['status'] == result['status']


def test_missing_model_cannot_become_hpa(tmp_path):
    from orchestrator import methods
    method = methods.get_method('confscale-pid')
    with patch.object(methods, 'UQ_MODELS_DIR', tmp_path), \
         patch.object(methods, 'kubectl_check', return_value=True), \
         patch.object(methods, 'kubectl'), \
         patch.object(methods.HPAMethod, 'configure', side_effect=AssertionError('fallback')):
        with pytest.raises(FileNotFoundError, match='refusing HPA fallback'):
            method.configure()


def test_unknown_method_override_fails():
    with pytest.raises(ValueError, match='Unknown method option'):
        runner._resolve_method_spec({'name': 'hpa-reactive', 'max_replica': 2})


def test_single_cell_dry_run_starts_no_forwarder():
    with patch.object(sys, 'argv', ['runner', '--method', 'hpa-reactive', '--workload', 'B', '--dry-run']), \
         patch.object(runner, 'PortForward', side_effect=AssertionError('created forwarder')):
        runner.main()


def test_scalar_queries_use_recorded_end_time():
    from collector.collect import PrometheusClient
    client = PrometheusClient('http://unused', evaluation_time=123)
    with patch.object(client, '_request', return_value={'data': {'result': []}}) as request:
        client.query('up')
    assert request.call_args.args == ('query', {'query': 'up', 'time': 123})


def test_seed_reproduces_workload_without_traffic(tmp_path):
    outputs = []
    for i in range(2):
        directory = tmp_path / str(i)
        subprocess.run([sys.executable, str(ROOT / 'reference/stage3_scale/workload_gen.py'),
                        'B', '--duration', '180', '--seed', '144', '--trace-only',
                        '--output-dir', str(directory)], check=True, capture_output=True,
                       env={**os.environ, 'CONFSCALE_ENABLE_REFERENCE_RUNTIME': '1'})
        outputs.append(next(directory.glob('*_timeseries.csv')).read_bytes())
    assert outputs[0] == outputs[1]


def test_render_pins_images_and_excludes_conflicting_hpa_and_crds(tmp_path):
    from confscale import cluster
    profile = cluster.load_profile(ROOT / 'configs/cluster.json')
    with patch.object(cluster, 'location', return_value=tmp_path):
        plan = cluster.render(profile, 'arm64')
    docs = list(yaml.safe_load_all((tmp_path / 'app.yaml').read_text()))
    assert all(d['kind'] not in ('PodMonitor', 'HorizontalPodAutoscaler') for d in docs)
    deployments = [d for d in docs if d['kind'] == 'Deployment']
    assert len(deployments) == 4
    for obj in deployments:
        container = obj['spec']['template']['spec']['containers'][0]
        if obj['metadata']['name'] != 'cache':
            assert container['imagePullPolicy'] == 'Never'
            assert {'name': 'CONFSCALE_ENABLE_REFERENCE_RUNTIME', 'value': '1'} in container['env']
    assert all('@sha256:' in image for image in plan['images'].values())
