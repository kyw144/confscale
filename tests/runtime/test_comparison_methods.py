"""Offline contracts for explicit comparison presets."""
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import subprocess
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from confscale.assurance import assess
from confscale.capabilities import (METHODS, LEGACY_METHODS, capabilities, capability_record,
                                   configure_method, preset_configuration)
from confscale.experiment import audit, load_config, plan, verify_models
from confscale.run_support import ROOT, sha256, write_json
from confscale.planning_numpy import PointBiasCorrector, compute_target_replicas

sys.path.insert(0, str(ROOT / 'reference/stage3_scale'))
methods = importlib.import_module('orchestrator.methods')
controller = importlib.import_module('orchestrator.controller')


def prediction(h0, h1=100):
    return {'point_forecast': np.array([h0, h1]), 'ci_lower': np.array([h0 - 1, h1 - 1]),
            'ci_upper': np.array([h0 + 1, h1 + 1]), 'confidence_score': .5}


def test_correction_is_causal_signed_and_has_no_warm_start():
    corrector = PointBiasCorrector(10)
    assert corrector.step(10, prediction(28)) == {
        'raw_h0': 28, 'bias_rps': 0, 'matched_errors': 0, 'corrected_rps': 28}
    assert corrector.step(21, prediction(35))['bias_rps'] == 7
    state = corrector.step(42, prediction(35))
    assert state['bias_rps'] == 0  # signed errors +7 and -7 cancel
    assert state['matched_errors'] == 2
    assert PointBiasCorrector(10).step(1e9, prediction(28))['corrected_rps'] == 28


def test_missing_steps_do_not_retain_unmatched_forecasts():
    corrector = PointBiasCorrector(10)
    corrector.step(10, prediction(28))
    state = corrector.step(None, None)
    assert state['matched_errors'] == 0
    state = corrector.step(21, prediction(35))
    assert state['matched_errors'] == 0  # late recovery cannot match the old 28
    assert corrector.step(28, prediction(42))['bias_rps'] == 7
    corrector.step(float('nan'), None)
    assert corrector.step(14, prediction(28))['matched_errors'] == 1


def test_buffer_retains_only_ten_actual_matches_and_clips_demand():
    corrector = PointBiasCorrector(10)
    for i in range(12):
        state = corrector.step(0, prediction(i * 2))
    assert state['matched_errors'] == 10
    assert state['bias_rps'] == 11  # preceding forecasts 2..20
    assert corrector.step(0, prediction(-10))['corrected_rps'] == 0
    assert PointBiasCorrector(0).step(0, prediction(28))['matched_errors'] == 0


@pytest.mark.parametrize('window', [-1, 11, True, 1.5])
def test_bad_bias_window_rejected(window):
    with pytest.raises(ValueError):
        PointBiasCorrector(window)


@pytest.mark.parametrize('confidence,tier', [(.1, 1), (.2, 2), (.5, 3)])
@pytest.mark.parametrize('point,bias,target', [(28, 7, 3), (28, -7, 5), (1, 20, 1), (999, 0, 20)])
def test_corrected_rule_ignores_second_horizon_and_upper_bound(confidence, tier, point, bias, target):
    assert compute_target_replicas(np.array([point, 10000]), confidence,
                                  np.array([1e6, 1e6]), h0_only=True, point_bias=bias) == (tier, target)


@pytest.mark.parametrize('confidence,expected', [(.1, (1, 4)), (.2, (2, 6)), (.5, (3, 5))])
def test_legacy_tier_targets_unchanged(confidence, expected):
    assert compute_target_replicas(np.array([14., 28.]), confidence, np.array([21., 35.])) == expected


def test_all_shared_planner_nodes_match_the_runtime_ast():
    trees = [ast.parse(path.read_text()) for path in
             [ROOT / 'confscale/planning_numpy.py', ROOT / 'reference/stage3_scale/orchestrator/controller.py']]
    for name in ('compute_target_replicas', 'PointBiasCorrector', 'HysteresisManager', '_tier_cooldown'):
        nodes = [next(node for node in tree.body if getattr(node, 'name', None) == name) for tree in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])


def test_plan_and_capabilities_are_offline_and_keep_legacy_monitoring():
    with patch('subprocess.run', side_effect=AssertionError('process during plan')):
        config, profile = load_config(ROOT / 'configs/comparison-smoke.json')
        cells = plan(config, profile)['cells']
    assert len(METHODS) == 18 and len(cells) == 4
    for cell in cells:
        assert cell['capabilities'] == capability_record(cell['method'])
        assert cell['preset_configuration'] == preset_configuration(cell['method'])
    for name in LEGACY_METHODS:
        assert capabilities(name).interval_monitor == name.startswith('confscale-')
    for name in ('confscale-point-corrected', 'confscale-scp-frozen'):
        method = configure_method(methods.get_method(name))
        state = method.get_state()
        assert all(state[key] == value for key, value in preset_configuration(name).items())
        assert capabilities(name).model_head == 'scp'
    assert configure_method(methods.get_method('confscale-scp')).coverage_monitor is True
    assert methods.get_method('confscale-scp').coverage_monitor is False  # registry untouched


def test_comparison_models_use_the_same_hash_pinned_scp_route(tmp_path):
    data = tmp_path / 'inputs/uq/bursty/scp/model.bin'
    data.parent.mkdir(parents=True)
    data.write_bytes(b'hash-pinned-fixture')
    write_json(tmp_path / 'provenance/omitted_inputs.json', {'inputs': [
        {'path': 'models/uq/bursty/scp/model.bin', 'sha256': sha256(data)}]})
    config = {'models_dir': str(tmp_path / 'inputs'), 'workloads': ['B'],
              'methods': ['confscale-point-corrected', 'confscale-scp-frozen', 'hpa-cpu-stock']}
    with patch('confscale.experiment.ROOT', tmp_path):
        assert verify_models(config) == {'uq/bursty/scp/model.bin': sha256(data)}
        data.write_bytes(b'changed')
        with pytest.raises(ValueError, match='Model hash mismatch'):
            verify_models(config)


@pytest.mark.parametrize('name', ['confscale-point-corrected', 'confscale-scp-frozen'])
def test_controller_command_keeps_repair_and_monitor_off(tmp_path, name):
    (tmp_path / 'models/diurnal/scp').mkdir(parents=True)
    method = configure_method(methods.get_method(name))
    method.run_dir = tmp_path / 'cell'
    proc = Mock()
    proc.poll.return_value = None
    with patch.object(methods, 'UQ_MODELS_DIR', tmp_path / 'models'), \
         patch.object(methods, 'kubectl'), patch.object(methods, 'kubectl_check', return_value=True), \
         patch.object(methods.subprocess, 'Popen', return_value=proc) as launch, patch.object(methods.time, 'sleep'):
        assert method.configure()
    command = launch.call_args.args[0]
    assert '--coverage-monitor' not in command and '--online-recal' not in command
    assert '--recalibrator' not in command and '--ladder' not in command
    assert ('--h0-only' in command) == (name == 'confscale-point-corrected')
    if name == 'confscale-point-corrected':
        assert command[command.index('--point-bias-window') + 1] == '10'


@pytest.mark.parametrize('extra', [['--coverage-monitor'], ['--ladder'], ['--recalibrator', 'pid'],
                                  ['--policy', 'ci-upper'], ['--lambda-risk', '.5']])
def test_corrected_cli_rejects_incompatible_modes(extra, capsys):
    with patch.object(sys, 'argv', ['controller', '--method', 'scp', '--model-dir', 'unused', '--h0-only', *extra]), \
         patch.object(controller, 'start_http_server', side_effect=AssertionError('started server')):
        with pytest.raises(SystemExit) as error:
            controller.main()
    assert error.value.code == 2
    assert 'h0 point control requires' in capsys.readouterr().err


CRITERIA = {'min_trace_ticks': 2, 'min_predictions': 2, 'min_validated_intervals': 1}


def comparison_fixture(path, name):
    expected = preset_configuration(name)
    write_json(path / 'execution.json', {'status': 'ok', 'workload_returncode': 0, 'reset_ok': True,
                                        'duration_s': 60, 'method': name})
    write_json(path / 'effective_method.json', {'method': name, 'capabilities': capability_record(name),
        'expected_config': expected, 'actual_config': {'method': name, **expected}})
    write_json(path / 'collection_summary.json', {'status': 'ok', 'total_requests': 20,
        'mean_replicas': 2, 'p95_ms': 20, 'e2e_p95_ms': 23, 'timeseries_datapoints': 20})
    (path / 'workload_B_timeseries.csv').write_text('elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\n'
        '0,10,10,10,0,12,23,30\n60,10,10,10,0,12,23,30\n')
    cfg = {('method' if key == 'uq_method' else key): value for key, value in expected.items()}
    (path / 'controller_config.yaml').write_text(yaml.safe_dump(cfg))
    write_json(path / 'controller_config.json', cfg)
    rows = []
    for i, (rps, h0, h1, raw, bias) in enumerate([(14, 28, 70, 4, 0), (21, 35, 90, 4, 7), (28, 42, 110, 5, 7)]):
        pred = prediction(h0, h1)
        target = raw if name == 'confscale-point-corrected' else [11, 13, 16][i]
        row = {'elapsed_s': i * 30, 'rps': rps, 'prediction_made': True,
               **{key: value.tolist() if hasattr(value, 'tolist') else value for key, value in pred.items()},
               'point_forecast_dtype': 'float64', 'ci_upper_dtype': 'float64',
               'raw_tier': 3, 'raw_target': target, 'target_replicas': target,
               'last_requested_replicas': target, 'observed_replicas': target}
        if name == 'confscale-point-corrected':
            row['point_correction'] = {'raw_h0': h0, 'bias_rps': bias, 'matched_errors': i,
                                       'corrected_rps': h0 - bias}
        rows.append(row)
    write_json(path / 'controller_scale_log.json', rows)
    return rows


@pytest.mark.parametrize('name', ['confscale-point-corrected', 'confscale-scp-frozen'])
def test_monitor_off_presets_require_targets_not_interval_counts(tmp_path, name):
    comparison_fixture(tmp_path, name)
    verdict = assess(tmp_path, True, CRITERIA, method_name=name)
    assert verdict['status'] == 'pass', verdict
    assert 'intervals_observed_later' not in verdict['checks']
    assert verdict['checks']['raw_targets_recomputed']
    assert not (tmp_path / 'operator_metrics_summary.json').exists()


@pytest.mark.parametrize('mutation', ['raw_target', 'bias', 'monitor', 'config', 'method', 'time'])
def test_bad_point_evidence_fails(tmp_path, mutation):
    name = 'confscale-point-corrected'
    rows = comparison_fixture(tmp_path, name)
    if mutation == 'raw_target':
        rows[1]['raw_target'] += 1
    elif mutation == 'bias':
        rows[1]['point_correction']['bias_rps'] = -7
    elif mutation == 'monitor':
        rows[1]['coverage'] = {}
    elif mutation == 'time':
        rows[1]['elapsed_s'] = 0
    elif mutation == 'config':
        cfg = json.loads((tmp_path / 'controller_config.json').read_text())
        cfg['coverage_monitor'] = True
        write_json(tmp_path / 'controller_config.json', cfg)
    elif mutation == 'method':
        result = json.loads((tmp_path / 'execution.json').read_text())
        result['method'] = 'confscale-scp'
        write_json(tmp_path / 'execution.json', result)
    write_json(tmp_path / 'controller_scale_log.json', rows)
    assert assess(tmp_path, True, CRITERIA, method_name=name)['status'] == 'fail'


def test_legacy_monitored_method_still_requires_coverage_evidence(tmp_path):
    comparison_fixture(tmp_path, 'confscale-scp-frozen')
    result = json.loads((tmp_path / 'execution.json').read_text())
    result['method'] = 'confscale-scp'
    write_json(tmp_path / 'execution.json', result)
    assert assess(tmp_path, True, CRITERIA, method_name='confscale-scp')['status'] == 'fail'


@pytest.mark.parametrize('name', ['hpa-cpu-tuned', 'hpa-cpu-stock'])
@pytest.mark.parametrize('mutation', [None, 'uid', 'cpu', 'health', 'survivor', 'unready', 'delete_failure',
                                      'wrong_delete', 'wrong_scale', 'wrong_readback', 'malformed_inventory'])
def test_native_assurance_requires_identity_spec_health_and_restoration(tmp_path, name, mutation):
    from test_assurance_artifacts import native_fixture
    hpa, reset, actual = native_fixture(tmp_path, name)
    if mutation == 'uid':
        hpa['metadata']['uid'] = 'replaced'
    elif mutation == 'cpu':
        hpa['spec']['metrics'][0]['resource']['target']['averageUtilization'] = 50
    elif mutation == 'health':
        hpa['status']['conditions'][0]['status'] = 'False'
    elif mutation == 'survivor':
        reset['delete']['attempts'][-1]['readback']['stdout'] = json.dumps({'items': [hpa]})
    elif mutation == 'unready':
        reset['deployment']['stdout'] = '{"spec": {"replicas": 1}, "status": {"replicas": 1, "readyReplicas": 0}}'
    elif mutation == 'delete_failure':
        reset['delete']['attempts'][-1]['delete']['returncode'] = 1
    elif mutation == 'wrong_delete':
        reset['delete']['attempts'][-1]['delete']['command'][2] = 'another-hpa'
    elif mutation == 'wrong_scale':
        reset['scale']['command'][2] = 'another-deployment'
    elif mutation == 'wrong_readback':
        reset['delete']['attempts'][-1]['readback']['command'][3] = 'another-namespace'
    elif mutation == 'malformed_inventory':
        reset['delete']['attempts'][-1]['readback']['stdout'] = '{"items": {}}'
    write_json(tmp_path / 'before_reset_status.json', {'items': [hpa]})
    write_json(tmp_path / 'hpa_reset_check.json', reset)
    verdict = assess(tmp_path, False, CRITERIA, method_name=name)
    assert verdict['status'] == ('pass' if mutation is None else 'fail'), verdict


@pytest.mark.parametrize('name', ['confscale-point-corrected', 'confscale-scp-frozen'])
def test_sealed_comparison_audit_remains_dependency_free(tmp_path, name):
    directory = tmp_path / 'cell'
    directory.mkdir()
    comparison_fixture(directory, name)
    config, profile = load_config(ROOT / 'configs/comparison-smoke.json')
    config['methods'] = [name]
    config['assurance'] = CRITERIA
    planned = plan(config, profile)
    write_json(tmp_path / 'plan.json', planned)
    receipt = {'status': 'pass', 'restoration': 'pass', 'runs': [
        {'cell': planned['cells'][0], 'directory': 'cell'}],
        'output_sha256': {str(p.relative_to(tmp_path)): sha256(p) for p in tmp_path.rglob('*') if p.is_file()}}
    write_json(tmp_path / 'receipt.json', receipt)
    assert audit(tmp_path)['status'] == 'pass'
    command = [sys.executable, '-S', '-m', 'confscale.experiment', 'audit', '--output', str(tmp_path)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['cells_verified'] == 1


@pytest.mark.parametrize('response,expected', [([], 1.0), ([{'value': [123, '4']}], 4.0),
                                             ([{'value': [123, 'NaN']}], None)])
def test_overhead_evidence_does_not_replace_legacy_values(response, expected):
    from collector.collect import _execute_scalar_queries
    from collector.queries import SCALAR_QUERIES
    query = next(q for q in SCALAR_QUERIES if q['name'] == 'overhead_replica_seconds')
    client = Mock()
    client.query.return_value = response
    evidence = {}
    result = _execute_scalar_queries(client, [query], 180, evidence)
    assert result[query['output_key']] == expected
    assert evidence['query_succeeded'] is True
    assert evidence['returned_data'] == bool(response)
    assert evidence['value'] == (expected if response else None)
    assert evidence['scrape_sample_count'] is None


def test_failed_cost_query_is_not_reported_as_measured_zero():
    from collector.collect import _execute_scalar_queries
    from collector.queries import SCALAR_QUERIES
    query = next(q for q in SCALAR_QUERIES if q['name'] == 'overhead_replica_seconds')
    client = Mock()
    client.query.side_effect = ConnectionError('unavailable')
    evidence = {}
    assert _execute_scalar_queries(client, [query], 180, evidence)[query['output_key']] == 1.0
    assert evidence['query_succeeded'] is False and evidence['value'] is None
