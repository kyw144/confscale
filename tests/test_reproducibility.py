import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from confscale.assurance import assess
from confscale.cluster import load_profile, kube_command
from confscale.experiment import audit, load_config, plan
from confscale.inputs import restore
from confscale.run_support import ROOT, sha256, write_json


class ModelRestoreTests(unittest.TestCase):
    def test_restore_direct_model_directory_and_archived_bundle(self):
        for layout in ('uq/diurnal/scp/model.json', 'archive/models/uq/diurnal/scp/model.json'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                data = b'{"model": "fixture"}\r\n'
                row = {'source': 'archive/models/uq/diurnal/scp/model.json',
                       'path': 'models/uq/diurnal/scp/model.json',
                       'sha256': hashlib.sha256(data).hexdigest()}
                write_json(root / 'provenance/omitted_inputs.json',
                           {'source_commit': 'fixture', 'inputs': [row]})
                original = root / 'bundle' / layout
                original.parent.mkdir(parents=True)
                original.write_bytes(data.replace(b'\r\n', b'\n'))
                with patch('confscale.inputs.ROOT', root):
                    result = restore(root / 'bundle', root / 'inputs/models')
                self.assertEqual(result['models_verified'], 1)
                self.assertEqual((root / 'inputs/models/uq/diurnal/scp/model.json').read_bytes(), data)

    def test_restore_rejects_mismatched_models_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            rows = []
            for name, actual in [('valid.json', b'expected'), ('invalid.json', b'wrong')]:
                source = root / 'bundle/uq' / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(actual)
                rows.append({'source': 'archive/uq/' + name, 'path': 'models/uq/' + name,
                             'sha256': hashlib.sha256(b'expected').hexdigest()})
            write_json(root / 'provenance/omitted_inputs.json', {'source_commit': 'fixture', 'inputs': rows})
            with patch('confscale.inputs.ROOT', root), self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
                restore(root / 'bundle', root / 'inputs/models')
            self.assertFalse((root / 'inputs/models').exists())


class PlanningTests(unittest.TestCase):
    def test_plan_is_offline_and_pairs_seeds(self):
        with patch('subprocess.run', side_effect=AssertionError('network/process during plan')):
            config, profile = load_config(ROOT / 'configs/smoke.json')
            cells = plan(config, profile)['cells']
        self.assertEqual(len(cells), 2)
        self.assertEqual([c['seed'] for c in cells], [144, 144])
        self.assertEqual([c['method'] for c in cells], ['hpa-reactive', 'confscale-pid'])

    def test_plan_without_site_packages(self):
        p = subprocess.run([sys.executable, '-S', '-m', 'confscale.experiment', 'plan'],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)['kind'], 'EXPERIMENT_PLAN_NOT_EXECUTED')

    def test_unknown_and_empty_config_fail(self):
        config = json.loads((ROOT / 'configs/smoke.json').read_text())
        for edit in ({'seed': 12}, {'methods': []}, {'methods': ['typo']},
                     {'seeds': [1, 1]}, {'seeds': [True]}, {'duration_s': 0}):
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.json'
                write_json(path, {**config, **edit})
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_cluster_must_be_named_and_ports_distinct(self):
        profile = load_profile(ROOT / 'configs/cluster.json')
        for edit in ({'name': 'unrelated-cluster'}, {'frontend_port': profile['prometheus_port']},
                     {'frontend_port': True}, {'frontned_port': 12345}):
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'cluster.json'
                write_json(path, {**profile, **edit})
                with self.assertRaises(ValueError):
                    load_profile(path)

    def test_kubectl_always_selects_private_kubeconfig_and_context(self):
        profile = load_profile(ROOT / 'configs/cluster.json')
        cmd = kube_command(profile, 'get', 'pods')
        self.assertIn('--kubeconfig', cmd)
        self.assertIn('kind-confscale-repro', cmd)
        self.assertTrue(cmd[2].startswith(str(ROOT / 'generated/clusters/')))


class AssuranceTests(unittest.TestCase):
    criteria = {'min_trace_ticks': 2, 'min_predictions': 2, 'min_validated_intervals': 1}

    def fixture(self, directory):
        write_json(directory / 'execution.json', {'status': 'ok', 'workload_returncode': 0, 'reset_ok': True, 'duration_s': 1})
        write_json(directory / 'collection_summary.json', {'status': 'ok', 'total_requests': 20,
                   'mean_replicas': 2, 'p95_ms': 20, 'e2e_p95_ms': 23, 'timeseries_datapoints': 20})
        (directory / 'workload_B_timeseries.csv').write_text(
            'elapsed_s,target_rps,actual_rps,ok,errors,p50_ms,p95_ms,p99_ms\n'
            '0,10,10,10,0,12,23,30\n1,10,10,10,0,12,23,30\n')
        entry = {'prediction_made': True, 'point_forecast': [10], 'ci_lower': [5], 'rps': 10,
                 'ci_upper': [15], 'last_requested_replicas': 2, 'observed_replicas': 1}
        write_json(directory / 'controller_scale_log.json', [{**entry, 'elapsed_s': i} for i in range(2)])
        write_json(directory / 'operator_metrics_summary.json', {'coverage_monitor': {'total_validated': 1, 'total_covered': 1}})

    def test_complete_predictive_evidence_passes_only_technical_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            verdict = assess(directory, True, self.criteria)
        self.assertEqual(verdict['status'], 'pass')
        self.assertIn('NOT_SCIENTIFIC_REPLICATION', verdict['kind'])

    def test_missing_failed_or_degenerate_evidence_fails(self):
        for filename, value in [
            ('execution.json', {'status': 'ok', 'workload_returncode': 1, 'reset_ok': True}),
            ('execution.json', {'status': 'ok', 'workload_returncode': 0, 'reset_ok': False}),
            ('collection_summary.json', {'status': 'degraded'}),
            ('operator_metrics_summary.json', {'coverage_monitor': {'total_validated': 0}}),
            ('controller_scale_log.json', []),
            ('controller_scale_log.json', None),
        ]:
            with self.subTest(filename=filename, value=value), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.fixture(directory)
                if value is None:
                    (directory / filename).unlink()
                else:
                    write_json(directory / filename, value)
                self.assertEqual(assess(directory, True, self.criteria)['status'], 'fail')

    def test_nonfinite_and_duplicate_time_traces_fail(self):
        for replacement in ('1,nan,10,10,0,12,23,30', '0,10,10,10,0,12,23,30'):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                self.fixture(directory)
                path = directory / 'workload_B_timeseries.csv'
                path.write_text(path.read_text().replace('1,10,10,10,0,12,23,30', replacement))
                self.assertEqual(assess(directory, True, self.criteria)['status'], 'fail')

    def test_hpa_requires_working_metrics_and_scaling_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            path = directory / 'execution.json'
            write_json(path, {**json.loads(path.read_text()), 'method': 'hpa-reactive'})
            hpa = {'kind': 'HorizontalPodAutoscaler', 'spec': {'scaleTargetRef': {'name': 'compute-worker'}},
                   'status': {'conditions': [{'type': k, 'status': 'True'} for k in ('AbleToScale', 'ScalingActive')]}}
            write_json(directory / 'before_reset_status.json', {'items': [hpa]})
            self.assertEqual(assess(directory, False, self.criteria)['status'], 'pass')
            hpa['status']['conditions'][1]['status'] = 'False'
            write_json(directory / 'before_reset_status.json', {'items': [hpa]})
            self.assertEqual(assess(directory, False, self.criteria)['status'], 'fail')

    def test_audit_rejects_altered_outputs_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cell_dir = directory / 'cell'
            cell_dir.mkdir()
            self.fixture(cell_dir)
            cell = {'method': 'confscale-pid', 'seed': 144, 'workload': 'B'}
            write_json(directory / 'plan.json', {'cells': [cell], 'config': {'assurance': self.criteria}})
            receipt = {'status': 'pass', 'restoration': 'pass', 'runs': [{'cell': cell, 'directory': 'cell'}],
                       'output_sha256': {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}}
            write_json(directory / 'receipt.json', receipt)
            self.assertEqual(audit(directory)['cells_verified'], 1)
            write_json(directory / 'receipt.json', {**receipt, 'runs': []})
            with self.assertRaisesRegex(ValueError, 'planned matrix'):
                audit(directory)
            write_json(directory / 'receipt.json', receipt)
            (cell_dir / 'execution.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'integrity failure'):
                audit(directory)


if __name__ == '__main__':
    unittest.main()
