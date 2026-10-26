import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from confscale.comparisons import (BATCHES, DIRECTORY, METRICS, ROOT, UNITS,
                                  _no_private_locations, load_comparisons,
                                  reproduce_comparisons, validate_batch, verify_comparisons)
from confscale.verify import RUNTIME_BASELINE_HASHES, verify_runtime_adaptations


class StoredComparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.manifest = load_comparisons()
        omitted = json.loads((ROOT / 'provenance/omitted_inputs.json').read_text())
        cls.artifact_inputs = {r['path'].removeprefix('models/uq/'): r['sha256']
                               for r in omitted['inputs'] if r['path'].startswith('models/uq/')}

    def test_pinned_files_and_complete_cohorts(self):
        expected = {'point-vs-tuned': '3fdbe8b76abcef15a8c4d7168e6cc77e08b06b182f1c9c5ca2edaa65e4f70860',
                    'point-vs-stock': 'f8dbbdeaee13783a73a0ef5ddf56deaf8e9a3837d757a4fa4691f41822b68cd5'}
        self.assertEqual(verify_comparisons(), {'comparison_files_verified': 2, 'comparison_cells_verified': 96,
                                               'comparison_source_hashes_recorded': 357})
        for record in self.manifest['records']:
            self.assertEqual(record['destination_sha256'], expected[record['batch']])
        for batch, (reps, methods) in BATCHES.items():
            data = self.data[batch]
            self.assertEqual(len(data['cells']), 12 * reps)
            self.assertEqual(len(data['summaries']), 12)
            self.assertEqual(set(data['cohort']['methods']), set(methods))
            self.assertEqual(data['cohort']['patterns'], list('ABCD'))
            self.assertFalse(data['cohort']['pool_with_other_batches'])
            self.assertEqual([data['metric_definitions'][m]['unit'] for m in METRICS], list(UNITS))
            self.assertEqual(data['metric_definitions']['tick_slo_violation_fraction']['threshold_ms'], 200)

    def test_saved_means_match_pinned_values_for_every_arm_and_pattern(self):
        # Columns are overhead replica-s/hour, run p95 ms, tick-violation
        # fraction, and sampled desired-count changes per run.
        expected = {
            'point-vs-tuned': [
                (20812.586666666666, 551.9399999999999, .95586, 13.6),
                (23993.36666666667, 546.4399999999999, .9540599999999999, 14.2),
                (23106.0, 554.0600000000001, .95472, 10.8),
                (29952.0, 1112.8400000000001, .7822199999999999, 39.8),
                (27696.0, 1105.3, .7754000000000001, 39.6),
                (28052.84333333333, 1110.6799999999998, .7787000000000001, 33.0),
                (57018.61333333334, 1225.16, .9625999999999999, 26.0),
                (47739.16333333333, 1232.62, .9628599999999998, 28.6),
                (47596.69666666667, 1230.3600000000001, .96646, 22.8),
                (19182.0, 966.18, .19324000000000002, 19.4),
                (16014.0, 958.5, .19332, 20.0),
                (14544.0, 1136.66, .1849, 38.4),
            ],
            'point-vs-stock': [
                (22343.96111111111, 545.9999999999999, .9551333333333333, 7.666666666666667),
                (22344.388888888887, 546.3333333333334, .9582666666666667, 8.0),
                (24024.44444444445, 549.3666666666667, .9591999999999999, 14.666666666666666),
                (37576.82777777778, 1085.1666666666667, .8405, 13.0),
                (37260.0, 1092.7666666666667, .8234666666666667, 12.333333333333334),
                (29570.0, 1089.3, .8401333333333333, 37.0),
                (46019.70165289257, 1258.8666666666666, .9722, 11.333333333333334),
                (45967.027777777774, 1253.2666666666667, .9692666666666666, 12.666666666666666),
                (47968.92222222222, 1268.6333333333332, .9632666666666667, 29.333333333333332),
                (48760.0, 992.9666666666667, .19323333333333334, 4.333333333333333),
                (49298.85555555555, 989.2666666666668, .1933666666666667, 4.333333333333333),
                (15280.0, 987.0333333333333, .19313333333333335, 19.333333333333332),
            ],
        }
        for batch, rows in expected.items():
            actual = [tuple(row['statistics'][metric]['mean'] for metric in METRICS) for row in self.data[batch]['summaries']]
            self.assertEqual(actual, rows)

    def test_failed_cost_criterion_and_mixed_service_ordering_retained(self):
        comparisons = {r['pattern']: r for r in self.data['point-vs-tuned']['recorded_comparisons']
                       if r['candidate'] == 'confscale-point-corrected' and r['reference'] == 'hpa-cpu-tuned'}
        d = comparisons['D']
        self.assertFalse(d['recorded_gate']['cost'])
        self.assertFalse(d['recorded_gate']['pattern_pass'])
        self.assertEqual(d['differences'][METRICS[0]]['welch_95'], {
            'low': -1180.1150300683794, 'high': 4120.115030068379, 'df': 7.825238843205227})
        self.assertLess(d['differences'][METRICS[1]]['difference'], 0)
        self.assertGreater(d['differences'][METRICS[2]]['difference'], 0)
        self.assertGreater(comparisons['C']['differences'][METRICS[1]]['difference'], 0)
        self.assertLess(comparisons['C']['differences'][METRICS[2]]['difference'], 0)

    def test_event_completeness_is_separate_from_sampled_changes(self):
        stock = self.data['point-vs-stock']
        self.assertTrue(stock['recorded_status']['cohort_complete'])
        self.assertFalse(stock['recorded_status']['hpa_event_log_complete'])
        self.assertFalse(stock['recorded_status']['proposal_observation_complete'])
        self.assertEqual(stock['method_configurations']['hpa-cpu-memory']['runtime_support'], 'stored-only')
        self.assertTrue(all(c['values']['desired_count_changes'] >= 0 for c in stock['cells']))
        tuned = self.data['point-vs-tuned']
        first = next(c for c in tuned['cells'] if c['id'] == 'a-confscale-point-corrected-rep-1')
        self.assertEqual(first['values']['desired_count_changes'], 15.0)
        self.assertIn('not the p95 of pooled requests', stock['metric_definitions']['e2e_run_p95_ms']['definition'])

    def test_checkpoint_identity_and_normalization_are_explicit(self):
        self.assertEqual(self.data['point-vs-tuned']['checkpoint_provenance'], {'status': 'configuration-only', 'files': []})
        inputs = self.data['point-vs-stock']['checkpoint_provenance']['files']
        self.assertEqual(sum(i['artifact_input_relation'] == 'identical' for i in inputs), 4)
        self.assertEqual(sum(i['artifact_input_relation'] == 'LF-to-CRLF-only' for i in inputs), 4)
        for item in inputs:
            self.assertEqual(item['sha256'] == item['artifact_input_sha256'], item['artifact_input_relation'] == 'identical')

    def test_outputs_repeat_and_include_failure_and_limitations(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(reproduce_comparisons(a), reproduce_comparisons(b))
            self.assertEqual(len(list(Path(a).iterdir())), 7)
            for p in Path(a).iterdir():
                self.assertEqual(p.read_bytes(), (Path(b) / p.name).read_bytes())
            tuned = (Path(a) / 'point-vs-tuned.md').read_text()
            self.assertIn('**FAIL**', tuned)
            self.assertIn('[-1180.12, 4120.12]', tuned)
            self.assertIn('higher mean p95 but lower tick-violation', tuned)
            self.assertIn('Recorded overall rule: **PASS**, 3/4', tuned)
            self.assertIn('request-error fraction must each be no more than reference +0.03', tuned)
            self.assertIn('event-log observation', (Path(a) / 'point-vs-stock.md').read_text())
            _no_private_locations(json.loads((Path(a) / 'summary.json').read_text()))

    def test_cli_has_no_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable, '-S', '-m', 'confscale', 'comparisons', '--output', d],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['kind'], 'STORED_RESULT_REPRODUCTION')

    def test_schema_rejects_cohort_unit_value_and_source_changes(self):
        sources = {s['id']: s for s in self.manifest['records'][0]['sources']}
        edits = [lambda d: d['cells'].pop(),
                 lambda d: d['cells'].append(copy.deepcopy(d['cells'][0])),
                 lambda d: d['cohort'].update(pool_with_other_batches=True),
                 lambda d: d['metric_definitions'][METRICS[0]].update(unit='replica-seconds'),
                 lambda d: d['cells'][0]['values'].update(e2e_run_p95_ms=float('nan')),
                 lambda d: d['cells'][0]['sources'].update(metrics='missing-source'),
                 lambda d: d.update(extra='unreviewed metadata'),
                 lambda d: d['cells'][0].update(path='/Users/example/private'),
                 lambda d: d['qualifications'].append('/home/example/private')]
        for edit in edits:
            data = copy.deepcopy(self.data['point-vs-tuned'])
            edit(data)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_batch(data, sources, self.artifact_inputs)

    def test_cross_artifact_and_recorded_result_consistency(self):
        for batch in BATCHES:
            sources = {s['id']: s for r in self.manifest['records'] if r['batch'] == batch for s in r['sources']}
            data = copy.deepcopy(self.data[batch])
            data['recorded_comparisons'][0]['differences'][METRICS[0]]['difference'] += 5
            with self.assertRaisesRegex(ValueError, 'difference contradicts'):
                validate_batch(data, sources, self.artifact_inputs)
            data = copy.deepcopy(self.data[batch])
            data['recorded_comparisons'][0]['differences'][METRICS[1]]['relative_percent'] += .1
            with self.assertRaisesRegex(ValueError, 'relative difference contradicts'):
                validate_batch(data, sources, self.artifact_inputs)
        sources = {s['id']: s for r in self.manifest['records'] if r['batch'] == 'point-vs-tuned' for s in r['sources']}
        data = copy.deepcopy(self.data['point-vs-tuned'])
        data['recorded_comparisons'][0]['recorded_gate'].update(service=False, pattern_pass=False)
        with self.assertRaisesRegex(ValueError, 'Overall recorded rule'):
            validate_batch(data, sources, self.artifact_inputs)
        sources = {s['id']: s for r in self.manifest['records'] if r['batch'] == 'point-vs-stock' for s in r['sources']}
        data = copy.deepcopy(self.data['point-vs-stock'])
        data['checkpoint_provenance']['files'][0]['artifact_input_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'artifact input manifest'):
            validate_batch(data, sources, self.artifact_inputs)

    def test_cost_and_strictly_cheaper_gates_cover_every_pairing(self):
        batch = 'point-vs-tuned'
        sources = {s['id']: s for r in self.manifest['records'] if r['batch'] == batch for s in r['sources']}
        data = copy.deepcopy(self.data[batch])
        nonprimary = next(r for r in data['recorded_comparisons'] if
                          (r['pattern'], r['candidate'], r['reference']) ==
                          ('A', 'confscale-point-corrected', 'confscale-scp-frozen'))
        self.assertGreater(nonprimary['differences'][METRICS[0]]['relative_percent'], 15)
        nonprimary['recorded_gate'].update(cost=True, pattern_pass=True)
        with self.assertRaisesRegex(ValueError, 'cost gate contradicts'):
            validate_batch(data, sources, self.artifact_inputs)
        for index, original in enumerate(self.data[batch]['recorded_comparisons']):
            for field, message in [('cost', 'cost gate contradicts'),
                                   ('strictly_cheaper', 'strictly-cheaper gate contradicts')]:
                data = copy.deepcopy(self.data[batch])
                gate = data['recorded_comparisons'][index]['recorded_gate']
                gate[field] = not gate[field]
                gate['pattern_pass'] = gate['cost'] and gate['service']
                with self.subTest(pattern=original['pattern'], candidate=original['candidate'],
                                  reference=original['reference'], field=field), self.assertRaisesRegex(ValueError, message):
                    validate_batch(data, sources, self.artifact_inputs)

    def test_generic_path_guard_handles_delimiters_without_rejecting_relative_paths_or_urls(self):
        private_locations = ['source=/Users/example/private', "source='/tmp/private'",
                             'path:(/tmp/private)', '{"source":"/tmp/private"}',
                             r'source=C:\Users\example\private', 'source=C:/temp/private',
                             '//tmp/private', '///tmp/private', 'source=///tmp/private',
                             "source='///tmp/private'", 'path:(///tmp/private)',
                             '{"source":"///tmp/private"}', 'source=https:///tmp/private',
                             'url=https://example.test/models/file;source=///tmp/private',
                             'https:\n//tmp/private', 'Note:\n//srv/private']
        for value in private_locations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _no_private_locations({'qualifications': [value]})
        for value in ['model=diurnal/scp/base_model.pt', 'fraction=3/4',
                      'https://example.test/models/file', 'url=https://example.test/models/file',
                      'HTTPS://example.test:8443/models/file', 'url=git+ssh://example.test/models/file']:
            with self.subTest(value=value):
                self.assertIsNone(_no_private_locations({'qualifications': [value]}))

    def test_manifest_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'comparisons', root / 'comparisons')
            (root / 'provenance').mkdir()
            manifest_path = root / 'provenance/comparison_manifest.json'
            shutil.copyfile(ROOT / 'provenance/comparison_manifest.json', manifest_path)
            shutil.copyfile(ROOT / 'provenance/omitted_inputs.json', root / 'provenance/omitted_inputs.json')
            path = root / DIRECTORY / 'point-vs-tuned.json'
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_comparisons(root)
            manifest = copy.deepcopy(self.manifest)
            manifest['records'][0]['destination'] = '../outside.json'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'destination'):
                load_comparisons(root)

    def test_new_metadata_has_only_neutral_ids_and_hashes(self):
        _no_private_locations(self.manifest)
        for data in self.data.values():
            _no_private_locations(data)
        for record in self.manifest['records']:
            for source in record['sources']:
                self.assertEqual(set(source), {'id', 'kind', 'sha256'})
                self.assertNotIn('/', source['id'])

    def test_original_paper_evidence_is_unchanged(self):
        manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
        rows = [r for r in manifest['records'] if r['destination'].startswith('evidence/')]
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(hashlib.sha256((ROOT / row['destination']).read_bytes()).hexdigest(), row['destination_sha256'])


class RuntimeAdaptationIntegrity(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        receipt = json.loads((ROOT / 'provenance/comparison_runtime_adaptations.json').read_text())
        source = json.loads((ROOT / 'provenance/source_manifest.json').read_text())
        source_rows = {r['destination']: r for r in source['records']}
        for row in receipt['changes']:
            destination = row['destination']
            path = root / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / destination, path)
            row['destination_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if destination in source_rows:
                source_rows[destination]['destination_sha256'] = row['destination_sha256']
        path = root / 'provenance/comparison_runtime_adaptations.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt))
        return root, receipt, source, path

    def test_all_runtime_destinations_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, source, _ = self.fixture(directory)
            self.assertEqual(verify_runtime_adaptations(root, source), {
                'runtime_adaptation_files_verified': 9, 'runtime_source_component_hashes_recorded': 5})
            for destination in RUNTIME_BASELINE_HASHES:
                path = root / destination
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    verify_runtime_adaptations(root, source)
                path.write_bytes(original)

    def test_inventory_lineage_and_manifest_contradictions_fail(self):
        edits = [lambda r: r['changes'].pop(),
                 lambda r: r['changes'].append(copy.deepcopy(r['changes'][0])),
                 lambda r: r['changes'][0].update(destination='../outside'),
                 lambda r: r['changes'][0].update(previous_shipped_sha256='0' * 64),
                 lambda r: r['source_components'].pop(),
                 lambda r: r.update(paper_snapshot_commit='0' * 40)]
        for edit in edits:
            with tempfile.TemporaryDirectory() as directory:
                root, receipt, source, path = self.fixture(directory)
                edit(receipt)
                path.write_text(json.dumps(receipt))
                with self.subTest(edit=edit), self.assertRaises(ValueError):
                    verify_runtime_adaptations(root, source)
        with tempfile.TemporaryDirectory() as directory:
            root, _, source, _ = self.fixture(directory)
            row = next(r for r in source['records'] if r['destination'] == 'confscale/planning_numpy.py')
            row['destination_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'manifests disagree'):
                verify_runtime_adaptations(root, source)


if __name__ == '__main__':
    unittest.main()
