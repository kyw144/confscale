import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from confscale import ACI, ConformalPID, CoverageMonitor, EscalationLadder, EmptyResidualBufferError
from confscale.demo import replay, upper_bound_target, write_demo
from confscale.reproduce import reproduce
from confscale.verify import verify

ROOT = Path(__file__).resolve().parents[1]

class CoreBehavior(unittest.TestCase):
    def test_aci_feedback_and_finite_sample_quantile(self):
        a = ACI(eta=.1, residual_buffer_size=5)
        with self.assertRaises(EmptyResidualBufferError): a.quantile()
        a.residuals.extend([1,2,3,4,5])
        self.assertEqual(a.quantile(), 5)
        a.update(-6, True)
        self.assertAlmostEqual(a.alpha, .01)
        self.assertEqual(list(a.residuals), [2,3,4,5,6])
        a.update(0, False)
        self.assertAlmostEqual(a.alpha, .02)
        self.assertEqual(a.quantile(), 6)

    def test_pid_is_aci_with_zero_integral_derivative(self):
        a, p = ACI(eta=.17), ConformalPID(k_p=.17, k_i=0, k_d=0)
        for residual, miss in [(2,False),(-8,True),(3,False),(20,True)]*50:
            a.update(residual, miss); p.update(residual, miss)
            self.assertEqual(a.alpha, p.alpha)
            self.assertEqual(a.quantile(), p.quantile())
        self.assertGreaterEqual(a.alpha, .0001)
        self.assertLessEqual(a.alpha, .5)

    def test_monitor_scores_previous_interval_fifo(self):
        m = CoverageMonitor(window_size=2)
        self.assertIsNone(m.validate_pending(42))
        m.record_prediction(0,10); m.record_prediction(100,110)
        self.assertTrue(m.validate_pending(10))
        self.assertFalse(m.validate_pending(99))
        self.assertEqual(m.trailing_coverage, .5)
        m.record_prediction(0,10); m.validate_pending(5)
        self.assertEqual(m.get_state()['validated'], 2)
        self.assertEqual(m.lifetime_validated, 3)
        self.assertEqual(m.lifetime_covered, 2)

    def test_ladder_persistence_middle_band_and_single_step_recovery(self):
        l = EscalationLadder(escalation_persistence=2, recovery_persistence=2)
        self.assertEqual([l.step(.7), l.step(.7)], [0,1])
        l.step(.7); l.step(.86)  # middle band discards partial escalation
        self.assertEqual(l.step(.7), 1)
        self.assertEqual(l.step(.7), 2)
        self.assertEqual(l.apply(10), 30)
        self.assertEqual([l.step(.9), l.step(.9)], [2,1])
        self.assertEqual(l.apply(10), 15)
        self.assertEqual([l.step(.9), l.step(.9)], [1,0])

    def test_upper_bound_rule_examples_and_masking(self):
        self.assertEqual(upper_bound_target([108,108]), 16)
        self.assertEqual(upper_bound_target([100.565,100.565]), 15)
        self.assertEqual(upper_bound_target([5,112]), upper_bound_target([20,112]))
        self.assertEqual(upper_bound_target([1e9]), 20)
        for bad in [[],[float('nan')],[float('inf')]]:
            with self.assertRaises(ValueError): upper_bound_target(bad)

    def test_demo_future_does_not_change_past(self):
        a,_ = replay(120,80); b,_ = replay(140,80)
        self.assertEqual(a,b[:len(a)])
        c,_ = replay(120,60)
        self.assertEqual(a[:60*5],c[:60*5])

    def test_demo_repeatable_and_labeled_separate_from_paper(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            sa, sb = write_demo(a), write_demo(b)
            self.assertEqual(sa,sb)
            self.assertEqual((Path(a)/'trace.csv').read_bytes(),(Path(b)/'trace.csv').read_bytes())
            self.assertIn('NOT_PAPER_RESULTS',sa['kind'])

class ArtifactIntegrity(unittest.TestCase):
    def test_source_manifest(self):
        self.assertGreaterEqual(verify()['source_derived_files_verified'], 100)

    def test_core_classes_equal_reference_excluding_docstrings(self):
        class StripDocs(ast.NodeTransformer):
            def visit_Expr(self,n):
                if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str): return None
                return self.generic_visit(n)
        for core, ref in [('aci.py','uq/aci.py'),('conformal_pid.py','uq/conformal_pid.py'),
                          ('coverage_monitor.py','orchestrator/coverage_monitor.py'),('escalation_ladder.py','baselines/escalation_ladder.py')]:
            def classes(p):
                tree = ast.parse(p.read_text(encoding='utf-8'))
                return [ast.dump(StripDocs().visit(n), include_attributes=False) for n in tree.body if isinstance(n,ast.ClassDef)]
            self.assertEqual(classes(ROOT/'confscale'/core),classes(ROOT/'reference/stage3_scale'/ref), core)

    def test_all_main_paper_tables_repeat_and_match_word_goldens(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ra, rb = reproduce(a), reproduce(b)
            self.assertEqual(ra,rb)
            self.assertEqual(set(ra['tables']),set('12345'))
            self.assertEqual(sum(x['data_cells_checked_against_paper'] for x in ra['tables'].values()),132)
            for i in range(1,6):
                self.assertEqual((Path(a)/f'table_{i}.csv').read_bytes(),(Path(b)/f'table_{i}.csv').read_bytes())

    def test_cli_works_without_site_packages(self):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([sys.executable,'-S','-m','confscale','reproduce','--output',d],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)

    def test_direct_runtime_launch_disabled_before_heavy_imports(self):
        env = os.environ.copy(); env.pop('CONFSCALE_ENABLE_REFERENCE_RUNTIME',None)
        for rel in ['stage3_scale/orchestrator/controller.py','stage3_scale/orchestrator/run_matrix.py',
                    'stage3_scale/workload_gen.py','studies/binding/driver.py',
                    'stage3_scale/load_gen.py',
                    'stage3_scale/infosys-benchmark/frontend/app.py',
                    'stage3_scale/infosys-benchmark/processor/app.py',
                    'stage3_scale/infosys-benchmark/compute-worker/app.py']:
            p = subprocess.run([sys.executable,'-S',str(ROOT/'reference'/rel)],cwd=ROOT,capture_output=True,text=True,env=env)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Reference runtime disabled',p.stderr)
            self.assertNotIn('ModuleNotFoundError',p.stderr)

    def test_load_generator_and_server_imports_stop_before_dependencies(self):
        env = os.environ.copy(); env.pop('CONFSCALE_ENABLE_REFERENCE_RUNTIME',None)
        for rel in ['load_gen.py','infosys-benchmark/frontend/app.py','infosys-benchmark/processor/app.py',
                    'infosys-benchmark/compute-worker/app.py']:
            code = 'import runpy,sys; runpy.run_path(sys.argv[1], run_name="artifact_import_probe")'
            p = subprocess.run([sys.executable,'-S','-c',code,str(ROOT/'reference/stage3_scale'/rel)],
                               cwd=ROOT,capture_output=True,text=True,env=env)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Reference runtime disabled',p.stderr)
            self.assertNotIn('ModuleNotFoundError',p.stderr)

    def test_all_python_sources_parse(self):
        for base in ['confscale','reference']:
            for p in (ROOT/base).rglob('*.py'):
                ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))

class OptionalPlannerParity(unittest.TestCase):
    def test_original_numpy_planner_and_stdlib_adapter_agree(self):
        try:
            import numpy as np
            from confscale.planning_numpy import compute_target_replicas
        except ImportError: self.skipTest('optional NumPy planner dependency not installed')
        for cap in [1.,10.,23.]:
            for util in [.3,.7,1.]:
                for upper in [[0,0],[100,108],[112,20],[1e9,2],[1,2],[7,14]]:
                    _, n = compute_target_replicas(np.array([0.,0.]),.5,np.array(upper),slo_capacity=cap,target_util=util,policy='ci-upper')
                    self.assertEqual(n,upper_bound_target(upper,cap,util))

    def test_planner_ast_matches_sanitized_reference(self):
        paths = [ROOT/'confscale/planning_numpy.py',ROOT/'reference/stage3_scale/orchestrator/controller.py']
        trees = [ast.parse(p.read_text(encoding='utf-8')) for p in paths]
        for name in ['compute_target_replicas','HysteresisManager','_tier_cooldown']:
            nodes = [next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name) for tree in trees]
            self.assertEqual(ast.dump(nodes[0],include_attributes=False),ast.dump(nodes[1],include_attributes=False))

if __name__ == '__main__': unittest.main()
