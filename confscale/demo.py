"""Deterministic synthetic replay; no trained model or latency simulation."""
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
from . import ACI, ConformalPID, CoverageMonitor, EscalationLadder


def upper_bound_target(upper, capacity=10.0, utilization=0.7, minimum=1, maximum=20):
    """Stdlib adaptation of the original planner's ci-upper branch (before hysteresis)."""
    if not upper or not all(math.isfinite(x) for x in upper):
        raise ValueError('upper must contain finite horizon values')
    if capacity <= 0 or not 0 < utilization <= 1 or not 1 <= minimum <= maximum:
        raise ValueError('invalid replica capacity/utilization/bounds')
    return max(minimum, min(maximum, math.ceil(max(upper) / (capacity * utilization))))


def quantile(residuals, alpha=0.1):
    """Same finite-sample rank convention as the source recalibrator."""
    if not residuals:
        raise ValueError('empty calibration sample')
    ordered = sorted(abs(x) for x in residuals)
    return ordered[max(0, min(len(ordered)-1, math.ceil((1-alpha)*(len(ordered)+1))-1))]


def residual_at(t, shift_at):
    # Formula rather than RNG keeps input bytes deterministic across runs.
    carrier = math.sin(t*1.73) + 0.35*math.cos(t*0.41)
    return carrier * (2.0 if t < shift_at else 12.0)


def replay(ticks=240, shift_at=80):
    if ticks < 40 or not 10 <= shift_at < ticks-10:
        raise ValueError('need ticks >= 40 and 10 <= shift_at < ticks-10')
    calibration = [residual_at(t, shift_at) for t in range(-120, 0)]
    q0 = quantile(calibration)
    methods = {}
    for name in ('frozen', 'rolling', 'aci', 'pid', 'aci_laddered'):
        recal = ACI() if name.startswith('aci') else ConformalPID() if name == 'pid' else None
        if recal:
            recal.residuals.extend(abs(x) for x in calibration)
        methods[name] = dict(recal=recal, monitor=CoverageMonitor(window_size=30),
                             ladder=EscalationLadder() if name.endswith('laddered') else None,
                             residuals=deque(map(abs, calibration), maxlen=200))
    rows = []
    for t in range(ticks):
        point = 55.0 + 8.0*math.sin(t/18.0)
        residual = residual_at(t, shift_at)
        observed = max(0.0, point+residual)
        # The second horizon is fixed-width on purpose, exposing max-over-horizons masking.
        point_h1 = point + 5.0
        upper_h1 = point_h1 + q0
        for name, state in methods.items():
            recal, monitor, ladder = state['recal'], state['monitor'], state['ladder']
            base_q = recal.quantile() if recal else quantile(state['residuals']) if name == 'rolling' else q0
            # Prior coverage drives this tick. The observation is scored only AFTER issuance.
            if ladder: ladder.step(monitor.trailing_coverage)
            q = ladder.apply(base_q) if ladder else base_q
            lower, upper = max(0.0, point-q), point+q
            requested = upper_bound_target([upper, upper_h1])
            monitor.record_prediction(lower, upper)
            covered = monitor.validate_pending(observed)
            if recal: recal.update(residual, not covered)
            state['residuals'].append(abs(residual))
            rows.append(dict(tick=t, phase='before_shift' if t < shift_at else 'after_shift', method=name,
                             point=round(point, 6), observed=round(observed, 6), lower=round(lower, 6),
                             upper=round(upper, 6), upper_h1=round(upper_h1, 6), width=round(upper-lower, 6),
                             covered=int(covered), trailing_coverage=round(monitor.trailing_coverage, 6),
                             alpha=round(recal.alpha, 6) if recal else 0.1,
                             ladder_level=ladder.level if ladder else 0,
                             h0_sets_max=int(upper >= upper_h1), requested_replicas=requested))
    summary = {'kind': 'ILLUSTRATIVE_SYNTHETIC_DEMO_NOT_PAPER_RESULTS',
               'ticks': ticks, 'shift_at': shift_at, 'calibration_points': len(calibration),
               'scope': 'supplied synthetic forecasts; one-step feedback; raw targets before hysteresis; no cluster or latency',
               'methods': {}}
    for name in methods:
        summary['methods'][name] = {}
        for phase in ('before_shift', 'after_shift'):
            subset = [r for r in rows if r['method'] == name and r['phase'] == phase]
            summary['methods'][name][phase] = {
                'coverage': sum(r['covered'] for r in subset)/len(subset),
                'mean_width': sum(r['width'] for r in subset)/len(subset),
                'mean_requested_replicas': sum(r['requested_replicas'] for r in subset)/len(subset),
                'h0_sets_max_fraction': sum(r['h0_sets_max'] for r in subset)/len(subset)}
    return rows, summary


def write_demo(output, ticks=240, shift_at=80):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = replay(ticks, shift_at)
    with (output/'trace.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    summary['trace_sha256'] = hashlib.sha256((output/'trace.csv').read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    (output/'README.txt').write_text('ILLUSTRATIVE DEMO. These values are not paper measurements.\n'
        'trace.csv records the interval before the observation is used to update it.\n'
        'requested_replicas is a computed target, not live replica execution or latency.\n', encoding='utf-8')
    return summary
