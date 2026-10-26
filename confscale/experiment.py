"""Plan/run explicit, isolated experiments; no cluster access during planning."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.parse import urlencode

from .assurance import assess
from .capabilities import (METHODS, LEGACY_METHODS, COMPARISON_METHODS, capabilities, capability_record,
                           preset_configuration, configure_method)
from .cluster import NAMESPACE, load_profile, location, kube, kube_command, ownership
from .run_support import ROOT, REFERENCE, environment, exact_keys, integer, sha256, write_json

MODEL_PATTERN = dict(zip('ABCD', ('diurnal', 'bursty', 'batch_ramp', 'signaling')))


def _validate_config(config):
    """Validate configuration in memory, including during offline audit."""
    exact_keys(config, ('schema_version', 'cluster', 'purpose', 'methods', 'workloads',
                        'seeds', 'duration_s', 'cooldown_s', 'complexity', 'models_dir', 'assurance'))
    if type(config['schema_version']) is not int or config['schema_version'] != 1 or config['purpose'] not in ('technical_smoke', 'replication_attempt'):
        raise ValueError('Unsupported schema or experiment purpose')
    if any(not isinstance(config[key], str) or not config[key] for key in ('cluster', 'models_dir')):
        raise ValueError('Cluster and model locations must be nonempty paths')
    for field, choices in [('methods', METHODS), ('workloads', 'ABCDEFGH')]:
        values = config[field]
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v in choices for v in values):
            raise ValueError(f'Invalid {field}: {values}')
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate {field}')
    if not isinstance(config['seeds'], list) or not config['seeds']:
        raise ValueError('seeds must be a nonempty list')
    for seed in config['seeds']:
        integer(seed, 0, 2**32 - 1, 'seed')
    if len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Duplicate seeds')
    for field, minimum, maximum in [('duration_s', 30, 86400), ('cooldown_s', 0, 3600), ('complexity', 1, 10000000)]:
        integer(config[field], minimum, maximum, field)
    exact_keys(config['assurance'], ('min_trace_ticks', 'min_predictions', 'min_validated_intervals'), label='assurance')
    for key, value in config['assurance'].items():
        integer(value, 1, 86400, key)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    _validate_config(config)
    config['cluster'] = str((path.parent / config['cluster']).resolve())
    config['models_dir'] = str((path.parent / config['models_dir']).resolve())
    profile = load_profile(config['cluster'])
    return config, profile


def _matrix_cells(config):
    _validate_config(config)
    return [{'method': method, 'workload': workload, 'replicate': i + 1, 'seed': seed,
             'model_pattern': MODEL_PATTERN.get(workload, 'diurnal'),
             'capabilities': capability_record(method),
             'preset_configuration': preset_configuration(method)}
            for i, seed in enumerate(config['seeds']) for workload in config['workloads']
            for method in config['methods']]


def plan(config, profile):
    return {'kind': 'EXPERIMENT_PLAN_NOT_EXECUTED', 'context': 'kind-' + profile['name'],
            'namespace': NAMESPACE, 'purpose': config['purpose'],
            'cells': _matrix_cells(config),
            'config': config, 'cluster_profile': profile}


def verify_models(config):
    manifest = json.loads((ROOT / 'provenance/omitted_inputs.json').read_text())
    expected = {r['path'].removeprefix('models/'): r['sha256']
                for r in manifest['inputs'] if r['path'].startswith('models/')}
    files = {}
    for workload in config['workloads']:
        pattern = MODEL_PATTERN.get(workload, 'diurnal')
        for method in config['methods']:
            uq = capabilities(method).model_head
            if uq is None:
                continue
            relative = Path('uq') / pattern / uq
            directory = Path(config['models_dir']) / relative
            required = {p: h for p, h in expected.items() if p.startswith(relative.as_posix() + '/')}
            if not directory.is_dir() or not required:
                raise ValueError(f'Required hash-pinned model is missing: {directory}; see confscale.inputs')
            for p, digest in required.items():
                if sha256(Path(config['models_dir']) / p) != digest:
                    raise ValueError(f'Model hash mismatch: {p}')
                files[p] = digest
    return files


@contextmanager
def forward(profile, output):
    port = profile['prometheus_port']
    # A pre-existing listener must never be mistaken for our metrics source.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    with (output / 'prometheus-forward.log').open('w') as log:
        proc = subprocess.Popen(kube_command(profile, '-n', 'monitoring', 'port-forward',
                                '--address', '127.0.0.1', 'svc/prometheus-kube-prometheus-prometheus',
                                f'{port}:9090'), stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 30
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Owned Prometheus forwarder exited; inspect its log')
                try:
                    with urlopen(url + '/-/ready', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Prometheus port forward not ready')
                time.sleep(.25)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def metrics_preflight(url):
    queries = {
        'app_scrapes': 'count(up{namespace="infosys-benchmark",app=~"frontend|processor|compute-worker"} == 1)',
        'replicas': 'kube_deployment_spec_replicas{namespace="infosys-benchmark",deployment="compute-worker"}',
        'cpu': 'count(container_cpu_usage_seconds_total{namespace="infosys-benchmark",container="compute"})',
    }
    results = {}
    for name, query in queries.items():
        with urlopen(url + '/api/v1/query?' + urlencode({'query': query}), timeout=10) as response:
            data = json.load(response)
        values = data.get('data', {}).get('result', [])
        if data.get('status') != 'success' or not values or float(values[0]['value'][1]) <= 0:
            raise RuntimeError(f'Missing required telemetry: {name}')
        results[name] = data
    return results


def snapshot(profile):
    # No Secrets or kubeconfig contents enter experiment receipts.
    return json.loads(kube(profile, '-n', NAMESPACE, 'get', 'deployments,pods,services,hpa', '-o', 'json'))


def restore_baseline(profile):
    kube(profile, '-n', NAMESPACE, 'delete', 'hpa', 'compute-worker', 'compute-worker-hpa', '--ignore-not-found=true')
    inventory = json.loads(kube(profile, '-n', NAMESPACE, 'get', 'hpa', '-o', 'json'))
    items = inventory.get('items') if isinstance(inventory, dict) else None
    if not isinstance(items, list):
        raise RuntimeError('Cannot prove baseline HPA absence: invalid inventory')
    for item in items:
        spec = item.get('spec') if isinstance(item, dict) else None
        target = spec.get('scaleTargetRef') if isinstance(spec, dict) else None
        if (not isinstance(target, dict) or not isinstance(target.get('kind'), str) or
                not target['kind'] or not isinstance(target.get('name'), str) or not target['name']):
            raise RuntimeError('Cannot prove baseline HPA absence: incomplete inventory')
        if target['kind'] == 'Deployment' and target['name'] == 'compute-worker':
            raise RuntimeError('Cannot restore baseline while an HPA still targets compute-worker')
    kube(profile, '-n', NAMESPACE, 'scale', 'deployment/compute-worker', '--replicas=1')
    kube(profile, '-n', NAMESPACE, 'rollout', 'status', 'deployment/compute-worker', '--timeout=120s', timeout=130)
    obj = json.loads(kube(profile, '-n', NAMESPACE, 'get', 'deployment/compute-worker', '-o', 'json'))
    if obj['spec']['replicas'] != 1 or obj.get('status', {}).get('readyReplicas') != 1:
        raise RuntimeError('Baseline restoration could not be verified')


def run(config, profile, output):
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Live runtime is locked for Python 3.12; use .venv/bin/python')
    inputs = verify_models(config)
    owner = ownership(profile)
    deployment = json.loads((location(profile) / 'deployment.json').read_text())
    if deployment['profile'] != profile or deployment['ownership'] != owner:
        raise ValueError('Profile differs from the deployed cluster receipt')
    for filename, digest in deployment['rendered_sha256'].items():
        if sha256(location(profile) / filename) != digest:
            raise ValueError(f'Deployed manifest changed: {filename}')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', profile['controller_metrics_port']))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = location(profile) / 'experiment.lock'
    # An exclusive file prevents concurrent experiments from racing resets.
    with lock.open('x') as stream:
        stream.write(str(output))
    receipt = {'kind': 'LIVE_TECHNICAL_VALIDATION_NOT_PAPER_RESULTS', 'status': 'running', 'runs': [],
               'started_at': datetime.now(timezone.utc).isoformat(), 'restoration': 'not_attempted'}
    env_keys = {'KUBECONFIG': str(location(profile) / 'kubeconfig'),
                'CONFSCALE_KUBE_CONTEXT': 'kind-' + profile['name'],
                'CONFSCALE_MODELS_DIR': config['models_dir'],
                'CONFSCALE_ENABLE_REFERENCE_RUNTIME': '1',
                'CONFSCALE_METRICS_PORT': str(profile['controller_metrics_port'])}
    old_env = {k: os.environ.get(k) for k in env_keys}
    try:
        write_json(output / 'plan.json', plan(config, profile))
        write_json(output / 'environment.json', environment())
        write_json(output / 'inputs.json', inputs)
        write_json(output / 'deployment.json', deployment)
        write_json(output / 'before.json', snapshot(profile))
        os.environ.update(env_keys)
        sys.path.insert(0, str(REFERENCE))
        from orchestrator import methods
        from orchestrator.run_matrix import execute_single_run
        methods.set_thread_kube_context(env_keys['CONFSCALE_KUBE_CONTEXT'])
        methods.UQ_MODELS_DIR = Path(config['models_dir']) / 'uq'
        restore_baseline(profile)
        with forward(profile, output) as prometheus_url:
            write_json(output / 'telemetry_preflight.json', metrics_preflight(prometheus_url))
            frontend_url = f'http://127.0.0.1:{profile["frontend_port"]}'
            with urlopen(frontend_url + '/health', timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError('Frontend unavailable')
            for cell in plan(config, profile)['cells']:
                method = configure_method(methods.get_method(cell['method']))
                if hasattr(method, 'prometheus_port'):
                    method.prometheus_port = profile['prometheus_port']
                result = execute_single_run(method, cell['workload'], cell['replicate'],
                           config['duration_s'], config['complexity'], frontend_url, prometheus_url,
                           output, workload_extra_args=['--seed', str(cell['seed'])])
                run_dir = Path(result['output_dir'])
                write_json(run_dir / 'execution.json', result)
                if cell['method'] in COMPARISON_METHODS:
                    write_json(run_dir / 'effective_method.json', {
                        'method': cell['method'], 'capabilities': capability_record(cell['method']),
                        'expected_config': preset_configuration(cell['method']),
                        'actual_config': method.get_state()})
                verdict = assess(run_dir, capabilities(cell['method']).forecasts, config['assurance'],
                                 method_name=cell['method'], expected_context='kind-' + profile['name'])
                # Retain the cell verdict even when outer cleanup subsequently fails.
                write_json(run_dir / 'assurance.json', verdict)
                receipt['runs'].append({'cell': cell, 'directory': run_dir.name, 'assurance': verdict})
                write_json(output / 'receipt.json', receipt)
                if verdict['status'] != 'pass':
                    raise RuntimeError(f'Cell failed technical assurance: {run_dir.name}')
                restore_baseline(profile)
                write_json(run_dir / 'after_reset.json', snapshot(profile))
                time.sleep(config['cooldown_s'])
        receipt['status'] = 'pass'
    except (Exception, KeyboardInterrupt) as error:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            restore_baseline(profile)
            write_json(output / 'after.json', snapshot(profile))
            receipt['restoration'] = 'pass'
        except Exception as error:
            receipt['status'] = 'failed'
            receipt['restoration'] = str(error)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        lock.unlink()
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            receipt['output_sha256'] = {relative: sha256(path) for relative, path in _output_files(output).items()}
        except (OSError, ValueError) as error:
            receipt['status'] = 'failed'
            receipt['sealing_error'] = str(error)
            receipt['output_sha256'] = {}
        write_json(output / 'receipt.json', receipt)
    return receipt


def _output_files(output):
    """Inventory every sealed file; only the root receipt contains its own hashes."""
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Output integrity failure: symlink {path.relative_to(output)}')
        if path.is_file() and path != output / 'receipt.json':
            files[path.relative_to(output).as_posix()] = path
    return files


def _audited_matrix(planned):
    try:
        expected = _matrix_cells(planned['config'])
        context = 'kind-' + planned['cluster_profile']['name']
        if (planned['context'] != context or planned['namespace'] != NAMESPACE or
                planned['purpose'] != planned['config']['purpose']):
            raise ValueError('Planned context, namespace or purpose is inconsistent')
        actual = planned['cells']
        # The original supported plan format predates capability metadata.
        legacy = [{key: value for key, value in cell.items()
                   if key not in ('capabilities', 'preset_configuration')} for cell in expected]
        valid_legacy = all(cell['method'] in LEGACY_METHODS for cell in expected) and actual == legacy
        if not expected or not isinstance(actual, list) or (actual != expected and not valid_legacy):
            raise ValueError('Planned matrix differs from its configuration')
        return actual, context
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete planned matrix configuration') from error


def audit(output):
    """Verify a sealed receipt and independently reassess retained cells, offline."""
    output = Path(output).resolve()
    files = _output_files(output)
    receipt = json.loads((output / 'receipt.json').read_text())
    if receipt.get('status') != 'pass' or receipt.get('restoration') != 'pass':
        raise ValueError('Run is not sealed with passing execution and restoration')
    hashes = receipt.get('output_sha256', {})
    if not isinstance(hashes, dict) or not hashes or set(hashes) != set(files):
        raise ValueError('Output integrity failure: incomplete or unexpected hash inventory')
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or sha256(path) != digest:
            raise ValueError(f'Output integrity failure: {relative}')
    planned = json.loads((output / 'plan.json').read_text())
    cells, context = _audited_matrix(planned)
    runs = receipt.get('runs')
    if (not isinstance(runs, list) or not runs or not all(isinstance(r, dict) for r in runs) or
            [r.get('cell') for r in runs] != cells):
        raise ValueError('Receipt cells differ from the planned matrix')
    seen_directories = set()
    for result in runs:
        if not isinstance(result.get('directory'), str) or not result['directory']:
            raise ValueError('Cell directory is missing')
        directory = (output / result['directory']).resolve()
        if directory == output or not directory.is_relative_to(output) or not directory.is_dir():
            raise ValueError('Cell directory escapes run or is missing')
        if directory in seen_directories:
            raise ValueError('Multiple planned cells share a cell directory')
        seen_directories.add(directory)
        name = result['cell']['method']
        if name in COMPARISON_METHODS:
            if (result['cell'].get('capabilities') != capability_record(name) or
                    result['cell'].get('preset_configuration') != preset_configuration(name)):
                raise ValueError('Comparison cell capabilities or preset differ from the runtime contract')
        verdict = assess(directory, capabilities(name).forecasts,
                         planned['config']['assurance'], method_name=name, expected_context=context)
        if verdict['status'] != 'pass':
            raise ValueError(f'Cell no longer passes assurance: {result["directory"]}')
    return {'kind': 'RETAINED_TECHNICAL_RUN_AUDIT', 'status': 'pass',
            'files_verified': len(hashes), 'cells_verified': len(receipt['runs'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'check-inputs', 'audit'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/smoke.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'audit':
        if args.output is None:
            parser.error('audit requires --output pointing at an existing run')
        print(json.dumps(audit(args.output), indent=2))
        return
    config, profile = load_config(args.config)
    if args.action == 'plan':
        result = plan(config, profile)
    elif args.action == 'check-inputs':
        result = {'verified_inputs': verify_models(config)}
    else:
        if args.output is None:
            parser.error('run requires a new --output directory')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        result = run(config, profile, args.output)
    print(json.dumps(result, indent=2))
    if args.action == 'run' and result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
