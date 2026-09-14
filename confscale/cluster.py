"""Render and provision the dedicated ConfScale kind testbed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from .run_support import ROOT, REFERENCE, checked, exact_keys, integer, sha256, write_json

SERVICES = ('compute-worker', 'processor', 'frontend')
NAMESPACE = 'infosys-benchmark'


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    exact_keys(profile, ('schema_version', 'name', 'frontend_port',
                         'prometheus_port', 'controller_metrics_port'))
    if profile['schema_version'] != 1:
        raise ValueError('Unsupported cluster schema_version')
    if not isinstance(profile['name'], str) or not re.fullmatch(r'confscale-[a-z0-9-]{1,40}', profile['name']):
        raise ValueError('Cluster name must start with confscale- and use lowercase DNS characters')
    ports = [integer(profile[k], 1024, 65535, k) for k in
             ('frontend_port', 'prometheus_port', 'controller_metrics_port')]
    if len(set(ports)) != len(ports):
        raise ValueError('Cluster host ports must be distinct')
    return profile


def location(profile):
    return ROOT / 'generated/clusters' / profile['name']


def kube_command(profile, *args):
    return ['kubectl', '--kubeconfig', str(location(profile) / 'kubeconfig'),
            '--context', 'kind-' + profile['name'], *args]


def kube(profile, *args, timeout=60):
    return checked(kube_command(profile, *args), timeout=timeout)


def ownership(profile):
    state = json.loads((location(profile) / 'ownership.json').read_text())
    uid = kube(profile, 'get', 'namespace', 'kube-system', '-o', 'jsonpath={.metadata.uid}')
    if state != {'name': profile['name'], 'cluster_uid': uid}:
        raise ValueError('Cluster ownership does not match this workspace')
    return state


def render(profile, architecture=None):
    import yaml
    architecture = architecture or {'aarch64': 'arm64', 'x86_64': 'amd64'}.get(
        platform.machine(), platform.machine())
    lock = json.loads((ROOT / 'cluster/images.lock.json').read_text())
    if architecture not in lock['architectures']:
        raise ValueError(f'Unsupported container architecture: {architecture}')
    images = lock['architectures'][architecture]
    directory = location(profile)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in [ROOT / 'cluster/Dockerfile', ROOT / 'requirements/services.lock']:
        digest.update(path.read_bytes())
    digest.update(images['python'].encode())
    app_images = {}
    for service in SERVICES:
        h = digest.copy()
        h.update((REFERENCE / 'infosys-benchmark' / service / 'app.py').read_bytes())
        app_images[service] = f'confscale-{service}:{h.hexdigest()[:20]}'
    kind = {'kind': 'Cluster', 'apiVersion': 'kind.x-k8s.io/v1alpha4',
            'nodes': [{'role': 'control-plane', 'extraPortMappings': [
                {'containerPort': 30080, 'hostPort': profile['frontend_port'],
                 'listenAddress': '127.0.0.1', 'protocol': 'TCP'}]}]}
    (directory / 'kind.yaml').write_text(yaml.safe_dump(kind))
    app = list(yaml.safe_load_all((REFERENCE / 'infosys-benchmark/k8s/manifests.yaml').read_text()))
    # The dedicated metrics stack scrapes annotations; no operator or CRDs.
    app = [o for o in app if o and o['kind'] not in ('PodMonitor', 'HorizontalPodAutoscaler')]
    for obj in app:
        if obj['kind'] == 'Deployment':
            container = obj['spec']['template']['spec']['containers'][0]
            name = obj['metadata']['name']
            container['image'] = images['redis'] if name == 'cache' else app_images[name]
            if name in SERVICES:
                container['imagePullPolicy'] = 'Never'
                container.setdefault('env', []).append({'name': 'CONFSCALE_ENABLE_REFERENCE_RUNTIME', 'value': '1'})
                if name == 'compute-worker':
                    container['command'] = ['gunicorn', '-w', '4', '-b', '0.0.0.0:8080', 'app:app']
    monitoring = list(yaml.safe_load_all((REFERENCE / 'orchestrator/simple-prometheus.yaml').read_text()))
    for obj in monitoring:
        if obj and obj['kind'] == 'Deployment':
            container = obj['spec']['template']['spec']['containers'][0]
            container['image'] = images['prometheus' if obj['metadata']['name'] == 'prometheus' else 'kube_state_metrics']
    metrics = list(yaml.safe_load_all((ROOT / 'cluster/vendor/metrics-server-v0.8.1.yaml').read_text()))
    for obj in metrics:
        if obj and obj['kind'] == 'Deployment':
            container = obj['spec']['template']['spec']['containers'][0]
            container['image'] = images['metrics_server']
            container['args'].append('--kubelet-insecure-tls')
    for name, objects in [('app', app), ('monitoring', monitoring), ('metrics-server', metrics)]:
        (directory / f'{name}.yaml').write_text(yaml.safe_dump_all(objects, sort_keys=False))
    plan = {'profile': profile, 'architecture': architecture, 'images': images,
            'app_images': app_images, 'kind_image': lock['kind_node'],
            'rendered_sha256': {p.name: sha256(p) for p in directory.glob('*.yaml')}}
    write_json(directory / 'plan.json', plan)
    return plan


def provision(profile, create=True):
    directory = location(profile)
    plan = render(profile)
    log_path = directory / 'provision.log'
    def run(command, timeout=600):
        with log_path.open('a') as log:
            log.write(json.dumps([str(c) for c in command]) + '\n')
            log.flush()
            subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, env={**os.environ, 'DOCKER_BUILDKIT': '0'})
    if create:
        existing = checked(['kind', 'get', 'clusters']).split()
        if profile['name'] in existing:
            raise ValueError('Cluster exists; use deploy for a workspace-owned cluster')
        run(['kind', 'create', 'cluster', '--name', profile['name'], '--image', plan['kind_image'],
             '--config', str(directory / 'kind.yaml'), '--kubeconfig', str(directory / 'kubeconfig'),
             '--wait', '120s', '--retain'])
        (directory / 'kubeconfig').chmod(0o600)
        uid = kube(profile, 'get', 'namespace', 'kube-system', '-o', 'jsonpath={.metadata.uid}')
        write_json(directory / 'ownership.json', {'name': profile['name'], 'cluster_uid': uid})
    ownership(profile)
    image_ids = {}
    for service, image in plan['app_images'].items():
        run(['docker', 'build', '-f', 'cluster/Dockerfile', '--build-arg', f'SERVICE={service}',
             '--build-arg', f'PYTHON_IMAGE={plan["images"]["python"]}', '-t', image, '.'])
        image_ids[service] = checked(['docker', 'image', 'inspect', image, '--format', '{{.Id}}']).strip()
        run(['kind', 'load', 'docker-image', image, '--name', profile['name']])
    for filename in ('metrics-server.yaml', 'monitoring.yaml', 'app.yaml'):
        run(kube_command(profile, 'apply', '-f', str(directory / filename)))
    for namespace, deployment in [('kube-system', 'metrics-server'), ('monitoring', 'prometheus'),
                                 ('monitoring', 'kube-state-metrics'),
                                 *[(NAMESPACE, n) for n in (*SERVICES, 'cache')]]:
        run(kube_command(profile, '-n', namespace, 'rollout', 'status', f'deployment/{deployment}', '--timeout=180s'))
    plan['image_ids'] = image_ids
    plan['status'] = 'deployed'
    plan['ownership'] = ownership(profile)
    write_json(directory / 'deployment.json', plan)
    return {'status': 'deployed', 'context': 'kind-' + profile['name'], 'receipt': str(directory / 'deployment.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['render', 'up', 'deploy'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/cluster.json')
    args = parser.parse_args()
    profile = load_profile(args.config)
    result = render(profile) if args.action == 'render' else provision(profile, create=args.action == 'up')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
