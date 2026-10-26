"""Read-only hash verification; source repository not required."""
import hashlib
import json
from pathlib import Path
import re


PAPER_SNAPSHOT = '7b43867bf644f4e6b1d27d0b5c17c9ce1f1a45a8'
RUNTIME_BASELINE_HASHES = {
    'reference/stage3_scale/orchestrator/controller.py': '87417f5cdbe23cec199549c25ed2eb3afdeb77b1f0fb9a1b09adf96f8fd5c936',
    'reference/stage3_scale/orchestrator/methods.py': '1804bbc57cc29acbce777ec01a7a3007935855957605284d9efbe277b487f9c1',
    'reference/stage3_scale/collector/collect.py': 'd2ceda51f923b7e31d4329f4f9ae0a58b4e0b606e5c3f46e400f71027806e88e',
    'confscale/planning_numpy.py': '55ad2735c8a9c4fb2570c411c206d30c80858b7fb1b9ccdf85f52d2f15d636c1',
    'confscale/experiment.py': 'da3e604480bdd81f7f4e32ba463c99f1fc8c720a8b1d51847c63341129d7614b',
    'confscale/assurance.py': '0f1b80cb8f9c8f42642eae4db77ba94a4b0f93cf0c98a00e13c0a0182645cb0e',
    'confscale/capabilities.py': None,
    'confscale/native_hpa.py': None,
    'configs/comparison-smoke.json': None,
}


def verify_runtime_adaptations(root, source_manifest):
    """Verify shipped runtime bytes and their recorded public-snapshot lineage."""
    from .comparisons import _hash, _json, _keys
    root = Path(root).resolve()
    adaptation = _json(root / 'provenance/comparison_runtime_adaptations.json')
    _keys(adaptation, ('date', 'paper_snapshot_commit', 'scope', 'pairing_semantics',
                       'source_components', 'changes'), 'runtime adaptation manifest')
    if adaptation['paper_snapshot_commit'] != PAPER_SNAPSHOT:
        raise ValueError('Unexpected runtime paper snapshot')
    if not isinstance(adaptation['date'], str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', adaptation['date']):
        raise ValueError('Invalid runtime adaptation date')
    for field in ('scope', 'pairing_semantics'):
        if not isinstance(adaptation[field], str) or not adaptation[field]:
            raise ValueError('Missing runtime adaptation scope')
    components = set()
    for row in adaptation['source_components']:
        _keys(row, ('component', 'sha256'), 'runtime source component')
        if not isinstance(row['component'], str) or row['component'] in components:
            raise ValueError('Invalid runtime source component identity')
        _hash(row['sha256'])
        components.add(row['component'])
    if components != {'signed_first_forecast_correction', 'first_forecast_target_rule',
                      'native_hpa_identity_lifecycle', 'native_hpa_exact_spec_adapter',
                      'overhead_query_response_provenance'}:
        raise ValueError('Incomplete runtime source component references')
    source_rows = {row['destination']: row for row in source_manifest['records']}
    destinations = set()
    for row in adaptation['changes']:
        _keys(row, ('destination', 'previous_shipped_sha256', 'destination_sha256', 'adaptation'), 'runtime adaptation')
        destination = row['destination']
        if destination not in RUNTIME_BASELINE_HASHES or destination in destinations:
            raise ValueError('Unexpected or duplicate runtime adaptation destination')
        if row['previous_shipped_sha256'] != RUNTIME_BASELINE_HASHES[destination]:
            raise ValueError('Runtime baseline lineage changed')
        if not isinstance(row['adaptation'], str) or not row['adaptation']:
            raise ValueError('Missing runtime adaptation description')
        _hash(row['destination_sha256'])
        target = (root / destination).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Runtime adaptation path escapes artifact')
        if hashlib.sha256(target.read_bytes()).hexdigest() != row['destination_sha256']:
            raise ValueError(f'Runtime adaptation hash mismatch: {destination}')
        if destination in source_rows and source_rows[destination]['destination_sha256'] != row['destination_sha256']:
            raise ValueError('Runtime and source manifests disagree')
        destinations.add(destination)
    if destinations != set(RUNTIME_BASELINE_HASHES):
        raise ValueError('Incomplete runtime adaptation inventory')
    return {'runtime_adaptation_files_verified': len(destinations),
            'runtime_source_component_hashes_recorded': len(components)}

def verify():
    from .comparisons import verify_comparisons
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'provenance/source_manifest.json').read_text(encoding='utf-8'))
    checked = 0
    for row in manifest['records']:
        target = (root/row['destination']).resolve()
        if not target.is_relative_to(root): raise ValueError('manifest path escapes artifact')
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != row['destination_sha256']: raise ValueError(f'Hash mismatch: {row["destination"]}')
        checked += 1
    return {'source_commit':manifest['source_commit'], 'source_derived_files_verified':checked,
            **verify_comparisons(root),
            **verify_runtime_adaptations(root, manifest),
            'scope':'integrity of shipped copies/adaptations, not verification of experiments or source authorship'}
