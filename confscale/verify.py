"""Read-only hash verification; source repository not required."""
import hashlib
import json
from pathlib import Path

def verify():
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
            'scope':'integrity of shipped copies/adaptations, not verification of experiments or source authorship'}
