"""Restore local models matching the input manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from .run_support import ROOT, sha256, write_json


def restore(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(ROOT / 'inputs'):
        raise ValueError('Restore destination must be under the ignored inputs/ directory')
    manifest = json.loads((ROOT / 'provenance/omitted_inputs.json').read_text())
    rows = [r for r in manifest['inputs'] if r['path'].startswith('models/')]
    verified = []
    for row in rows:
        relative = Path(row['path']).relative_to('models')
        original = (source / relative).resolve()
        if not original.is_file():
            original = (source / row['source']).resolve()
        if not original.is_relative_to(source):
            raise ValueError('Input path escapes source directory')
        data = original.read_bytes()
        source_hash = hashlib.sha256(data).hexdigest()
        conversion = 'none'
        if source_hash != row['sha256'] and original.suffix in ('.yaml', '.yml', '.json', '.csv'):
            # Accept line-ending conversion only when the complete expected hash matches.
            lf = data.replace(b'\r\n', b'\n')
            for label, candidate in [('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))]:
                if hashlib.sha256(candidate).hexdigest() == row['sha256']:
                    data, conversion = candidate, label
                    break
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Input hash mismatch: {row["path"]}')
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Input path escapes destination directory')
        if target.exists() and sha256(target) != row['sha256']:
            raise ValueError(f'Refusing to replace different local input: {target}')
        verified.append((data, target, {**row, 'local_source_sha256': source_hash,
                                       'line_ending_conversion': conversion}))
    for data, target, _ in verified:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    receipt = {'source_commit': manifest['source_commit'], 'files': [r for _, _, r in verified]}
    write_json(destination / 'restore_receipt.json', receipt)
    return {'models_verified': len(verified), 'destination': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Model directory containing gru/ and uq/')
    parser.add_argument('--destination', type=Path, default=ROOT / 'inputs/models')
    args = parser.parse_args()
    print(json.dumps(restore(args.source, args.destination), indent=2))


if __name__ == '__main__':
    main()
