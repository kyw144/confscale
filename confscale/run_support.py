"""Shared file/provenance helpers for explicit live commands."""
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import distributions
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'reference/stage3_scale'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def checked(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, text=True,
                          capture_output=True, timeout=kwargs.pop('timeout', 60), **kwargs).stdout


def environment():
    paths = sorted(set(checked(['git', 'ls-files', '--cached', '--others',
                                '--exclude-standard'], cwd=ROOT).splitlines()))
    return {
        'commit': checked(['git', 'rev-parse', 'HEAD'], cwd=ROOT).strip(),
        'dirty': bool(checked(['git', 'status', '--porcelain'], cwd=ROOT).strip()),
        'files': {p: sha256(ROOT / p) for p in paths if (ROOT / p).is_file()},
        'python': sys.version, 'platform': platform.platform(),
        'machine': platform.machine(),
        'packages': sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions()),
    }


def exact_keys(value, required, optional=(), label='configuration'):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f'{label}: missing {sorted(missing)}, unknown {sorted(extra)}')


def integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{label} must be an integer in [{minimum}, {maximum}]')
    return value
