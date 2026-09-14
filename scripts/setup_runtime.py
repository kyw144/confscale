"""Install the hashed Python 3.12 environment using the correct PyTorch index."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sync_command(python, platform):
    command = ['uv', 'pip', 'sync', '--python', str(python), '--require-hashes']
    # Universal compilation records the PyPI Mac wheel hashes. The CPU index
    # also serves a Mac wheel with the same version but different bytes.
    if platform != 'darwin':
        command += ['--torch-backend', 'cpu']
    return command + ['requirements/runtime.lock', 'requirements/test.lock']


def main():
    python = ROOT / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    env = dict(os.environ)
    env.pop('UV_TORCH_BACKEND', None)
    if not python.exists():
        subprocess.run(['uv', 'venv', '--python', '3.12', '.venv'], cwd=ROOT, check=True, env=env)
    subprocess.run([str(python), '-c',
                    'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 .venv"'], check=True)
    subprocess.run(sync_command(python, sys.platform), cwd=ROOT, check=True, env=env)


if __name__ == '__main__':
    main()
