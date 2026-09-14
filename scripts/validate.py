"""Run offline checks and retain a machine-readable command receipt."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-tests', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/validation')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, '-S', '-m', 'confscale', 'verify'],
        [sys.executable, '-S', '-m', 'confscale', 'reproduce', '--output', str(output / 'paper')],
        [sys.executable, '-S', '-m', 'confscale', 'demo', '--output', str(output / 'demo')],
        [sys.executable, '-S', '-m', 'confscale.experiment', 'plan'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    ]
    if args.runtime_tests:
        commands.append([sys.executable, '-m', 'pytest', 'tests', 'reference', '-q'])
    receipt = {'kind': 'OFFLINE_SOFTWARE_CHECKS_NOT_LIVE_EXPERIMENTS', 'python': sys.version,
               'platform': platform.platform(), 'started_at': datetime.now(timezone.utc).isoformat(),
               'commands': []}
    failed = False
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (output / f'{i:02d}.stdout.log').write_text(result.stdout, encoding='utf-8')
        (output / f'{i:02d}.stderr.log').write_text(result.stderr, encoding='utf-8')
        receipt['commands'].append({'command': command, 'returncode': result.returncode})
        print(f'{"PASS" if result.returncode == 0 else "FAIL"}: {" ".join(command)}')
        failed |= result.returncode != 0
    receipt['status'] = 'fail' if failed else 'pass'
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
