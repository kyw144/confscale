"""Offline entrypoint: imports no cluster runtime and never sends requests."""
import argparse
import json
from pathlib import Path
from .demo import write_demo
from .reproduce import reproduce

def main():
    parser = argparse.ArgumentParser(description='ConfScale local paper artifact: offline only')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='new illustrative synthetic replay; not paper evidence')
    demo.add_argument('--output', type=Path, default=Path('generated/demo'))
    demo.add_argument('--ticks', type=int, default=240)
    demo.add_argument('--shift-at', type=int, default=80)
    rep = sub.add_parser('reproduce', help='regenerate all five paper tables from shipped frozen aggregates')
    rep.add_argument('--output', type=Path, default=Path('generated/paper'))
    sub.add_parser('verify', help='check packaged source/evidence hashes')
    args = parser.parse_args()
    if args.command == 'demo': result = write_demo(args.output, args.ticks, args.shift_at)
    elif args.command == 'reproduce': result = reproduce(args.output)
    else:
        from .verify import verify
        result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__': main()
