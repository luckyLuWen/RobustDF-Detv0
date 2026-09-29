"""Evaluate a trained checkpoint on the configured held-out test split."""
import argparse
import subprocess
import sys
from pathlib import Path
from _paths import MMDET, MAIN_CONFIG, config_path, environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', default=MAIN_CONFIG)
    args, extra = parser.parse_known_args()
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    command = [sys.executable, str(MMDET / 'tools/test.py'), str(config_path(args.config)),
               str(args.checkpoint.resolve()), *extra]
    return subprocess.call(command, cwd=MMDET, env=environment())

if __name__ == '__main__':
    raise SystemExit(main())
