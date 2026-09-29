"""Train one configuration with the tested MMDetection entry point."""
import argparse
import subprocess
import sys
from pathlib import Path
from _paths import ROOT, MMDET, MAIN_CONFIG, config_path, environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default=MAIN_CONFIG)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--cfg-options', nargs='+', default=[])
    args, extra = parser.parse_known_args()
    config = config_path(args.config)
    work_dir = args.work_dir or ROOT / 'work_dirs' / config.stem / f'seed_{args.seed}'
    options = args.cfg_options + [f'randomness.seed={args.seed}', 'randomness.diff_rank_seed=False']
    command = [sys.executable, str(MMDET / 'tools/train.py'), str(config),
               '--work-dir', str(work_dir.resolve()), *extra, '--cfg-options', *options]
    return subprocess.call(command, cwd=MMDET, env=environment())

if __name__ == '__main__':
    raise SystemExit(main())
