"""Run the original 17 + 3 configurations over the five reported seeds."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from _paths import ROOT

GROUPS = {'main': 'run_manifest.txt', 'core': 'run_manifest_RDFcore15.txt'}
SEEDS = [0, 1, 42, 3407, 1234]

def read_manifest(name):
    text = (ROOT / 'HAZ-ViT/configs_seeds5' / name).read_text(encoding='utf-8')
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['main', 'core', 'all'], default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'work_dirs')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    groups = ['main', 'core'] if args.group == 'all' else [args.group]
    jobs = [(config, seed) for group in groups for config in read_manifest(GROUPS[group]) for seed in args.seeds]
    print(f'{len(jobs)} runs selected.', flush=True)
    failures = []
    for index, (config, seed) in enumerate(jobs, 1):
        output = args.work_dir.resolve() / Path(config).stem / f'seed_{seed}'
        print(f'[{index}/{len(jobs)}] {config}, seed={seed}', flush=True)
        if args.list_only:
            continue
        if (output / 'done.ok').exists():
            print('Completed run found; skipping.', flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / 'tools/train.py'), config,
                   '--seed', str(seed), '--work-dir', str(output)]
        with (output / 'train.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append((config, seed))
            print(f'Run failed; inspect {output / "train.log"}', flush=True)
        else:
            (output / 'done.ok').write_text('Training completed.\n', encoding='ascii')
    if failures:
        raise SystemExit(f'{len(failures)} training runs failed: {failures}')

if __name__ == '__main__':
    main()
