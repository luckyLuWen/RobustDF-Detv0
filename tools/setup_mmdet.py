"""Fetch the tested MMDetection revision and install the RDF-Det overlay."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'HAZ-ViT/mmdetection'
COMMIT = '44ebd17b145c2372c4b700bfb9cb20dbd28ab64a'
URL = 'https://github.com/open-mmlab/mmdetection.git'

def run(*args):
    subprocess.run([str(a) for a in args], check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Copy an existing v3.3.0 checkout instead of downloading it.')
    parser.add_argument('--skip-install', action='store_true', help='Prepare source without running pip.')
    args = parser.parse_args()
    if not DEST.exists():
        if args.source:
            source = args.source.resolve()
            head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            if head != COMMIT:
                raise SystemExit(f'Expected MMDetection commit {COMMIT}, got {head}.')
            shutil.copytree(source, DEST, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
            (DEST / '.rdfdet_upstream_revision').write_text(COMMIT + '\n', encoding='ascii')
        else:
            run('git', 'clone', '--branch', 'v3.3.0', '--depth', '1', URL, DEST)
    if (DEST / '.git').exists():
        head = subprocess.check_output(['git', '-C', str(DEST), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        marker = DEST / '.rdfdet_upstream_revision'
        head = marker.read_text(encoding='ascii').strip() if marker.exists() else ''
    if head != COMMIT:
        raise SystemExit(f'{DEST} is not the expected MMDetection revision. Keep it intact and use a fresh checkout.')
    overlay = ROOT / 'projects/ViTDet/vitdet'
    target = DEST / 'projects/ViTDet/vitdet'
    target.mkdir(parents=True, exist_ok=True)
    for path in overlay.glob('*.py'):
        shutil.copy2(path, target / path.name)
    patch_root = ROOT / 'patches/mmdetection'
    for path in patch_root.rglob('*.py'):
        target_file = DEST / path.relative_to(patch_root)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_file)
    if not args.skip_install:
        run(sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '-e', DEST)
    print(f'MMDetection {COMMIT[:12]} prepared at {DEST}')

if __name__ == '__main__':
    main()
