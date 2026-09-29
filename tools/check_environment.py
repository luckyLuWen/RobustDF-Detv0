"""Check the training environment, required assets, and a resolved configuration."""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path
from _paths import ROOT, MMDET, MAIN_CONFIG, config_path, environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=MAIN_CONFIG)
    args = parser.parse_args()
    environment()
    sys.path.insert(0, str(MMDET))
    import torch
    for name, expected in [('torch', '2.1.2'), ('torchvision', '0.16.2'),
                           ('mmcv', '2.1.0'), ('mmengine', '0.10.7'), ('mmdet', '3.3.0')]:
        module = importlib.import_module(name)
        actual = module.__version__.split('+')[0]
        print(f'{name}: {module.__version__} ({module.__file__})')
        if actual != expected:
            raise SystemExit(f'Expected {name} {expected}; found {actual}.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Check the GPU driver and the CUDA-enabled PyTorch installation.')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    from mmengine.config import Config
    cfg = Config.fromfile(str(config_path(args.config)))
    expected_classes = ['car_fire', 'car_nofire', 'lkyw_fire', 'lkyw_nofire']
    for split in ['train', 'val', 'test']:
        path = ROOT / 'datasets/fire2/coco_4c/annotations' / f'{split}.json'
        if not path.is_file():
            raise SystemExit(f'Missing {path}. See the data download section in README.md.')
        data = json.loads(path.read_text(encoding='utf-8'))
        categories = [c['name'] for c in sorted(data['categories'], key=lambda c: c['id'])]
        if categories != expected_classes:
            raise SystemExit(f'Unexpected class order in {path}: {categories}')
        image_dir = ROOT / 'datasets/fire2/coco_4c/images'
        missing = [i['file_name'] for i in data['images'] if not (image_dir / i['file_name']).is_file()]
        if missing:
            raise SystemExit(f'{split}: {len(missing)} images are missing, starting with {missing[0]}.')
        print(f'{split}: {len(data["images"])} images, {len(data["annotations"])} annotations')
    print(f'Configuration resolved: {cfg.filename}')

if __name__ == '__main__':
    main()
