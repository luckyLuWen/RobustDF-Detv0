"""Resolve repository paths without depending on the caller's directory."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MMDET = ROOT / 'HAZ-ViT/mmdetection'
CONFIGS = ROOT / 'HAZ-ViT/configs'
MAIN_CONFIG = 'rtmdet_l_fire_4c_innov_dualpath_globalgate_dinov3.py'

def config_path(value):
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path, CONFIGS / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Configuration not found: {value}')

def environment():
    if not (MMDET / 'tools/train.py').is_file():
        raise RuntimeError('Run python tools/setup_mmdet.py before training or evaluation.')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(MMDET) + os.pathsep + env.get('PYTHONPATH', '')
    return env
