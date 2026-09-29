"""Extract validation peaks and final scores from MMDetection training logs."""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL_RE = re.compile(
    r'Epoch\(val\)\s*\[(\d+)\]\[\d+/\d+\].*?'
    r'coco/bbox_mAP:\s*([0-9.\-]+).*?'
    r'coco/bbox_mAP_50:\s*([0-9.\-]+).*?'
    r'coco/bbox_mAP_75:\s*([0-9.\-]+)')

def collect(root):
    aliases = json.loads((ROOT / 'results/config_names.json').read_text(encoding='utf-8'))
    rows = []
    for log in sorted(root.rglob('train.log')):
        if not log.parent.name.startswith('seed_'):
            continue
        config = log.parent.parent.name + '.py'
        if config not in aliases:
            raise ValueError(f'Unknown configuration in {log}; add its name to results/config_names.json.')
        values = [(int(epoch), float(ap), float(ap50), float(ap75))
                  for epoch, ap, ap50, ap75 in VAL_RE.findall(log.read_text(encoding='utf-8', errors='replace'))]
        if not values:
            raise ValueError(f'No validation records found in {log}.')
        best_ap = max(values, key=lambda x: (x[1], -x[0]))
        best_ap50 = max(values, key=lambda x: (x[2], -x[0]))
        final = values[-1]
        family = 'RTMDet' if config.startswith('rtmdet_') else 'YOLOX' if config.startswith('yolox_') else 'Faster R-CNN'
        rows.append(dict(config=config, model=aliases[config], family=family,
            seed=int(log.parent.name.split('_')[1]), best_ap=best_ap[1], best_ap50=best_ap50[2],
            ap50_at_best_ap=best_ap[2], best_ap_epoch=best_ap[0], best_ap50_epoch=best_ap50[0],
            final_ap=final[1], final_ap50=final[2], final_epoch=final[0]))
    if not rows:
        raise ValueError(f'No runs found below {root}.')
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_root', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/seed_metrics.csv')
    args = parser.parse_args()
    rows = collect(args.log_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} runs to {args.output}')

if __name__ == '__main__':
    main()
