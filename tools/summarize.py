"""Recompute result tables from compact per-seed validation metrics."""
from __future__ import annotations
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {'RTMDet': 'RTM-Base', 'YOLOX': 'YOLOX-Base', 'Faster R-CNN': 'FRCNN-Base'}
FAMILIES = list(BASELINES)

def mean_ci(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half = float(student_t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values)) if len(values) > 1 else float('nan')
    return mean, std, mean - half, mean + half

def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [float('nan')] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
        result[index] = previous
    return result

def paired(values, baseline):
    differences = [a - b for a, b in zip(values, baseline)]
    mean, std, low, high = mean_ci(differences)
    if len(differences) < 2:
        return mean, low, high, float('nan'), float('nan')
    if std == 0:
        return mean, low, high, (0.0 if mean == 0 else math.copysign(float('inf'), mean)), (1.0 if mean == 0 else 0.0)
    dz = mean / std
    p = float(2 * student_t.sf(abs(dz * math.sqrt(len(differences))), len(differences) - 1))
    return mean, low, high, dz, p

def load(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        identity = (row['config'], int(row['seed']))
        if identity in seen:
            raise ValueError(f'Duplicate run: {identity}')
        seen.add(identity)
        row['seed'] = identity[1]
        for key in ['best_ap', 'best_ap50', 'final_ap', 'final_ap50']:
            row[key] = float(row[key])
    if not rows:
        raise ValueError('The metrics file is empty.')
    return rows

def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['model']].append(row)
    summaries = []
    for model, runs in groups.items():
        mean_ap, std_ap, lo, hi = mean_ci([r['best_ap'] for r in runs])
        mean_ap50, std_ap50, _, _ = mean_ci([r['best_ap50'] for r in runs])
        baseline = groups.get(BASELINES[runs[0]['family']])
        delta = (mean_ap - statistics.mean(r['best_ap'] for r in baseline)) * 100 if baseline else float('nan')
        summaries.append(dict(model=model, family=runs[0]['family'], config=runs[0]['config'],
            runs=len(runs), ap=mean_ap, ap_std=std_ap, ap_ci_low=lo, ap_ci_high=hi,
            ap50=mean_ap50, ap50_std=std_ap50, delta_ap_points=delta,
            final_ap=statistics.mean(r['final_ap'] for r in runs),
            final_ap50=statistics.mean(r['final_ap50'] for r in runs),
            ap_gap_points=statistics.mean(r['best_ap'] - r['final_ap'] for r in runs) * 100,
            ap50_gap_points=statistics.mean(r['best_ap50'] - r['final_ap50'] for r in runs) * 100))
    summaries.sort(key=lambda r: (FAMILIES.index(r['family']), -r['ap']))
    comparisons = []
    for family, baseline_name in BASELINES.items():
        if baseline_name not in groups:
            continue
        baseline = {r['seed']: r for r in groups[baseline_name]}
        family_rows = []
        for model, runs in groups.items():
            if model == baseline_name or runs[0]['family'] != family:
                continue
            by_seed = {r['seed']: r for r in runs}
            seeds = sorted(set(by_seed) & set(baseline))
            if len(seeds) < 2:
                raise ValueError(f'At least two paired seeds are required for {model}.')
            row = dict(family=family, model=model, baseline=baseline_name, paired_seeds=len(seeds))
            for metric in ['ap', 'ap50']:
                mean, low, high, dz, p = paired(
                    [by_seed[s]['best_' + metric] for s in seeds],
                    [baseline[s]['best_' + metric] for s in seeds])
                row.update({f'delta_{metric}_points': mean * 100, f'{metric}_delta_ci_low_points': low * 100,
                    f'{metric}_delta_ci_high_points': high * 100, f'{metric}_dz': dz, f'{metric}_p': p})
            family_rows.append(row)
        for metric in ['ap', 'ap50']:
            for row, corrected in zip(family_rows, holm([r[metric + '_p'] for r in family_rows])):
                row[metric + '_holm_p'] = corrected
                row[metric + '_significance'] = '***' if corrected < .001 else '**' if corrected < .01 else '*' if corrected < .05 else 'ns'
        comparisons.extend(family_rows)
    return summaries, comparisons

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k: f'{v:.10g}' if isinstance(v, float) else v for k, v in row.items()})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT / 'results/seed_metrics.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/results')
    args = parser.parse_args()
    summaries, comparisons = summarize(load(args.metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'main_results.csv', summaries)
    write_csv(args.output_dir / 'baseline_comparisons.csv', comparisons)
    print('| Model | AP | AP50 | Delta AP (points) |')
    print('|---|---:|---:|---:|')
    for row in summaries:
        print(f"| {row['model']} | {row['ap']:.4f} | {row['ap50']:.4f} | {row['delta_ap_points']:+.2f} |")

if __name__ == '__main__':
    main()
