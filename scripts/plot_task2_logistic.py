"""Render first-logistic comparison and reliability from working metric JSON.

Preview figures stay in ignored work storage, not submission/Git folders.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT/'work/task2/logistic/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT/'work/task2/logistic/comparison.png')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    groups = ['calendar_station', 'own_history', 'neighbors']
    names = ['Time + station', '+ Own history', '+ Neighbors']
    thresholds = [0, 5, 25]
    colors = ['#9ba7b1', '#1a9b8c', '#2967b2']
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for j, q in enumerate(thresholds):
        ax = axes[0, j]
        values = [summary['results'][g]['models'][str(q)]['test']['pr_auc'] for g in groups]
        bars = ax.bar(names, values, color=colors)
        base = summary['results'][groups[0]]['models'][str(q)]['test']['prevalence']
        ax.axhline(base, color='#aa3d3d', ls='--', label=f'Constant baseline: {base:.4f}')
        ax.bar_label(bars, fmt='%.4f', padding=3)
        ax.set_ylim(0, max(values)*1.2)
        ax.set_title(f'Next 30 min > {q} mm: test PR-AUC')
        ax.tick_params(axis='x', labelrotation=12)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        upper = 0
        for g, name, color in zip(groups, names, colors):
            curve = summary['results'][g]['models'][str(q)]['test']['calibration_curve']
            x = [r['mean_probability'] for r in curve]
            y = [r['observed_frequency'] for r in curve]
            upper = max(upper, max(x), max(y))
            ax.plot(x, y, 'o-', color=color, label=name, markersize=3)
        ax.plot([0, upper*1.05], [0, upper*1.05], '--', color='#777')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed event frequency')
        ax.set_title(f'> {q} mm: raw-probability reliability')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('Spark MLlib L2 Logistic Regression | 5% train (2017–2022), full test (2024)', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
