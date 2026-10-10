"""Plot the benchmark from working metrics; keep previews outside deliverables."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ['logistic', 'svm', 'xgb']
NAMES = ['L2 Logistic', 'Linear SVM', 'Spark XGBoost']
POLICIES = ['natural', 'balanced', 'smote']
COLORS = ['#767f8c', '#249a88', '#d07b38']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT/'work/task2/model_comparison_full/summary.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'work/task2/model_comparison_full/preview')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    assert data['preparation']['train_rows'] == 34542285
    assert data['preparation']['training_policy'] == 'all_eligible_2017_2022'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = ['0', '5', '25']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for q, ax in zip(thresholds, axes):
        for k, policy in enumerate(POLICIES):
            values = [data['candidates'][a][policy]['models'][q]['validation']['pr_auc'] for a in ALGORITHMS]
            bars = ax.bar(np.arange(len(ALGORITHMS))+(k-1)*.24, values, width=.24,
                          label=['Natural', 'Class weights', 'Constrained SMOTE'][k], color=COLORS[k])
            ax.bar_label(bars, fmt='%.4f' if q == '25' else '%.3f', fontsize=7, rotation=90, padding=3)
        rate = data['candidates']['logistic']['natural']['models'][q]['validation']['prevalence']
        ax.axhline(rate, ls='--', color='#bd4444', label='Event prevalence')
        ax.set_xticks(np.arange(len(ALGORITHMS)), NAMES, rotation=12, fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1]*1.17)
        ax.set_title(f'Next 30 min > {q} mm')
        ax.set_ylabel('PR-AUC (higher is better)')
        ax.grid(axis='y', alpha=.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=4, fontsize=9)
    fig.suptitle('2023 Jul-Dec validation | identical full features; training policies compared', fontsize=12)
    fig.savefig(args.output_dir/'validation_policies.png', dpi=170); plt.close(fig)
    if args.validation_only:
        print(args.output_dir/'validation_policies.png')
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    for j, q in enumerate(thresholds):
        test = [data['test'][a]['test'][q] for a in ALGORITHMS]
        ax = axes[0, j]
        bars = ax.bar(NAMES, [r['pr_auc'] for r in test], color=COLORS)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.axhline(test[0]['prevalence'], color='#bd4444', ls='--', label=f"Event prevalence: {test[0]['prevalence']:.5f}")
        ax.set_ylim(0, ax.get_ylim()[1]*1.17)
        ax.set_title(f'> {q} mm: test PR-AUC')
        ax.tick_params(axis='x', rotation=12, labelsize=9)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        brier = [r['brier'] for r in test]
        bars = ax.bar(NAMES, brier, color=COLORS)
        ax.bar_label(bars, fmt='%.7f' if q == '25' else '%.5f', padding=3, fontsize=8)
        # Constant training-rate probability benchmark, evaluated on this test block.
        n = data['preparation']['train_rows']
        p = data['preparation']['train_positives'][q]/n
        prevalence = test[0]['prevalence']
        baseline = prevalence*(1-p)**2+(1-prevalence)*p**2
        digits = 7 if q == '25' else 5
        ax.axhline(baseline, ls='--', color='#bd4444', label=f'Constant train-rate Brier: {baseline:.{digits}f}')
        ax.set_ylim(0, max(max(brier), baseline)*1.22)
        ax.set_title(f'> {q} mm: calibrated test Brier (lower is better)')
        ax.tick_params(axis='x', rotation=12, labelsize=9)
        ax.legend(fontsize=8)
    fig.suptitle('Full 2024 test | policy chosen on 2023 Jul-Dec; calibration fitted on Jan-Jun', fontsize=12)
    fig.savefig(args.output_dir/'test_comparison.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for q, ax in zip(thresholds, axes):
        upper = .001
        for a, name, color in zip(ALGORITHMS, NAMES, COLORS):
            curve = data['test'][a]['test'][q]['calibration_curve']
            x = [r['mean_probability'] for r in curve]
            y = [r['observed_frequency'] for r in curve]
            upper = max(upper, max(x), max(y))
            policy = data['test'][a]['test'][q]['selected_policy']
            ax.plot(x, y, 'o-', color=color, label=f'{name}: {policy}', markersize=3)
        axis_upper = min(1, upper*1.05)
        ax.plot([0, axis_upper], [0, axis_upper], '--', color='#777', label='Perfect calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed event frequency')
        ax.set_title(f'> {q} mm: test reliability')
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    fig.suptitle('Monotone Platt-calibrated probabilities | approximate equal-mass reliability bins', fontsize=12)
    fig.savefig(args.output_dir/'test_calibration.png', dpi=170); plt.close(fig)
    print(args.output_dir)


if __name__ == '__main__':
    main()
