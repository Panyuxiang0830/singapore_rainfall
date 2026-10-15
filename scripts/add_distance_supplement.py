"""Add the bilingual distance-aware feature experiment to the reviewed reports.

The distance experiment is intentionally a supplement.  It does not replace the
reviewed grid-XGBoost prediction interface or its primary results.  Keeping the
content in this small idempotent script also means a later English-report sync
cannot silently discard the Chinese/English supplemental section.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = '%% DISTANCE_SUPPLEMENT_START'
END = '%% DISTANCE_SUPPLEMENT_END'
INSERT_BEFORE = r'\clearpage\section*{10.'


def _metric_rows():
    """Read the committed frozen 2024 metrics for both model families."""
    rows = []
    for threshold, distance_policy in ((0, 'natural'), (5, 'balanced'), (25, 'balanced')):
        grid = json.loads((ROOT / 'models' / 'grid_xgb' / f'gt{threshold}' / 'test_2024.json').read_text())
        distance = json.loads((ROOT / 'results' / 'distance_full' / 'test' / f'xgb_gt_{threshold}mm.json').read_text())
        rows.append((threshold, grid, distance, distance_policy))
    return rows


def _table(language: str) -> str:
    lines = []
    for threshold, grid, distance, _policy in _metric_rows():
        lines.append(
            f"{threshold} & {grid['pr_auc']:.5f} & {grid['roc_auc']:.5f} & {grid['brier']:.6f} & "
            f"{distance['pr_auc']:.5f} & {distance['roc_auc']:.5f} & {distance['brier']:.6f}\\\\"
        )
    header = (
        '阈值 (mm) & \u7f51格 PR-AUC & \u7f51格 ROC-AUC & \u7f51格 Brier & 距离版 PR-AUC & 距离版 ROC-AUC & 距离版 Brier'
        if language == 'cn' else
        'Threshold (mm) & Grid PR-AUC & Grid ROC-AUC & Grid Brier & Distance PR-AUC & Distance ROC-AUC & Distance Brier'
    )
    return '\n'.join([
        r'\begin{center}\small',
        r'\begin{tabular}{r|rrr|rrr}\toprule',
        header + r'\\\midrule',
        *lines,
        r'\bottomrule\end{tabular}',
        r'\end{center}',
    ])


def _section(language: str) -> str:
    table = _table(language)
    if language == 'cn':
        title = '9.1. 补充实验：距离感知的五邻站特征'
        text = r'''\textbf{目的与边界。} 本节是在既有网格 XGBoost 主线之外增加的补充实验；默认预测入口和主要结论仍采用第 7.1 节的网格模型。它检验「相邻站点距离不同，邻站降雨的参考价值是否也不同」这一问题。实验只使用预测时点及之前可获得的站点坐标和五分钟降雨读数；不使用未来降雨、验证期或测试期信息来构造任一特征。

\textbf{特征与训练。} 树模型的原 73 维向量增加五个最近邻站的 14 个降雨量/可用性变量及其 14 个缺失标记（28 维），再增加五个邻站距离、最近可报告邻站距离、反距离加权雨量、以 5 km 为尺度的指数衰减加权雨量及其 8 个缺失标记（16 维），得到 117 维。距离和邻接关系按在该时点已知的站点坐标建立；填充值只在训练期拟合。实验使用 2017--2022 年的 34,542,285 个合格窗口训练，2023 H1 校准、2023 H2 在 27 个算法/类别策略候选中选择，随后固定模型评估完整的 2024 年。完整距离训练运行于 Google Cloud Compute Engine 的 Ubuntu 24.04 \texttt{e2-custom-20-65536} 虚拟机（20 vCPU、64 GiB RAM、200 GB 持久磁盘），使用 Python 3.11.10、Java 17.0.20.1、Spark 3.5.8 与 XGBoost 3.2.0。

\textbf{2024 年冻结模型结果。} 三个阈值均选为 XGBoost；0 mm 使用自然类别比例，5 mm 与 25 mm 使用类别加权。'''
        conclusion = r'''\textbf{解释。} 距离版在该补充实验的 2024 年结果中三个阈值的 PR-AUC 和 ROC-AUC 都较高，Brier 都较低，说明把距离作为显式信息值得进一步研究。两组模型的特征、参数搜索和模型选择协议并不相同，表中的数值只是补充性对照，不能归因于单一特征，也不构成用测试集重新选择默认模型。2024 年此前已在原实验中查看过；本轮是在固定方案后的后续评估，而非研究者完全未见的封存测试。'''
    else:
        title = '9.1. Supplementary experiment: distance-aware five-neighbour features'
        text = r'''\textbf{Purpose and boundary.} This is a supplementary experiment alongside the reviewed grid-XGBoost main line. The default prediction interface and primary conclusions remain the grid models in Section 7.1. It asks whether neighbouring gauges have different predictive value when their distances differ. Every feature uses only station coordinates and five-minute readings available at the prediction time; no future rainfall, validation rows or test rows enter feature construction.

\textbf{Features and training.} The original 73-dimensional tree vector receives 14 measurements/availability variables for five nearest neighbours and their 14 missingness indicators (28 dimensions), then five neighbour distances, nearest reporting-neighbour distance, inverse-distance-weighted rainfall, exponentially decayed rainfall with a 5 km scale, and their eight missingness indicators (16 dimensions), giving 117 dimensions. Distances and graph edges use station coordinates known at the time; imputation values fit on training rows only. The experiment trains on 34,542,285 eligible 2017--2022 windows, calibrates on 2023 H1, selects among 27 algorithm/class-policy candidates on 2023 H2, and evaluates frozen models on all of 2024. Full distance training ran on a Google Cloud Compute Engine Ubuntu 24.04 \texttt{e2-custom-20-65536} VM (20 vCPU, 64 GiB RAM and 200 GB persistent disk), using Python 3.11.10, Java 17.0.20.1, Spark 3.5.8 and XGBoost 3.2.0.

\textbf{Frozen-model results for 2024.} XGBoost is selected for all three thresholds: natural class proportions at 0 mm and class weighting at 5 and 25 mm.'''
        conclusion = r'''\textbf{Interpretation.} In this supplementary experiment, the distance version has higher PR-AUC and ROC-AUC and lower Brier score at every threshold in 2024, so explicit distance information merits further study. The model families differ in features, parameter search and model-selection protocol; this table is a supplementary descriptive comparison, not evidence attributable to one feature or a test-set-driven replacement of the default model. The 2024 period had already been inspected in the original experiment, so this is follow-up evaluation after the specification was frozen, rather than a fully unseen sealed test.'''
    return '\n'.join([
        START,
        r'\clearpage',
        rf'\section*{{{title}}}',
        '', text, '', table, '', conclusion, END, '',
    ])


def strip_distance_supplement(content: str) -> str:
    """Return a report without a previous generated supplemental block."""
    start = content.find(START)
    if start < 0:
        return content
    end = content.find(END, start)
    if end < 0:
        raise ValueError('Distance supplement has a start marker but no end marker')
    return content[:start] + content[end + len(END):]


def upsert(path: Path, language: str) -> None:
    content = strip_distance_supplement(path.read_text(encoding='utf-8'))
    point = content.find(INSERT_BEFORE)
    if point < 0:
        raise ValueError(f'Cannot locate Section 10 in {path}')
    path.write_text(content[:point] + _section(language) + content[point:], encoding='utf-8')


def main() -> None:
    upsert(ROOT / 'reports' / 'Task1_Report_CN.tex', 'cn')
    upsert(ROOT / 'reports' / 'Task1_Report.tex', 'en')
    print('DISTANCE_SUPPLEMENT_WRITTEN: Chinese and English report sources')


if __name__ == '__main__':
    main()
