"""Embed verified final-model ROC/PR coordinates in both existing reports.

Run task2_curves.py on gpushare first. Pass its working curves directory here;
no standalone chart or statistical table is retained in the submission.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = r'\clearpage\section*{图 9：最终选定模型的 ROC 与 PR 曲线}'
CALIBRATION = r'\clearpage\section*{图 10：校准曲线与概率质量}'


def coordinates(points):
    return ' '.join(f'({x:.9g},{y:.9g})' for x, y in points)


def make_section(values):
    for q, value in zip([0, 5, 25], values):
        assert value['threshold_mm'] == q and value['algorithm'] in ['xgb', 'logistic', 'svm']
        assert value['n'] == 6374476 and value['training_rows'] == 34542285
        for kind in ['roc', 'pr']:
            assert abs(value[kind+'_plot_area']-value[kind+'_auc']) <= min(1e-4, value[kind+'_auc']*.002)
    body = [HEADER, r'\begin{center}\begin{tikzpicture}']
    for kind, y in [('roc', 6.4), ('pr', 0.)]:
        for i, value in enumerate(values):
            q = value['threshold_mm']
            name = {'xgb': 'XGB', 'logistic': 'LR', 'svm': 'SVM'}[value['algorithm']]
            label_x = '假正例率（FPR）' if kind == 'roc' else 'Recall'
            label_y = '真正例率（TPR）' if kind == 'roc' else 'Precision'
            legend_position = 'at={(0.97,0.05)},anchor=south east' if kind == 'roc' else 'at={(0.97,0.97)},anchor=north east'
            y_limits = r'ymin=0,ymax=0.15,ytick={0,0.03,0.06,0.09,0.12,0.15},yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},' if kind == 'pr' and q == 25 else 'ymin=0,ymax=1,ytick={0,0.2,0.4,0.6,0.8,1},'
            body.append(r'\begin{axis}[reportaxis,'
                        f'at={{({i*8.3:.1f}cm,{y:.1f}cm)}},anchor=south west,width=7.4cm,height=5.4cm,'
                        f'title={{$R_{{30}}>{q}$ mm: {kind.upper()} ({name})}},xlabel={{{label_x}}},ylabel={{{label_y}}},'
                        r'xmin=0,xmax=1,xtick={0,0.2,0.4,0.6,0.8,1},'+y_limits+
                        r'scaled ticks=false,'
                        'legend style={'+legend_position+r',draw=none,fill=white,font=\scriptsize}]')
            body.append(r'\addplot[teal,thick,no marks] coordinates {'+coordinates(value[kind])+r'};')
            body.append(r'\addlegendentry{'+f'AUC = {value[kind+"_auc"]:.5f}'+r'}')
            reference = [(0., 0.), (1., 1.)] if kind == 'roc' else [(0., value['prevalence']), (1., value['prevalence'])]
            body.append(r'\addplot[orange!85!black,dashed,thick,no marks] coordinates {'+coordinates(reference)+r'};')
            body.append(r'\addlegendentry{随机参考}')
            body.append(r'\end{axis}')
    body.append(r'\end{tikzpicture}\end{center}')
    policies_cn = {'natural': '自然分布', 'balanced': '类别加权', 'smote': 'SMOTE'}
    policies_en = {'natural': 'natural-distribution', 'balanced': 'class-weighted', 'smote': 'SMOTE'}
    algorithms = {'logistic': 'Logistic', 'svm': 'SVM', 'xgb': 'XGBoost'}
    models_cn = '、'.join(policies_cn[v['policy']]+' '+algorithms[v['algorithm']] for v in values)
    models_en = ', '.join(policies_en[v['policy']]+' '+algorithms[v['algorithm']] for v in values)
    scope = r'\textbf{评价对象。}三列依次为'+models_cn+r'，由 2023 年下半年验证集选定。所有模型使用 2017--2022 年全部 34,542,285 条有效训练记录。曲线及 AUC 使用完整 2024 年测试集的 6,374,476 个窗口。ROC 图中虚线为 $y=x$，PR 图中虚线为测试集正例比例。25 mm 的 PR 面板将纵轴放大至 0--0.15，以展示低 Precision 区间。'
    roc = '、'.join(f"{v['roc_auc']:.5f}" for v in values)
    pr = '、'.join(f"{v['pr_auc']:.5f}" for v in values)
    conclusion = r'\textbf{排序表现。}三个目标的 ROC-AUC 分别为 '+roc+'，PR-AUC 分别为 '+pr+'。较高雨量阈值对应的事件更稀少，需结合 PR 曲线评价模型能否同时取得较高 Precision 和 Recall。'
    reviewed = {
        HEADER: r'\clearpage\section*{Figure 9: ROC and PR curves of the final selected models}',
        CALIBRATION: r'\clearpage\section*{Figure 10: reliability curves and probability quality}',
        scope: r'\textbf{Evaluation scope.} The columns show '+models_en+r', selected using 2023 H2 validation. All models fit the complete 34,542,285 eligible training records from 2017--2022. Curves and AUCs use all 6,374,476 eligible 2024 test windows. Dashed references are $y=x$ for ROC and test prevalence for PR. The 25 mm PR panel magnifies precision to 0--0.15.',
        conclusion: r'\textbf{Ranking performance.} ROC-AUCs are '+roc.replace('、', ', ')+', and PR-AUCs are '+pr.replace('、', ', ')+'. Higher rainfall thresholds describe rarer events. PR curves assess whether models achieve high precision and recall together.'}
    body += ['', scope, '', conclusion, '']
    return '\n'.join(body), reviewed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves-dir', type=Path, default=ROOT/'work/task2/model_comparison_full/curves')
    args = parser.parse_args()
    values = [json.loads((args.curves_dir/f'final_gt_{q}mm.json').read_text()) for q in [0, 5, 25]]
    section, reviewed = make_section(values)
    source = ROOT/'reports/Task1_Report_CN.tex'
    text = source.read_text()
    old_calibration = r'\clearpage\section*{图 9：校准曲线与概率质量}'
    if HEADER in text:
        begin = text.index(HEADER); end = text.index(CALIBRATION, begin)
        text = text[:begin]+section+text[end:]
    else:
        assert old_calibration in text
        text = text.replace(old_calibration, section+CALIBRATION, 1)
    translation_path = ROOT/'scripts/report_task2_content.py'
    spec = importlib.util.spec_from_file_location('reviewed_content', translation_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pairs = dict(module.LINE_PAIRS)
    pairs.update(reviewed)
    labels = dict(module.LABELS)
    labels.update({'假正例率（FPR）': 'False positive rate (FPR)',
                   '真正例率（TPR）': 'True positive rate (TPR)', '随机参考': 'Random'})
    apply_function = translation_path.read_text().split('\ndef apply(', 1)[1]
    translation_path.write_text('"""Reviewed bilingual report content; synchronize Chinese and English edits here."""\nimport hashlib\n\n'+
                               'LINE_PAIRS = '+json.dumps(pairs, ensure_ascii=False, indent=2)+'\n\n'+
                               'OVERRIDES = '+json.dumps(module.OVERRIDES, ensure_ascii=False, indent=2)+'\n\n'+
                               'LABELS = '+json.dumps(labels, ensure_ascii=False, indent=2)+'\n\ndef apply('+apply_function)
    source.write_text(text)
    from report_translations import translate
    (ROOT/'reports/Task1_Report.tex').write_text(translate(text))
    print('Updated both existing reports with verified final-model ROC/PR curves.')


if __name__ == '__main__':
    main()
