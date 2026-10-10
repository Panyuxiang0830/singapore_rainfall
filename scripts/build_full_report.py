"""Update the existing bilingual report from verified full-training results.

Before results exist, --methods-only produces an explicitly incomplete methods
draft and removes obsolete prediction results. No scores are relabelled.
The Task 1 source and inline figures are preserved.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

from build_task2_curve_report import make_section

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ['logistic', 'svm', 'xgb']
POLICIES = ['natural', 'balanced', 'smote']
NAMES = {'logistic': 'Logistic', 'svm': 'SVM', 'xgb': 'XGBoost'}
POLICY_CN = {'natural': '自然分布', 'balanced': '类别加权', 'smote': 'SMOTE'}
POLICY_EN = {'natural': 'Natural', 'balanced': 'Balanced', 'smote': 'SMOTE'}
QS = [0, 5, 25]


class Writer:
    def __init__(self):
        self.lines = []
        self.pairs = {}

    def line(self, cn='', en=None):
        self.lines.append(cn)
        if en is not None:
            self.pairs[cn] = en

    def section(self, cn, en):
        self.line(r'\clearpage\section*{'+cn+'}', r'\clearpage\section*{'+en+'}')

    def paragraph(self, cn_title, cn, en_title, en):
        self.line(r'\textbf{'+cn_title+'。}'+cn, r'\textbf{'+en_title+'.} '+en)
        self.line()

    def table(self, columns, headers_cn, headers_en, rows):
        self.line(r'\begin{center}\begin{tabular}{'+columns+r'}\toprule')
        self.line(' & '.join(headers_cn)+r'\\\midrule', ' & '.join(headers_en)+r'\\\midrule')
        for cn, en in rows:
            self.line(' & '.join(cn)+r'\\', ' & '.join(en)+r'\\')
        self.line(r'\bottomrule\end{tabular}\end{center}')
        self.line()


def f(value, digits=5):
    return f'{value:.{digits}f}'


def bar_panels(w, values, legends=None, references=None):
    w.line(r'\begin{center}\begin{tikzpicture}')
    for i, q in enumerate(QS):
        extra = ',ybar,bar width=7pt,legend style={at={(0.5,-0.18)},anchor=north,legend columns=3}' if legends else ''
        w.line(r'\begin{axis}[reportaxis,'+f'at={{({i*8.3:.1f}cm,0)}},anchor=south west,width=7.4cm,height=8cm,title={{$R_{{30}}>{q}$ mm}},'+r'ylabel={PR-AUC},xtick={0,1,2},xticklabels={LR,SVM,XGB},ymin=0'+extra+']')
        series = values[str(q)]
        colours = ['teal', 'orange!85!black', 'blue!65!black']
        for j, ys in enumerate(series):
            coord = ' '.join(f'({k},{v:.9g})' for k, v in enumerate(ys))
            w.line(r'\addplot[ybar,bar width='+('7pt' if legends else '20pt')+',fill='+colours[j]+r',draw=none] coordinates {'+coord+'};')
        if references:
            v = references[str(q)]
            w.line(r'\addplot[orange!85!black,dashed,thick] coordinates {'+f'(-0.5,{v:.9g}) (2.5,{v:.9g})'+'};')
        if legends:
            w.line(r'\legend{自然分布,类别加权,SMOTE}', r'\legend{Natural,Balanced,SMOTE}')
        w.line(r'\end{axis}')
    w.line(r'\end{tikzpicture}\end{center}')
    w.line()


def methods(w, prepared, baseline):
    assert prepared['train_rows'] == 34542285
    assert prepared['config']['training_policy'] == 'all_eligible_2017_2022'
    w.section('4. Task 2：预测目标与时间切分', '4. Task 2: target and chronological splits')
    w.paragraph('预测目标', r'对于站点 $s$、时刻 $t$ 和雨量阈值 $q$，估计未来 30 分钟累计雨量严格超过 $q$ 的概率。设置 $q\in\{0,5,25\}$ mm，每个目标分别训练分类器；同一站点时刻保留一份特征与三个标签列。',
                'Prediction target', r'For station $s$, time $t$ and rainfall threshold $q$, estimate the probability that rainfall in the next 30 minutes strictly exceeds $q$. Use $q\in\{0,5,25\}$ mm and one classifier per target, with one feature row and three label columns per station/time.')
    w.line(r'\[R_{s,t}^{30}=\sum_{k=1}^{6}r_{s,t+5k\,\mathrm{min}},\qquad y_{s,t}^{(q)}=\mathbf{1}(R_{s,t}^{30}>q),\qquad \hat p_q=P(y^{(q)}=1\mid x_{s,t}).\]')
    w.paragraph('标签构造', r'将时间戳作为五分钟区间的终点，累计 $(t,t+30\mathrm{min}]$ 内六个有效区间的雨量，保留九位小数后与阈值比较。当前区间不计入目标；任一未来区间缺测时不构造标签。累计雨量等于阈值时标签为 0。',
                'Label construction', r'Treat timestamps as five-minute interval ends. Sum six valid intervals in $(t,t+30\mathrm{min}]$ and round to nine decimal places before comparison. Exclude the current interval and reject windows missing any future interval. Equality with the threshold is negative.')
    pos = prepared['train_positives']
    data = [(['2017--2022：全部训练记录', '34,542,285', *[f"{int(pos[str(q)]):,}" for q in QS]], ['2017--2022: all training records', '34,542,285', *[f"{int(pos[str(q)]):,}" for q in QS]]),
            (['2023 上半年：概率校准', '3,325,499', '251,794', '30,699', '1,532'], ['2023 H1: calibration', '3,325,499', '251,794', '30,699', '1,532']),
            (['2023 下半年：模型选择', '3,393,373', '197,358', '28,000', '2,487'], ['2023 H2: selection', '3,393,373', '197,358', '28,000', '2,487']),
            (['2024：测试', '6,374,476', '405,430', '53,193', '3,517'], ['2024: test', '6,374,476', '405,430', '53,193', '3,517'])]
    w.table('lrrrr', ['用途', '有效行数', '$>0$ 正例', '$>5$ 正例', '$>25$ 正例'], ['Purpose', 'Rows', '$>0$ positives', '$>5$ positives', '$>25$ positives'], data)
    w.paragraph('训练与验证', '所有模型及预处理均使用 2017--2022 年全部 34,542,285 条有效训练记录。2023 年上半年拟合概率校准函数，下半年选择模型及概率判别阈值；2024 年用于测试。标签窗口不得跨上述时间边界。校准、选择和测试保留自然类别分布。',
                'Training and validation', 'All classifiers and preprocessing use all 34,542,285 eligible 2017--2022 training records. Fit calibration on 2023 H1, select models and decision cutoffs on H2, and test on 2024. Label windows cannot cross these boundaries. Calibration, selection and test preserve natural class proportions.')
    w.section('5. 从 Task 1 到特征工程', '5. From Task 1 to feature engineering')
    w.line(r'\begingroup\small')
    w.table('p{5.2cm}p{9.0cm}p{9.0cm}', ['Task 1 依据', '构造方式', '实际输入'], ['Task 1 finding', 'Construction', 'Inputs'], [
        (['下午高峰随月份变化', '小时、月份周期编码与交互', '8 个日历特征；不加入年份 ID'], ['Afternoon peak varies by month', 'Cyclic hour/month and interactions', '8 calendar features; no year ID']),
        (['站点差异与空间相关', '站点 ID、经纬度、三个最近邻站', '邻站雨量均值、最大值、有雨比例'], ['Station differences and spatial correlation', 'Station ID, coordinates, three closest neighbours', 'Neighbour rain means, maxima and wet fractions']),
        (['短时降雨持续性', '最新读数及 5/10/15/30/60 分钟滞后', '15/30/60 分钟累计、变化和雨段长度'], ['Short-term persistence', 'Latest reading and 5/10/15/30/60-minute lags', '15/30/60-minute totals, changes and wet-spell length'])])
    w.paragraph('历史与邻站特征', '只使用预测时已可获得的观测，观测时刻和保留的更新时间均不晚于 $t$。从过去 90 分钟检索历史；最近读数超过 30 分钟时标记不可用。邻站由当时已知坐标确定，聚合三个最近合格邻站的雨量、有效观测数、读数年龄及距离。Spark 窗口函数和连接处理全量记录。',
                'History and neighbours', 'Use only observations available by prediction time: both observation and retained update timestamps must be no later than $t$. Search the past 90 minutes; mark the latest reading unavailable after 30 minutes. Use coordinates known by $t$ to identify three closest eligible neighbours and aggregate rainfall, available counts, ages and distances. Spark windows and joins process all records.')
    w.paragraph('时间编码', '令 $h$ 为当地小时（包含分钟的小数部分），$m$ 为月份。小时和月份使用下列周期编码，并将两组编码的四个乘积作为交互项，共形成八个特征。这样 23:00 与 00:00、12 月与 1 月在特征空间中保持接近。',
                'Cyclic time encoding', 'Let $h$ be local hour including fractional minutes and $m$ the month. Use the cyclic encodings below and their four pairwise products, giving eight features. Adjacent 23:00/00:00 and December/January stay close in feature space.')
    w.line(r'\[h_{\sin}=\sin(2\pi h/24),\quad h_{\cos}=\cos(2\pi h/24),\qquad m_{\sin}=\sin[2\pi(m-1)/12],\quad m_{\cos}=\cos[2\pi(m-1)/12].\]')
    w.paragraph('输入处理', '缺失值填补、缺失标志和站点类别词表仅基于完整训练期拟合。线性模型对站点进行 one-hot 编码，树模型使用类别索引；常量特征排除。Logistic 与 SVM 使用特征标准化。通过九个 Logistic 模型比较日历与站点、加入本站历史、再加入邻站历史三组特征。',
                'Input processing', 'Fit imputation, missingness indicators and station vocabulary on the complete training period only. Use one-hot station encoding for linear models and a categorical index for trees; exclude constant features. Logistic and SVM standardize features. Nine logistic models compare calendar/station, added own history, and added neighbour history.')
    if baseline:
        config = prepared['config']
        capped = sum(m['iteration_cap_reached'] for g in baseline['results'].values() for m in g['models'].values())
        w.paragraph('特征组比较设置', f"九个特征组比较模型使用 L2 正则参数 {config['reg_param']}，最多迭代 {config['max_iter']} 次，收敛容差为 $10^{{-6}}$。其中 {capped} 个模型达到设定的迭代上限。正式模型比较中的 Logistic 与 SVM 最多迭代 100 次。", 'Feature comparison settings', f"The nine ablation classifiers use L2 parameter {config['reg_param']}, at most {config['max_iter']} iterations and tolerance $10^{{-6}}$. {capped} reach this iteration cap. Logistic and SVM in the main algorithm comparison allow 100 iterations.")
        linear_dimension = baseline['results']['neighbors']['models']['0']['vector_dimension']
        tree_dimension = len(prepared['active_numeric'])+len(prepared['missing_indicators'])+1
        w.paragraph('特征维度', f"完整输入包含 {len(prepared['active_numeric'])} 个数值特征和 {len(prepared['missing_indicators'])} 个缺失标志，加上站点编码后，线性模型为 {linear_dimension} 维，树模型为 {tree_dimension} 维。", 'Feature dimensions', f"The complete input has {len(prepared['active_numeric'])} numeric features and {len(prepared['missing_indicators'])} missingness indicators. With station encoding, linear models have {linear_dimension} dimensions and tree models {tree_dimension}.")
        rows=[]
        for g, cn, en in [('calendar_station', '日历与站点', 'Calendar/station'), ('own_history', '加本站历史', 'Add own history'), ('neighbors', '再加邻站', 'Add neighbours')]:
            vals=[f(baseline['results'][g]['models'][str(q)]['test']['pr_auc']) for q in QS]
            rows.append(([cn, *vals], [en, *vals]))
        w.table('lrrr', ['L2 Logistic 特征组', '测试 PR-AUC：0 mm', '5 mm', '25 mm'], ['L2 logistic features', 'Test PR-AUC: 0 mm', '5 mm', '25 mm'], rows)
        gains = []
        for q in QS:
            own = baseline['results']['own_history']['models'][str(q)]['test']['pr_auc']
            near = baseline['results']['neighbors']['models'][str(q)]['test']['pr_auc']
            gains.append(near-own)
        changes_cn = '、'.join(f'{v:+.5f}' for v in gains)
        changes_en = ', '.join(f'{v:+.5f}' for v in gains)
        w.paragraph('邻站信息的增量', '相对于本站历史模型，加入邻站历史后三个目标的测试 PR-AUC 依次变化 '+changes_cn+'。结果与 Task 1 的空间相关性分析一致。',
                    'Neighbour contribution', 'Adding neighbour history changes test PR-AUC by '+changes_en+' relative to own-history models, consistent with Task 1 spatial correlations.')
    w.line(r'\endgroup')
    w.section('6. 模型、不平衡处理与概率校准', '6. Models, imbalance and calibration')
    rows=[(['L2 Logistic（MLlib）', r'$\lambda=0.001$，标准化，最大迭代 100，容差 $10^{-6}$'], ['L2 logistic (MLlib)', r'$\lambda=0.001$, standardization, 100 iterations, tolerance $10^{-6}$']),
          (['Linear SVM（MLlib）', r'$\lambda=0.001$，标准化，最大迭代 100，容差 $10^{-6}$'], ['Linear SVM (MLlib)', r'$\lambda=0.001$, standardization, 100 iterations, tolerance $10^{-6}$']),
          (['XGBoost（Spark 接口）', r'80 棵树，深度 4，学习率 0.1，64 bins，子节点权重 20；$L_2=1$'], ['XGBoost (Spark interface)', r'80 trees, depth 4, learning rate 0.1, 64 bins, child weight 20; $L_2=1$'])]
    w.table('p{6cm}p{17cm}', ['模型', '参数'], ['Model', 'Parameters'], rows)
    w.paragraph('训练策略', r'比较自然分布、类别加权和受约束 SMOTE 三种策略。类别权重为 $w_+=N/(2P)$、$w_-=N/[2(N-P)]$。SMOTE 目标正例占比为 20\%，正例总数最多为原始正例的 10 倍。每种策略均保留全部真实训练记录；合成观测仅添加到训练集。',
                'Training policies', r'Compare natural distribution, balanced weighting and constrained SMOTE. Weights are $w_+=N/(2P)$ and $w_-=N/[2(N-P)]$. SMOTE targets 20\% positives and caps their total at ten times the original count. Every policy retains all real training observations; synthetic rows enter training only.')
    w.paragraph('SMOTE 构造', '所有训练正例均可作为合成来源，按站点依次处理以控制内存。只在同站点、相同缺失模式、离散状态及雨量零值模式下检索最多五个近邻。对 16 个连续雨量特征用同一个系数插值，时间编码和其他离散信息保持原值；无合适近邻时复制原样本。',
                'SMOTE construction', 'All training positives are available as donors, processed sequentially by station to bound memory. Search up to five neighbours with identical station, missingness, discrete profile and rain-zero pattern. Interpolate 16 continuous rain features with one coefficient, preserving calendar and discrete information; copy isolated donors.')
    w.paragraph('概率校准', r'利用 2023 年上半年观测拟合 Platt 函数 $\hat p=1/[1+\exp(-(az+b))]$，其中 $z$ 为基础模型分数，约束 $a\geq0$。该函数把 SVM 分数转换成概率，并调整类别加权或重采样后的概率尺度。若某组预测概率约为 10\%，理想情况下该组实际正例比例也应接近 10\%。',
                'Probability calibration', r'Fit Platt calibration $\hat p=1/[1+\exp(-(az+b))]$ on 2023 H1, where $z$ is the base score and $a\geq0$. This converts SVM scores to probabilities and adjusts weighting/resampling probability scales. A group predicted near 10\% should ideally contain about 10\% actual positives.')
    w.section('7. 评价方法与验证集选择', '7. Evaluation and validation selection')
    w.paragraph('评价指标', r'ROC-AUC 评价正负例排序，随机排序的参考值为 0.5。PR-AUC 综合 Precision 与 Recall，适合正例较少的目标；随机排序的 Precision 参考水平约为正例比例 $\pi$。两种面积均由 Spark 精确积分计算。Brier score 为预测概率与标签的均方差，越小越好。',
                'Evaluation metrics', r'ROC-AUC measures ranking, with random-ranking reference 0.5. PR-AUC combines precision and recall and is useful for rare positives; random-ranking precision is approximately prevalence $\pi$. Spark integrates both areas exactly. Brier score is mean squared probability error; lower is better.')
    w.paragraph('模型选择', '在 2023 年下半年比较各算法和雨量阈值的训练策略，主要按 PR-AUC 选择，Brier score 作为并列时的依据。概率判别阈值在该验证集上按 F1 选择，用于计算测试 Precision、Recall、F1 和 Accuracy；它与定义标签的雨量阈值不同。测试结果不参与模型选择。',
                'Model selection', 'Compare policies for each algorithm/threshold on 2023 H2, select primarily by PR-AUC and break ties with Brier. Choose decision cutoffs by validation F1 for test precision, recall, F1 and accuracy; these differ from rainfall thresholds defining labels. Test results do not select models.')


def results(w, summary, benchmark):
    assert summary['status']=='COMPLETE' and summary['preparation']['train_rows']==34542285
    selected=summary['selection']; prep=summary['preparation']
    assert selected['training_data']['rows']==34542285
    rows=[]
    values={str(q):[] for q in QS}
    for a in ALGORITHMS:
        for p in POLICIES:
            ys=[summary['candidates'][a][p]['models'][str(q)]['validation']['pr_auc'] for q in QS]
            rows.append(([NAMES[a]+' / '+POLICY_CN[p], *[f(v) for v in ys]], [NAMES[a]+' / '+POLICY_EN[p], *[f(v) for v in ys]]))
    w.table('lrrr', ['算法 / 策略', '选择 PR-AUC：0 mm', '5 mm', '25 mm'], ['Algorithm / policy', 'Selection PR-AUC: 0 mm', '5 mm', '25 mm'], rows)
    chosen=[]
    for q in QS:
        a=selected['selected_algorithm_by_threshold'][str(q)];p=selected['selected_policy_by_algorithm_and_threshold'][a][str(q)]
        chosen.append((q,a,p))
    cn='；'.join(f'{q} mm：{POLICY_CN[p]} {NAMES[a]}' for q,a,p in chosen)
    en='; '.join(f'{q} mm: {POLICY_EN[p]} {NAMES[a]}' for q,a,p in chosen)
    w.paragraph('选定模型', '验证集选出的最终模型为 '+cn+'。各算法在三个目标上的最优策略，共 9 个模型，均在完整测试集上评价。', 'Selected models', 'Validation selects '+en+'. Evaluate all 9 algorithm/threshold policy winners on the full test set.')
    w.section('图 7：训练策略的验证集比较', 'Figure 7: training policies on validation')
    for q in QS:
        values[str(q)]=[[summary['candidates'][a][p]['models'][str(q)]['validation']['pr_auc'] for a in ALGORITHMS] for p in POLICIES]
    bar_panels(w,values,legends=True)
    improved = {}
    for policy in ['balanced', 'smote']:
        improved[policy] = sum(summary['candidates'][a][policy]['models'][str(q)]['validation']['pr_auc'] > summary['candidates'][a]['natural']['models'][str(q)]['validation']['pr_auc'] for a in ALGORITHMS for q in QS)
    w.paragraph('验证比较', f"相对于自然分布训练，类别加权在 9 个算法与目标组合中的 {improved['balanced']} 个组合提高 PR-AUC，SMOTE 在 {improved['smote']} 个组合提高 PR-AUC。因此，不平衡处理的收益需要根据算法与预测目标分别评价。所有比较采用相同完整训练期与相同验证记录。", 'Validation comparison', f"Relative to natural-distribution training, weighting improves PR-AUC in {improved['balanced']} of 9 algorithm/target combinations and SMOTE in {improved['smote']}. Evaluate imbalance treatment by algorithm and target. Every comparison uses the same complete training period and identical validation observations.")
    w.section('8. 2024 年测试结果', '8. Test results in 2024')
    rows=[]
    for q in QS:
        for a in ALGORITHMS:
            t=summary['test'][a]['test'][str(q)];p=t['selected_policy']
            nums=[f(t['roc_auc']),f(t['pr_auc']),f(t['brier'],7)]
            rows.append(([str(q),NAMES[a],POLICY_CN[p],*nums],[str(q),NAMES[a],POLICY_EN[p],*nums]))
    w.table('rllrrr', ['阈值 mm','算法','验证选定策略','ROC-AUC','PR-AUC','Brier'], ['Threshold mm','Algorithm','Selected policy','ROC-AUC','PR-AUC','Brier'], rows)
    w.paragraph('测试样本', r'每个目标均评价 6,374,476 个真实窗口，三个目标的正例比例为 6.3602\%、0.83447\% 和 0.055173\%。所有概率指标基于校准后的预测。高雨量目标的正例较少，Accuracy 需结合 PR-AUC 和概率评价解释。', 'Test observations', r'Each target uses 6,374,476 real windows; prevalences are 6.3602\%, 0.83447\% and 0.055173\%. All probability metrics use calibrated predictions. High-threshold events are rare, so interpret accuracy together with PR-AUC and probability quality.')
    rows=[]
    for q,a,p in chosen:
        train=summary['train_final'][str(q)];val=summary['candidates'][a][p]['models'][str(q)]['validation'];test=summary['test'][a]['test'][str(q)]
        assert train['n']==34542285
        for name_cn,name_en,m in [('训练','Training',train),('验证','Validation',val),('测试','Test',test)]:
            nums=[f(m['roc_auc']),f(m['pr_auc']),f(m['brier'],7)]
            rows.append(([str(q),NAMES[a],name_cn,*nums],[str(q),NAMES[a],name_en,*nums]))
    w.table('rllrrr', ['阈值 mm','最终模型','数据集','ROC-AUC','PR-AUC','Brier'], ['Threshold mm','Final model','Data','ROC-AUC','PR-AUC','Brier'], rows)
    w.paragraph('训练与测试比较', '训练指标使用全部真实训练记录，不包含 SMOTE 合成观测；验证与测试也使用全部合格记录。各时期正例比例不同，PR-AUC 的跨时期差异同时受到模型泛化与事件发生频率影响。', 'Training and test comparison', 'Training diagnostics use all real training observations, excluding SMOTE rows. Validation and test also use all eligible records. Cross-period PR-AUC differences reflect both generalization and changes in event prevalence.')
    w.section('图 8：测试集 PR-AUC 比较', 'Figure 8: test PR-AUC comparison')
    values={str(q):[[summary['test'][a]['test'][str(q)]['pr_auc'] for a in ALGORITHMS]] for q in QS}
    references={str(q):summary['test']['logistic']['test'][str(q)]['prevalence'] for q in QS}
    bar_panels(w,values,references=references)
    w.paragraph('测试比较', '橙色虚线为测试正例比例，即随机排序的近似参考水平。最终模型依据验证集选择；测试集用于评价既定选择，不根据测试成绩重新更换模型。', 'Test comparison', 'Orange dashed lines show test prevalence, the approximate random-ranking reference. Final models are chosen on validation; test evaluates those choices without replacing them based on test scores.')
    curve_values=[json.loads((benchmark/'curves'/f'final_gt_{q}mm.json').read_text()) for q in QS]
    curve_section,reviewed=make_section(curve_values)
    w.lines.extend(curve_section.splitlines());w.pairs.update(reviewed)
    w.section('图 10：校准曲线与概率质量', 'Figure 10: reliability and probability quality')
    w.line(r'\begin{center}\begin{tikzpicture}')
    high=[]
    for i,(q,a,p) in enumerate(chosen):
        t=summary['test'][a]['test'][str(q)];points=t['calibration_curve']
        largest=max(max(v['mean_probability'],v['observed_frequency'])*100 for v in points)
        maximum=100 if q==0 else math.ceil(largest*1.1/(5 if largest>5 else .1))*(5 if largest>5 else .1)
        coords=' '.join(f"({v['mean_probability']*100:.9g},{v['observed_frequency']*100:.9g})" for v in points)
        axis=r'\begin{axis}[reportaxis,'+f'at={{({i*8.3:.1f}cm,0)}},anchor=south west,width=7.4cm,height=8cm,title={{$R_{{30}}>{q}$ mm}},'+r'xlabel={平均预测概率（\%）},ylabel={实际发生比例（\%）},xmin=0,ymin=0,'+f'xmax={maximum:.9g},ymax={maximum:.9g},'+r'scaled ticks=false,legend style={at={(0.5,-0.24)},anchor=north,legend columns=1}]'
        w.line(axis,axis.replace('平均预测概率','Mean predicted probability').replace('实际发生比例','Observed frequency'))
        w.line(r'\addplot[gray,dashed,thick] coordinates {'+f'(0,0) ({maximum:.9g},{maximum:.9g})'+'};')
        w.line(r'\addplot[teal,mark=*,mark size=1.6pt,thick] coordinates {'+coords+'};')
        w.line(r'\legend{理想校准,'+NAMES[a]+'}',r'\legend{Ideal calibration,'+NAMES[a]+'}')
        w.line(r'\end{axis}')
        high.append(max(points,key=lambda x:x['mean_probability']))
    w.line(r'\end{tikzpicture}\end{center}\vspace{10mm}')
    w.paragraph('概率质量', r'将相近的预测概率分组，以组内平均预测概率为横轴，实际正例比例为纵轴。灰色虚线表示理想关系 $y=x$；曲线高于虚线表示风险低估，低于虚线表示风险高估。图中坐标均为百分比，三个目标的显示范围不同。', 'Probability quality', r'Group similar forecasts and plot mean probability against observed positive frequency. The dashed ideal is $y=x$: points above it underestimate risk; points below it overestimate risk. Coordinates are percentages, with different displayed ranges across targets.')
    last=high[2]
    w.paragraph('稀有事件校准', f"25 mm 目标的最高平均预测概率组包含 {last['n']:,} 个窗口，平均预测概率为 {last['mean_probability']*100:.3f}\\%，实际超越比例为 {last['observed_frequency']*100:.3f}\\%。该比较揭示模型概率与实际发生频率的差异，与排序指标反映的能力不同。", 'Rare-event calibration', f"For 25 mm, the highest-mean-probability group contains {last['n']:,} windows: mean forecast {last['mean_probability']*100:.3f}\\% versus observed frequency {last['observed_frequency']*100:.3f}\\%. This measures probability reliability rather than ranking alone.")
    w.section('9. 分类指标与补充分析', '9. Classification metrics and further analysis')
    rows=[]
    for q,a,p in chosen:
        t=summary['test'][a]['test'][str(q)]
        row=[str(q),NAMES[a],f(t['probability_cutoff'],4),f(t['precision']),f(t['recall']),f(t['f1']),f(t['accuracy'])]
        rows.append((row,row))
    w.table('rlrrrrr',['阈值 mm','模型','判别阈值','Precision','Recall','F1','Accuracy'],['Threshold mm','Model','Cutoff','Precision','Recall','F1','Accuracy'],rows)
    w.paragraph('分类评价', '表中判别阈值由验证集确定。高阈值事件中负例占比很大，因此高 Accuracy 不代表有效识别了强降雨；主要比较 ROC-AUC、PR-AUC，并以 Brier score 和校准曲线评价概率。', 'Classification evaluation', 'Cutoffs are validation-selected. Negative cases dominate high-threshold targets, so high accuracy alone does not establish effective heavy-rain detection. Compare ROC-AUC and PR-AUC primarily, with Brier and reliability for probabilities.')
    rows=[]
    for q in QS:
        r=summary['smote'][str(q)]
        row=[str(q),f"{r['original_positives']:,}",f"{r['added_rows']:,}",f(r['augmented_positive_fraction']*100,3),f"{r['numerically_changed_rows']:,}",f"{r['isolated_profile_duplicate_rows']:,}"]
        rows.append((row,row))
    w.table('rrrrrr',['阈值 mm','真实正例','合成行数',r'增广正例（\%）','雨量变化行数','孤立状态复制'],['Threshold mm','Real positives','Added rows',r'Augmented positives (\%)','Changed rain rows','Isolated copies'],rows)
    w.paragraph('空间与时间依赖', '相邻站点可能受同一降雨过程影响，半小时窗口也存在重叠。后续可按日期或降雨事件分组评价，进一步比较模型在不同站点与事件中的稳定性。三个阈值独立训练，后续可研究跨阈值概率的单调约束。', 'Spatial and temporal dependence', 'Neighbouring stations can share a rainfall event and half-hour windows overlap. Event/day-grouped evaluation can assess stability across stations and events. Independent threshold classifiers also motivate future monotone cross-threshold probability constraints.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'work/task2/logistic_full')
    parser.add_argument('--benchmark-dir',type=Path,default=ROOT/'work/task2/model_comparison_full')
    parser.add_argument('--methods-only',action='store_true')
    args=parser.parse_args()
    prepared=json.loads((args.baseline_dir/'preparation.json').read_text())
    baseline=json.loads((args.baseline_dir/'summary.json').read_text()) if (args.baseline_dir/'summary.json').exists() else None
    source=ROOT/'reports/Task1_Report_CN.tex';old=source.read_text()
    prefix=old.split(r'\clearpage\section*{4.',1)[0]
    suffix=r'\clearpage\section*{10.'+old.split(r'\clearpage\section*{10.',1)[1]
    w=Writer();methods(w,prepared,baseline)
    if args.methods_only:
        w.paragraph('实验说明','预测实验正在以完整训练数据重新估计，本稿不列示尚未完成的模型评价结果。','Experiment note','Prediction experiments are being estimated on the complete training data; this draft does not report unfinished model evaluations.')
    else:
        summary=json.loads((args.benchmark_dir/'summary.json').read_text())
        results(w,summary,args.benchmark_dir)
    text=prefix+'\n'.join(w.lines)+'\n'+suffix
    old_runtime=next(line for line in prefix.splitlines() if line.startswith(r'\textbf{Spark 配置与分工。}'))
    runtime=r'\textbf{Spark 配置与分工。}Task 1 与初始特征组比较采用单机双线程模式 \texttt{local[2]}。完整特征的模型比较与评价采用 \texttt{local[20]}，shuffle partitions 为 80，default parallelism 为 20。XGBoost 使用两个工作进程，每个任务配置 10 个 CPU。不使用 GPU，以低优先级串行执行各项实验。Spark DataFrame/SQL 与窗口函数用于全量清洗、时间聚合、标签和特征构造；模型训练与评价使用 Spark ML 管线。'
    text=text.replace(old_runtime,runtime)
    w.pairs[runtime]=r'\textbf{Spark configuration and responsibilities.} Task 1 and the initial feature-group comparisons use single-machine \texttt{local[2]}. Full-feature model comparison and evaluation use \texttt{local[20]}, 80 shuffle partitions and default parallelism 20. XGBoost uses two workers with ten CPUs per task. Experiments run sequentially at low priority without a GPU. Spark DataFrame/SQL and windows process full-data cleaning, aggregation, labels and features; Spark ML pipelines train and evaluate models.'
    expanded_runtime=runtime+r'一般阶段 Java 堆为 6 GiB，进程总内存停止上限为 12 GiB；XGBoost 训练和评价的总内存上限为 24 GiB。'
    text=text.replace(runtime,expanded_runtime)
    w.pairs[expanded_runtime]=w.pairs.pop(runtime)+r' Ordinary phases use a 6 GiB Java heap and a 12 GiB process-tree stop limit. XGBoost fitting and evaluation have a 24 GiB process-tree limit.'
    path=ROOT/'scripts/report_task2_content.py'
    spec=importlib.util.spec_from_file_location('reviewed',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_environment=next(line for line in prefix.splitlines() if line.startswith(r'\textbf{计算环境。}'))
    environment=old_environment.replace('本项目限定使用两个 CPU', '完整特征的模型训练限定使用 20 个 CPU').replace('driver heap 为 2 GiB', '默认 driver heap 为 6 GiB')
    text=text.replace(old_environment,environment)
    w.pairs[environment]=module.LINE_PAIRS[old_environment].replace('this project uses two CPUs', 'full-feature model training uses 20 CPUs').replace('a 2 GiB driver heap', 'a 6 GiB default driver heap')
    pairs={k:v for k,v in module.LINE_PAIRS.items() if k in text};pairs.update(w.pairs)
    overrides={k:v for k,v in module.OVERRIDES.items() if v[0] in text}
    labels={k:v for k,v in module.LABELS.items() if k in text}
    labels.update({'假正例率（FPR）':'False positive rate (FPR)','真正例率（TPR）':'True positive rate (TPR)','随机参考':'Random'})
    apply=path.read_text().split('\ndef apply(',1)[1]
    path.write_text('"""Reviewed bilingual report content generated from verified full-training results."""\nimport hashlib\n\nLINE_PAIRS = '+json.dumps(pairs,ensure_ascii=False,indent=2)+'\n\nOVERRIDES = '+json.dumps(overrides,ensure_ascii=False,indent=2)+'\n\nLABELS = '+json.dumps(labels,ensure_ascii=False,indent=2)+'\n\ndef apply('+apply)
    source.write_text(text)
    from report_translations import translate
    english=translate(text)
    assert re.findall(r'coordinates\s*\{([^}]*)\}',text)==re.findall(r'coordinates\s*\{([^}]*)\}',english)
    (ROOT/'reports/Task1_Report.tex').write_text(english)
    print('Updated existing bilingual sources:', 'methods draft' if args.methods_only else 'complete full-training report')


if __name__=='__main__':
    main()
