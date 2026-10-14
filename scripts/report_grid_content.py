"""Reviewed bilingual grid-model report content and embedded actual results."""
import hashlib

PAIRS = [('\\textbf{预测结果。}原算法比较采用 Spark MLlib Logistic、Linear SVM 及外部 Spark XGBoost 集成。最终三个 XGBoost 模型分别在 '
  '16 组完整参数网格上调参；2017--2021 年训练、2022 年选参数，再用全部 2017--2022 年重训。2023 年上半年校准，下半年选择概率管线和判别阈值。2024 '
  '年最终测试 PR-AUC 为 0.61218、0.22511、0.02118。0 mm 保留未校准概率，5 与 25 mm 使用校准概率。',
  '\\textbf{Prediction results.} The original algorithm comparison uses Spark MLlib Logistic, '
  'Linear SVM and external Spark XGBoost integration. Each final XGBoost model searches a complete '
  '16-combination parameter grid using 2017--2021 training and 2022 parameter selection, then '
  'refits on all eligible 2017--2022 rows. The first half of 2023 calibrates probabilities; the '
  'second half selects probability pipelines and decision cutoffs. Final 2024 test PR-AUCs are '
  '0.61218, 0.22511, 0.02118. The 0 mm model retains uncalibrated probabilities; 5 and 25 mm use '
  'calibration.'),
 ('\\textbf{训练与验证。}原算法比较与最终重训均使用全部 34,542,285 条有效 2017--2022 年记录。网格候选模型及其预处理使用全部 27,434,009 条有效 '
  '2017--2021 年记录，以 2022 年选择参数。2023 年上半年拟合概率校准，下半年选择概率管线和判别阈值；2024 '
  '年评价固定后的模型。标签窗口不得跨划分边界，校准、验证和测试保留自然类别分布。',
  '\\textbf{Training and validation.} The original algorithm comparison and final refits use all '
  '34,542,285 eligible 2017--2022 rows. Grid candidates and their preprocessing use all 27,434,009 '
  'eligible 2017--2021 rows, with parameters selected on 2022. The first half of 2023 fits '
  'probability calibration and the second half selects probability pipelines and decision cutoffs; '
  '2024 evaluates frozen models. Target windows cannot cross split boundaries. Calibration, '
  'validation and testing retain natural class proportions.'),
 ('\\textbf{特征维度。}原比较模型使用 45 个数值特征、27 个筛选后的缺失标志及站点编码：线性模型为 162 维，树模型为 73 维。最终网格模型对全部 46 '
  '个候选数值变量设置缺失标志，排除常量数值变量后使用 45 个数值特征和一个站点类别索引，共 92 维。所有填补和编码参数仅在对应训练期拟合，因此与原比较的差异不能全部归因于调参。',
  '\\textbf{Feature dimensions.} The original comparison uses 45 numeric features, 27 selected '
  'missingness flags and station encoding: 162 dimensions for linear models and 73 for trees. '
  'Final grid models flag missingness for all 46 candidate numeric variables and retain 45 '
  'nonconstant numeric features plus one station category index, giving 92 dimensions. Imputation '
  'and encoding fit only the corresponding training period; differences from the original '
  'comparison cannot be attributed solely to tuning.'),
 ('\\textbf{选定模型。}原算法比较在三个目标上均选择 XGBoost：0 mm 使用自然分布，5 与 25 mm 使用类别加权。图 7 和图 8 保留该原始比较。最终预测入口统一采用第 '
  '7.1 节的三个网格搜索模型，其指标和曲线在下文单独呈现。',
  '\\textbf{Selected models.} The original algorithm comparison selects XGBoost for all targets: '
  'natural proportions for 0 mm and class weights for 5 and 25 mm. Figures 7 and 8 retain this '
  'original comparison. The final prediction interface uniformly uses the three grid-search models '
  'in Section 7.1, with their metrics and curves presented separately below.'),
 ('\\textbf{测试样本。}上表为原算法比较；下表为最终网格模型。每个目标在 2024 年均评价 6,374,476 个有效窗口，正例比例分别为 6.3602\\%、0.83447\\% '
  '和 0.055173\\%。最终 0 mm 概率未经校准，5 与 25 mm 已校准。',
  '\\textbf{Test population.} The upper table reports the original algorithm comparison; the lower '
  'table reports final grid models. Each target evaluates 6,374,476 eligible 2024 windows, with '
  'positive proportions 6.3602\\%, 0.83447\\% and 0.055173\\%. Final 0 mm probabilities are '
  'uncalibrated; 5 and 25 mm are calibrated.'),
 ('\\textbf{跨时期比较。}2022 年指标对应 2017--2021 年训练的候选模型；2023 H2 与 2024 年指标对应 2017--2022 '
  '年重训后的固定模型。各时期正例比例不同，PR-AUC 变化同时受到模型与事件发生频率影响，不能将它直接解释为性能提升或下降。',
  '\\textbf{Across-period comparison.} The 2022 metrics belong to candidates trained on '
  '2017--2021; 2023 H2 and 2024 metrics belong to frozen models refitted on 2017--2022. Positive '
  'proportions differ between periods, so PR-AUC changes reflect both models and event frequency, '
  'rather than directly proving improvement or deterioration.'),
 ('\\textbf{测试比较。}图 8 为原算法比较，三个 XGBoost 的测试 PR-AUC 为 0.60809、0.21839、0.02089。最终网格模型分别为 '
  '0.61218、0.22511、0.02118，均略高；但 25 mm 的 ROC-AUC 与 F1 下降，因此不能称为所有性能均改善。最终 5 mm 的 PR-AUC '
  '也低于此前单独调参模型的 0.22640。测试结果用于报告，不用于重新选择参数或模型。',
  '\\textbf{Test comparison.} Figure 8 shows the original algorithm comparison, where XGBoost test '
  'PR-AUCs are 0.60809, 0.21839 and 0.02089. Final grid values are 0.61218, 0.22511 and 0.02118, '
  'all slightly higher; however, 25 mm ROC-AUC and F1 decline, so this is not an improvement '
  'across all measures. Final 5 mm PR-AUC also falls below the earlier individually tuned model at '
  '0.22640. Test results are reported without reselecting parameters or models.'),
 ('\\textbf{评价对象。}曲线对应固定的三个网格模型，按 0、5、25 mm 排列；全部使用 2017--2022 年重训，并在完整 2024 '
  '年测试窗口上评价。图中坐标是经有界简化保留的实际曲线点，AUC 由全部分数精确计算。ROC 虚线为 $y=x$，PR 虚线为测试正例比例；25 mm 的 PR 纵轴放大至 0--0.15。',
  '\\textbf{Evaluation objects.} Curves correspond to frozen grid models for 0, 5 and 25 mm, '
  'refitted on 2017--2022 and evaluated on every eligible 2024 window. Plotted coordinates retain '
  'actual curve points through bounded simplification; AUCs use exact full-score integration. ROC '
  'reference lines are $y=x$; PR references are test prevalence. The 25 mm PR axis is enlarged to '
  '0--0.15.'),
 ('\\textbf{排序表现。}最终三个模型的 ROC-AUC 分别为 0.88002、0.88159、0.90309，PR-AUC 分别为 '
  '0.61218、0.22511、0.02118。较高雨量阈值对应更稀少的事件，应结合 PR 曲线和正例比例解释较高的 ROC-AUC。',
  '\\textbf{Ranking performance.} Final ROC-AUCs are 0.88002, 0.88159, 0.90309, and PR-AUCs are '
  '0.61218, 0.22511, 0.02118. Higher rainfall thresholds correspond to rarer events; high ROC-AUC '
  'must be interpreted alongside PR curves and prevalence.'),
 ('\\textbf{概率质量。}三个目标的平均预测概率为 6.1816\\%、0.8603\\%、0.0426\\%，实际发生比例为 '
  '6.3602\\%、0.8345\\%、0.0552\\%。每个点汇总一组相近预测概率的窗口，灰色虚线表示理想校准。整体平均值接近并不保证各风险区间均准确。',
  '\\textbf{Probability quality.} Mean predicted probabilities are 6.1816\\%, 0.8603\\%, '
  '0.0426\\%, versus observed proportions 6.3602\\%, 0.8345\\% and 0.0552\\%. Each point '
  'aggregates windows with similar probabilities; grey lines represent ideal calibration. Similar '
  'overall means do not guarantee accuracy at every risk level.'),
 ('\\textbf{校准选择。}2023 H2 的 0 mm 未校准 Brier 为 0.0343148，优于校准后的 0.0344561，因此保留未校准概率。5 与 25 mm 校准后 '
  'Brier 分别为 0.0070153、0.0007187，明显低于其加权模型未校准误差 0.1144999、0.0994519。管线选择只使用验证集，测试集仅评价选定结果。',
  '\\textbf{Calibration selection.} On 2023 H2, 0 mm uncalibrated Brier is 0.0343148, better than '
  'calibrated 0.0344561, so uncalibrated probabilities are retained. Calibrated Brier for 5 and 25 '
  'mm is 0.0070153 and 0.0007187, substantially below uncalibrated weighted-model errors of '
  '0.1144999 and 0.0994519. Pipeline selection uses validation only; testing evaluates the chosen '
  'results.'),
 ('\\textbf{稀有事件校准。}25 mm 的最高平均预测概率组包含 170,814 个窗口，平均预测概率为 0.891\\%，实际发生比例为 '
  '1.171\\%。该分组结果应结合有限正例数量解释；未计算事件分块置信区间，不宣称统计显著改善。',
  '\\textbf{Rare-event calibration.} The highest mean-probability group for 25 mm contains 170,814 '
  'windows, with mean prediction 0.891\\% and observed frequency 1.171\\%. Interpret this group '
  'alongside limited positive counts. No event-block confidence intervals are computed, so no '
  'statistically significant improvement is claimed.'),
 ('\\textbf{模型选择。}原算法比较按 2023 H2 的 PR-AUC 选择算法与训练策略，以 Brier 作为并列判据。最终网格模型的超参数按 2022 年 PR-AUC '
  '选择，校准与未校准管线按 2023 H2 Brier 选择。所有分类判别阈值在相应验证预测上按 F1 确定，用于测试 Precision、Recall、F1 和 '
  'Accuracy；它与定义标签的雨量阈值不同。2024 年结果不参与本轮选择。',
  '\\textbf{Model selection.} The original comparison selects algorithms and policies by 2023 H2 '
  'PR-AUC, with Brier as tie-breaker. Final grid hyperparameters use 2022 PR-AUC, while calibrated '
  'versus uncalibrated pipelines use 2023 H2 Brier. All decision cutoffs maximize F1 on the '
  'corresponding validation predictions and determine test Precision, Recall, F1 and Accuracy; '
  'they differ from rainfall thresholds defining labels. The 2024 results do not enter this round '
  'of selection.'),
 ('\\textbf{小组成员。}潘禹翔（A0355687J）、侯佳成（A0353127J）、吴嘉蕊（A0262371J）。',
  '\\textbf{Group members.} Pan Yuxiang (A0355687J), Hou Jiacheng (A0353127J), Wu Jiarui '
  '(A0262371J).')]

BLOCKS = [('\\clearpage\\section*{7.1. XGBoost 网格搜索与最终模型}\n'
  '\n'
  '\\textbf{网格设计。}固定 XGBoost 算法，对每个雨量阈值独立遍历四个参数候选集合的完整笛卡尔积，共 $2^4=16$ 组，三个目标共 48 个候选拟合。树数量为 100 或 '
  '200，深度为 4 或 6，学习率为 0.05 或 0.1，L2 正则为 1 或 5。其余固定为 128 bins、最小子节点权重 20、L1 正则 0、行和列采样比例 1。0 mm '
  '使用自然分布，5 与 25 mm 使用类别加权。使用按年份划分的验证集，而非随机折叠交叉验证；不抽样训练行，不使用 early stopping。\n'
  '\n'
  '\\textbf{参数选择与重训。}每个候选使用完整 2017--2021 年数据，并按 2022 年 Spark 精确 PR-AUC 排序，Brier 作为并列判据。选定参数后，在全部 '
  '2017--2022 年记录上重新拟合预处理和分类器。2023 年上半年拟合单调 Platt 校准，下半年按 Brier 在校准与未校准管线中选择，并按 F1 '
  '选择分类判别阈值。最终模型在读取本轮 2024 年测试结果前固定。\n'
  '\\begin{center}\\begin{tabular}{rrrrrlr}\\toprule\n'
  '雨量阈值 mm & 树数量 & 深度 & 学习率 & L2 正则 & 最终概率 & 2022 PR-AUC\\\\\\midrule\n'
  '0 & 100 & 6 & 0.05 & 5 & 未校准 & 0.61137\\\\\n'
  '5 & 100 & 6 & 0.05 & 5 & Platt 校准 & 0.25174\\\\\n'
  '25 & 100 & 4 & 0.05 & 5 & Platt 校准 & 0.03050\\\\\n'
  '\\bottomrule\\end{tabular}\\end{center}\n'
  '\n'
  '\\textbf{计算环境。}本地 Mac 构造的特征上传至 Google Compute Engine e2-highmem-8：8 vCPU、64 GB 配置内存（系统约 62 '
  'GiB）、100 GB Balanced Persistent Disk、Ubuntu 22.04.5 LTS x86\\_64。Python 3.11、Java 17、Spark '
  '3.5.8、XGBoost 3.2.0；单机 local[8]，两个 XGBoost worker，每任务 4 CPU，driver heap 6 GiB，不使用 '
  'GPU。本配置用于网格搜索、重训、校准和最终测试；原 Task 1 与算法比较的执行环境另行记录。\n',
  '\\clearpage\\section*{7.1. XGBoost grid search and final models}\n'
  '\n'
  '\\textbf{Grid design.} Fixing the XGBoost algorithm, each rainfall target traverses the full '
  'Cartesian product of four parameter sets: $2^4=16$ combinations per target and 48 candidate '
  'fits overall. The numbers of trees are 100 or 200, depths 4 or 6, learning rates 0.05 or 0.1, '
  'and L2 penalties 1 or 5. Fixed settings are 128 bins, minimum child weight 20, L1 penalty 0, '
  'and row/column sampling fractions 1. The 0 mm target uses natural proportions; 5 and 25 mm use '
  'class weights. Validation is chronological holdout, not random-fold cross-validation. No '
  'training rows are subsampled and no early stopping is used.\n'
  '\n'
  '\\textbf{Parameter selection and refitting.} Each candidate uses all eligible 2017--2021 rows '
  'and is ranked by exact Spark PR-AUC on 2022, with Brier as tie-breaker. Preprocessing and '
  'classifiers are then refitted on all eligible 2017--2022 rows. Monotone Platt calibration fits '
  '2023 H1; 2023 H2 selects calibrated versus uncalibrated pipelines by Brier and decision cutoffs '
  'by F1. Final models are frozen before reading this round of 2024 test results.\n'
  '\\begin{center}\\begin{tabular}{rrrrrlr}\\toprule\n'
  'Rainfall threshold mm & Trees & Depth & Learning rate & L2 penalty & Final probabilities & 2022 '
  'PR-AUC\\\\\\midrule\n'
  '0 & 100 & 6 & 0.05 & 5 & Uncalibrated & 0.61137\\\\\n'
  '5 & 100 & 6 & 0.05 & 5 & Platt calibrated & 0.25174\\\\\n'
  '25 & 100 & 4 & 0.05 & 5 & Platt calibrated & 0.03050\\\\\n'
  '\\bottomrule\\end{tabular}\\end{center}\n'
  '\n'
  '\\textbf{Computing environment.} Features prepared on the local Mac are uploaded to Google '
  'Compute Engine e2-highmem-8: 8 vCPUs, 64 GB configured RAM (about 62 GiB reported), a 100 GB '
  'Balanced Persistent Disk and Ubuntu 22.04.5 LTS x86\\_64. The runtime uses Python 3.11, Java '
  '17, Spark 3.5.8 and XGBoost 3.2.0; single-machine local[8], two XGBoost workers, four CPUs per '
  'task, a 6 GiB driver heap and no GPU. This configuration runs grid search, refitting, '
  'calibration and final testing; original Task 1 and algorithm-comparison environments are '
  'recorded separately.\n'),
 ('\\textbf{算法比较设置。}本节固定参数表与三种训练策略对应原 27 次算法比较，不是最终网格模型。Logistic 与 Linear SVM 使用 Spark '
  'MLlib；XGBoost 使用外部 Spark 接口。原比较的结果保留为基线，最终 XGBoost 参数在第 7.1 节给出。',
  "\\textbf{Algorithm comparison setup.} This section's fixed-parameter table and three policies "
  'belong to the original 27-fit algorithm comparison, not final grid models. Logistic and Linear '
  'SVM use Spark MLlib; XGBoost uses an external Spark interface. Original results are retained as '
  'baselines; final XGBoost parameters appear in Section 7.1.'),
 ('\\begin{center}\\begin{tabular}{rllrrr}\\toprule\n'
  '阈值 mm & 网格模型 & 数据集 & ROC-AUC & PR-AUC & Brier\\\\\\midrule\n'
  '0 & XGBoost & 2023 H2 验证 & 0.88141 & 0.58987 & 0.0343148\\\\\n'
  '0 & XGBoost & 2024 测试 & 0.88002 & 0.61218 & 0.0361268\\\\\n'
  '5 & XGBoost & 2023 H2 验证 & 0.89386 & 0.23927 & 0.0070153\\\\\n'
  '5 & XGBoost & 2024 测试 & 0.88159 & 0.22511 & 0.0071700\\\\\n'
  '25 & XGBoost & 2023 H2 验证 & 0.91441 & 0.03934 & 0.0007187\\\\\n'
  '25 & XGBoost & 2024 测试 & 0.90309 & 0.02118 & 0.0005448\\\\\n'
  '\\bottomrule\\end{tabular}\\end{center}',
  '\\begin{center}\\begin{tabular}{rllrrr}\\toprule\n'
  'Threshold mm & Grid model & Dataset & ROC-AUC & PR-AUC & Brier\\\\\\midrule\n'
  '0 & XGBoost & 2023 H2 validation & 0.88141 & 0.58987 & 0.0343148\\\\\n'
  '0 & XGBoost & 2024 test & 0.88002 & 0.61218 & 0.0361268\\\\\n'
  '5 & XGBoost & 2023 H2 validation & 0.89386 & 0.23927 & 0.0070153\\\\\n'
  '5 & XGBoost & 2024 test & 0.88159 & 0.22511 & 0.0071700\\\\\n'
  '25 & XGBoost & 2023 H2 validation & 0.91441 & 0.03934 & 0.0007187\\\\\n'
  '25 & XGBoost & 2024 test & 0.90309 & 0.02118 & 0.0005448\\\\\n'
  '\\bottomrule\\end{tabular}\\end{center}'),
 ('\\textbf{原 SMOTE 统计。}下表对应原比较的合成样本；最终网格模型不使用 SMOTE。',
  '\\textbf{Original SMOTE counts.} The following table belongs to the original comparison; final '
  'grid models use no synthetic observations.'),
 ('\\begin{center}\\begin{tabular}{rlrrrrr}\\toprule\n'
  '阈值 mm & 网格模型 & 判别阈值 & Precision & Recall & F1 & Accuracy\\\\\\midrule\n'
  '0 & XGBoost & 0.285102 & 0.62658 & 0.56163 & 0.59233 & 0.95083\\\\\n'
  '5 & XGBoost & 0.194984 & 0.30804 & 0.32470 & 0.31615 & 0.98828\\\\\n'
  '25 & XGBoost & 0.031623 & 0.04585 & 0.10918 & 0.06458 & 0.99825\\\\\n'
  '\\bottomrule\\end{tabular}\\end{center}',
  '\\begin{center}\\begin{tabular}{rlrrrrr}\\toprule\n'
  'Threshold mm & Grid model & Decision cutoff & Precision & Recall & F1 & Accuracy\\\\\\midrule\n'
  '0 & XGBoost & 0.285102 & 0.62658 & 0.56163 & 0.59233 & 0.95083\\\\\n'
  '5 & XGBoost & 0.194984 & 0.30804 & 0.32470 & 0.31615 & 0.98828\\\\\n'
  '25 & XGBoost & 0.031623 & 0.04585 & 0.10918 & 0.06458 & 0.99825\\\\\n'
  '\\bottomrule\\end{tabular}\\end{center}'),
 ('\\textbf{历史调参与空间对照。}此前 5 mm 六组配置搜索在 2023 H2 的 PR-AUC 为 0.239991、Brier 为 0.00701053，略优于最终网格模型的 '
  '0.239268、0.00701530。为统一三个目标的实验流程，本项目仍固定采用网格模型，并不宣称 5 mm 网格搜索优于此前模型。固定此前参数增加八个空间交互特征后，2023 H2 '
  'PR-AUC 为 0.238817、Brier 为 0.00701540，因此未采用该扩展。\n'
  '\n'
  '\\textbf{预测入口与限制。}输入站点、五分钟网格上的预测时间及 0、5、25 mm '
  '阈值，加载对应网格管线并输出超越概率；可显式选择原比较模型。仍依赖已准备的历史特征行，其样本范围受完整未来标签限制，尚非实时服务。三个目标独立训练，无跨阈值概率单调保证。2024 '
  '年此前已被原实验查看过；本轮参数与最终模型在重新测试前固定，但此评价属于后续实验，不是研究者完全未见的封存测试。\n',
  '\\textbf{Earlier tuning and spatial comparison.} The earlier six-configuration 5 mm search '
  'achieved 2023 H2 PR-AUC 0.239991 and Brier 0.00701053, slightly better than final grid-model '
  'values 0.239268 and 0.00701530. The project nevertheless fixes the grid model for a uniform '
  'three-target procedure and does not claim it outperforms the earlier 5 mm model. Adding eight '
  'spatial interaction features at the earlier fixed parameters produced 2023 H2 PR-AUC 0.238817 '
  'and Brier 0.00701540, so that extension was not adopted.\n'
  '\n'
  '\\textbf{Prediction interface and limitations.} Station, prediction time on the five-minute '
  'grid and a threshold of 0, 5 or 25 mm select the corresponding grid pipeline and exceedance '
  'probability; the original comparison model remains explicitly selectable. Prepared historical '
  'feature rows are still required, and their sample scope depends on complete future labels; this '
  'is not yet a live service. Targets are trained independently without cross-threshold '
  'probability monotonicity guarantees. The original experiment previously inspected 2024; this '
  'round freezes parameters and models before retesting, but remains a follow-up experiment rather '
  'than a researcher-unseen sealed test.\n')]

LABELS = {'10. AI 使用声明与参考文献': '10. AI declaration and references', '图 8：原算法比较的测试集 PR-AUC': 'Figure 8: test PR-AUC in the original algorithm comparison', '阈值 mm & 原算法 & 验证选定策略': 'Threshold mm & Original algorithm & Validation-selected policy', '2026 年 10 月 14 日 / 实证分析与预测': '14 October 2026 / Empirical analysis and prediction'}

def apply(paragraphs, hashes, labels):
    for cn,en in PAIRS:
        prefix=cn[:cn.index('}')+1]
        paragraphs[prefix]=en
        hashes[prefix]=hashlib.sha256(cn.encode()).hexdigest()
    labels.update(LABELS)

def translate_blocks(source):
    for cn,en in BLOCKS:
        source=source.replace(cn,en)
    return source
