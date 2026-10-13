"""Reviewed bilingual text for the verified additional 5 mm experiment."""
import hashlib

SUMMARY_CN = r'\textbf{预测结果。}使用 2017--2022 年全部 34,542,285 条有效记录训练模型，2023 年用于概率校准和选择，2024 年用于测试。原模型比较选定三个 XGBoost 模型；后续 5 mm 调参模型的测试 PR-AUC 从 0.21839 提高至 0.22640，Brier 从 0.0072009 降至 0.0071609。当前预测入口采用该 5 mm 模型，0 与 25 mm 沿用原模型。额外空间交互特征未改善 2023 年验证表现，未采用。'
SUMMARY_EN = r'\textbf{Prediction findings.} Models use all 34,542,285 eligible 2017--2022 records for training, 2023 for calibration and selection, and 2024 for testing. The original comparison selected three XGBoost models. Subsequent 5 mm tuning increased test PR-AUC from 0.21839 to 0.22640 and reduced Brier from 0.0072009 to 0.0071609. The prediction interface uses this tuned 5 mm model and retains the original 0 and 25 mm models. Additional spatial interactions did not improve 2023 validation performance and were not adopted.'

SECTION_CN = r'''
\clearpage\section*{10. 补充实验：5 mm 调参与最终预测入口}
\textbf{实验设计。}候选模型使用全部 27,434,009 条有效 2017--2021 年记录，2022 年用于调参。预处理仅在对应训练期拟合，剔除跨越划分边界的目标窗口。比较六组固定配置，按 PR-AUC 选择，以 Brier 作为并列判据；无抽样或 early stopping。选定配置后，在全部 34,542,285 条有效 2017--2022 年记录上重训。2023 年上半年拟合单调 Platt 校准，下半年选择概率管线及判别阈值；随后固定模型评价 2024 年。
\textbf{选定参数。}类别加权 XGBoost：200 棵树、深度 6、学习率 0.05、128 bins、最小子节点权重 20、L2 正则 5、L1 正则 0。2022 年 PR-AUC 从原参数配置的 0.24617 升至 0.25049。新流程对全部数值特征设置缺失标志，与原流程筛选缺失标志的方式略有不同，收益不能全部归因于调参。原第 8--9 节表格与曲线保留为原比较实验结果。
\begin{center}\begin{tabular}{lrr}\toprule
2024 测试指标 & 原比较实验 5 mm 模型 & 调参后的最终 5 mm 模型\\\midrule
PR-AUC & 0.21839 & 0.22640\\
ROC-AUC & 0.87952 & 0.87908\\
Brier score & 0.0072009 & 0.0071609\\
Precision & 0.29634 & 0.29402\\
Recall & 0.33540 & 0.34657\\
F1 & 0.31466 & 0.31814\\
\bottomrule\end{tabular}\end{center}
\textbf{结果解释。}测试集共 6,374,476 个有效窗口，53,193 个正例，正例比例为 0.83447\%。PR-AUC 相对提高约 3.7\%，Brier 略降，但 ROC-AUC 与 Precision 略降。分类指标使用各流程验证选定的判别阈值，并非固定误报率比较。新模型判别阈值为 0.169824，平均预测概率为 0.87725\%；Brier 低于训练期正例率常数基线 0.0082751。未计算事件分块置信区间，不宣称统计显著提升。
\textbf{计算环境。}本地 Mac 构造的特征上传至 Google Compute Engine e2-highmem-8：8 vCPU、64 GB 配置内存（系统约 62 GiB）、100 GB Balanced Persistent Disk、Ubuntu 22.04.5 LTS x86\_64。Python 3.11、Java 17、Spark 3.5.8、XGBoost 3.2.0；单机 local[8]，两个 XGBoost worker，每任务 4 CPU，driver heap 6 GiB，不使用 GPU。本配置用于补充调参、重训、校准与测试，不改变原 Task 1 和原模型比较的执行环境记录。
\clearpage\section*{10.1. 未采用的空间特征与应用限制}
\textbf{空间交互对照。}固定上述参数，增加八个由已有历史特征派生的交互项：本站与邻站最新及半小时雨量差、邻站最大与平均雨量差、邻站有雨而本站干燥的比例、读数年龄差、本站最近 15 分钟占半小时雨量比例及邻站雨量对数。缺失输入保持缺失，不使用未来观测；这些变量不代表风向或风暴速度。
\begin{center}\begin{tabular}{lrr}\toprule
验证指标 & 调参模型 & 加空间交互特征\\\midrule
2022 PR-AUC & 0.250486 & 0.250632\\
2023 下半年 PR-AUC & 0.239991 & 0.238817\\
2023 下半年 Brier & 0.00701053 & 0.00701540\\
\bottomrule\end{tabular}\end{center}
\textbf{选择结论。}较新的验证期排序和概率误差均未改善，因此不采用空间特征扩展，也不根据 2024 年成绩选择该扩展。保留调参模型作为 5 mm 预测模型。2024 年此前已被原实验查看过；本次参数选择未读取它，但它不是研究者完全未见的封存集。
\textbf{预测入口。}输入站点、五分钟网格上的预测时间及 0、5、25 mm 阈值，输出超越概率。5 mm 默认加载调参校准管线；0 与 25 mm 沿用原验证选定模型。可显式选择原模型作对照。预测仍需要已准备的历史特征行，不是实时服务，且其历史样本范围依赖完整未来标签。三个独立目标尚无跨阈值单调保证；未知站点编码的处理不能证明新站点预测精度。
\textbf{AI 使用补充。}Codex 辅助补充实验脚本、云端部署、文件检查及双语报告整理。结果来自实际完整数据运行；空间对照结果来自云端验证输出。小组成员负责审核，人工小数据仅用于实现检查。
'''

SECTION_EN = r'''
\clearpage\section*{10. Additional experiment: 5 mm tuning and prediction interface}
\textbf{Design.} Candidates used all 27,434,009 eligible 2017--2021 records, with 2022 for tuning. Preprocessing was fitted only on the corresponding training block; future targets crossing split boundaries were purged. Six fixed configurations were ranked by PR-AUC, with Brier as a tie-breaker, without subsampling or early stopping. The selected configuration was refitted on all 34,542,285 eligible 2017--2022 records. Monotone Platt calibration used 2023 H1; probability-pipeline and cutoff selection used 2023 H2. The model was then frozen for 2024 evaluation.
\textbf{Parameters.} Balanced XGBoost used 200 trees, depth 6, learning rate 0.05, 128 bins, minimum child weight 20, L2 regularisation 5 and L1 regularisation 0. The 2022 PR-AUC increased from 0.24617 for the original parameter configuration to 0.25049. The new procedure assigns missing indicators to every numeric feature, rather than selecting indicators as in the original procedure; gains cannot be attributed entirely to parameter tuning. Tables and curves in Sections 8--9 remain results of the original comparison experiment.
\begin{center}\begin{tabular}{lrr}\toprule
2024 test metric & Original comparison: 5 mm & Tuned final 5 mm model\\\midrule
PR-AUC & 0.21839 & 0.22640\\
ROC-AUC & 0.87952 & 0.87908\\
Brier score & 0.0072009 & 0.0071609\\
Precision & 0.29634 & 0.29402\\
Recall & 0.33540 & 0.34657\\
F1 & 0.31466 & 0.31814\\
\bottomrule\end{tabular}\end{center}
\textbf{Interpretation.} The test population contains 6,374,476 eligible windows, with 53,193 positives (0.83447\%). PR-AUC increased by approximately 3.7\% relatively and Brier decreased slightly, while ROC-AUC and precision declined slightly. Classification metrics use each procedure's validation-selected cutoff, not a common false-alarm rate. The new cutoff is 0.169824; mean predicted probability is 0.87725\%. Brier is below the constant training-prevalence baseline of 0.0082751. Event-block confidence intervals were not computed, so statistical significance is not established.
\textbf{Environment.} Features constructed on the local Mac were uploaded to a Google Compute Engine e2-highmem-8 VM: 8 vCPUs, 64 GB configured RAM (approximately 62 GiB reported), 100 GB Balanced Persistent Disk and Ubuntu 22.04.5 LTS x86\_64. Software used Python 3.11, Java 17, Spark 3.5.8 and XGBoost 3.2.0. Spark ran in local[8] mode, with two XGBoost workers and four CPUs per task, a 6 GiB driver heap and no GPU. This describes the additional tuning, refitting, calibration and test experiment, not the original Task 1 or model-comparison runs.
\clearpage\section*{10.1. Rejected spatial interactions and application limits}
\textbf{Spatial ablation.} With the selected parameters fixed, eight interactions were derived from existing historical inputs: neighbour/own latest and half-hour rain gaps, neighbour peak/mean gap, neighbour wet fraction while the own station is dry, reading-age difference, own recent-15-minute rain share and log neighbour rainfall. Missing inputs remain missing; no future observations are used. These features do not measure wind direction or storm velocity.
\begin{center}\begin{tabular}{lrr}\toprule
Validation metric & Tuned model & Additional spatial interactions\\\midrule
2022 PR-AUC & 0.250486 & 0.250632\\
2023 H2 PR-AUC & 0.239991 & 0.238817\\
2023 H2 Brier & 0.00701053 & 0.00701540\\
\bottomrule\end{tabular}\end{center}
\textbf{Selection.} Ranking and probability error did not improve on the more recent validation block, so the spatial extension was rejected without using 2024 results to select it. The tuned model remains the 5 mm predictor. The original experiment had already inspected 2024; although the new parameter selection did not read it, this is not an entirely unseen holdout for the researchers.
\textbf{Prediction interface.} Station, five-minute-grid prediction time and a 0, 5 or 25 mm threshold produce an exceedance probability. The tuned calibrated pipeline is the default for 5 mm; the original validation-selected models remain for 0 and 25 mm. An explicit option selects original models for comparison. Inference still requires a prepared archived feature row and is not a real-time service; historical row eligibility depends on complete future labels. Independent targets have no guaranteed cross-threshold probability ordering. Handling unknown station codes does not establish accuracy at new stations.
\textbf{Additional AI use.} Codex assisted supplementary experiment scripts, cloud deployment, artifact checks and bilingual report editing. Metrics come from actual full-data runs; the spatial comparison comes from cloud validation output. Group members are responsible for review, and artificial fixtures are used only for implementation checks.
'''


THRESHOLD_CN = '\\textbf{阈值选择。}本项目预设 0、5 和 25 mm 三个雨量阈值，分别考察未来半小时是否出现降雨、是否超过较低的正雨量门槛，以及是否发生较大累计雨量事件。选择不同门槛旨在比较事件发生频率降低时，模型的排序能力和概率质量如何变化。这些阈值是本项目的实验设置，并非官方降雨等级。每个阈值分别训练和选择分类器，当前预测接口仅支持这三个值，不支持任意连续雨量阈值。'
THRESHOLD_EN = '\\textbf{Threshold choices.} We preset rainfall thresholds of 0, 5 and 25 mm to examine any rainfall, exceedance of a lower positive accumulation threshold, and a larger accumulation event over the next 30 minutes. These targets allow comparison of model ranking and probability quality as events become rarer. The thresholds are experimental choices, not official rainfall categories. A classifier is trained and selected separately for each threshold; the current interface supports only these three values rather than arbitrary continuous thresholds.'


def apply(paragraphs, hashes, labels):
    paragraphs[r'\textbf{阈值选择。}'] = THRESHOLD_EN
    hashes[r'\textbf{阈值选择。}'] = hashlib.sha256(THRESHOLD_CN.encode()).hexdigest()
    paragraphs[r'\textbf{预测结果。}'] = SUMMARY_EN
    hashes[r'\textbf{预测结果。}'] = hashlib.sha256(SUMMARY_CN.encode()).hexdigest()
    labels['2026 年 10 月 13 日 / 实证分析与预测'] = '13 October 2026 / Empirical analysis and prediction'
    labels['11. AI 使用声明与参考文献'] = '11. AI declaration and references'
