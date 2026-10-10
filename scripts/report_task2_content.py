"""Reviewed bilingual report content generated from verified full-training results."""
import hashlib

LINE_PAIRS = {
  "\\clearpage\\section*{4. Task 2：预测目标与时间切分}": "\\clearpage\\section*{4. Task 2: target and chronological splits}",
  "用途 & 有效行数 & $>0$ 正例 & $>5$ 正例 & $>25$ 正例\\\\\\midrule": "Purpose & Rows & $>0$ positives & $>5$ positives & $>25$ positives\\\\\\midrule",
  "2023 上半年：概率校准 & 3,325,499 & 251,794 & 30,699 & 1,532\\\\": "2023 H1: calibration & 3,325,499 & 251,794 & 30,699 & 1,532\\\\",
  "2023 下半年：模型选择 & 3,393,373 & 197,358 & 28,000 & 2,487\\\\": "2023 H2: selection & 3,393,373 & 197,358 & 28,000 & 2,487\\\\",
  "\\clearpage\\section*{5. 从 Task 1 到特征工程}": "\\clearpage\\section*{5. From Task 1 to feature engineering}",
  "Task 1 依据 & 构造方式 & 实际输入\\\\\\midrule": "Task 1 finding & Construction & Inputs\\\\\\midrule",
  "\\clearpage\\section*{6. 模型、不平衡处理与概率校准}": "\\clearpage\\section*{6. Models, imbalance and calibration}",
  "L2 Logistic（MLlib） & $\\lambda=0.001$，标准化，最大迭代 100，容差 $10^{-6}$\\\\": "L2 logistic (MLlib) & $\\lambda=0.001$, standardization, 100 iterations, tolerance $10^{-6}$\\\\",
  "Linear SVM（MLlib） & $\\lambda=0.001$，标准化，最大迭代 100，容差 $10^{-6}$\\\\": "Linear SVM (MLlib) & $\\lambda=0.001$, standardization, 100 iterations, tolerance $10^{-6}$\\\\",
  "GBT（MLlib） & 40 轮，深度 4，步长 0.1，128 bins，叶节点最少 20 行\\\\": "GBT (MLlib) & 40 rounds, depth 4, step 0.1, 128 bins, minimum 20 rows/node\\\\",
  "XGBoost（Spark 接口） & 80 棵树，深度 4，学习率 0.1，64 bins，子节点权重 20；$L_2=1$\\\\": "XGBoost (Spark interface) & 80 trees, depth 4, learning rate 0.1, 64 bins, child weight 20; $L_2=1$\\\\",
  "\\clearpage\\section*{7. 评价方法与验证集选择}": "\\clearpage\\section*{7. Evaluation and validation selection}",
  "\\item Apache Spark 3.5.8，\\href{https://spark.apache.org/docs/3.5.8/ml-classification-regression.html}{MLlib 分类与回归文档}，用于 Logistic、Linear SVM 与 GBT 实现；评价采用 Spark BinaryClassificationEvaluator。": "\\item Apache Spark 3.5.8, \\href{https://spark.apache.org/docs/3.5.8/ml-classification-regression.html}{MLlib classification and regression documentation}, for logistic, linear SVM and GBT implementation. Evaluation uses Spark BinaryClassificationEvaluator.",
  "\\item XGBoost，\\href{https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/spark_estimator.html}{PySpark Estimator 文档}，用于外部 Spark XGBoost 集成；实际安装版本为 3.2.0。": "\\item XGBoost, \\href{https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/spark_estimator.html}{PySpark estimator documentation}, for external Spark integration; the actual installed version is 3.2.0.",
  "\\textbf{研究内容。}本项目利用 2017--2024 年的新加坡降雨观测数据分析时间与空间变化规律，并建立模型，预测监测站在未来 30 分钟内累计雨量超过给定阈值的概率。": "\\textbf{Study objectives.} Using Singapore rainfall observations from 2017--2024, we analyse temporal and spatial patterns and develop models to predict the probability that station rainfall accumulated over the next 30 minutes exceeds a given threshold.",
  "\\textbf{分析样本。}各项分析根据比较目的采用相应的站点范围与数据完整性筛选条件，具体见下表。": "\\textbf{Analysis samples.} Station scope and completeness criteria are chosen according to the comparison, as summarized below.",
  "\\textbf{数据核对。}下载完整年度 CSV 后，将 Spark 读入的记录数与年度 API 元数据的记录数核对，并检查重复观测、时间对齐和雨量取值。": "\\textbf{Data checks.} After downloading complete annual CSV files, we reconcile Spark input row counts with annual API metadata and check duplicate observations, timestamp alignment and rainfall values.",
  "\\textbf{清洗结果。}86 条重复记录全部来自 2017 年；八年数据中未发现负雨量、偏离五分钟网格超过两秒的时间戳或跨文件重复。缺失时段不计为零雨量。对超过 50 mm/5min 的读数进行额外筛查，本次没有触发该条件的记录。": "\\textbf{Cleaning results.} All 86 duplicate records occur in 2017. The eight years contain no negative rainfall, timestamps more than two seconds from the five-minute grid, or cross-file duplicates. Missing intervals are not counted as zero rainfall. Additional screening of readings above 50 mm/5min identifies none.",
  "\\textbf{计算环境。}Task 1 使用 Apple Silicon Mac：10 个逻辑 CPU、16 GiB RAM，macOS 26.5.2 ARM64，Python 3.12.14、Java Temurin 17.0.20.1+1、Spark 3.5.8。Task 2 在 gpushare 运行，环境为 Python 3.11.10、Azul Zulu Java 17.0.20.1、Spark 3.5.8、XGBoost 3.2.0。gpushare 容器配额为 42 CPU、约 667.68 GiB RAM；本项目限定使用两个 CPU，driver heap 为 2 GiB。": "\\textbf{Computing environment.} Task 1 runs on an Apple Silicon Mac with 10 logical CPUs, 16 GiB RAM, macOS 26.5.2 ARM64, Python 3.12.14, Java Temurin 17.0.20.1+1 and Spark 3.5.8. Task 2 runs on gpushare with Python 3.11.10, Azul Zulu Java 17.0.20.1, Spark 3.5.8 and XGBoost 3.2.0. The gpushare container quota is 42 CPUs and about 667.68 GiB RAM; this project uses two CPUs and a 2 GiB driver heap.",
  "\\textbf{Task 2 特征构造（月份）。}月度雨频与雨强的变化模式不同，因而使用月份的正弦、余弦编码表示年内周期，并与小时编码组合，刻画季节与日内降雨规律的共同作用。": "\\textbf{Task 2 features (month).} Monthly frequency and wet-interval intensity vary differently. We encode month using sine and cosine to represent the annual cycle and combine them with hour encoding to capture seasonal variation in diurnal rainfall.",
  "\\textbf{Task 2 特征构造（年份）。}训练、验证和测试按年份划分。模型使用月份与小时的周期特征，不加入年份编号。NEA 的 2019 年气候回顾将当年偏干与强正位相印度洋偶极子联系起来，为年际变化提供外部背景。": "\\textbf{Task 2 features (year).} Training, validation and test are split chronologically. Models use cyclic month and hour features without year identifiers. NEA relates the dry conditions in 2019 to a strong positive Indian Ocean Dipole, providing external context for annual variation.",
  "\\textbf{Task 2 特征构造（邻站）。}邻近站点的日雨量呈现较强相关性，因此加入三个最近邻站的历史雨量均值、最大值和有雨比例。Task 2 通过加入邻站特征前后的模型比较，评价其对半小时预测的贡献。": "\\textbf{Task 2 features (neighbours).} Stronger daily-rainfall correlation among nearby stations motivates mean rainfall, maximum rainfall and wet fractions from the three closest stations. Task 2 compares models before and after adding neighbour features to evaluate their contribution to half-hour prediction.",
  "\\noindent 21 个站点形成 210 对站点组合。距离不超过 5 km 的 17 对站点，日雨量平均 Pearson 相关系数为 0.809；距离超过 30 km 的 12 对为 0.336。每 5 km 距离组的平均相关性总体随距离增加而降低，表明降雨具有空间相关性。": "\\noindent The 21 stations form 210 pairs. The mean Pearson correlation of daily rainfall is 0.809 for 17 pairs separated by at most 5 km, compared with 0.336 for 12 pairs more than 30 km apart. Mean correlations in 5 km distance groups generally decrease with separation, indicating spatial dependence.",
  "\\noindent \\textbf{有雨区间的雨量分布。}八年全部站点的有雨五分钟观测中，41.72\\% 的雨量位于 $(0,0.2]$ mm，37.16\\% 位于 $(0.2,1]$ mm。合计 78.88\\% 的有雨区间雨量不超过 1 mm，表明多数降雨区间的累计雨量较小。全部有效五分钟观测中的无雨比例为 97.08\\%。": "\\noindent \\textbf{Rainfall distribution in wet intervals.} Across all stations and eight years, 41.72\\% of wet five-minute observations fall in $(0,0.2]$ mm and 37.16\\% in $(0.2,1]$ mm. Together, 78.88\\% have at most 1 mm, indicating small accumulations in most wet intervals. Dry intervals account for 97.08\\% of all valid five-minute observations.",
  "\\textbf{半小时累计雨量的超越概率。}2024 年共有 6,374,476 个完整后续六区间窗口。后续 30 分钟累计雨量严格超过 0 mm 和 5 mm 的窗口比例分别为 6.36\\% 和 0.83\\%，超越概率随雨量阈值升高而降低。累计雨量按九位小数比较，以避免阈值边界的浮点误差。图中横轴按阈值类别排列。": "\\textbf{Exceedance probability of half-hour rainfall.} There are 6,374,476 complete future six-interval windows in 2024. Subsequent 30-minute totals strictly exceed 0 mm in 6.36\\% of windows and 5 mm in 0.83\\%; exceedance becomes less frequent as the rainfall threshold rises. Totals are compared at nine decimal places to avoid floating-point errors at threshold boundaries. The horizontal axis uses threshold categories.",
  "\\textbf{相邻区间的降雨持续性。}2024 年同站点连续五分钟观测显示，前一区间无雨时，后一区间有雨的条件概率为 1.04\\%；前一区间有雨时，该概率为 67.03\\%。降雨具有明显的短时持续性，支持在预测模型中纳入近期雨量及连续有雨时长。": "\\textbf{Rainfall persistence in consecutive intervals.} For consecutive five-minute observations at the same station in 2024, the probability of rain is 1.04\\% following a dry interval and 67.03\\% following a wet interval. This substantial short-term persistence supports recent rainfall and wet-spell duration as predictors.",
  "\\item NEA，\\href{https://data.gov.sg/collections/2279/view}{Historical Rainfall across Singapore}，2017--2024 年度数据集。": "\\item NEA, \\href{https://data.gov.sg/collections/2279/view}{Historical Rainfall across Singapore}, annual datasets 2017--2024.",
  "\\item scikit-learn，\\href{https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html}{Precision--Recall 文档}与\\href{https://scikit-learn.org/stable/modules/calibration.html}{概率校准文档}。": "\\item scikit-learn, \\href{https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html}{Precision--Recall documentation} and \\href{https://scikit-learn.org/stable/modules/calibration.html}{probability-calibration documentation}.",
  "\\textbf{Task 2 特征启示（持续性与稀有事件）。}实际特征包括最近读数、过去 15/30/60 分钟累计、近期雨势变化和连续有雨时长。只使用预测时已可获得的记录，缺测不补为无雨。下文逐阈值给出正例数、PR-AUC、Brier score 和校准曲线。": "\\textbf{Task 2 feature implications (persistence and rare events).} Implemented features include the latest reading, 15/30/60-minute totals, recent rainfall changes and wet-spell duration. Only records available at prediction time are used; missing readings are not dry readings. The following sections report positive counts, PR-AUC, Brier score and calibration curves separately by threshold.",
  "\\textbf{AI 使用声明。}本项目使用 OpenAI Codex 辅助研究设计、数据处理和模型代码编写、结果核对与报告文字整理。报告中的降雨统计与模型评价结果均来自实际数据处理和模型运行；SMOTE 合成样本仅用于训练。": "\\textbf{AI-use declaration.} We use OpenAI Codex to assist study design, data-processing and model code, result checks and report editing. Reported rainfall statistics and model evaluations come from actual processing and model runs; SMOTE examples are used only for training.",
  "\\textbf{小组成员。}潘禹翔、侯佳成、吴嘉蕊。": "\\textbf{Group members.} Pan Yuxiang, Hou Jiacheng, Wu Jiarui.",
  "\\item Meteorological Service Singapore，\\href{https://www.weather.gov.sg/climate-climate-of-singapore}{Climate of Singapore}。": "\\item Meteorological Service Singapore, \\href{https://www.weather.gov.sg/climate-climate-of-singapore}{Climate of Singapore}.",
  "\\item geoBoundaries，\\href{https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/}{Singapore ADM0}；轮廓源自 URA 2016 年资料，ODbL 1.0。": "\\item geoBoundaries, \\href{https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/}{Singapore ADM0}, based on URA 2016 data, ODbL 1.0.",
  "\\vspace{5mm}\\noindent 固定站点样本中，合格站点年月的覆盖率至少为 80\\%，先对站点等权平均，再对可用年份等权平均。月度平均雨频以 11 月最高（4.28\\%）、7 月最低（1.85\\%）；有雨区间平均雨量以 3 月约 1.02 mm、1 月约 0.67 mm。雨频与有雨区间雨量呈现不同的季节变化。各月包含 6--8 个年份，每个合格年月包含 17--21 个站点。": "\\vspace{5mm}\\noindent Eligible station-months from the fixed panel require at least 80\\% coverage, with equal averaging over stations followed by available years. Mean monthly rain frequency is highest in November (4.28\\%) and lowest in July (1.85\\%). Mean wet-interval amounts are about 1.02 mm in March and 0.67 mm in January. Frequency and wet-interval amount show different seasonal patterns. Each month includes 6--8 years, and each eligible year-month includes 17--21 stations.",
  "\\noindent 各年采用相同的 21 个站点进行等权比较，每个站点的年度覆盖率至少为 80\\%。平均雨量 / 观测24h 从 2019 年的 4.81 mm 到 2021 年的 7.84 mm。2019 年的雨频为 1.97\\%，明显低于其余年份，而有雨区间雨量为 0.841 mm，与相邻年份接近，表明该年偏低的归一化雨量主要对应较低的降雨频率。": "\\noindent Each year compares the same 21 equally weighted stations, each with annual coverage of at least 80\\%. Mean rainfall per observed 24 hours ranges from 4.81 mm in 2019 to 7.84 mm in 2021. Rain frequency in 2019 is 1.97\\%, below the other years, while its wet-interval amount of 0.841 mm is close to adjacent years. Lower rainfall frequency is thus the main component associated with its lower normalized rainfall.",
  "\\noindent 降雨频率在 16:00--17:00 达到日内高峰，且高峰时段与强度随月份变化。11--12 月的下午高峰较明显，6 月的较高雨频更接近上午至中午。统计对覆盖率至少 80\\% 的站点、年份、月份、小时组合等权平均。该结果支持同时使用小时、月份与二者交互项作为预测特征。": "\\noindent Rain frequency peaks at 16:00--17:00, with peak timing and strength varying by month. The afternoon peak is more pronounced in November--December, while June has elevated frequency from morning toward midday. Statistics equally weight eligible station-year-month-hour strata with at least 80\\% coverage. These findings motivate hour, month and their interactions as predictors.",
  "\\textbf{预测目标。}对于站点 $s$、时刻 $t$ 和雨量阈值 $q$，估计未来 30 分钟累计雨量严格超过 $q$ 的概率。设置 $q\\in\\{0,5,25\\}$ mm，每个目标分别训练分类器；同一站点时刻保留一份特征与三个标签列。": "\\textbf{Prediction target.} For station $s$, time $t$ and rainfall threshold $q$, estimate the probability that rainfall in the next 30 minutes strictly exceeds $q$. Use $q\\in\\{0,5,25\\}$ mm and one classifier per target, with one feature row and three label columns per station/time.",
  "\\textbf{标签构造。}将时间戳作为五分钟区间的终点，累计 $(t,t+30\\mathrm{min}]$ 内六个有效区间的雨量，保留九位小数后与阈值比较。当前区间不计入目标；任一未来区间缺测时不构造标签。累计雨量等于阈值时标签为 0。": "\\textbf{Label construction.} Treat timestamps as five-minute interval ends. Sum six valid intervals in $(t,t+30\\mathrm{min}]$ and round to nine decimal places before comparison. Exclude the current interval and reject windows missing any future interval. Equality with the threshold is negative.",
  "2017--2022：全部训练记录 & 34,542,285 & 2,074,430 & 279,801 & 19,787\\\\": "2017--2022: all training records & 34,542,285 & 2,074,430 & 279,801 & 19,787\\\\",
  "2024：测试 & 6,374,476 & 405,430 & 53,193 & 3,517\\\\": "2024: test & 6,374,476 & 405,430 & 53,193 & 3,517\\\\",
  "\\textbf{训练与验证。}所有模型及预处理均使用 2017--2022 年全部 34,542,285 条有效训练记录。2023 年上半年拟合概率校准函数，下半年选择模型及概率判别阈值；2024 年用于测试。标签窗口不得跨上述时间边界。校准、选择和测试保留自然类别分布。": "\\textbf{Training and validation.} All classifiers and preprocessing use all 34,542,285 eligible 2017--2022 training records. Fit calibration on 2023 H1, select models and decision cutoffs on H2, and test on 2024. Label windows cannot cross these boundaries. Calibration, selection and test preserve natural class proportions.",
  "下午高峰随月份变化 & 小时、月份周期编码与交互 & 8 个日历特征；不加入年份 ID\\\\": "Afternoon peak varies by month & Cyclic hour/month and interactions & 8 calendar features; no year ID\\\\",
  "站点差异与空间相关 & 站点 ID、经纬度、三个最近邻站 & 邻站雨量均值、最大值、有雨比例\\\\": "Station differences and spatial correlation & Station ID, coordinates, three closest neighbours & Neighbour rain means, maxima and wet fractions\\\\",
  "短时降雨持续性 & 最新读数及 5/10/15/30/60 分钟滞后 & 15/30/60 分钟累计、变化和雨段长度\\\\": "Short-term persistence & Latest reading and 5/10/15/30/60-minute lags & 15/30/60-minute totals, changes and wet-spell length\\\\",
  "\\textbf{历史与邻站特征。}只使用预测时已可获得的观测，观测时刻和保留的更新时间均不晚于 $t$。从过去 90 分钟检索历史；最近读数超过 30 分钟时标记不可用。邻站由当时已知坐标确定，聚合三个最近合格邻站的雨量、有效观测数、读数年龄及距离。Spark 窗口函数和连接处理全量记录。": "\\textbf{History and neighbours.} Use only observations available by prediction time: both observation and retained update timestamps must be no later than $t$. Search the past 90 minutes; mark the latest reading unavailable after 30 minutes. Use coordinates known by $t$ to identify three closest eligible neighbours and aggregate rainfall, available counts, ages and distances. Spark windows and joins process all records.",
  "\\textbf{时间编码。}令 $h$ 为当地小时（包含分钟的小数部分），$m$ 为月份。小时和月份使用下列周期编码，并将两组编码的四个乘积作为交互项，共形成八个特征。这样 23:00 与 00:00、12 月与 1 月在特征空间中保持接近。": "\\textbf{Cyclic time encoding.} Let $h$ be local hour including fractional minutes and $m$ the month. Use the cyclic encodings below and their four pairwise products, giving eight features. Adjacent 23:00/00:00 and December/January stay close in feature space.",
  "\\textbf{输入处理。}缺失值填补、缺失标志和站点类别词表仅基于完整训练期拟合。线性模型对站点进行 one-hot 编码，树模型使用类别索引；常量特征排除。Logistic 与 SVM 使用特征标准化。通过九个 Logistic 模型比较日历与站点、加入本站历史、再加入邻站历史三组特征。": "\\textbf{Input processing.} Fit imputation, missingness indicators and station vocabulary on the complete training period only. Use one-hot station encoding for linear models and a categorical index for trees; exclude constant features. Logistic and SVM standardize features. Nine logistic models compare calendar/station, added own history, and added neighbour history.",
  "模型 & 参数\\\\\\midrule": "Model & Parameters\\\\\\midrule",
  "\\textbf{训练策略。}比较自然分布、类别加权和受约束 SMOTE 三种策略。类别权重为 $w_+=N/(2P)$、$w_-=N/[2(N-P)]$。SMOTE 目标正例占比为 20\\%，正例总数最多为原始正例的 10 倍。每种策略均保留全部真实训练记录；合成观测仅添加到训练集。": "\\textbf{Training policies.} Compare natural distribution, balanced weighting and constrained SMOTE. Weights are $w_+=N/(2P)$ and $w_-=N/[2(N-P)]$. SMOTE targets 20\\% positives and caps their total at ten times the original count. Every policy retains all real training observations; synthetic rows enter training only.",
  "\\textbf{SMOTE 构造。}所有训练正例均可作为合成来源，按站点依次处理以控制内存。只在同站点、相同缺失模式、离散状态及雨量零值模式下检索最多五个近邻。对 16 个连续雨量特征用同一个系数插值，时间编码和其他离散信息保持原值；无合适近邻时复制原样本。": "\\textbf{SMOTE construction.} All training positives are available as donors, processed sequentially by station to bound memory. Search up to five neighbours with identical station, missingness, discrete profile and rain-zero pattern. Interpolate 16 continuous rain features with one coefficient, preserving calendar and discrete information; copy isolated donors.",
  "\\textbf{概率校准。}利用 2023 年上半年观测拟合 Platt 函数 $\\hat p=1/[1+\\exp(-(az+b))]$，其中 $z$ 为基础模型分数，约束 $a\\geq0$。该函数把 SVM 分数转换成概率，并调整类别加权或重采样后的概率尺度。若某组预测概率约为 10\\%，理想情况下该组实际正例比例也应接近 10\\%。": "\\textbf{Probability calibration.} Fit Platt calibration $\\hat p=1/[1+\\exp(-(az+b))]$ on 2023 H1, where $z$ is the base score and $a\\geq0$. This converts SVM scores to probabilities and adjusts weighting/resampling probability scales. A group predicted near 10\\% should ideally contain about 10\\% actual positives.",
  "\\textbf{评价指标。}ROC-AUC 评价正负例排序，随机排序的参考值为 0.5。PR-AUC 综合 Precision 与 Recall，适合正例较少的目标；随机排序的 Precision 参考水平约为正例比例 $\\pi$。两种面积均由 Spark 精确积分计算。Brier score 为预测概率与标签的均方差，越小越好。": "\\textbf{Evaluation metrics.} ROC-AUC measures ranking, with random-ranking reference 0.5. PR-AUC combines precision and recall and is useful for rare positives; random-ranking precision is approximately prevalence $\\pi$. Spark integrates both areas exactly. Brier score is mean squared probability error; lower is better.",
  "\\textbf{模型选择。}在 2023 年下半年比较各算法和雨量阈值的训练策略，主要按 PR-AUC 选择，Brier score 作为并列时的依据。概率判别阈值在该验证集上按 F1 选择，用于计算测试 Precision、Recall、F1 和 Accuracy；它与定义标签的雨量阈值不同。测试结果不参与模型选择。": "\\textbf{Model selection.} Compare policies for each algorithm/threshold on 2023 H2, select primarily by PR-AUC and break ties with Brier. Choose decision cutoffs by validation F1 for test precision, recall, F1 and accuracy; these differ from rainfall thresholds defining labels. Test results do not select models.",
  "\\textbf{实验说明。}预测实验正在以完整训练数据重新估计，本稿不列示尚未完成的模型评价结果。": "\\textbf{Experiment note.} Prediction experiments are being estimated on the complete training data; this draft does not report unfinished model evaluations.",
  "\\textbf{Spark 配置与分工。}采用单机双线程模式 \\texttt{local[2]}，shuffle partitions 为 8，default parallelism 为 2，不使用 GPU。Spark DataFrame/SQL 与窗口函数用于全量清洗、时间聚合、标签和特征构造；模型训练与评价使用 Spark ML 管线。gpushare 进程使用两个 CPU，并以低优先级串行执行各项训练。": "\\textbf{Spark configuration and responsibilities.} Use single-machine \\texttt{local[2]}, eight shuffle partitions, default parallelism two and no GPU. Spark DataFrame/SQL and windows process full-data cleaning, aggregation, labels and features. Spark ML pipelines train and evaluate models. On gpushare, two CPUs run training phases sequentially at low priority."
}

OVERRIDES = {
  "\\textbf{Task 2 特征启示（持续性与稀有事件）。}": [
    "\\textbf{Task 2 特征启示（持续性与稀有事件）。}实际特征包括最近读数、过去 15/30/60 分钟累计、近期雨势变化和连续有雨时长。只使用预测时已可获得的记录，缺测不补为无雨。下文逐阈值给出正例数、PR-AUC、Brier score 和校准曲线。",
    "\\textbf{Task 2 feature implications (persistence and rare events).} Implemented features include the latest reading, 15/30/60-minute totals, recent rainfall changes and wet-spell duration. Only records available at prediction time are used; missing readings are not dry readings. The following sections report positive counts, PR-AUC, Brier score and calibration curves separately by threshold."
  ],
  "\\textbf{AI 使用声明。}": [
    "\\textbf{AI 使用声明。}本项目使用 OpenAI Codex 辅助研究设计、数据处理和模型代码编写、结果核对与报告文字整理。报告中的降雨统计与模型评价结果均来自实际数据处理和模型运行；SMOTE 合成样本仅用于训练。",
    "\\textbf{AI-use declaration.} We use OpenAI Codex to assist study design, data-processing and model code, result checks and report editing. Reported rainfall statistics and model evaluations come from actual processing and model runs; SMOTE examples are used only for training."
  ],
  "\\textbf{小组成员。}": [
    "\\textbf{小组成员。}潘禹翔、侯佳成、吴嘉蕊。",
    "\\textbf{Group members.} Pan Yuxiang, Hou Jiacheng, Wu Jiarui."
  ],
  "\\item Meteorological Service": [
    "\\item Meteorological Service Singapore，\\href{https://www.weather.gov.sg/climate-climate-of-singapore}{Climate of Singapore}。",
    "\\item Meteorological Service Singapore, \\href{https://www.weather.gov.sg/climate-climate-of-singapore}{Climate of Singapore}."
  ],
  "\\item geoBoundaries": [
    "\\item geoBoundaries，\\href{https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/}{Singapore ADM0}；轮廓源自 URA 2016 年资料，ODbL 1.0。",
    "\\item geoBoundaries, \\href{https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/}{Singapore ADM0}, based on URA 2016 data, ODbL 1.0."
  ]
}

LABELS = {
  "新加坡降雨2017--2024：探索与预测": "Singapore Rainfall 2017--2024: Exploration and Prediction",
  "新加坡降雨 2017--2024 / Task 1 与 2": "Singapore rainfall 2017--2024 / Tasks 1 and 2",
  "2026 年 10 月 9 日 / 实证分析与预测": "9 October 2026 / Empirical analysis and prediction",
  "新加坡历史降雨：时空规律与概率预测": "Singapore rainfall: temporal-spatial patterns and prediction",
  "Task 1 与 2：2017--2024 年 Spark 探索与概率预测": "Tasks 1 and 2: Spark exploration and probability prediction, 2017--2024",
  "图 4：降雨频率的日内变化": "Figure 4: Diurnal variation in rainfall frequency",
  "图 6：雨量分布、半小时超越概率与降雨持续性": "Figure 6: Rainfall distribution, half-hour exceedance and persistence",
  "有雨区间的五分钟雨量分布": "Five-minute rainfall when wet",
  "后续半小时累计雨量超越概率": "Half-hour rainfall exceedance",
  "相邻区间的条件降雨概率": "Conditional probability of rain",
  "10. AI 使用声明与参考文献": "10. AI use and references",
  "降雨频率": "Rainfall frequency",
  "有雨区间平均雨量": "Mean wet-interval rainfall",
  "假正例率（FPR）": "False positive rate (FPR)",
  "真正例率（TPR）": "True positive rate (TPR)",
  "随机参考": "Random"
}

def apply(paragraphs, hashes, labels):
    for chinese, english in LINE_PAIRS.items():
        paragraphs[chinese] = english
        hashes[chinese] = hashlib.sha256(chinese.encode()).hexdigest()
    for prefix, (chinese, english) in OVERRIDES.items():
        paragraphs[prefix] = english
        hashes[prefix] = hashlib.sha256(chinese.encode()).hexdigest()
    labels.update(LABELS)
