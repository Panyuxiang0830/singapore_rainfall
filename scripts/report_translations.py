"""Translate the reviewed Chinese report, preserving all embedded plot data.

Content edits must be made in both languages. Unknown Chinese text stops the
English build so it cannot silently publish an outdated or partial translation.
"""
import re
import hashlib

PARAGRAPHS = {'\\textbf{范围与状态。}': '\\textbf{Scope and status.} This report covers Task 1. Task 2 model training and evaluation '
                     "remain to be completed. The final course report must include all group members\\' names and "
                     'student IDs.',
 '\\textbf{数据与方法。}': '\\textbf{Data and methods.} The eight NEA / data.gov.sg annual datasets contain 48,538,713 '
                     'source records and 48,538,627 unique valid observations from 91 station IDs. Spark performs '
                     'full-data cleaning and aggregation. Annual comparisons use 21 stable stations to reduce '
                     'changes caused by the monitoring network.',
 '\\textbf{主要结果。}': '\\textbf{Main findings.} Of all valid five-minute observations, 97.08\\% are dry. Eligible '
                    'diurnal samples show the highest rain frequency at 16:00--17:00 Singapore time. '
                    'Available-year monthly averages have the highest frequency in November and the lowest in '
                    'July, with varying eligible years and stations. In the fixed panel, duration-normalized '
                    'rainfall is lowest in 2019 and highest in 2021. Nearby stations generally have more strongly '
                    'correlated daily rainfall; eight years alone do not establish long-term climate change.',
 '\\noindent 表中各年均': '\\noindent Each year in this table uses the same 21 stations, with equal station weights. '
                     'Duration-normalized rainfall is neither an official annual total nor an actual daily total. '
                     'Eligible monthly and hourly samples can vary.\\clearpage\\section*{1. Data sources, fields '
                     'and cleaning}',
 '\\textbf{来源与字段。}': '\\textbf{Sources and fields.} Records contain observation and update timestamps, station '
                     'ID/name/device ID, coordinates, reading update timestamp, rainfall amount, type, unit and '
                     'date fields. There are no temperature, humidity or wind observations. API previews identify '
                     'fields and annual IDs; weather findings come from the complete annual files.',
 '\\textbf{来源追踪。}': '\\textbf{Source tracking.} During execution, each full CSV receives a manifest containing '
                    'its official dataset ID, API row count, retrieval time, byte count and SHA256 of the '
                    'original CSV bytes. The read row count is reconciled against the recorded annual API count.',
 '\\textbf{清洗步骤。}': '\\textbf{Cleaning.} Spark parses timestamps in Asia/Singapore. It accepts finite, '
                    'nonnegative mm readings with the expected five-minute type and valid station ID, timestamp '
                    'and CSV record. Timestamps are aligned only if within two seconds of the five-minute grid; '
                    'original epochs are retained. Station-time duplicates use the latest reading update, then '
                    'deterministic tie-breakers. Combined annual data are checked again for cross-file '
                    'duplicates.',
 '\\textbf{实际审计。}': '\\textbf{Cleaning results.} All 86 duplicates occur in 2017. There are no negative readings '
                    'or timestamps more than two seconds off-grid, and no cross-file duplicates. No station-year '
                    'with at least 90\\% coverage has all-zero rainfall. Missing intervals are never zero-filled. '
                    'Readings above 50 mm per five minutes are retained and flagged for sensitivity analysis; '
                    'none occur in this run. This is a screening value rather than an official quality threshold; '
                    'valid records can still contain sensor errors.',
 '设站点在给定时间段内': 'For a station and time period, let $N$ be the number of valid observations, $r_i\\geq0$ the '
               'rainfall amount, $W=\\sum_i\\mathbf{1}(r_i>0)$ the number of wet intervals, and $E$ the expected '
               'calendar slots. We use:',
 '预期日历槽数显式': 'Calendar denominators handle leap years. Low coverage can reflect a station operating for only part '
             'of a year. Intensity averages wet intervals only. For one station and period, normalized rainfall '
             'equals $288\\times\\text{frequency}/100\\times\\text{wet-interval intensity}$. This product '
             'generally does not hold after separate station averaging. Normalized rainfall is not an official '
             'daily or annual total and does not reconstruct missed storms.',
 '\\textbf{年度固定样本。}': '\\textbf{Fixed annual panel.} Of 91 station IDs across eight years, 21 satisfy all '
                      'criteria: observations in every year, annual coverage of at least 80\\% in each year, and '
                      'latitude/longitude spans no larger than $0.005^\\circ$. The other 70 are excluded only '
                      'from fixed-panel annual comparisons; their valid observations were not deleted during '
                      'cleaning. The 21 stations receive equal weights. The next page lists the station scope for '
                      'each analysis.',
 '\\textbf{月度与日内样本。}': '\\textbf{Monthly and diurnal samples.} Figures 2 and 4 further screen sufficiently '
                       'complete months or hourly strata from the 21-station pool. Monthly summaries average '
                       'stations equally, then available years equally. Each eligible year-month has 17--21 '
                       'stations, and each month has 6--8 available years. Requiring the same station-month to '
                       'qualify in all eight years leaves only May, June, July, September, October and December, '
                       'used as an additional comparison.',
 '\\textbf{空间比较。}': '\\textbf{Spatial samples.} Figure 1 uses 2024, which has 70 stations. Of these, 56 have '
                    'annual coverage of at least 90\\% and enter map colouring; the other 14 are shown in grey. '
                    'Figure 5 uses daily totals from the fixed 21 stations, requiring at least 95\\% daily '
                    'coverage and 365 common valid days per pair. The 80\\%, 90\\% and 95\\% completeness '
                    'thresholds are project choices.',
 '\\textbf{实际计算环境。}': '\\textbf{Actual computing environment.} Full-data analysis ran locally on an Apple Silicon '
                      'Mac with 10 logical CPUs and 16 GiB RAM: macOS 26.5.2 ARM64, Python 3.12.14, Java Temurin '
                      '17.0.20.1+1 and Spark 3.5.8.',
 '\\textbf{Spark 配置与分工。}': '\\textbf{Spark configuration and role.} Spark uses \\texttt{local[2]}: two worker '
                           'threads on one machine. Driver heap is 2 GiB, shuffle partitions 8 and default '
                           'parallelism 2. Annual processing is sequential and uses no GPU. Spark DataFrame/SQL '
                           'and windows perform cleaning, deduplication, temporal grouping and full-data '
                           'aggregation; small summaries are used for figures and daily-rainfall correlations.',
 '\\textbf{各项分析用了哪些站点？}': '\\textbf{Which stations enter each analysis?} Screening applies to a specific '
                          'comparison; it does not remove all stations outside the fixed panel.',
 '分析 & 使用范围': 'Analysis & Station scope & Screening rule\\\\\\midrule',
 '图 1：2024': 'Figure 1: 2024 station map & 56 of 70 stations that year & Annual coverage $\\geq90\\%$; 14 shown '
             'in grey\\\\',
 '图 2：月度变化': 'Figure 2: monthly patterns & Eligible subsets of the fixed 21 & 17--21 stations per eligible '
             'year-month\\\\',
 '图 3：年度变化': 'Figure 3: annual changes & The same 21 stations & Other 70 excluded from this comparison only\\\\',
 '图 4：日内变化': 'Figure 4: diurnal patterns & Eligible strata of the fixed 21 & Year-month-hour coverage '
             '$\\geq80\\%$\\\\',
 '图 5：距离': 'Figure 5: distance and correlation & Fixed 21 stations; 210 pairs & Daily coverage $\\geq95\\%$; '
           '$\\geq365$ shared days\\\\',
 '图 6 左：': 'Figure 6 left: wet amounts & All 91 stations across eight years & Wet five-minute observations '
           'only\\\\',
 '图 6 中、右：': 'Figure 6 middle/right: exceedance and persistence & Valid 2024 observations from all 70 stations & '
             'Six complete future slots or two adjacent slots\\\\',
 '\\textbf{清洗核对。}': '\\textbf{Cleaning checks.} Source row counts match API counts; 86 duplicate records were '
                    'removed, and missing slots are not zero-filled. The overall dry fraction uses all 91 '
                    'stations, rather than only the 21-station panel.',
 '\\noindent 2024 年有': '\\noindent In 2024, 56 of 70 stations have annual coverage of at least 90\\%. Coloured '
                       'points show individual gauge statistics; grey crosses indicate lower coverage. S33 '
                       '(Jurong Pier Road) has the highest rain frequency, 3.55\\%. No spatial interpolation is '
                       'performed. The geoBoundaries land outline is derived from URA 2016 data under ODbL 1.0 '
                       'and serves as a geographic reference.',
 '\\vspace{5mm}\\noindent 从年度': '\\vspace{5mm}\\noindent Eligible station-year-months from the fixed annual pool '
                                'require at least 80\\% coverage. Stations are averaged equally, then available '
                                'years equally. Mean frequency is highest in November (4.28\\%) and lowest in '
                                'July (1.85\\%). Mean wet-interval intensity is about 1.02 mm in March and 0.67 '
                                'mm in January. Frequency and intensity need not share the same seasonal pattern. '
                                'Each month has 6--8 available years and varying eligible stations. MSS '
                                'descriptions of wetter November--January conditions provide context rather than '
                                'proof of monsoon causation.',
 '\\noindent 相同 21 个': '\\noindent The same 21 stations enter every year, each with at least 80\\% annual '
                       'coverage. Mean rainfall per observed 24 hours ranges from 4.81 mm in 2019 to 7.84 mm in '
                       '2021. Frequency and wet-interval intensity change differently, helping explain annual '
                       'differences. Coverage normalization does not recover missed storms; mean coverage in 2018 '
                       'is about 82.9\\%, lower than in most years. These are descriptive annual comparisons, '
                       'without a long-term climate trend or significance claim.',
 '\\noindent 雨频最高的小时': '\\noindent Frequency peaks at 16:00--17:00 Singapore time. The right panel shows how the '
                       'cycle varies by month; June has elevated frequencies closer to morning and midday. '
                       'Eligible station-year-month-hour strata require at least 80\\% coverage and receive equal '
                       'weights; their composition can vary. MSS afternoon-rainfall descriptions provide '
                       'background. No additional temperature, wind or humidity observations are available to '
                       'explain mechanisms.',
 '\\noindent 每个点代表': '\\noindent Each point is a station pair: separation is on the horizontal axis and Pearson '
                     'correlation of daily totals on shared valid dates is on the vertical axis. The dark line '
                     'averages correlations within 5 km distance groups. The 21 stations form 210 pairs. Mean '
                     'correlation is 0.809 for 17 pairs no more than 5 km apart, and 0.336 for 12 pairs more than '
                     '30 km apart. \\textbf{Nearby stations generally show stronger daily-rainfall co-movement.} '
                     'This is an overall pattern, not an ordering that holds for every pair.',
 '\\textbf{站点并非独立。}': '\\textbf{Stations are not independent.} Nearby gauges may experience the same rainfall '
                      'event, and different pairs can share a station. Thus, 91 gauges are not 91 independent '
                      'weather samples, and the 210 correlations are not 210 independent samples. Shared-day '
                      'screening follows the criteria given earlier.',
 '\\noindent \\textbf{左图：': '\\noindent \\textbf{Left: rainfall amounts when wet.} The denominator contains '
                            '\\textbf{wet} five-minute records from all stations across eight years. Bars show '
                            'the proportions in rainfall ranges: 41.72\\% are in $(0,0.2]$ mm and 37.16\\% in '
                            '$(0.2,1]$ mm. Together, about 78.88\\% of wet intervals have no more than 1 mm. Dry '
                            'records are excluded from this panel; they represent 97.08\\% of all valid '
                            'observations.',
 '\\textbf{中图：': '\\textbf{Middle: how often a half-hour total exceeds a threshold.} For 6,374,476 complete 2024 '
                 'windows, the sum of six five-minute readings in $(t,t+30\\text{min}]$ defines the subsequent '
                 'total. The horizontal axis lists thresholds and the vertical axis shows the fraction strictly '
                 'exceeding each threshold. Totals exceed 0 mm in 6.36\\% of windows and 5 mm in 0.84\\%. Larger '
                 'thresholds are rarer. The five categories have equal display spacing, not equal numerical '
                 'spacing.',
 '\\textbf{右图：': '\\textbf{Right: does rainfall continue?} For adjacent five-minute intervals at the same station '
                 'in 2024, a wet interval follows a dry one in 1.04\\% of cases, versus 67.03\\% after a wet one. '
                 'Recent rainfall therefore motivates features for Task 2. The middle and right panels are '
                 'historical statistics, \\textbf{not trained-model performance}.',
 '\\textbf{AI 使用声明。}': '\\textbf{AI-use declaration.} OpenAI Codex assisted with planning, API inspection, '
                       'runtime configuration, PySpark cleaning and aggregation code, execution checks, figure '
                       'and dashboard code, report writing and translation, and interpretation. Weather values '
                       'come from the actual eight annual files processed with Spark; previews, synthetic data '
                       'and AI-generated numbers are not used as annual results. Both reports render the same '
                       'verified aggregate values as inline TikZ/PGFPlots data. Group members should review the '
                       'methods, source code and results, and declare AI assistance in the final submission.',
 '\\item NEA，': '\\item NEA, Historical Rainfall across Singapore, annual datasets 2017--2024: '
                '\\href{https://data.gov.sg/collections/2279/view}{data.gov.sg collection 2279}. Source and '
                'checksum records are generated during downloads.',
 '\\item Meteorological Service Singapore，': '\\item Meteorological Service Singapore, '
                                             '\\href{https://www.weather.gov.sg/climate-climate-of-singapore}{Climate '
                                             'of Singapore}. Used only for seasonal and diurnal context, not as a '
                                             'replacement for project observations or proof of causation.',
 '\\item geoBoundaries，': '\\item geoBoundaries, '
                          '\\href{https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/}{Singapore ADM0 '
                          'metadata}. The URA-derived outline represents 2016, under ODbL 1.0. Release metadata '
                          'and GeoJSON are retained in \\texttt{reports/assets/}; they are not a current '
                          'coastline survey.',
 '\\textbf{可复现材料。}': '\\textbf{Reproducibility.} Source and run instructions are in \\texttt{scripts/} and '
                     '\\texttt{README.md}; both language reports retain LaTeX sources. Separate images, CSV '
                     'tables, interactive pages and audit logs are not retained, as requested. Running the '
                     'analysis regenerates them. The current LaTeX embeds chart values and compiles without '
                     'downloading the eight annual datasets again.',
 '\\textbf{课程提交状态。}': '\\textbf{Submission status.} This is a Task 1 report, not the complete project submission. '
                      'Task 2, model evaluation, group names and student IDs must be added. At full-project '
                      'completion, push to \\url{https://github.com/Panyuxiang0830/singapore_rainfall.git}, '
                      'verify durable code/report/model backups, then clean only the project directory on the '
                      'server.'}

# Reviewed Chinese paragraph hashes: update these together with translations.
PARAGRAPH_SHA256 = {'\\textbf{范围与状态。}': 'a9769a5200938cc8caba1370e847f72c952b6cd83be1c1e99cdfdec2f8cc0778',
 '\\textbf{数据与方法。}': '35a32577f45c3d1d9c1d12fed9f4bc2df23368d0bb4e2d48cfb3e9fbfe47d073',
 '\\textbf{主要结果。}': 'ce4c7f706897d5118b1423888a9ec2ae80b1872ee6eb7dbebb74b10c590824a6',
 '\\noindent 表中各年均': 'aaff7dd80abf75d1e7b420a64bed20880011f3fafe277385625d7acfb265bb04',
 '\\textbf{来源与字段。}': 'abe15602fdba6f66bc997914fe584dbbc3d69feaae762133b4a62b11196ef7e2',
 '\\textbf{来源追踪。}': '239b8c2fbd180a95e42a39f46e3534c93b1c95bae7c89c675743582572e5efb0',
 '\\textbf{清洗步骤。}': 'f8f2d719190a8d82f3cd2e9af81c3680144fad68fac33f0bb876f45983fc6b05',
 '\\textbf{实际审计。}': '24af3af6bfdb9163e82159962a70285b9e2c3ffb2be46ec6a27bd4c4292b4a8a',
 '设站点在给定时间段内': 'a18a26a4b1e88157dd27c924e187b963025baa01ec705d333376f7f9e0ccd854',
 '预期日历槽数显式': '58b883c931f64311e6bd8bfba08bb96c393738d398605d89413d3c8f4d757eef',
 '\\textbf{年度固定样本。}': '14eb5c48cfecfc2a9810314beb6edd41e281d29f524f050c1eaee4fded76f044',
 '\\textbf{月度与日内样本。}': 'c6247c1affa3e1e5499303b01b5175ea70a2a252ee21aa4d8e1d0cda5089d936',
 '\\textbf{空间比较。}': '07635ee59d48617ae67d709b8010c9fdd5e320c932b153a5f95210ddb1c01128',
 '\\textbf{实际计算环境。}': 'd5fddc822249d92b4e2e5c8921e66d4356ce5c5c12b6ee8e181ce9ad9dc15664',
 '\\textbf{Spark 配置与分工。}': 'bbd6f3fc2946418662ebe6ad8a61abfdbfeacd84ea3b177f7c238b8fedaf875e',
 '\\textbf{各项分析用了哪些站点？}': '6ea5b8798251764ae2453c22943a99c73e4138c819525aebcdce04891674c4d8',
 '分析 & 使用范围': '8d1d64501ce7e302661c6a22fdf2d30830c346ca3e8a4dfa38aca4e0ef6f5556',
 '图 1：2024': '1b18dd1d4e8686d64e6722715e04da3f8f7e650015c9194be82dd07f5eada6b8',
 '图 2：月度变化': '58a29053fcb617dbd0dcac655587cd724870ea9fd9e322153a10aed3adaa29a4',
 '图 3：年度变化': 'ffe330ec0da9bf279bbefa4efb10eda43dd3551c05a45603fce73c978e19d81a',
 '图 4：日内变化': 'ab041f46e7101d1b29837f66a8efe5a15e97fa744b4fe140579c0b9a4b7e6ff1',
 '图 5：距离': '7404eb7adc382d05f00718162a92b7b293f35679c62ca56873cc7e4ca2d67303',
 '图 6 左：': 'e785c1225384e4a077f6f9b954a297d711dec8f9947b563aeed743b423180ec3',
 '图 6 中、右：': '77177478fc4524282926de9edc28b16a3b6407dcab935e9d92f93b1983a337c0',
 '\\textbf{清洗核对。}': '058b421c4f158f707574cd1cb2368edc427243302208dca5cedbbb32b843b7a0',
 '\\noindent 2024 年有': 'f242c2ba577ac9833de15e371454d8d3b6e38a930e5775d392948cbff67e606e',
 '\\vspace{5mm}\\noindent 从年度': '14c9b5a0ffe200e6c02dab5460894f137db3b8c4ac4dc0c070290bc5c85feadb',
 '\\noindent 相同 21 个': 'e2fa3a633e7ec5f2b92417955274677fb18bb72b8b4b949773ad85aaccd8d55c',
 '\\noindent 雨频最高的小时': 'c9fb08f00a930573fa1a6c93f45342bf25f35a8a3c23ec7c7a7ccf6ab49064ca',
 '\\noindent 每个点代表': 'b0d838977b052149e5db9cafa3251f6d0f3b87990e15aae4ce4d574d4b97c21c',
 '\\textbf{站点并非独立。}': '06a9e048090dbfe205baf0e9712b59031e8e9067b8867845909eb8d9a58bc2f4',
 '\\noindent \\textbf{左图：': 'f14a7b99cb76a66d05beabebfa5a9b522f8ada50264e1e6e36ee9dc0fb18c2d8',
 '\\textbf{中图：': 'a793523b2c3d2cbd972be5a576f821425278e8c4e49c0104322bbd18394de9bb',
 '\\textbf{右图：': '96291ef49ea4ed6533f3692229946f6ce4781ea394f445bae33eb3460b8d35a8',
 '\\textbf{AI 使用声明。}': '8945af4634177d0ac52c9911bcb2c08525b25a3cd1ada8baac0c4e59288b940b',
 '\\item NEA，': '4ff226c66b098d8a171f37deeea3bb7b0cc647d77f12edd74ae7cf9d7e7d22d6',
 '\\item Meteorological Service Singapore，': '9b160b9c65c582dc4c29deeacf0b31ab6f7da7df437943e0069739c81be6fd21',
 '\\item geoBoundaries，': 'adde2a05fcde847a892ddd8a0cf2eb6a2a84f2bbcbef99cc5e85bec64dedc2a3',
 '\\textbf{可复现材料。}': '993ab5ecf1e406f860f86c34fc82f68a89a8dac64d67176d813d0f52f1ed7ed8',
 '\\textbf{课程提交状态。}': '1aa8068fd3435580c2491d71a94cf6775e5f266e34515953ada0075ba9a5fc3e'}

LABELS = {'新加坡降雨2017--2024：Task 1中文报告': 'Singapore Rainfall 2017--2024: Task 1 Report',
 '项目小组（Codex协助）': 'Project group (Codex assistance)',
 '新加坡降雨 2017--2024 / Task 1': 'Singapore rainfall 2017--2024 / Task 1',
 '中文技术报告': 'Technical report (English)',
 '2026 年 10 月 8 日 / 真实数据全量分析': '8 October 2026 / Full-data analysis',
 '新加坡历史降雨的时间与空间规律': 'Temporal and spatial patterns in Singapore rainfall',
 'Task 1：2017--2024 年全量 Spark 探索分析': 'Task 1: Full-data Spark exploration, 2017--2024',
 '1. 数据来源、字段与清洗': '1. Data sources, fields and cleaning',
 '2. 指标定义与公平比较': '2. Metrics and comparable samples',
 '3. 计算环境与各项分析的样本范围': '3. Computing environment and station scope',
 '4. AI 使用声明、来源与复现': '4. AI use, sources and reproducibility',
 '图 1：2024 年观测站降雨地图': 'Figure 1: Station rainfall maps in 2024',
 '图 2：一年内的月度规律': 'Figure 2: Monthly patterns within a year',
 '图 3：固定站点的年度变化': 'Figure 3: Annual changes in the fixed panel',
 '图 4：一天内何时更容易下雨': 'Figure 4: When during the day does it rain?',
 '图 5：站点距离与日雨量的相关性': 'Figure 5: Distance and daily-rainfall correlation',
 '图 6：雨下多大、半小时超阈值与短时持续性': 'Figure 6: Rainfall amounts, half-hour exceedance and persistence',
 '平均覆盖率': 'Mean coverage',
 '合格样本平均覆盖率': 'Mean eligible-sample coverage',
 '雨量 / 观测24h': 'Rain / observed 24h',
 '湿区间雨强': 'Wet-interval intensity',
 '日历天数': 'calendar days',
 '覆盖率至少90': 'Coverage at least 90',
 '有有效观测': 'Any valid observations',
 '有雨五分钟区间': 'Wet five-minute intervals',
 '下雨的频率': 'Rain frequency',
 '下雨时的强度': 'Intensity when wet',
 '经度（东经）': 'Longitude (E)',
 '纬度（北纬）': 'Latitude (N)',
 '雨频与雨强衡量不同现象': 'Frequency versus intensity',
 '雨频最高与最低的各10站': 'Ten highest and ten lowest frequencies',
 '有雨区间': 'Wet intervals',
 '湿五分钟': 'wet five-minute interval',
 '月度降雨频率': 'Monthly rain frequency',
 '月度湿区间雨强': 'Monthly wet-interval intensity',
 '月份（新加坡时间）': 'Month (Singapore time)',
 '可用年份平均': 'Available-year mean',
 '日内雨频变化': 'Diurnal rain frequency',
 '小时（新加坡时间）': 'Hour (Singapore time)',
 '月份与小时的雨频': 'Frequency by month and hour',
 '站点年度覆盖率': 'Station-year coverage',
 '监测网络随年份变化': 'Monitoring network by year',
 '距离与日雨量相关性': 'Distance and daily-rainfall correlation',
 '站点距离（km）': 'Station separation (km)',
 'Pearson相关系数': 'Pearson correlation',
 '左：下雨时，每五分钟下多少？': 'Wet-interval rainfall amounts',
 '中：后续30分钟累计超过多少？': 'Subsequent 30-minute exceedance',
 '右：前一段下雨，下一段呢？': 'Rain after a dry or wet interval',
 '五分钟雨量（mm）': 'Five-minute rainfall (mm)',
 '有雨记录中的占比': 'Share of wet records',
 '窗口比例': 'Windows exceeding threshold',
 '当前区间有雨': 'Next interval wet',
 '前一段无雨': 'Previous interval dry',
 '前一段有雨': 'Previous interval wet',
 '阈值（mm）': 'Threshold (mm)',
 '浅色': 'Light',
 '深色': 'dark',
 '灰色为无合格观测': 'grey: no eligible sample',
 '源记录数': 'Source rows',
 '删除重复数': 'Duplicates removed',
 '偏离网格超过2秒': 'Off-grid over 2 s',
 '负雨量数': 'Negative readings',
 '资料来源': 'Sources',
 '观测24h': 'observed 24h',
 '年份': 'Year',
 '站点数': 'Stations',
 '雨频': 'Wet frequency',
 '覆盖率': 'Coverage',
 '雨量': 'Rainfall',
 '（': ' (',
 '）': ')',
 '：': ': ',
 '；': '; ',
 '，': ', ',
 '。': '.'}


PARAGRAPHS = {**{'\\textbf{Task 2 特征启示（站点）。}': '\\textbf{Task 2 feature implications (stations).} Candidate features '
                               'include station ID and coordinates to represent station differences. '
                               'Task 2 should screen windows for historical-feature and future-label '
                               'completeness, rather than automatically restrict training to the '
                               '21-station annual panel.',
 '\\textbf{Task 2 特征启示（月份）。}': '\\textbf{Task 2 feature implications (months).} Retain cyclic month or '
                               'day-of-year features and test their contribution by ablation. Singapore '
                               'is affected by monsoons but has no sharply distinct wet and dry '
                               'seasons. Less frequent rain does not mean no rain, and frequency is '
                               'different from accumulated rainfall.',
 '\\textbf{Task 2 特征启示（年份）。}': '\\textbf{Task 2 feature implications (years).} Initially omit '
                               'calendar-year identifiers from the main model to reduce memorization of '
                               'individual years. NEA documented dry conditions and a strong positive '
                               'Indian Ocean Dipole in 2019; this is external climate context rather '
                               'than causal evidence from these data. Use chronological training, '
                               'validation and test splits, and report performance across years.',
 '\\textbf{Task 2 特征启示（小时与月份）。}': '\\textbf{Task 2 feature implications (hour and month).} Encode '
                                  'Singapore local hour cyclically and consider month--hour '
                                  'interactions: the strength and timing of the afternoon peak vary by '
                                  'month. Logistic regression and linear SVM need explicit interaction '
                                  'features; trees can learn such combinations. Their predictive '
                                  'contribution must still be tested by ablation.',
 '\\textbf{Task 2 特征启示（邻站）。}': '\\textbf{Task 2 feature implications (neighbours).} Start with '
                               'neighbour rainfall available at prediction time: distance-weighted '
                               'means, maxima, wet fractions and counts of available neighbours. Defer '
                               'graph neural networks. Daily correlation motivates candidate features '
                               'but does not establish their usefulness for half-hour prediction; '
                               'compare models with and without neighbour features.',
 '\\textbf{Task 2 特征启示（持续性与稀有事件）。}': '\\textbf{Task 2 feature implications (persistence and rare '
                                     'events).} Candidates include recent five-minute readings, '
                                     '15/30/60-minute totals, recent rainfall changes and wet-spell '
                                     'duration. Use only records available at prediction time and do '
                                     'not treat missing readings as dry. For rarer high-threshold '
                                     'events, report positive counts, PR-AUC, Brier score and '
                                     'calibration curves separately by threshold. These are planned '
                                     'model experiments.',
 '\\item NEA，\\href': '\\item NEA, '
                      '\\href{https://www.nea.gov.sg/media/news/news/index/2019-is-singapore-s-joint-warmest-year-on-record}{Review '
                      'of the climate in 2019}. External context on dry conditions and the positive '
                      'Indian Ocean Dipole, not an input feature in the proposed model.'}, **PARAGRAPHS}
PARAGRAPH_SHA256.update({'\\textbf{Task 2 特征启示（站点）。}': 'e29b985ff6c292f9af121dedaf55c28d8a4b12c8f1376fbf56fa7ad49c9a0aa2',
 '\\textbf{Task 2 特征启示（月份）。}': '360ce371dc2f63c0e78d59445fe9a5a490ec3a1aa5e46278f4b2b760c797555a',
 '\\textbf{Task 2 特征启示（年份）。}': '86c3477d03ef2095945aef52be8021ee16de908dbdc6dbd2fcdbea59cda2f96d',
 '\\textbf{Task 2 特征启示（小时与月份）。}': '309ff9fecfcac707677fa13945d0e3446bd5de38024b531bff300776fd81f215',
 '\\textbf{Task 2 特征启示（邻站）。}': '73243b1861454a7017284be0e3c306db64f7b0548789110c7fa5fc9c3f365135',
 '\\textbf{Task 2 特征启示（持续性与稀有事件）。}': 'b98afc2aa974c874310d1c2ef870dfc3206dd46fde47c9c5b6def81b5b2f3e3e',
 '\\item NEA，\\href': 'df1d07e29f187ee67b9fdf222e92bd9e54ba77758552f278cb6e044b1db864e8'})

PARAGRAPHS.update({'\\textbf{中图：': '\\textbf{Middle: how often a half-hour total exceeds a threshold.} For 6,374,476 '
                 'complete 2024 windows, the sum of six five-minute readings in '
                 '$(t,t+30\\text{min}]$ defines the subsequent total. Sums are rounded to nine '
                 'decimal places to avoid floating-point errors at exact thresholds. The '
                 'horizontal axis lists thresholds and the vertical axis shows the fraction '
                 'strictly exceeding each threshold. Totals exceed 0 mm in 6.36\\% of windows and '
                 '5 mm in 0.83\\%. Larger thresholds are rarer. The five categories have equal '
                 'display spacing, not equal numerical spacing.',
 '\\textbf{实际计算环境。}': '\\textbf{Actual computing environment.} The main Task 1 analysis previously '
                      'ran on an Apple Silicon Mac with 10 logical CPUs and 16 GiB RAM: macOS '
                      '26.5.2 ARM64, Python 3.12.14, Java Temurin 17.0.20.1+1 and Spark 3.5.8. On '
                      '9 October 2026, Task 2 labels and the half-hour exceedance statistics in '
                      'Figure 6 were checked on gpushare using Python 3.11.10, Azul Zulu Java '
                      '17.0.20.1 and Spark 3.5.8. The container quota is 42 CPUs and approximately '
                      '667.68 GiB RAM; project execution uses only two CPUs, a 2 GiB driver heap '
                      'and a sampled 4 GiB process-tree RSS stop bound.'})
PARAGRAPH_SHA256.update({'\\textbf{中图：': 'a408a0e24cf638412147b494279178e9707e0a5e1b75ec2ab9513b0c1c19052d',
 '\\textbf{实际计算环境。}': 'bb871788560157652ae502190a70fc538a34f07f28c3d3955f3172b7b5b3c490'})

from report_task2_content import apply as apply_full_report_content
apply_full_report_content(PARAGRAPHS, PARAGRAPH_SHA256, LABELS)
from report_tuning_content import apply as apply_tuning_content, SECTION_CN, SECTION_EN
apply_tuning_content(PARAGRAPHS, PARAGRAPH_SHA256, LABELS)

from report_grid_content import apply as apply_grid_content, translate_blocks
apply_grid_content(PARAGRAPHS, PARAGRAPH_SHA256, LABELS)

def translate(source):
    source = translate_blocks(source)
    source = source.replace(SECTION_CN, SECTION_EN)
    lines = []
    for text in source.splitlines():
        for prefix in sorted(PARAGRAPHS, key=len, reverse=True):
            english = PARAGRAPHS[prefix]
            if text.startswith(prefix):
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest != PARAGRAPH_SHA256[prefix]:
                    raise ValueError('Chinese content changed; review its English translation and update its hash: ' + prefix)
                text = english.replace("\\'", "'")
                break
        lines.append(text)
    translated = '\n'.join(lines) + '\n'
    for chinese in sorted(LABELS, key=len, reverse=True):
        translated = translated.replace(chinese, LABELS[chinese])
    unknown = [line for line in translated.splitlines() if re.search(r'[\u4e00-\u9fff]', line)]
    if unknown:
        raise ValueError('Update English translations for these changed Chinese lines:\n' + '\n'.join(unknown))
    # Keep long English sample-scope cells within the landscape page.
    translated = translated.replace('\\begin{tabular}{lll}', '\\begin{tabular}{p{7.0cm}p{6.8cm}p{10.5cm}}')
    return translated
