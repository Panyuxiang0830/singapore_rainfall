# Singapore rainfall, 2017–2024

Task 1 explores rainfall across time and monitoring stations with PySpark.
Task 2 predicts next-30-minute exceedance probabilities for 0, 5 and 25 mm.
Both tasks and the 36-candidate full-feature model/imbalance comparison are complete.
Task 2 labels, features, training, calibration and evaluation ran on gpushare on
9 October 2026. The main Task 1 analysis used the authorized local Spark fallback;
its half-hour exceedance statistics were subsequently checked on gpushare.

The complete bilingual reports retain the legacy filenames `Task1_Report_CN` and
`Task1_Report`, but cover **both tasks**. Group members: 潘禹翔 (Pan Yuxiang),
侯佳成 (Hou Jiacheng), 吴嘉蕊 (Wu Jiarui). Report student IDs are omitted as requested.
Source, README, standalone LaTeX, PDF reports and all trained native pipelines
are retained. Raw datasets, generated summary tables, chart images, dashboards,
audit logs and temporary caches are excluded from the deliverable and Git.

## Data

Official collection: https://data.gov.sg/collections/2279/view

The eight annual datasets contain 48,538,713 source records according to
the API metadata inspected for this project. Fields describe observation
and update times, station IDs/names/devices/coordinates, rainfall, type and
unit. They do not contain wind, humidity or temperature.

`inspect_data.py` discovers annual IDs and checks a bounded preview. Preview
rows are sequential, not a representative sample. Full-year downloads are
performed by `download_year.py`; manifests record original-byte SHA256,
retrieval time and API row counts. Do not commit raw annual files.

## Environment and resource policy

The project was initialized in `/home/research_pyx/singapore` on gpushare,
but intermittent SSH/JupyterLab failures prevented reliable access. The
user authorized a local fallback. The verified full-data Task 1 run uses
an Apple Silicon Mac with 10 logical CPUs and 16 GiB RAM, macOS 26.5.2,
Python 3.12.14, Java Temurin 17.0.20.1+1 and Spark 3.5.8.

Spark uses **local[2]**, JVM ActiveProcessorCount=2, 2 GiB driver heap and
eight shuffle partitions; BLAS/OpenMP use one thread. This is one machine
with two Spark worker threads, not a multi-node cluster. macOS does not
expose Linux CPU affinity, so no affinity guarantee is claimed. No GPU is
used. All full-data transformations remain in Spark.

Java 17 and PySpark 3.5.8 are installed under `.runtime/`, with no global
package changes. Bundled pandas/numpy and project-local extra packages
are used. The historical configuration is documented here and in the retained reports.
The Spark code records actual versions and hardware in reports/environment.json
on each new run; that generated record is not distributed.
The preliminary gpushare instance quota was 42 CPU cores and
716,915,937,280 bytes of RAM; that is not the execution environment of the
verified Task 1 results.

The runner starts one subprocess at a time, samples its process-tree RSS
and filesystem free space every five seconds, and stops that subprocess
if RSS exceeds 4 GiB or free disk falls below 5 GiB. This is a monitoring
guard, not a kernel-enforced memory/disk quota. Priority and CPU affinity
must be set with the launch command below. Downloads default to 8 MiB/s.
All temporary files stay in the project directory.

## Run

Python 3.12 is recommended. On a fresh environment, create a project
environment and install the pinned requirements. `configure_runtime.py`
selects checksum-pinned official Java metadata for macOS ARM64 or Linux
x86_64. `setup_runtime.py --java-only` installs Java inside the project.
The source package does not include generated Parquet or large binaries.

```bash
cd singapore_rainfall
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/configure_runtime.py
.venv/bin/python scripts/setup_runtime.py --java-only
.venv/bin/python scripts/inspect_data.py --limit 200
nice -n 19 .venv/bin/python -u scripts/run_task1.py
MPLCONFIGDIR=work/matplotlib nice -n 19 .venv/bin/python scripts/plot_task1.py
.venv/bin/python scripts/build_dashboard.py
.venv/bin/python scripts/validate_results.py
.venv/bin/python scripts/build_task1_report_cn.py
.venv/bin/python scripts/build_task1_report.py
```

On Linux, prefix the Spark command with `taskset -c 0,1` and `ionice -c 3`
where those CPU IDs are available. The runner keeps Java/Spark scratch in
the project and monitors only its own subprocesses. No recurring service
or automatic server restart is configured.

The runner first checks future-only time windows, gaps, strict thresholds
and timestamp tolerance with a small independent Spark example. It then
downloads and cleans each annual dataset, reconciles CSV counts with API
metadata, and removes its reproducible raw gzip only after successful
Parquet output and a matching audit. Completed annual stages are skipped
on reruns. Interrupted downloads are restarted, not appended.

For a long run, use a terminal multiplexer or a one-off `nohup` command
with output redirected to `work/task1.log`. Before starting another run,
verify the existing PID, command line and working directory to avoid
duplicate jobs. An SSH failure does not establish whether a detached job
is still running.

## Generated outputs and analysis

These products are created during execution and are excluded from Git.
They are not currently included in this cleaned directory. Only the PDF/LaTeX
reports and static map inputs remain under reports/.

- `reports/downloads/`: source provenance and byte checksums.
- `reports/quality/`: annual cleaning audits and count reconciliation.
- `reports/resource_usage.jsonl`: sampled process RSS and disk reserve.
- `reports/tables/`: small Spark aggregates and derived comparisons.
- `reports/task1_overview.json` and `task1_findings.json`: actual statistics.
- `reports/figures/`: nine analytical figures, PNG and editable SVG.
- `reports/task1_dashboard.html`: standalone viewer with maps for every
  year, selectable metrics, station history and a complete station table.
- `reports/Task1_Report.pdf`: complete English report covering Tasks 1 and 2.
- `reports/Task1_Report_CN.pdf`: synchronized complete Chinese report.

Read `TASK1_METHOD.md` for cleaning and interpretation. Missing intervals
are never treated as zero rainfall. Annual comparisons use the same
stations, with at least 80% coverage in every year and a coordinate-span
screen. The 2024 map colours stations with at least 90% coverage. Frequency
and intensity are distinguished. Rainfall per observed 24 hours normalizes
by available observation duration; it is not an official annual total.

Monthly plots use >=80%-covered station-year-months from the annual panel;
membership varies. Gaps are displayed explicitly. The stricter all-eight-year
month comparison is also included, but only six months qualify.

Source records have not undergone official climate quality control.
Readings over 50 mm per five minutes are retained and examined in a
sensitivity table. The map boundary is a 2016 geoBoundaries/URA outline,
licensed under ODbL 1.0, and used only as a geographic reference. Eight
years are insufficient to establish a long-term climate trend.

## AI use

OpenAI Codex assisted with planning, API inspection, cleaning and analysis
code, runtime setup, execution checks and visualization code. Results
must be produced by the real Spark run and checked against saved tables;
no generated or preview data may be presented as full-dataset findings.
Group members must review the methodology and declare AI assistance in
the final report. Codex also assisted with Task 2 feature engineering, model training code,
execution and evaluation. Results come from recorded runs, not generated
example data.

## Backup and eventual cleanup

Git remote: https://github.com/Panyuxiang0830/singapore_rainfall.git

At completion of the entire project, push source code, documentation,
standalone LaTeX sources and final PDF reports. Do not retain generated tables,
separate chart images, dashboards or audit logs in the submission. The saved native pipelines are small and tracked directly under `models/`;
retain their metadata/data/CRC files together when copying or loading.
Verify all remote assets before deleting only this project directory.
**Do not delete the server project after Task 1.**

## Synchronized Chinese and English LaTeX reports

`reports/Task1_Report_CN.tex` is a standalone source with Chinese text,
formulas and nine inline TikZ/PGFPlots figure groups covering both tasks. Its compiled PDF is
`reports/Task1_Report_CN.pdf` (20 pages). Compilation and editing instructions
are in `reports/LATEX_CN_README.md`. The existing source is standalone: it
can compile without deleted figures or CSV files. The current source is the reviewed complete report. Edit it in place, update the
reviewed translation in `report_task2_content.py` or `report_translations.py`,
and run `python scripts/build_task1_report.py` to synchronize English.
The legacy Chinese Task 1 generator refuses to overwrite a source containing Task 2.

The English source is `reports/Task1_Report.tex`, translated from the reviewed
Chinese source by `scripts/build_task1_report.py`. Its compiled output is
`reports/Task1_Report.pdf` (20 pages). Content changes must be made in both
languages: update `report_translations.py` or `report_task2_content.py` and their reviewed content.
The build stops if a reviewed Chinese paragraph changes without this review.
Both reports embed the same chart values; neither needs deleted CSV files to
compile. The English builder now creates LaTeX rather than a ReportLab PDF.

Compile the saved sources in the built-in LaTeX editor, or export using an
already available Tectonic/XeLaTeX installation:

```bash
tectonic --untrusted --outdir reports reports/Task1_Report_CN.tex
tectonic --untrusted --outdir reports reports/Task1_Report.tex
```

To update only the English source from the current Chinese report, run
`python scripts/build_task1_report.py` without rerunning the full Spark pipeline.
Do not run the Chinese generator for a prose-only edit: it rebuilds the source.

Original report Figures 2 (station ranking), 4 (year-month heatmaps) and 7
(coverage/network plots) are omitted as requested. Retained Task 1 report figures
are numbered 1--6; Task 2 adds Figures 7--9. The exploratory plotting code may still generate optional
charts; they are not retained in the report or distributed as image files.

## Execution preference for subsequent work

As requested, subsequent full-data analysis and Task 2 training should run on
gpushare in `/home/research_pyx/singapore`. Start within the existing two-thread,
2 GiB driver-heap and 4 GiB monitored process-tree bounds, with no GPU or global
installation. Inspect any existing project job before restarting.

The main Task 1 results were produced by the local fallback. Both reports now
distinguish that execution from the server label preparation and Figure 6
half-hour-statistics check. Update the actual configuration and results after
any further verified server run, keeping both report languages synchronized.
A single gpushare VM using local[2] is not a multi-node Spark cluster.

## Task 2: label-preparation stage

The server project remains accessible at `/home/research_pyx/singapore`. Its
physical directory is now `/hy-tmp/singapore_rainfall_pyx`, on the data disk.
All 1,533 copied files (1,250,203,771 bytes) were SHA256-verified before removing
the system-disk copy. No other project was altered. Java and Python packages
remain project-local; GPU use is disabled.
The actual server label stage uses Python 3.11.10, Spark 3.5.8, Azul Zulu Java
17.0.20.1 and psutil 6.0.0. The container has a 42-CPU quota and a
716,915,937,280-byte memory limit; project execution is restricted to two CPU
IDs, 2 GiB JVM heap and the 4 GiB sampled process-tree RSS stop bound.

From the server project directory:

```bash
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 -u scripts/run_task2_labels.py
```

The runner limits affinity to two available CPU IDs, uses the existing 2 GiB
Spark driver heap and 4 GiB process-tree RSS monitor, and leaves 5 GiB disk
space free. It runs each year sequentially and uses a lock to prevent duplicate
label jobs. It reuses complete audited annual Parquet inputs.

Targets sum readings at t+5, t+10, ..., t+30 minutes for each station. Exactly
six distinct slots are required; missing readings invalidate the label. The
current observation is excluded, and exceedance uses strict `>` rather than
`>=`. Sums are rounded to nine decimal places (mm) to avoid binary floating-point
artifacts at exact threshold values; this rule is covered by a Spark fixture. Reading labels are provisionally treated as interval ends; the physical
interval convention and publication latency must be reviewed before modelling.
Stored reading update timestamps are diagnosed, but are not automatically
interpreted as first publication timestamps.

Existing observation times are prediction anchors. The planned chronological
splits are 2017--2022 for training, 2023 for validation and 2024 for testing.
Targets ending at or after the next split's start are purged. Padding from the
next annual file preserves horizons across year boundaries within training.
The final six slots of 2024 cannot be labelled without 2025 data.

Working targets are written to `data/task2/labels/`; threshold diagnostics
for 0, 5, 25 and 50 mm are written under `work/task2/`. These are resumable
working files excluded from Git and the desktop submission copy. Threshold
selection is based on training-period support; 2024 was previously explored
for Task 1 and is not described as wholly untouched. Counts of positive days
are coverage diagnostics, not counts of independent storms.

The completed label stage checked all eight annual inputs: 48,538,627 candidate
anchors yield 47,636,035 valid targets, with 901,794 incomplete windows and
798 windows purged at split boundaries. Training has 34,542,285 targets.
Its positive counts for thresholds 0, 5, 25 and 50 mm are respectively
2,074,430; 279,801; 19,787; and 488. Based on training support, the provisional
main experiments use 0, 5 and 25 mm. The 50 mm experiment is deferred
and has no label column in the feature dataset. These amounts are modelling thresholds, not official half-hour
weather categories. Sampled peak process-tree RSS was 2.158 GiB. Feature
engineering must first handle the diagnosed observation/update-time lag;
stored update times alone do not prove first publication times.

Model comparison will prioritize ROC-AUC and PR-AUC, with Brier score and
calibration curves for probability quality. Precision, recall, F1 and accuracy
will be supplementary at a validation-selected probability cutoff. Planned
experiments compare unweighted training, class weights and constrained
SMOTE; validation/calibration/test retain natural class proportions.
Graph neural networks are deferred.

## Task 2: feature engineering

Run on gpushare with the same resource bounds as label preparation:

```bash
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 -u scripts/run_task2_features.py
```

Use `--years 2017 --months 1` for a single-month pilot, `--check-only` for
meaningful Spark fixtures, or rerun normally to resume completed months.
This stage does not train any model. Each station/time has one feature row
and three separate target columns for >0, >5 and >25 mm; thresholds do not
multiply feature rows. The continuous future total is retained for verification
but is excluded, along with all target and audit columns, from model inputs.

`task2_features.py` is the shared feature definition for training and subsequent
inference. Full observation transformations, regular-calendar windows and
neighbor joins use Spark SQL/DataFrames. Only the small station-coordinate
metadata table leaves Spark for deterministic nearest-neighbor geometry.
The explicit model input lists contain 46 numerical features and station ID
as one categorical input. Do not assemble every numeric column automatically.

- Calendar: cyclic hour and month plus four hour/month interactions, using
  Asia/Singapore time. No raw year or arbitrary station-ID ordering.
- Station: first known coordinates and a recent geometry-change flag.
- Own history: latest available five-minute reading, age and wet indicator;
  exact lags of 5/10/15/30/60 minutes with their availability indicators; 15/30/60-minute totals ending at the
  latest available reading, their valid-slot counts, 30-minute wet fraction
  and maximum, five-minute change, 15-minute trend, and a wet run capped at
  twelve slots with a cap indicator. A gap stops the run. Totals require all
  expected slots; observed wet fraction/max carry their observation count.
- Spatial: the nearest three other stations among those whose coordinate
  metadata is already available. Use their same-prediction-time snapshots,
  latest-rain mean/max and wet fraction, complete 30-minute total mean/max,
  available station counts, mean reading age and mean distance. If first-known
  coordinates differ from the latest available coordinates by >0.005 degree,
  disable that station's spatial contribution rather than assert a current
  accurate neighbor geometry. The graph contains no target information.

For a prediction at t, a retained reading is eligible only if its observation,
rounded calendar slot and stored update times are all <=t. Calendar grids
prevent missing observations from shifting lags. The lookback is 90 minutes;
a reading older than 30 minutes is unavailable for latest/rolling/spatial rain
features. Totals end at the latest eligible reading (not necessarily at t),
so its age is essential. Missing values remain null and availability counts
remain zero; they are not silently converted to no rain. Imputation, categorical
encoding and scaling will be fitted on training data only during modelling.

This is a conservative simulation using retained archive versions: update
timestamps do not prove first publication, and revisions absent from the
archive cannot be reconstructed. No near-real-time operational performance
is claimed. Monthly integrity checks enforce one row per target, unchanged
label counts, update/observation times <=t and at most three available neighbors.
The fixtures also change unpublished/future rain and verify identical current
features, check missing totals and exact lags, and exclude a not-yet-known
station from historical neighbor rankings.

Generated feature Parquet and metadata/statistics remain under server `data/`
and `work/`, excluded from Git and the desktop submission copy. Monthly
snapshot intermediates are deleted after successful checks. Annual observation
inputs are temporarily partitioned by month to avoid repeated archive scans,
and those temporary input files are removed after each requested annual job. Saved targets
and monthly features are resumable working data for the next training stage.

Feature-stage completion: all 96 months have been verified, totaling
47,636,035 unique station/time rows: 34,542,285 train; 6,719,274 validation;
6,374,476 test. Every annual row and threshold-positive count matches the
prepared labels. All monthly as-of/neighbor integrity checks passed. Recent
own readings are available for 45,830,161 rows, and at least one neighbor for
45,591,491 rows. The retained feature working files occupy 991,186,114 bytes
(about 0.92 GiB). Sampled annual-job peak process-tree RSS was 2.879 GiB;
minimum free disk was 29.956 GiB. No feature jobs remain active, and temporary
monthly snapshots and partitioned observation inputs have been removed.
The next stage fits preprocessing on training only and compares regularized
logistic baselines before the SVM/tree experiments.


## Task 2: first logistic feature comparison

```bash
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 -u scripts/run_task2_logistic.py
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 -u scripts/run_task2_logistic.py --check-only
```

The first baseline uses a deterministic, label-independent 5% station/time
sample from 2017--2022, identical across all classifiers and feature groups.
It preserves natural proportions in expectation; exact class counts are saved
with the models. This is a sampled training experiment, not full-data training.
Validation (2023) and test (2024) retain every eligible row and natural class
proportions. Temporal windows have already been purged across these splits.
Task 1 explored 2024, so it is not described as wholly untouched data.

Three cumulative groups test the hypotheses from Task 1: calendar/station,
plus own recent history, plus nearest-station features. Each group fits one
binary classifier per >0, >5 and >25 mm threshold, using Spark MLlib
LogisticRegression with L2 regParam=0.001, elasticNetParam=0,
standardization=True, maxIter=80, tol=1e-6. This regularization strength is
fixed for the initial ablation; no optimal hyperparameter claim is made.
No SMOTE, class weights or negative undersampling are used in this first
baseline. Those are separate experiments to compare after this reference.

Preprocessing fits only the training sample: train-constant/all-null numeric
values are omitted, informative missingness indicators are retained, observed
numeric means impute remaining nulls, and station IDs are one-hot encoded
with an unknown-station category. MLlib applies training-based scaling during
LR optimization. Input columns come from explicit feature lists, never future
totals/labels/audit timestamps. Each trained PipelineModel includes fitted
preprocessing and is reloaded to verify identical probabilities.

ROC-AUC and PR-AUC use Spark BinaryClassificationEvaluator with numBins=0,
without score downsampling. Spark's trapezoidal PR area is not sklearn's
average precision. Training, validation and test also include Brier score,
log loss, mean probability, reliability points and a constant training-rate
baseline. Reliability curves diagnose raw logistic probabilities; no post-hoc
calibrator has been fitted. The supplementary classification cutoff maximizes
F1 over a fine logarithmic probability grid on 2023 only, then stays fixed for
2024. Feature-group selection uses validation PR-AUC only. Multiple 2024
ablation results are descriptive comparisons, not repeated test-set tuning.
Overlapping targets/stations mean rows are not independent observations.

Saved models live in `models/logistic_baseline/{group}/gt_{threshold}mm/`;
`manifest.json` stores fitted feature metadata, settings and the group selected
on validation for each supported rainfall threshold. Model binaries are
excluded from ordinary Git tracking by the project-wide models/ ignore rule,
but are retained in the desktop deliverable as required for submission.
Intermediate samples, metric JSON and score caches stay in ignored server
work storage. The runner is sequential, resumable and uses the existing
2-CPU/2-GiB heap/4-GiB sampled process-tree RSS and disk guards.

Score a prepared historical feature row with a saved model:

```bash
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 scripts/run_task2_logistic.py \
  --predict-station STATION_ID --predict-time '2024-01-01T12:00:00+08:00' --threshold 5
```

The scorer accepts only the three trained rainfall thresholds and reads the
station/time row from prepared causal archive features. Real-time history
retrieval and arbitrary unseen thresholds are not implemented by this entry
point. Independent threshold classifiers can produce probabilities that fail
p(>25)<=p(>5)<=p(>0); test counts are recorded instead of silently hiding this.


First baseline completed on gpushare: nine models, 1,727,966 sampled training
rows (103,477 / 13,712 / 926 positives for >0 / >5 / >25 mm), all 6,719,274
validation rows and 6,374,476 test rows. No model reached the 80-iteration cap.
Nine fits took 106.78 seconds combined; the original full run took 1,020.64
seconds (about 17 minutes), including preprocessing and full train/validation/
test evaluation. Sampled process-tree peak RSS was 2.566 GiB; minimum free
project disk was 29.881 GiB. All jobs have finished.

2024 test results (ROC-AUC / PR-AUC):

| Features | >0 mm | >5 mm | >25 mm |
|---|---|---|---|
| Time + station | 0.6618 / 0.1165 | 0.6713 / 0.01647 | 0.7216 / 0.001358 |
| + Own history | 0.8437 / 0.5306 | 0.8284 / 0.1579 | 0.8303 / 0.01485 |
| + Neighbors | 0.8550 / 0.5598 | 0.8496 / 0.1791 | 0.8596 / 0.01734 |

Validation PR-AUC selected the neighbors group for all three thresholds.
Its test Brier scores are 0.0394482 / 0.00749509 / 0.000553572; the corresponding
constant training-rate baselines are 0.0595707 / 0.00827522 / 0.000551427.
Recent own history gives the largest improvement, with additional gains from
neighbors. The >25 mm classifier ranks events better but has worse Brier than
the constant reference; this first baseline does not establish accurate rare-
event probabilities. Regularization, a larger rare-event training sample and
post-hoc calibration on chronological validation portions remain experiments.
Independent neighbors classifiers had 21 cross-threshold probability-order
violations among 6,374,476 test rows; raw scores are retained transparently.

Metric fixtures passed hand-calculated Brier/AUC cases, constant-score ties,
invalid probability/missing-label rejection and parity with Spark's evaluator.
All nine saved pipelines passed score round trips; the selected models also
scored without any target columns and handled an unseen station category.
The station/time/threshold prediction CLI reproduced the loaded-model score.
Model and desktop copies were checked byte-for-byte with SHA256.

Generate a comparison/reliability preview from working metrics (keep the
preview outside the submission folder):

```bash
PYTHONPATH=.runtime/python python3 scripts/plot_task2_logistic.py \
  --summary work/task2/logistic/summary.json --output work/task2/logistic/comparison.png
```


## Task 2: full-feature model and imbalance comparison

Install the optional model dependencies in a project environment on a new
machine (`pip install -r requirements-task2.txt`). The actual gpushare run
reuses already available XGBoost 3.2.0, scikit-learn 1.9.0, SciPy 1.17.1 and
PyArrow 25.0.0; it does not install global packages or use a GPU.

```bash
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 -u scripts/run_task2_models.py
```

This sequential, resumable benchmark compares regularized MLlib logistic
regression, MLlib LinearSVC, MLlib GBTClassifier and the external
XGBoost Spark estimator. XGBoost participates in Spark ML pipelines but is
not a built-in MLlib algorithm. Each algorithm fits three independent
classifiers (>0, >5, >25 mm) under natural proportions, balanced class weights
and constrained SMOTE: 36 candidates with all full-feature information.
The station is one-hot encoded for linear classifiers; tree classifiers use
a categorical station index, with Spark metadata or XGBoost categorical
feature types. A station index is not treated as an ordinal numerical variable.

All candidates share the frozen 5% real training sample used by the first
logistic baseline (2017--2022, 1,727,966 rows). Training means, missingness
indicators and station vocabulary are also frozen from that sample. MLlib
linear estimators standardize their features during fitting. The constant
geometry-change value is omitted; its informative missingness indicator is
retained. No feature groups are removed for this benchmark.

- Natural policy: every real training observation has weight 1.
- Balanced policy: weight N/(2P) for positives and N/(2(N-P)) for negatives;
  training weights average 1. Validation and test are not reweighted.
- Constrained SMOTE: target positive fraction 20%, positive count capped at
  ten times the original. Only training positives at the same station and
  with identical missing/discrete profiles and rainfall zero patterns may be paired. Rainfall summaries
  interpolate with one common random coefficient; station, calendar,
  missingness and discrete values are preserved. Zero patterns prevent interpolation from
creating extra visible wet intervals while retaining the original wet count.
Singleton profiles use random
  duplication, counted separately. This is a constrained SMOTE variant,
  not a claim to implement standard SMOTE or SMOTENC. The bounded minority
  pool (at most 103,477 real positives) is exported by Spark and processed
  with single-thread scikit-learn nearest neighbors. All full-data fitting
  and evaluation remain Spark-based. Synthetic rows are training vectors,
  not asserted historical rainfall observations.

2023 Jan--Jun with natural class proportions fits a separate nonnegative-slope
Platt logistic calibrator to each classifier's raw margin. SVM margins are
not called probabilities before calibration. 2023 Jul--Dec selects the best
imbalance policy for each algorithm/threshold by PR-AUC, then Brier for ties.
Half-hour target windows crossing the July or January seam are purged.
Only these 12 selected models receive full 2024 testing (6,374,476 rows).
The overall algorithm choice for inference is also made by validation only.
Training diagnostics use an identical deterministic 20% subset of the real
training sample, without synthetic observations. Scores are exact Spark
ROC/PR areas (numBins=0); Spark PR-AUC is not average precision. Supplementary
F1/precision/recall/accuracy use a validation-selected cutoff, fixed for test.
All candidates use fixed documented settings; no optimized-hyperparameter
or convergence claim is made merely because fitting finished.

Artifacts are native calibrated Spark PipelineModels in
`models/model_comparison/{algorithm}/{policy}/gt_{threshold}mm/`.
Their manifest stores calibration, fitted model parameters, validation-based
choices and runtime information. Vector caches, synthetic examples, scores,
metric JSON and previews remain in ignored `work/`, outside the submission.
`task2_models.py` handles fitting, calibration, selection and evaluation;
`task2_smote.py` builds bounded training-only augmentation;
`plot_task2_models.py` regenerates comparisons and reliability plots from
working metrics. The benchmark also checks save/load equality and label-free
prediction on small real Spark fixtures for all four estimators.

After the complete benchmark, predict a prepared archived feature row:

```bash
PYTHONPATH=.runtime/python ionice -c 3 nice -n 19 python3 scripts/run_task2_models.py \
  --predict-station STATION_ID --predict-time '2024-03-01T12:00:00+08:00' --threshold 5
```

Optional `--predict-algorithm xgb` uses that algorithm's validation-selected
sampling policy. Only the three trained thresholds are accepted. This
entry point reads prepared causal archive features; operational retrieval of
new readings and arbitrary threshold inference are separate future work.
The resulting probabilities are independently calibrated across thresholds;
cross-threshold probability ordering is checked and reported.


### Completed benchmark results

All 36 candidates finished on gpushare. The 12 validation-selected models were
evaluated on all 6,374,476 eligible 2024 rows, retaining natural proportions.
All 12 complete saved pipelines passed label-free and unseen-station checks.
The source/desktop deliverable contains all 36 trained candidate pipelines;
the manifest routes prediction to the validation-selected algorithm and policy.

| Algorithm | Threshold (mm) | Selected policy | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---:|---|---:|---:|---:|
| logistic | >0 | natural | 0.855000 | 0.559780 | 0.039275746 |
| logistic | >5 | smote | 0.850042 | 0.181622 | 0.007543821 |
| logistic | >25 | smote | 0.856513 | 0.018049 | 0.000548293 |
| svm | >0 | smote | 0.860490 | 0.573874 | 0.039970560 |
| svm | >5 | smote | 0.776010 | 0.200927 | 0.007721968 |
| svm | >25 | balanced | 0.848400 | 0.015503 | 0.000548372 |
| gbt | >0 | natural | 0.877491 | 0.603380 | 0.036884328 |
| gbt | >5 | natural | 0.861000 | 0.208915 | 0.007432868 |
| gbt | >25 | natural | 0.765161 | 0.018807 | 0.000561785 |
| xgb | >0 | natural | 0.879863 | 0.607967 | 0.036495851 |
| xgb | >5 | balanced | 0.875258 | 0.212986 | 0.007229151 |
| xgb | >25 | smote | 0.888894 | 0.019180 | 0.000546104 |

The defaults chosen on validation are XGBoost/natural (>0 mm),
XGBoost/balanced (>5 mm), and logistic/SMOTE (>25 mm). They are not changed
after seeing 2024: XGBoost has slightly higher test PR-AUC at all three
thresholds, but this is a descriptive test comparison, not a new selection.

Validation PR-AUC for every policy (2023 Jul--Dec):

| Algorithm / policy | >0 mm | >5 mm | >25 mm |
|---|---:|---:|---:|
| logistic / natural | 0.545472 | 0.209322 | 0.032209 |
| logistic / balanced | 0.539676 | 0.212157 | 0.034935 |
| logistic / smote | 0.542132 | 0.215811 | 0.035197 |
| svm / natural | 0.515459 | 0.008251 | 0.004221 |
| svm / balanced | 0.535415 | 0.174281 | 0.031909 |
| svm / smote | 0.553754 | 0.222021 | 0.022118 |
| gbt / natural | 0.583316 | 0.224210 | 0.034601 |
| gbt / balanced | 0.579323 | 0.216424 | 0.019336 |
| gbt / smote | 0.580729 | 0.211412 | 0.033347 |
| xgb / natural | 0.583682 | 0.224129 | 0.031714 |
| xgb / balanced | 0.581865 | 0.230020 | 0.027006 |
| xgb / smote | 0.577959 | 0.225315 | 0.033099 |

SMOTE helped the SVM >0 and >5 mm tasks, while class weights were better
for SVM >25 mm. Natural proportions won all three GBT tasks. Gains from
imbalance handling depend on algorithm and threshold; feature-group gains
in the earlier ablation do not prove each individual feature is beneficial.

The final augmentation is version `same_station_profile_zero_pattern_smote_cap10_v2`.
An earlier trial allowed different rainfall zero patterns to interpolate and
occasionally contradicted the fixed wet-interval count. Those SMOTE samples
and fitted models were removed and regenerated; none of those trial results
appear in the final tables. Final generation checks also require visible wet
slots not to exceed the stored 30-minute wet count.

| Threshold | Added rows | Augmented positive fraction | Rows with changed rain features | Isolated-profile copies |
|---|---:|---:|---:|---:|
| >0 mm | 302,646 | 20.000% | 129,231 | 73,685 |
| >5 mm | 123,408 | 7.406% | 34,344 | 48,033 |
| >25 mm | 8,334 | 0.533% | 773 | 6,061 |

Even some interpolations between distinct records produce identical rain
features. Thus synthetic row count is not independent information. Only 773
of the 8,334 added >25 mm rows change rain features. The real sampled training
set has 926 positive >25 mm rows, many with overlapping rainfall windows.

All nine SVM fits and the weighted logistic >0 and >25 mm fits reached the
100-iteration cap. Their results are fixed-parameter baselines, not proof of
convergence or optimal hyperparameters. Selected logistic fits did not reach
the cap. Extreme-event probability quality remains weak: the selected GBT
>25 mm Brier (0.0005618) is worse than the constant training-rate reference
(0.0005514), despite better-than-chance ranking. The selected XGBoost and
logistic Briers at this threshold improve only modestly on that reference.

Per-algorithm independent-threshold probability-order violation counts on
2024 are logistic 1, SVM 2,448, GBT 528 and XGBoost 247. These counts do not
describe the mixed-algorithm default. No cross-threshold ordering guarantee
is claimed; these classifiers are not a complete rainfall distribution model.

The 36 base fits took 1922.97 seconds combined.
The final resumed invocation took 2936.83 seconds
(about 49 minutes), reusing previously completed natural/weighted groups;
this is not a from-scratch wall-time claim. Sampled process-tree peak RSS
for benchmark attempts was 2.719 GiB, minimum free project disk 26.984 GiB,
CPU affinity [0,1], 2 GiB driver heap, local[2], and no GPU. Interrupted
SMOTE trials are included in resource accounting. No benchmark job remains
active. The guard is sampled monitoring rather than a kernel-enforced quota.

The prediction CLI was verified for station S07 at 2024-03-01 12:00 Singapore
time, threshold 5 mm: XGBoost/balanced returned probability 0.0061816927.
Archived feature retrieval is required; this is not a deployed live service.

Preview generation (kept in ignored work storage):

```bash
python3 scripts/plot_task2_models.py
```

The complete Task 2 results have not yet been incorporated into the final
bilingual course PDF. Group names/student IDs, report integration and verified
Git backup remain before full-project completion and server cleanup.

## Final deliverable layout

- `scripts/`: reproducible download, PySpark data processing, feature/label code,
  training/evaluation, prediction, plotting and report translation source.
- `models/logistic_baseline/`: nine feature-ablation models and fitted preprocessing.
- `models/model_comparison/`: all 36 trained comparison pipelines, preprocessing and
  validation-selected default routing manifest. These are required model artifacts.
- `reports/Task1_Report_CN.tex` and `.pdf`: full Chinese report, Tasks 1 and 2.
- `reports/Task1_Report.tex` and `.pdf`: synchronized full English report.
- `reports/assets/`: static geographic boundary and its source metadata.
- `requirements*.txt`, `runtime_metadata/`: dependency versions and pinned runtime assets.
- `AI_USAGE.md`: AI assistance declaration.

Reports compile without raw datasets or generated statistical tables. Re-training and
creating features for a new archive prediction require downloading/rebuilding data
with the preceding commands. The bundled prediction interface supports only the
three trained thresholds, not an arbitrary continuous rainfall threshold.

Submission deadline: 18 October 2026. The assignment originally asks for student IDs;
the present PDFs omit them following the group's explicit preference.
