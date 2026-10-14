# Singapore rainfall: Spark exploration and probability prediction

Group members: 潘禹翔、侯佳成、吴嘉蕊.

The project uses the eight annual Historical Rainfall across Singapore datasets,
2017–2024, from [data.gov.sg collection 2279](https://data.gov.sg/collections/2279/view).
Task 1 describes rainfall patterns across time and monitoring stations.
Task 2 predicts the probability that next-30-minute accumulated rainfall strictly
exceeds 0, 5 or 25 mm, given station, prediction time and rainfall threshold.

## Execution environment

Run data processing and training on gpushare in `/home/research_pyx/singapore`.
This logical directory points to project-owned `/hy-tmp/singapore_rainfall_pyx`.
The pinned environment uses Python 3.11.10, Spark 3.5.8, Java 17 and XGBoost 3.2.0.
Task 1's original analysis used the authorized Mac fallback; Task 2 runs on gpushare.
The full-training runner uses Spark `local[20]` on one machine with 20 worker
threads, 80 shuffle partitions and default parallelism 20. It is not a multi-node cluster.
The training runner pins 20 CPU threads, uses low process and I/O priority,
disables GPU use, and monitors memory and disk. Default full-training driver heap is 6 GiB;
The process-tree memory guard is 12 GiB for ordinary Spark phases and 40 GiB for XGBoost fitting/evaluation; free-disk reserve is 5 GiB. These are stop limits,
not reservations. Each phase requires eight GiB of free-memory headroom above its limit.
XGBoost disables Python-worker reuse to release native buffers between jobs.
The Spark XGBoost adapter stores all dense training batches before quantizing them;
the larger bound accommodates the complete 73-feature matrix without dropping rows.
XGBoost keeps two workers and sets `spark.task.cpus=10`, giving each native
worker ten threads within the same 20-CPU limit. Other phases use one CPU per task.
The completed feature-group comparisons used the initial two-CPU configuration;
the remaining full-feature comparison and evaluation use the 20-CPU configuration.
The reproducible project runtime belongs in `.runtime/`; no global package changes are needed.

For a fresh checkout, use Python 3.11, inspect `runtime_metadata/`, and run:

```bash
git clone https://github.com/Panyuxiang0830/singapore_rainfall.git singapore
cd singapore
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-task2.txt
python scripts/configure_runtime.py
python scripts/setup_runtime.py --java-only
```

Use Python 3.11 for the pinned environment. This fresh-checkout setup installs
packages inside `.venv/` and Java inside `.runtime/`; it changes no global packages.
The original gpushare project directory was removed after verified final backups.
A fresh checkout needs the isolated runtime and data rebuilt. Runtime and working
data are ignored by Git. XGBoost phase startup requires eight GiB of free memory
above its configured process-tree limit.

## Full data processing and Task 1

```bash
python scripts/run_task1.py
```

The resumable runner downloads annual official files, verifies hashes and row
counts, cleans timestamps and rain amounts, deduplicates station/time rows,
then performs Spark SQL aggregation. The cleaned data contains 48,538,627
records from 91 station IDs. Detailed temporary outputs are working data;
retain source code, static geographic boundary inputs and final reports.

Analysis compares monthly, yearly and diurnal rain frequency and intensity,
station locations and distance-dependent daily rainfall correlation. The annual
comparison uses the same 21 qualifying stations across all eight years;
other analyses use their explicitly defined station and coverage criteria.
No station's rain is interpreted as an area-weighted national total.

## Features and labels

```bash
python scripts/run_task2_features.py
```

Use `--check-only` for the temporal-window fixtures. Features contain cyclic
hour/month encodings and interactions, station identity/coordinates, recent
rain and availability, lags, rolling amounts, changes and wet-spell length,
and summaries of the three closest eligible stations. Feature retrieval is
causal: observation, calendar slot and retained update times cannot exceed
prediction time. Missing observations remain missing, rather than zero rain.

One station/time feature row carries three labels. Each is defined by six
valid future slots at t+5, t+10, …, t+30 minutes; the current slot is excluded.
Round the total to nine decimal places and use strict `>` for threshold comparison.
The feature data contains 47,636,035 eligible rows:

| Use | Years | Eligible rows |
|---|---|---:|
| Full training | 2017–2022 | 34,542,285 |
| Calibration and selection | 2023 | 6,719,274 |
| Test | 2024 | 6,374,476 |

The 2023 H1 calibration and H2 selection windows are purged at their shared
boundary; all calibration/selection/test sets retain natural class proportions.

## Full training, selection and evaluation

**Every eligible 2017–2022 row is used for training and fitting preprocessing.**
All algorithms and feature groups use the same complete training period.
Imputation, station vocabulary and feature
choices are fitted on that period only. Validation and test are never added to training.

```bash
PYTHONPATH=.runtime/python python scripts/run_full_training.py --cpu-threads 20
```

The runner performs these steps sequentially with checkpoints:

1. Fit preprocessing on all training rows.
2. Fit nine MLlib L2 Logistic feature-group models: calendar/station,
   own-station history, and additional neighbour history, each at three thresholds.
3. Run estimator and SMOTE fixtures, then prepare full feature vectors.
4. Generate constrained SMOTE observations by station. Every training minority
   donor remains available; station partitions are processed sequentially to
   bound memory. Continuous rain features share one interpolation coefficient;
   station, time, missingness, zero pattern and discrete profiles remain fixed.
5. Fit 27 models: three algorithms × three imbalance policies × three thresholds.
   Algorithms are MLlib Logistic, Linear SVM and the XGBoost Spark estimator.
   Policies are natural distribution, balanced class weights and constrained SMOTE.
   Every fit retains all original eligible training observations.
6. Fit monotone Platt probability calibration on 2023 H1.
7. Select policies and final algorithms by 2023 H2 PR-AUC, with Brier as tie-breaker.
8. Evaluate all validation-selected models on the full 2024 test set and the
   final three models on all real training rows.
9. Check model serialization and unknown-station handling; recover final ROC/PR curves.

`work/task2/full_run/status.json` records the current phase; phase logs, metrics
and temporary vectors are under `work/`. Re-running the command resumes this
full-data experiment. It does not reuse incompatible older model checkpoints.
No two project runs should execute concurrently.
Training vectors are written to Parquet in 256 partitions. MLlib bounds the
file-scan partition size before caching, retaining every training row without
an additional wide-vector shuffle.

ROC-AUC and PR-AUC measure ranking. Brier score and reliability curves evaluate
probability quality. Precision, recall, F1 and accuracy use a probability cutoff
selected on validation. Spark computes exact AUC with `numBins=0`.
Synthetic observations are added only to training. Their target positive fraction
is 20%, with positive count capped at ten times the original count.

## Prediction

After the full run completes, load a saved pipeline through the prediction CLI:

```bash
python scripts/run_task2_models.py \
  --predict-station S07 --predict-time '2024-03-01T12:00:00+08:00' --threshold 5
```

The CLI uses the frozen grid-search XGBoost models at all three thresholds: uncalibrated at 0 mm and Platt calibrated at 5/25 mm. Add `--original-model` to reproduce the original algorithm comparison. Grid artifacts are in `models/grid_xgb/gt0`, `gt5`, and `gt25`.
Use `--predict-algorithm logistic|svm|xgb` to choose an algorithm's validation-selected
policy. Saved pipelines include fitted preprocessing and probability calibration. The output includes `model_version` to distinguish the selected pipeline.
An archived prediction requires the corresponding prepared causal feature row.
The three supported rainfall thresholds are 0, 5 and 25 mm.

## Historical original comparison run

The gpushare run completed all 27 full-data comparison fits and nine feature-group
fits. Each base classifier used all 34,542,285 eligible training observations.
Full test evaluation used 6,374,476 observations per target. Validation selected
natural XGBoost for 0 mm and balanced XGBoost for 5 and 25 mm.

All nine algorithm/threshold winners passed loading and end-to-end probability
checks on real label-free rows and previously unseen station IDs; predictions
matched the saved preprocessing/base/calibration sequence exactly. The original prediction
command (now requiring `--original-model`) was executed successfully for all three thresholds. For S07
at 2024-03-01 12:00 Singapore time, it returned probabilities 0.0275664259,
0.0047685057 and 0.0001335948 for 0, 5 and 25 mm, respectively.

Both report sources compile successfully, and final ROC/PR curves use the complete
test set. The saved models and reports are ready to load or read. To recreate
historical feature rows in a fresh checkout, run data processing and feature
construction first; the annual archives and feature Parquet are not bundled.

## Bilingual LaTeX reports and curves

Edit the existing `reports/Task1_Report_CN.tex` in place. Despite legacy names,
this source and `Task1_Report.tex` cover both tasks. Update reviewed translations
in `scripts/report_task2_content.py` / `scripts/report_translations.py` and run:

```bash
# On the local submission checkout after the server run completes:
python scripts/collect_full_training.py
python scripts/build_full_report.py
tectonic --untrusted --outdir reports reports/Task1_Report_CN.tex
tectonic --untrusted --outdir reports reports/Task1_Report.tex
```

Use the built-in LaTeX compiler to check both sources before exporting PDFs.
Plot data are embedded in TikZ/PGFPlots. No separate chart images, statistical
CSV files or HTML dashboards are needed to compile the reports.
Curves use every test row, with bounded coordinates collected only for display;
exact AUCs are checked against saved evaluation results.

## Files retained for submission

- `scripts/`: data processing, features, full training, evaluation, prediction and report code.
- `models/logistic_baseline_full/`: full-training feature-group pipelines and preprocessing.
- `models/model_comparison_full/`: full-training comparison pipelines and selection manifest.
- `reports/Task1_Report_CN.tex` and `.pdf`: Chinese report.
- `reports/Task1_Report.tex` and `.pdf`: English report.
- `reports/assets/`: static geographic inputs with provenance.
- `requirements*.txt`, `runtime_metadata/`: environment versions and pinned runtime metadata.
- `AI_USAGE.md`: declared AI assistance.

The original annual files, cached vectors, scores, logs and intermediate tables
are working data and do not belong in Git or the submission. The reports list
the three group members; student IDs are omitted as explicitly requested.
The assignment deadline is 18 October 2026.

## Final three-target grid search

`TUNING_README.md` describes the complete 16-combination Cartesian grid per target.
The algorithm remains XGBoost. Candidates fit all eligible 2017–2021 rows and
select parameters on 2022; final models refit all eligible 2017–2022 rows.
2023 H1 calibrates probabilities, and H2 chooses raw versus calibrated pipelines
and classification cutoffs. The three grid models were frozen before follow-up
2024 evaluation. The 5 mm model is slightly worse than the earlier six-preset
winner on H2 validation; it is retained to use one grid-search procedure for all
three targets. 2024 had been inspected previously and is disclosed as follow-up
evaluation rather than a researcher-unseen holdout.

Models and verified metrics are tracked under `models/grid_xgb/gt<q>/`.
The bilingual report integrates methods and results in the original model and
evaluation sections. Historical six-preset artifacts in `models/tuned_xgb_gt5/`
and the `Tuning_Addendum_*.md` files describe the earlier experiment only.
`export_grid_curves.py` exports actual bounded ROC/PR coordinates without fitting;
exact AUCs still use all eligible test scores. No 2024 result affects selection.

Label-free loading checks on S07 at 2024-03-01 12:00 Singapore time passed for
all final grid pipelines. Their outputs are 0.0263253860, 0.0057297214, 0.0001255024
for 0, 5 and 25 mm respectively; these replace the default example outputs.

Verified follow-up 2024 metrics (6,374,476 windows per target):

| Rainfall threshold | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| 0 mm | 0.61218037 | 0.88001840 | 0.03612682 |
| 5 mm | 0.22511155 | 0.88159029 | 0.00717005 |
| 25 mm | 0.02118424 | 0.90308515 | 0.00054482 |
