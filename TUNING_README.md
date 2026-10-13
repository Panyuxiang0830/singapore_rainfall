# Rainfall XGBoost tuning

This is a new experiment script, not an already trained or proven better model.
It does not download data or construct features. Use the original repository's
monthly `data/task2/features/YYYY/MM` Parquet partitions, Python 3.11,
PySpark 3.5.8, XGBoost 3.2.0, Java 17 and `requirements-task2.txt`.

Copy `tune_rainfall_xgb.py` into the repository's `scripts/` folder, or pass
`--repo-root /absolute/path/to/singapore_rainfall` when running it elsewhere.

Preview without Spark or data:

```bash
python scripts/tune_rainfall_xgb.py --plan-only
```

Tune six parameter configurations (no training subsampling):

```bash
python scripts/tune_rainfall_xgb.py --threshold 5 --policy balanced \
  --cpu-threads 2 --driver-heap-gib 2
```

To use the original server's 20-CPU configuration and also produce final models:

```bash
python scripts/tune_rainfall_xgb.py --threshold 5 --policy balanced \
  --cpu-threads 20 --driver-heap-gib 6 --final-fit
```

Set JAVA_HOME appropriately and activate the repository environment first.
The full feature matrix is large. Two threads do not imply small memory usage.
Run full training on a suitable Linux server; the original experiment used a
40 GiB process-tree stop bound for XGBoost. This script does not implement that
resource watchdog. Do not run it alongside another full experiment. XGBoost's
Spark integration is officially tested on Linux; macOS execution is unverified.

## Experimental design

- Tune: all eligible 2017-2021 rows; validate on eligible 2022 rows.
- Refit preprocessing separately on each training period. No validation rows
  fit imputation or station encoding. Future labels, timestamps and audit columns
  never enter the feature vector.
- Compare six configurations by Spark exact PR-AUC, with Brier as tie-breaker.
  This finite search does not use early stopping.
- `original_params` uses the original XGBoost parameter values, but preprocessing
  and the development training period belong to this new experiment. Its result
  is not a reproduction of the report's full-training baseline.
- Optional final fit: all eligible 2017-2022 rows, using the selected parameters.
- Calibrate a monotone Platt transform on 2023 H1; compare raw and calibrated
  probabilities on 2023 H2 and select by Brier.
- Purge future target windows crossing training/validation seams.
- Never read 2024. No automatic test evaluation or original model replacement.
- `--threshold 0|5|25` selects one target. `--policy natural|balanced` selects
  one weighting policy; compare policies through separate runs if desired.

Outputs are in a fresh `work/task2/xgb_tuning/<run-id>/` directory:

- `plan.json`, `comparison.csv`, `results.json`, `winner.json`.
- `candidates/`: loadable Spark pipelines, uncalibrated probabilities in
  `base_probability`.
- With final fitting: `final_uncalibrated/`, `final_calibrated/`,
  `validation_2023H2.json`, `final_selection.json`.
- Final uncalibrated probabilities: `base_probability`; calibrated: `probability`.
  In both cases take vector element 1 for P(y=1).
- Pipelines expect station ID, prediction epoch and the original numeric history
  features. They do not fetch raw history or construct inference features.

The existing historical feature eligibility depends on complete future labels;
this script retains that evaluation population. It does not fix that separate
inference-interface limitation. Independent targets do not guarantee probability
ordering across thresholds.

Validation performed for this deliverable: Python syntax and no-dependency
`--plan-only` CLI execution. Full Spark/XGBoost training has not been run here;
no performance improvement is claimed. AI assistance: OpenAI Codex wrote this
new script and instructions; group members should review and record actual runs.
