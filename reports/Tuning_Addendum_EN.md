# Additional experiment: XGBoost tuning for the 5 mm target

This experiment optimised the probability that rainfall accumulated over the next 30 minutes exceeds 5 mm, using the existing rainfall features. Candidate classifiers used all 27,434,009 eligible records from 2017–2021, with 2022 as the chronological tuning validation period. Imputation, station vocabulary and feature selection were fitted only on the corresponding training block. Future targets crossing split boundaries were purged. Six fixed configurations were compared without training subsampling or early stopping. Validation PR-AUC selected the configuration, with Brier score as a tie-breaker.

The selected configuration used 200 trees, maximum depth 6, learning rate 0.05, 128 histogram bins, minimum child weight 20, L2 regularisation 5 and L1 regularisation 0. Balanced class weights were applied while retaining every real training observation. The 2022 validation PR-AUC increased from 0.24617 for the original parameter configuration to 0.25049. This experiment assigned missing-value indicators to every numeric feature, whereas the original report selected indicators according to training-period missingness. Consequently, comparison with the original final model reflects the new training procedure as a whole; the difference cannot be attributed solely to hyperparameter changes.

After fixing the configuration, preprocessing and the base classifier were refitted on all 34,542,285 eligible 2017–2022 observations. The first half of 2023 fitted monotone Platt calibration; the second half compared calibrated and uncalibrated probability quality and selected the classification cutoff. The calibrated pipeline and cutoff of 0.169824 were frozen before evaluating all 6,374,476 eligible 2024 windows. There were 53,193 positive windows, a prevalence of 0.83447%.

| 2024 test metric | Original report: balanced XGBoost | New final model |
|---|---:|---:|
| PR-AUC | 0.21839 | 0.22640 |
| ROC-AUC | 0.87952 | 0.87908 |
| Brier score | 0.0072009 | 0.0071609 |
| Precision | 0.29634 | 0.29402 |
| Recall | 0.33540 | 0.34657 |
| F1 | 0.31466 | 0.31814 |

PR-AUC improved by approximately 3.7% relative to the original report, and Brier score decreased slightly. ROC-AUC and precision declined slightly, while recall and F1 increased; the metrics did not improve uniformly. Classification metrics use each procedure's validation-selected cutoff, rather than a common false-alarm rate. The new Brier score was below the constant training-prevalence baseline of 0.0082751. Mean predicted probability was 0.87725%, slightly above observed prevalence. These are modest ranking and probability-quality gains. Event-block confidence intervals were not computed, so statistical significance is not established.

The experiment ran on one Google Compute Engine e2-highmem-8 VM, with 8 vCPUs, 64 GB configured RAM (approximately 62 GiB reported by the operating system), a 100 GB Balanced Persistent Disk and Ubuntu 22.04.5 LTS x86_64. The software environment used Python 3.11, Java 17, PySpark 3.5.8 and XGBoost 3.2.0. Spark ran in local[8] mode. XGBoost used two workers with four CPUs per task; the Java driver heap was 6 GiB and no GPU was used. Features were constructed on the local Mac and uploaded. The Google VM performed this tuning, refitting, calibration and test evaluation; it did not perform the original Task 1 analysis.

This additional experiment covers only the 5 mm target. The original models remain the results for 0 and 25 mm. The original experiment had already evaluated 2024. Although this new parameter selection did not read 2024 data, the test period is not an entirely unseen holdout for the researchers. Further development should use training-period time blocks and 2023 validation data, without choosing models through repeated inspection of 2024 performance.

Additional AI declaration: OpenAI Codex assisted with tuning and evaluation scripts, deployment instructions, file-integrity checks and this report text. Metrics come from actual full-data runs; artificial fixtures were used only for code checks. Group members should review the methods, run records and report. The final classifier is saved as a Spark PipelineModel. Integrating it with the original prediction CLI requires updating the selection manifest and probability output-column mapping.
