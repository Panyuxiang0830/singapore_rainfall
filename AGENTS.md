# Project working agreements

- Every report content change must be applied to both Chinese and English versions.
- Edit the existing open Chinese LaTeX source in place; keep its editor open.
- Update report_translations.py for changed text, regenerate Task1_Report.tex,
  compile both reports, and synchronize the desktop Project2 copy.
- Do not retain generated CSV summaries, separate chart images, HTML dashboards,
  audit logs, previews or ZIP backups in the deliverable directory or Git.
  Retain source code, required static inputs, LaTeX sources and PDF reports.
- Run subsequent full-data work on gpushare under /home/research_pyx/singapore.
  Full training now uses 20 CPU threads (explicit user authorization on 10 Oct),
  with 6 GiB driver heap (the 20-task full-data cache exhausted the initial 2 GiB),
  low process priority, no GPU,
  and a 12 GiB process-tree monitoring bound for 20-way parallel workers
  (the measured 20-CPU fixture peak exceeded the original 4 GiB guard).
  Require at least eight GiB free-memory headroom above the phase bound. Avoid duplicate jobs;
  inspect this project's existing processes before launching a new run.
- Full-data XGBoost fitting may use a 24 GiB process-tree stop bound because
  its Spark adapter retains dense batches. Keep 20 CPUs and low priority;
  use two Spark XGBoost workers with ten CPUs per task rather than 20 workers;
  require at least 32 GiB free memory. XGBoost evaluation has the same bound;
  other phases use the 12 GiB bound.
- Record actual execution hardware, Spark deployment mode and resource limits.
  Historical Task 1 results were computed on the Mac; describe them as gpushare
  results only after the corresponding analysis is actually rerun there.
  Synchronize any resulting report content changes in both languages.
- A gpushare VM running Spark local[20] is a single-machine deployment, not a
  multi-node cluster. Do not install global packages or alter other projects.
- Task 1 is complete. Task 2 is being retrained using all eligible training rows.
  Push and verify final backups before removing the server project
  at full-project completion; report/backup work alone does not authorize early cleanup.

- Server storage: /home/research_pyx/singapore is a symlink to the project-owned
  /hy-tmp/singapore_rainfall_pyx directory after checksum-verified migration.
  Keep temporary data in that project directory. Final cleanup must account for
  the physical directory and its logical link after verified backups.
- Compare models primarily using ROC-AUC and PR-AUC; retain Brier and calibration
  curves. Precision/recall/F1/accuracy are supplementary at a validation-selected
  probability cutoff, not the main model-selection criteria.

- Task 2 active thresholds are 0, 5 and 25 mm; defer the 50 mm experiment.
- Keep one station/time feature row with separate target columns; do not expand
  observation features once per rainfall threshold.
- Feature availability is conservative: observation, calendar slot and stored
  update times must all be <= prediction time. Do not call update time first
  publication time. Rolling totals end at the latest eligible observation and
  include its age; missing values are not no-rain observations.
- Fit imputation, scaling and categorical encoding on training data only.
  Use task2_features.NUMERIC_FEATURES/CATEGORICAL_FEATURES explicitly; never
  assemble target, future-total or audit columns as predictors.

- Group report names are 潘禹翔、侯佳成、吴嘉蕊. Omit student IDs as explicitly requested.
- Task1_Report_CN.tex / Task1_Report.tex now contain the complete project report,
  despite legacy filenames. Do not overwrite them with the legacy Task 1 generator.
- Native model pipelines in models/ are required submission artifacts and tracked in Git.

- Reports should use concise student-project academic prose. Exclude work logs,
  Git/Desktop synchronization, compilation internals, submission status and
  conversational headings such as 怎样读图/默认路由. State findings directly.
- Keep the requested deletions from the 9 October editorial review. Explain
  full training data, cyclic time encoding, feature-group comparisons
  and Platt probability calibration clearly in both languages.

- User requires all eligible 2017--2022 rows for every model fit and preprocessing fit.
  Remove training subsampling options and obsolete experiment results from current deliverables.
  Preserve 2023 calibration/selection and 2024 testing. Never resume incompatible old models.
