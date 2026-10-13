# AI assistance

OpenAI Codex assisted study design, data-processing and model code, result checks,
and report editing. Rainfall statistics and model evaluations are computed from
actual runs on the real 2017--2024 data. Constrained SMOTE examples are used only
for training, not as validation/test observations.

The main Task 1 run used the authorized local Spark fallback. Task 2 labels,
features, model training, calibration and evaluation ran on gpushare. The reports
state the actual environments and methods. Members are responsible for reviewing
the code, numerical results and final report.

Group members: 潘禹翔 (Pan Yuxiang), 侯佳成 (Hou Jiacheng), 吴嘉蕊 (Wu Jiarui).

Additional 5 mm tuning experiment (13 October 2026): features were rebuilt on the local Mac; tuning, final fitting, calibration and full 2024 evaluation ran on Google Compute Engine e2-highmem-8 (8 vCPU, 64 GB RAM, Ubuntu 22.04). Codex assisted the new scripts, deployment, artifact checks and bilingual report addenda. This is an additional experiment; the original gpushare models are preserved.
