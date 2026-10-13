#!/usr/bin/env bash
# Run from the repository's activated Linux environment; stop on failure.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/task2/xgb_grid/logs
run_stamp=$(date +%Y%m%d_%H%M%S)
for threshold in 0 5 25; do
    policy=balanced
    if [[ "$threshold" == 0 ]]; then policy=natural; fi
    python -u scripts/tune_rainfall_xgb.py \
        --configuration grid --feature-set baseline \
        --threshold "$threshold" --policy "$policy" \
        --cpu-threads 8 --driver-heap-gib 6 --final-fit \
        --output-root "work/task2/xgb_grid/gt${threshold}" \
        2>&1 | tee "work/task2/xgb_grid/logs/${run_stamp}_gt${threshold}.log"
done
