"""Evaluate a frozen tuned rainfall pipeline on all eligible 2024 rows.

Place in repository scripts/. Uses its own saved 2023 H2 cutoff, never tunes
against 2024. Output is separate from original models and tuning artifacts.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True, help='Completed tuning run directory')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.cpu_threads <= 20:
        p.error('CPU threads must be 1-20')
    if args.output.exists():
        p.error('Output already exists; do not overwrite an existing test result')
    selection = json.loads((args.model_dir / 'final_selection.json').read_text())
    validation = json.loads((args.model_dir / 'validation_2023H2.json').read_text())
    if selection['model'] not in ['final_calibrated', 'final_uncalibrated']:
        p.error('Unsupported selected pipeline')
    q = selection['threshold_mm']
    key = 'calibrated' if selection['model'] == 'final_calibrated' else 'uncalibrated'
    cutoff = validation[key]['probability_cutoff']
    probability = 'probability' if key == 'calibrated' else 'base_probability'
    root = Path(__file__).resolve().parents[1]
    paths = [root / f'data/task2/features/2024/{m:02d}' for m in range(1, 13)]
    if any(not (path / '_SUCCESS').exists() for path in paths):
        p.error('All twelve prepared 2024 feature partitions are required')
    os.environ['RAINFALL_CPU_THREADS'] = str(args.cpu_threads)
    os.environ['RAINFALL_DRIVER_HEAP_GIB'] = '6'
    os.environ['RAINFALL_TASK_CPUS'] = str(max(1, args.cpu_threads // min(2, args.cpu_threads)))
    os.environ['RAINFALL_PYTHON_WORKER_REUSE'] = 'false'
    import xgboost.spark
    from pyspark.ml import PipelineModel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from task1_spark import start_spark
    from task2_features import NUMERIC_FEATURES
    from task2_logistic import metrics
    spark = start_spark('FrozenTunedModel2024Test')
    try:
        frame = spark.read.parquet(*map(str, paths)).select(
            'station_id', 'slot_epoch', *NUMERIC_FEATURES,
            F.col(f'label_gt_{q}mm').alias('y'))
        model = PipelineModel.load(str(args.model_dir / selection['model']))
        # The pipeline receives only historical inputs; label is attached for scoring.
        scores = model.transform(frame).select(
            vector_to_array(probability)[1].alias('p'), 'y')
        # Compute training prevalence directly, without refitting any stage.
        trainpaths = [str(root / f'data/task2/features/{y}/{m:02d}')
                      for y in range(2017, 2023) for m in range(1, 13)]
        train_rate = spark.read.parquet(*trainpaths).agg(
            F.avg(f'label_gt_{q}mm').alias('rate')).first().rate
        result = metrics(scores, train_rate, cutoff=cutoff)
        if result['n'] != 6374476:
            raise RuntimeError(f"Unexpected test row count: {result['n']}")
        result.update(threshold_mm=q, selected_model=selection['model'],
                      selected_parameters=selection['params'],
                      cutoff_source='2023 H2; frozen before test',
                      test_year=2024, model_refitted=False,
                      training_rows=validation['training_rows'],
                      training_policy='all eligible 2017-2022 rows',
                      baseline_comparison='Compare only identical target and evaluation population')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
        print(json.dumps({k: result[k] for k in ['n','pr_auc','roc_auc','brier','precision','recall']}, indent=2))
        print('TEST_COMPLETE', args.output)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
