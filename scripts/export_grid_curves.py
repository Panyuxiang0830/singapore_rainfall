"""Export bounded exact-score ROC/PR curves for frozen grid models; no fitting."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve previously exported curves')
    selection = json.loads((args.model_dir / 'final_selection.json').read_text())
    q = selection['threshold_mm']
    name = selection['model']
    if q not in (0, 5, 25) or name not in ('final_calibrated', 'final_uncalibrated'):
        parser.error('Unsupported saved selection')
    os.environ.update(RAINFALL_CPU_THREADS='8', RAINFALL_DRIVER_HEAP_GIB='6',
                      RAINFALL_TASK_CPUS='4', RAINFALL_PYTHON_WORKER_REUSE='false')
    import xgboost.spark
    from pyspark.ml import PipelineModel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from task1_spark import start_spark
    from task2_features import NUMERIC_FEATURES
    from task2_curves import curves
    root = Path(__file__).resolve().parents[1]
    spark = start_spark('FrozenGrid2024Curves')
    try:
        frame = spark.read.parquet(*[str(root / f'data/task2/features/2024/{m:02d}')
                                    for m in range(1, 13)])
        inputs = frame.select('station_id', 'slot_epoch', *NUMERIC_FEATURES,
                              F.col(f'label_gt_{q}mm').alias('y'))
        model = PipelineModel.load(str(args.model_dir / name))
        probability = 'probability' if name == 'final_calibrated' else 'base_probability'
        scores = model.transform(inputs).select(vector_to_array(probability)[1].alias('p'), 'y')
        from pyspark import StorageLevel
        scores = scores.persist(StorageLevel.DISK_ONLY)
        n = scores.count()
        if n != 6374476:
            raise RuntimeError(f'Unexpected test population: {n}')
        result = dict(threshold_mm=q, n=n, selected_model=name, **curves(spark, scores))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
        print('CURVES_COMPLETE', args.output, flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
