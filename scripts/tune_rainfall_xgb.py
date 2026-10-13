#!/usr/bin/env python3
"""Chronological XGBoost tuning for singapore_rainfall (no 2024 access).

Copy to the repository's scripts/ directory. Requires existing monthly feature
Parquet, Python 3.11, Spark 3.5.8 and XGBoost 3.2.0. All eligible rows are used.
Default: fit 2017-2021, rank candidates on 2022. --final-fit refits the winner
on 2017-2022, calibrates on 2023 H1, and evaluates on 2023 H2. Neither stage
reads 2024. Each invocation creates a new directory; old models are untouched.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Allow execution outside scripts/ when --repo-root is supplied.
boot = argparse.ArgumentParser(add_help=False)
boot.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
known, _ = boot.parse_known_args()
ROOT = known.repo_root.resolve()
sys.path.insert(0, str(ROOT / 'scripts'))


def boundary(year, month=1):
    return int(datetime(year, month, 1, tzinfo=ZoneInfo('Asia/Singapore')).timestamp())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def candidates():
    baseline = dict(n_estimators=80, max_depth=4, learning_rate=.1,
                    max_bin=64, min_child_weight=20, reg_lambda=1., reg_alpha=0.)
    return [
        ('original_params', baseline),
        ('slow_depth4', dict(baseline, n_estimators=300, learning_rate=.05, max_bin=128)),
        ('depth5', dict(baseline, n_estimators=200, max_depth=5, learning_rate=.05, max_bin=128)),
        ('depth6_regularized', dict(baseline, n_estimators=200, max_depth=6,
                                  learning_rate=.05, max_bin=128, reg_lambda=5.)),
        ('depth3', dict(baseline, n_estimators=250, max_depth=3, learning_rate=.05, max_bin=128)),
        ('depth4_small_child', dict(baseline, n_estimators=200, learning_rate=.05,
                                  max_bin=128, min_child_weight=5, reg_lambda=5.)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[boot])
    parser.add_argument('--threshold', type=int, choices=[0, 5, 25], default=5)
    parser.add_argument('--policy', choices=['natural', 'balanced'], default='balanced')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--driver-heap-gib', type=int, default=2)
    parser.add_argument('--candidates', type=int, choices=range(1, 7), default=6,
                        help='First N configurations, always including original parameters')
    parser.add_argument('--final-fit', action='store_true')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--plan-only', action='store_true', help='Print plan without importing Spark')
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 20 or args.driver_heap_gib < 1:
        parser.error('Use 1-20 CPU threads and a positive driver heap')
    plan = dict(threshold_mm=args.threshold, policy=args.policy,
                tuning_train_years=list(range(2017, 2022)), tuning_validation_year=2022,
                final_train_years=list(range(2017, 2023)), calibration='2023 H1',
                final_validation='2023 H2', test_2024_read=False, subsampling=False,
                selection='2022 Spark PR-AUC, Brier tie-breaker; no early stopping',
                candidates=[dict(name=n, **p) for n, p in candidates()[:args.candidates]])
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    if not (ROOT / 'scripts/task2_features.py').exists():
        parser.error('--repo-root must point to a singapore_rainfall checkout')
    # Preflight every required partition before starting an expensive job.
    years = range(2017, 2024 if args.final_fit else 2023)
    missing = [str(ROOT / f'data/task2/features/{y}/{m:02d}')
               for y in years for m in range(1, 13)
               if not (ROOT / f'data/task2/features/{y}/{m:02d}/_SUCCESS').exists()]
    if missing:
        parser.error('Build monthly feature Parquet first; missing: ' + ', '.join(missing[:5]))
    import pyspark
    import xgboost
    if pyspark.__version__ != '3.5.8' or xgboost.__version__ != '3.2.0':
        parser.error('Use repository versions: pyspark==3.5.8, xgboost==3.2.0')
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, VectorSlicer
    from pyspark.ml.functions import vector_to_array
    from pyspark.ml.linalg import DenseMatrix
    from pyspark.sql import SparkSession, functions as F
    from xgboost.spark import SparkXGBClassifier
    from task2_features import NUMERIC_FEATURES, FEATURE_VERSION
    from task2_logistic import metrics

    outroot = args.output_root or ROOT / 'work/task2/xgb_tuning'
    out = outroot / (datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}_gt{args.threshold}')
    out.mkdir(parents=True, exist_ok=False)
    plan.update(feature_version=FEATURE_VERSION, numeric_features=NUMERIC_FEATURES,
                cpu_threads=args.cpu_threads, driver_heap_gib=args.driver_heap_gib,
                python=sys.version, spark=pyspark.__version__, xgboost=xgboost.__version__)
    save(out / 'plan.json', plan)
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    workers = 1 if args.cpu_threads == 1 else 2
    task_cpus = max(1, args.cpu_threads // workers)
    spark = (SparkSession.builder.appName('RainfallXGBTuning')
             .master(f'local[{args.cpu_threads}]')
             .config('spark.task.cpus', str(task_cpus))
             .config('spark.driver.memory', f'{args.driver_heap_gib}g')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .config('spark.sql.shuffle.partitions', str(max(8, args.cpu_threads * 4)))
             .config('spark.python.worker.reuse', 'false')
             .config('spark.sql.execution.arrow.maxRecordsPerBatch', '4096')
             .config('spark.local.dir', str(out / 'spark_tmp'))
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    label = f'label_gt_{args.threshold}mm'

    def read(years):
        paths = [str(ROOT / f'data/task2/features/{y}/{m:02d}')
                 for y in years for m in range(1, 13)]
        return spark.read.parquet(*paths).select('station_id', 'slot_epoch',
                                                *NUMERIC_FEATURES, F.col(label).alias('label'))

    def counts(raw):
        r = raw.agg(F.count('*').alias('n'), F.sum('label').alias('p')).first()
        if not 0 < r.p < r.n:
            raise ValueError('Training needs both classes')
        return int(r.n), int(r.p)

    def preprocessing(raw, destination):
        # Refit on this stage's training block: never reuse full-period preprocessing
        # for a 2022 validation experiment.
        expr = []
        for i, c in enumerate(NUMERIC_FEATURES):
            expr += [F.count(c).alias(f'n{i}'), F.min(c).alias(f'lo{i}'), F.max(c).alias(f'hi{i}')]
        stats = raw.agg(*expr).first().asDict()
        active = [c for i, c in enumerate(NUMERIC_FEATURES)
                  if stats[f'n{i}'] and stats[f'lo{i}'] != stats[f'hi{i}']]
        # Keep explicit missingness for every numeric feature, including constants.
        from pyspark.ml.feature import SQLTransformer
        statement = 'SELECT *, ' + ', '.join(
            f'CASE WHEN `{c}` IS NULL OR isnan(`{c}`) THEN 1.0 ELSE 0.0 END AS `{c}__missing`'
            for c in NUMERIC_FEATURES) + ' FROM __THIS__'
        columns = [c + '__filled' for c in active] + [c + '__missing' for c in NUMERIC_FEATURES] + ['station_index']
        stages = [SQLTransformer(statement=statement)]
        if active:
            stages.append(Imputer(strategy='mean', inputCols=active,
                                  outputCols=[c + '__filled' for c in active]))
        stages += [StringIndexer(inputCol='station_id', outputCol='station_index',
                                 handleInvalid='keep', stringOrderType='alphabetAsc'),
                   VectorAssembler(inputCols=columns, outputCol='features')]
        front = Pipeline(stages=stages).fit(raw)
        front.write().save(str(destination))
        return front, len(columns)

    def vectors(front, raw, name):
        path = out / name
        front.transform(raw).select('features', 'label').repartition(max(16, args.cpu_threads * 4)).write.parquet(str(path))
        return spark.read.parquet(str(path))

    def fit(frame, params, dimension, n, p):
        weight = (F.when(F.col('label') == 1, n / (2 * p)).otherwise(n / (2 * (n - p)))
                  if args.policy == 'balanced' else F.lit(1.))
        est = SparkXGBClassifier(features_col='features', label_col='label', weight_col='weight',
            probability_col='base_probability', raw_prediction_col='base_raw', prediction_col='base_prediction',
            num_workers=workers, device='cpu', tree_method='hist',
            feature_types=['q'] * (dimension - 1) + ['c'],
            subsample=1., colsample_bytree=1., random_state=5208, verbosity=0, **params)
        return est.fit(frame.withColumn('weight', weight))

    def evaluate(scored, probability, rate):
        scores = scored.select(vector_to_array(probability)[1].alias('p'), F.col('label').alias('y'))
        return metrics(scores, rate)

    try:
        train = read(range(2017, 2022)).filter(F.col('slot_epoch') + 1800 < boundary(2022))
        valid = read([2022]).filter(F.col('slot_epoch') + 1800 < boundary(2023))
        n, p = counts(train)
        front, dim = preprocessing(train, out / 'tuning_preprocessing')
        trainv, validv = vectors(front, train, 'tuning_train'), vectors(front, valid, 'tuning_valid')
        results = []
        best = None
        for name, params in candidates()[:args.candidates]:
            print('FIT', name, params, flush=True)
            begin = time.monotonic()
            model = fit(trainv, params, dim, n, p)
            result = evaluate(model.transform(validv), 'base_probability', p / n)
            result.update(name=name, params=params, fit_and_evaluate_seconds=time.monotonic()-begin,
                          training_rows=n, training_positives=p)
            PipelineModel(stages=front.stages + [model]).write().save(str(out / 'candidates' / name))
            results.append(result)
            save(out / 'results.json', results)
            print(json.dumps({k: result[k] for k in ['name', 'pr_auc', 'roc_auc', 'brier']}), flush=True)
            if best is None or (result['pr_auc'], -result['brier']) > (best['pr_auc'], -best['brier']):
                best = result
            del model
        save(out / 'winner.json', best)
        with (out / 'comparison.csv').open('w', newline='') as f:
            keys = ['name', 'pr_auc', 'roc_auc', 'brier', 'log_loss', 'training_rows', 'fit_and_evaluate_seconds']
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows({k: r[k] for k in keys} for r in results)
        if args.final_fit:
            raw = read(range(2017, 2023)).filter(F.col('slot_epoch') + 1800 < boundary(2023))
            fn, fp = counts(raw)
            finalfront, finaldim = preprocessing(raw, out / 'final_preprocessing')
            finalv = vectors(finalfront, raw, 'final_train')
            base = fit(finalv, best['params'], finaldim, fn, fp)
            year2023 = read([2023])
            calraw = year2023.filter(F.col('slot_epoch') + 1800 < boundary(2023, 7))
            selraw = year2023.filter((F.col('slot_epoch') >= boundary(2023, 7)) &
                                     (F.col('slot_epoch') + 1800 < boundary(2024)))
            slicer = VectorSlicer(inputCol='base_raw', outputCol='cal_features', indices=[1])
            calibrator = LogisticRegression(labelCol='label', featuresCol='cal_features',
                probabilityCol='probability', rawPredictionCol='cal_raw', predictionCol='prediction',
                regParam=1e-6, maxIter=100, tol=1e-7,
                lowerBoundsOnCoefficients=DenseMatrix(1, 1, [0.])).fit(
                    slicer.transform(base.transform(finalfront.transform(calraw))))
            uncal = PipelineModel(stages=finalfront.stages + [base])
            calibrated = PipelineModel(stages=uncal.stages + [slicer, calibrator])
            uncal.write().save(str(out / 'final_uncalibrated'))
            calibrated.write().save(str(out / 'final_calibrated'))
            vm = dict(uncalibrated=evaluate(uncal.transform(selraw), 'base_probability', fp / fn),
                      calibrated=evaluate(calibrated.transform(selraw), 'probability', fp / fn),
                      training_rows=fn, params=best['params'], test_2024_read=False)
            save(out / 'validation_2023H2.json', vm)
            chosen = 'final_calibrated' if vm['calibrated']['brier'] < vm['uncalibrated']['brier'] else 'final_uncalibrated'
            save(out / 'final_selection.json', dict(model=chosen, criterion='2023 H2 Brier',
                 threshold_mm=args.threshold, params=best['params'], test_2024_read=False,
                 inference='Supply station_id, slot_epoch and NUMERIC_FEATURES; no label needed'))
        print('DONE:', out, flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
