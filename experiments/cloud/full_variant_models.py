"""Run the complete Task 2 benchmark with causal spatial feature extensions.

This reuses the official full-training benchmark's fitted 2017--2022 front end,
then appends either 14 five-neighbour measurements (101 tree dimensions) or
those measurements plus eight distance-aware measurements (117 dimensions).
All appended imputation means are fitted only on training rows.  Synthetic
SMOTE rows use the causal neighbour snapshot of their recorded source row;
no future observation or validation/test row is consulted.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import task2_models as benchmark
import task2_smote as smote
from experiment_spatial_features import SPATIAL_NAMES
from experiment_distance_features import DISTANCE_NAMES
from task1_spark import save_json, start_spark
from task2_features import THRESHOLDS_MM
from task2_logistic import LABELS, read_years
from task2_curves import curves


VARIANTS = ('spatial5', 'distance')


def configure(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    work = ROOT / 'work/cloud/experiments/full_variants' / variant
    models = ROOT / 'models/full_variants' / variant
    names = list(SPATIAL_NAMES) + (list(DISTANCE_NAMES) if variant == 'distance' else [])
    benchmark.WORK = work
    benchmark.MODELS = models
    benchmark.VERSION = 'full_%s_cal2023H1_select2023H2_v1' % variant
    smote.WORK = work
    smote.VERSION = 'full_%s_station_partitioned_smote_v1' % variant
    original_front = benchmark.front_model

    def enhanced_front(kind):
        model, columns = original_front(kind)
        return model, columns + [x for name in names for x in (name + '__filled', name + '__missing')]

    benchmark.front_model = enhanced_front
    benchmark.cache_vectors = lambda spark, raw, split: cache_vectors(spark, raw, split, work, names, variant, original_front)
    benchmark.prepare_smote_vectors = lambda spark: cache_smote_vectors(spark, work, names, variant, original_front)
    original_estimator = benchmark.estimator

    def categorical_estimator(algorithm, q, dimension):
        model = original_estimator(algorithm, q, dimension)
        if algorithm == 'xgb' and dimension >= 73:
            # Station index occupies dimension 72 in the tree representation;
            # appended spatial variables follow it and are all quantitative.
            model.setParams(feature_types=['q'] * 72 + ['c'] + ['q'] * (dimension - 73))
        return model

    benchmark.estimator = categorical_estimator
    return work, models, names, original_front


def sidecar_paths(variant, years):
    root = ROOT / 'work/cloud/experiments' / ('spatial' if variant == 'spatial5' else 'distance') / 'features'
    paths = [root / str(year) / f'{month:02d}' for year in years for month in range(1, 13)]
    missing = [str(path) for path in paths if not (path / '_SUCCESS').exists()]
    if missing:
        raise RuntimeError('Missing causal feature sidecars: ' + ', '.join(missing[:3]))
    return paths


def sidecars(spark, variant, years, names):
    if variant == 'spatial5':
        return spark.read.parquet(*map(str, sidecar_paths(variant, years))).select('station_id', 'slot_epoch', *names)
    spatial = spark.read.parquet(*map(str, sidecar_paths('spatial5', years))).select('station_id', 'slot_epoch', *SPATIAL_NAMES)
    distance = spark.read.parquet(*map(str, sidecar_paths('distance', years))).select('station_id', 'slot_epoch', *DISTANCE_NAMES)
    return spatial.join(distance, ['station_id', 'slot_epoch'], 'inner')


def filled(frame, names, means, feature_col='features'):
    columns = [feature_col]
    for name in names:
        missing = F.col(name).isNull() | F.isnan(F.col(name))
        frame = frame.withColumn(name + '__missing', F.when(missing, F.lit(1.0)).otherwise(F.lit(0.0)))
        frame = frame.withColumn(name + '__filled', F.when(missing, F.lit(float(means[name]))).otherwise(F.col(name)))
        columns += [name + '__filled', name + '__missing']
    return VectorAssembler(inputCols=columns, outputCol='enhanced_features', handleInvalid='error').transform(frame).drop(feature_col).withColumnRenamed('enhanced_features', feature_col)


def feature_state(spark, work, names, variant):
    target = work / 'feature_preprocessing.json'
    if target.exists():
        state = json.loads(target.read_text())
        if state['feature_names'] != names or state['variant'] != variant:
            raise RuntimeError('Feature-preprocessing marker is incompatible')
        return state
    raw = read_years(spark, range(2017, 2023)).select('station_id', 'slot_epoch')
    joined = raw.join(sidecars(spark, variant, range(2017, 2023), names), ['station_id', 'slot_epoch'], 'left')
    counts = joined.agg(F.count('*').alias('n'), F.sum(F.when(F.col(names[0]).isNull(), 1).otherwise(0)).alias('first_missing'),
                        *[F.avg(name).alias(name) for name in names]).first().asDict()
    if counts['n'] != 34542285:
        raise RuntimeError('Training sidecar row count mismatch: %s' % counts['n'])
    means = {name: float(counts[name]) if counts[name] is not None and math.isfinite(float(counts[name])) else 0.0 for name in names}
    state = {'variant': variant, 'feature_names': names, 'mean_by_feature': means,
             'fit_rows': counts['n'], 'fit_years': '2017--2022',
             'policy': 'Means fit on training rows only; source sidecars use causal as-of snapshots.'}
    save_json(target, state)
    print('ENHANCED_FEATURE_STATE_COMPLETE', variant, counts['n'], counts['first_missing'], flush=True)
    return state


def cache_vectors(spark, raw, split, work, names, variant, original_front):
    state = feature_state(spark, work, names, variant)
    years = range(2017, 2023) if split == 'train' else ([2024] if split == 'test' else [2023])
    attached = raw.join(sidecars(spark, variant, years, names), ['station_id', 'slot_epoch'], 'left')
    for kind in ('linear', 'tree'):
        target = work / 'vectors' / split / kind
        if (target / '_SUCCESS').exists():
            continue
        front, _ = original_front(kind)
        frame = front.transform(attached).select('station_id', 'slot_epoch', 'features', *LABELS, *names)
        frame = filled(frame, names, state['mean_by_feature']).select('station_id', 'slot_epoch', 'features', *LABELS)
        if split == 'train':
            frame = frame.repartition(256)
        frame.write.mode('overwrite').parquet(str(target))
        row = spark.read.parquet(str(target)).select('features').limit(1).first()
        expected = (73 if kind == 'tree' else None)
        if kind == 'tree' and row['features'].size != expected + 2 * len(names):
            raise RuntimeError('Unexpected enhanced tree vector dimension')
        print('ENHANCED_VECTORS_COMPLETE', variant, split, kind, row['features'].size, flush=True)


def cache_smote_vectors(spark, work, names, variant, original_front):
    state = feature_state(spark, work, names, variant)
    lookup = sidecars(spark, variant, range(2017, 2023), names).withColumnRenamed('slot_epoch', 'source_slot_a')
    for q in THRESHOLDS_MM:
        raw = spark.read.parquet(str(work / 'smote' / f'gt_{q}mm' / 'synthetic.parquet'))
        for kind in ('linear', 'tree'):
            target = work / 'smote_vectors' / f'gt_{q}mm' / kind
            if (target / '_SUCCESS').exists():
                continue
            front, _ = original_front(kind)
            frame = front.transform(raw).select('station_id', 'source_slot_a', 'features', 'label')
            frame = frame.join(lookup, ['station_id', 'source_slot_a'], 'left')
            missing = frame.filter(F.col(names[0]).isNull()).limit(1).count()
            if missing:
                raise RuntimeError('Synthetic row lacks its causal source sidecar')
            frame = filled(frame, names, state['mean_by_feature']).select('features', 'label')
            frame.write.mode('overwrite').parquet(str(target))
            print('ENHANCED_SMOTE_VECTORS_COMPLETE', variant, q, kind, flush=True)


def run_curves(spark, work, models):
    manifest = json.loads((work / 'selection.json').read_text())
    for q in THRESHOLDS_MM:
        target = work / 'curves' / f'final_gt_{q}mm.json'
        if target.exists():
            continue
        algorithm = manifest['selected_algorithm_by_threshold'][str(q)]
        policy = manifest['selected_policy_by_algorithm_and_threshold'][algorithm][str(q)]
        kind = benchmark.representation(algorithm)
        frame = spark.read.parquet(str(work / 'vectors' / 'test' / kind))
        pipeline = PipelineModel.load(str(models / algorithm / policy / f'gt_{q}mm'))
        for stage in pipeline.stages[-3:]:
            frame = stage.transform(frame)
        scores = frame.select(vector_to_array(f'prob_{q}')[1].alias('p'), F.col(f'label_gt_{q}mm').alias('y')).persist(StorageLevel.DISK_ONLY)
        try:
            result = curves(spark, scores)
            stats = scores.agg(F.count('*').alias('n'), F.sum('y').alias('positives')).first().asDict()
            test = json.loads((work / 'test' / f'{algorithm}_gt_{q}mm.json').read_text())
            if stats['n'] != 6374476 or any(not math.isclose(result[k], test[k], abs_tol=1e-8) for k in ('roc_auc', 'pr_auc')):
                raise RuntimeError('Recovered curve metrics do not match the untouched test scores')
            result.update(stats, threshold_mm=q, algorithm=algorithm, policy=policy,
                          feature_variant=json.loads((work / 'feature_preprocessing.json').read_text())['variant'])
            save_json(target, result)
            print('ENHANCED_CURVES_COMPLETE', q, algorithm, policy, result['pr_auc'], flush=True)
        finally:
            scores.unpersist(blocking=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--phase', choices=['prepare', 'smote', 'smote-vectors', 'fit', 'select', 'test', 'train-diagnostics', 'curves'], required=True)
    parser.add_argument('--threshold', type=int, choices=THRESHOLDS_MM)
    parser.add_argument('--algorithm', choices=benchmark.ALGORITHMS)
    parser.add_argument('--policy', choices=benchmark.POLICIES)
    args = parser.parse_args()
    work, models, names, original_front = configure(args.variant)
    work.mkdir(parents=True, exist_ok=True)
    if args.phase == 'select':
        benchmark.select_models()
        return
    spark = start_spark('RainfallFullVariant_' + args.variant + '_' + args.phase)
    try:
        if args.phase == 'prepare':
            feature_state(spark, work, names, args.variant)
            benchmark.prepare(spark)
        elif args.phase == 'smote':
            smote.generate(args.threshold)
        elif args.phase == 'smote-vectors':
            benchmark.prepare_smote_vectors(spark)
        elif args.phase == 'fit':
            benchmark.fit_group(spark, args.algorithm, args.policy)
        elif args.phase == 'test':
            cache_vectors(spark, read_years(spark, [2024]), 'test', work, names, args.variant, original_front)
            benchmark.test_algorithm(spark, args.algorithm)
        elif args.phase == 'train-diagnostics':
            benchmark.train_diagnostics(spark)
        elif args.phase == 'curves':
            run_curves(spark, work, models)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
