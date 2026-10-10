"""Full-feature Spark classifier and imbalance benchmark with independent calibration.

Fits: all eligible training rows, 2017--2022. Calibration: 2023 Jan--Jun.
Selection: 2023 Jul--Dec. Half-hour target windows cannot cross these seams.
Only the validation-selected policy per algorithm/threshold sees full 2024 test.
"""
import argparse
import json
import math
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import DenseMatrix
from pyspark.sql import functions as F

from task1_spark import ROOT, save_json, start_spark
from task2_features import NUMERIC_FEATURES, THRESHOLDS_MM
from task2_logistic import read_years, metrics, TRAIN_PARTITIONS

WORK = ROOT/'work/task2/model_comparison_full'
MODELS = ROOT/'models/model_comparison_full'
VERSION = 'full_training_cal2023H1_select2023H2_v3'
ALGORITHMS = ['logistic', 'svm', 'xgb']
POLICIES = ['natural', 'balanced', 'smote']
BASE_WORK = ROOT/'work/task2/logistic_full'
BASE_MODELS = ROOT/'models/logistic_baseline_full'


def log(event, **details):
    print(event, json.dumps(details, default=str), flush=True)


def boundary(year, month=1):
    return int(datetime(year, month, 1, tzinfo=ZoneInfo('Asia/Singapore')).timestamp())


def representation(algorithm):
    return 'linear' if algorithm in ['logistic', 'svm'] else 'tree'


def front_model(kind):
    prior = json.loads((BASE_WORK/'preparation.json').read_text())
    preprocess = PipelineModel.load(str(BASE_MODELS/'preprocessing'))
    columns = [f'{c}__filled' for c in prior['active_numeric']]
    columns += [f'{c}__missing' for c in prior['missing_indicators']]
    columns += ['station_vector' if kind == 'linear' else 'station_index']
    assembler = VectorAssembler(inputCols=columns, outputCol='features', handleInvalid='error')
    return PipelineModel(stages=preprocess.stages+[assembler]), columns


def vector_path(split, kind):
    return WORK/'vectors'/split/kind


def cache_vectors(spark, raw, split):
    from task2_logistic import LABELS
    for kind in ['linear', 'tree']:
        target = vector_path(split, kind)
        if (target/'_SUCCESS').exists():
            continue
        front, _ = front_model(kind)
        frame = front.transform(raw).select('station_id', 'slot_epoch', 'features', *LABELS)
        if split == 'train':
            frame = frame.repartition(TRAIN_PARTITIONS)
        frame.write.mode('overwrite').parquet(str(target))
        log('VECTORS_COMPLETE', split=split, representation=kind)


def prepare(spark):
    marker = WORK/'preparation.json'
    if marker.exists():
        stored = json.loads(marker.read_text())
        if stored['version'] != VERSION:
            raise RuntimeError('Benchmark version mismatch')
        log('BENCHMARK_PREPARATION_RESUMED')
        return
    prior = json.loads((BASE_WORK/'preparation.json').read_text())
    train = read_years(spark, range(2017, 2023))
    if prior['config']['training_policy'] != 'all_eligible_2017_2022' or prior['train_rows'] != 34542285:
        raise RuntimeError('Full training-only preprocessing is required')
    scale_row = train.agg(*[F.stddev_pop(c).alias(c) for c in NUMERIC_FEATURES]).first().asDict()
    scales = {c: float(v) if v and math.isfinite(v) and v > 1e-10 else 1.0 for c, v in scale_row.items()}
    validation = read_years(spark, [2023])
    calibration = validation.filter(F.col('slot_epoch')+1800 < boundary(2023, 7))
    selection = validation.filter((F.col('slot_epoch') >= boundary(2023, 7)) &
                                 (F.col('slot_epoch')+1800 < boundary(2024)))
    cache_vectors(spark, train, 'train')
    cache_vectors(spark, calibration, 'calibration')
    cache_vectors(spark, selection, 'selection')
    train_check = spark.read.parquet(str(vector_path('train', 'tree'))).agg(
        F.count('*').alias('n'), *[F.sum(f'label_gt_{q}mm').alias(str(q)) for q in THRESHOLDS_MM]).first().asDict()
    if train_check['n'] != prior['train_rows'] or any(train_check[str(q)] != prior['train_positives'][str(q)] for q in THRESHOLDS_MM):
        raise RuntimeError('Full training row counts changed')
    for q in THRESHOLDS_MM:
        target = WORK/'minority'/f'gt_{q}mm'
        if not (target/'_SUCCESS').exists():
            train.filter(F.col(f'label_gt_{q}mm') == 1).select('station_id', 'slot_epoch', *NUMERIC_FEATURES).write.mode('overwrite').partitionBy('station_id').parquet(str(target))
    counts = {}
    for name in ['calibration', 'selection']:
        frame = spark.read.parquet(str(vector_path(name, 'tree')))
        record = frame.agg(F.count('*').alias('n'), *[F.sum(f'label_gt_{q}mm').alias(str(q)) for q in THRESHOLDS_MM],
                           F.min('slot_epoch').alias('first_epoch'), F.max('slot_epoch').alias('last_epoch')).first().asDict()
        counts[name] = record
    result = {'version': VERSION, 'train_rows': prior['train_rows'], 'train_positives': prior['train_positives'],
              'training_policy': 'all_eligible_2017_2022', 'distance_scales': scales, 'validation_blocks': counts,
              'active_numeric': prior['active_numeric'], 'missing_indicators': prior['missing_indicators'],
              'preprocessing_policy': 'Imputer, missing indicators and station vocabulary fitted on all eligible training rows',
              'calibration_policy': 'Natural-proportion 2023 Jan--Jun; monotone Platt logistic fit separately from base classifier',
              'selection_policy': 'Natural-proportion 2023 Jul--Dec PR-AUC, then Brier only for ties; never test selection',
              'training_diagnostics': 'Full real training data for the final three validation-selected models', 'test_policy': 'Full eligible 2024, selected policy for each algorithm/threshold',
              'thresholds_mm': list(THRESHOLDS_MM), 'smote_target_fraction': 0.2, 'smote_positive_multiplier_cap': 10}
    save_json(marker, result)
    log('BENCHMARK_PREPARATION_COMPLETE', **{k: v for k, v in result.items() if k != 'distance_scales'})


def prepare_smote_vectors(spark):
    for q in THRESHOLDS_MM:
        raw = spark.read.parquet(str(WORK/'smote'/f'gt_{q}mm'/'synthetic.parquet'))
        for kind in ['linear', 'tree']:
            target = WORK/'smote_vectors'/f'gt_{q}mm'/kind
            if (target/'_SUCCESS').exists():
                continue
            front, _ = front_model(kind)
            front.transform(raw).select('features', 'label').write.mode('overwrite').parquet(str(target))
            log('SMOTE_VECTORS_COMPLETE', threshold=q, representation=kind)


def estimator(algorithm, q, dimension):
    shared = dict(labelCol=f'label_gt_{q}mm', featuresCol='features', weightCol='weight',
                  rawPredictionCol=f'raw_{q}', predictionCol=f'base_pred_{q}')
    if algorithm == 'logistic':
        return LogisticRegression(**shared, probabilityCol=f'base_prob_{q}', regParam=0.001,
                                  elasticNetParam=0, standardization=True, maxIter=100, tol=1e-6)
    if algorithm == 'svm':
        return LinearSVC(**shared, regParam=0.001, standardization=True, maxIter=100, tol=1e-6)
    if algorithm == 'xgb':
        from xgboost.spark import SparkXGBClassifier
        # Categorical station index avoids numeric station ordering and fractional one-hot IDs.
        # CPU task count controls threads in the Spark estimator; nthread/n_jobs are not set.
        return SparkXGBClassifier(features_col='features', label_col=f'label_gt_{q}mm',
                   weight_col='weight', raw_prediction_col=f'raw_{q}', prediction_col=f'base_pred_{q}',
                   probability_col=f'base_prob_{q}', num_workers=min(2, int(os.environ.get('RAINFALL_CPU_THREADS', '2'))), device='cpu', tree_method='hist',
                   feature_types=['q']*(dimension-1)+['c'], n_estimators=80,
                   max_depth=4, learning_rate=0.1, max_bin=64, min_child_weight=20,
                   reg_lambda=1.0, reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0,
                   random_state=5208, verbosity=0)
    raise ValueError(algorithm)


def platt_estimator(q):
    return LogisticRegression(labelCol=f'label_gt_{q}mm', featuresCol=f'cal_features_{q}',
               probabilityCol=f'prob_{q}', rawPredictionCol=f'cal_raw_{q}', predictionCol=f'pred_{q}',
               regParam=1e-6, elasticNetParam=0.0, standardization=True, maxIter=100, tol=1e-7,
               lowerBoundsOnCoefficients=DenseMatrix(1, 1, [0.0]))


def fit_group(spark, algorithm, policy):
    prepared = json.loads((WORK/'preparation.json').read_text())
    marker = WORK/'results'/f'{algorithm}_{policy}.json'
    if marker.exists():
        log('MODEL_GROUP_RESUMED', algorithm=algorithm, policy=policy)
        return
    begun = time.monotonic()
    kind = representation(algorithm)
    front, assembled = front_model(kind)
    # Bound file-scan tasks directly. Repartitioning these wide vectors again
    # materializes a large shuffle despite the existing 256-file Parquet cache.
    # File splitting changes only task layout; every training row is retained.
    spark.conf.set('spark.sql.files.maxPartitionBytes', 8 * 1024 * 1024)
    spark.conf.set('spark.sql.files.minPartitionNum', TRAIN_PARTITIONS)
    stored_vectors = spark.read.parquet(str(vector_path('train', kind)))
    dimension = stored_vectors.first()['features'].size
    # The complete vectors already have a compressed Parquet cache. Let MLlib
    # cache its own instance blocks instead of also duplicating all vectors as
    # a much larger SQL row cache. This changes storage, never training rows.
    train = stored_vectors
    log('MODEL_GROUP_START', algorithm=algorithm, policy=policy, rows=prepared['train_rows'],
        dimension=dimension, file_scan_partitions=train.rdd.getNumPartitions(),
        spark_master=spark.sparkContext.master,
        task_cpus=spark.sparkContext.getConf().get('spark.task.cpus'),
        jvm_processors=spark.sparkContext._jvm.java.lang.Runtime.getRuntime().availableProcessors())
    bases, results = {}, {}
    try:
        for q in THRESHOLDS_MM:
            state = WORK/'results'/f'{algorithm}_{policy}_gt_{q}mm.json'
            base_path = WORK/'base_models'/algorithm/policy/f'gt_{q}mm'
            if state.exists():
                result = json.loads(state.read_text())
                bases[q] = PipelineModel.load(str(base_path)).stages[0]
                results[str(q)] = result
                continue
            label = f'label_gt_{q}mm'
            fitting = train.select('features', label)
            p, n = prepared['train_positives'][str(q)], prepared['train_rows']
            if policy == 'balanced':
                fitting = fitting.withColumn('weight', F.when(F.col(label) == 1, n/(2*p)).otherwise(n/(2*(n-p))))
            else:
                fitting = fitting.withColumn('weight', F.lit(1.0))
            if policy == 'smote':
                added = spark.read.parquet(str(WORK/'smote_vectors'/f'gt_{q}mm'/kind)).select(
                    'features', F.col('label').alias(label)).withColumn('weight', F.lit(1.0))
                fitting = fitting.unionByName(added)
            start = time.monotonic()
            base = estimator(algorithm, q, dimension).fit(fitting)
            elapsed = time.monotonic()-start
            PipelineModel(stages=[base]).write().overwrite().save(str(base_path))
            base_config = {p.name: v for p, v in base.extractParamMap().items() if p.name not in ['arbitrary_params_dict']}
            optimization = {}
            if algorithm in ['logistic', 'svm']:
                training_summary = base.summary() if algorithm == 'svm' else base.summary
                optimization = {'iterations': training_summary.totalIterations,
                                'iteration_cap_reached': training_summary.totalIterations >= 100}
            result = {'algorithm': algorithm, 'policy': policy, 'threshold_mm': q, 'feature_representation': kind,
                      'vector_dimension': dimension, 'base_fit_seconds': elapsed, **optimization,
                      'assembled_features': assembled, 'base_params': base_config,
                      'training_rows': n+(json.loads((WORK/'smote'/f'gt_{q}mm'/'generation.json').read_text())['added_rows'] if policy == 'smote' else 0),
                      'real_training_rows': n, 'real_training_positives': p}
            result['fit_runtime'] = {
                'spark_master': spark.sparkContext.master,
                'task_cpus': int(spark.sparkContext.getConf().get('spark.task.cpus')),
                'jvm_processors': spark.sparkContext._jvm.java.lang.Runtime.getRuntime().availableProcessors(),
                'jvm_max_heap_gib': spark.sparkContext._jvm.java.lang.Runtime.getRuntime().maxMemory() / 1024**3,
            }
            save_json(state, result)
            bases[q] = base; results[str(q)] = result
            log('BASE_FIT_COMPLETE', algorithm=algorithm, policy=policy, threshold=q, fit_seconds=elapsed, **optimization)
        cal = spark.read.parquet(str(vector_path('calibration', kind)))
        for q, base in bases.items():
            cal = base.transform(cal)
        cal = cal.select(*[f'label_gt_{q}mm' for q in THRESHOLDS_MM],
                         *[f'raw_{q}' for q in THRESHOLDS_MM]).persist(StorageLevel.DISK_ONLY)
        parts = {}
        try:
            for q in THRESHOLDS_MM:
                result = results[str(q)]
                path = MODELS/algorithm/policy/f'gt_{q}mm'
                if result.get('calibration'):
                    pipeline = PipelineModel.load(str(path))
                    parts[q] = pipeline.stages[-2:]
                    continue
                slice_stage = VectorSlicer(inputCol=f'raw_{q}', outputCol=f'cal_features_{q}', indices=[1])
                start = time.monotonic()
                calibrator = platt_estimator(q).fit(slice_stage.transform(cal))
                pipeline = PipelineModel(stages=front.stages+[bases[q], slice_stage, calibrator])
                pipeline.write().overwrite().save(str(path))
                result['calibration'] = {'slope': float(calibrator.coefficients[0]), 'intercept': calibrator.intercept,
                                         'iterations': calibrator.summary.totalIterations,
                                         'seconds': time.monotonic()-start,
                                         'rows': prepared['validation_blocks']['calibration']['n'],
                                         'time_block': '2023 Jan--Jun, purged at July seam', 'natural_proportions': True,
                                         'nonnegative_slope_constraint': True}
                result['model_relative_path'] = str(path.relative_to(ROOT))
                parts[q] = [slice_stage, calibrator]
                save_json(WORK/'results'/f'{algorithm}_{policy}_gt_{q}mm.json', result)
                log('CALIBRATION_COMPLETE', algorithm=algorithm, policy=policy, threshold=q, **result['calibration'])
        finally:
            cal.unpersist(blocking=True)
        # Selection first so every confusion matrix uses the same validation-selected cutoff.
        for split, frame in [('validation', spark.read.parquet(str(vector_path('selection', kind))))]:
            scored = frame
            for q in THRESHOLDS_MM:
                scored = bases[q].transform(scored)
                for part in parts[q]:
                    scored = part.transform(scored)
            columns = [F.col(f'label_gt_{q}mm') for q in THRESHOLDS_MM]+[
                vector_to_array(f'prob_{q}')[1].alias(f'p_{q}') for q in THRESHOLDS_MM]
            if algorithm != 'svm':
                columns += [vector_to_array(f'base_prob_{q}')[1].alias(f'uncal_{q}') for q in THRESHOLDS_MM]
            scored = scored.select(*columns).persist(StorageLevel.DISK_ONLY)
            try:
                for q in THRESHOLDS_MM:
                    result = results[str(q)]
                    if split in result:
                        continue
                    cutoff = None if split == 'validation' else result['validation']['probability_cutoff']
                    rate = prepared['train_positives'][str(q)]/prepared['train_rows']
                    result[split] = metrics(scored.select(F.col(f'p_{q}').alias('p'), F.col(f'label_gt_{q}mm').alias('y')), rate, cutoff)
                    if algorithm != 'svm':
                        raw = scored.agg(F.avg(F.pow(F.col(f'uncal_{q}')-F.col(f'label_gt_{q}mm'), 2)).alias('brier'),
                                         F.avg(f'uncal_{q}').alias('mean_probability')).first().asDict()
                        result[split]['uncalibrated_probability_quality'] = raw
                    save_json(WORK/'results'/f'{algorithm}_{policy}_gt_{q}mm.json', result)
                    log('BENCHMARK_METRICS_COMPLETE', algorithm=algorithm, policy=policy, threshold=q, split=split,
                        **{k: v for k, v in result[split].items() if k not in ['calibration_curve']})
            finally:
                scored.unpersist(blocking=True)
        result = {'version': VERSION, 'algorithm': algorithm, 'policy': policy, 'models': results,
                  'group_seconds': time.monotonic()-begun, 'train_metrics': 'Full real training diagnostics computed after validation selection for final three models',
                  'validation_metrics': 'Full natural-proportion 2023 Jul--Dec, no calibration-fitting rows',
                  'fixed_parameters': True, 'hyperparameter_optimization_claim': False}
        save_json(marker, result)
        log('MODEL_GROUP_COMPLETE', algorithm=algorithm, policy=policy, group_seconds=result['group_seconds'])
    finally:
        train.unpersist(blocking=True)


def select_models():
    prepared = json.loads((WORK/'preparation.json').read_text())
    results = {a: {p: json.loads((WORK/'results'/f'{a}_{p}.json').read_text()) for p in POLICIES} for a in ALGORITHMS}
    selected = {}
    for a in ALGORITHMS:
        selected[a] = {}
        for q in THRESHOLDS_MM:
            # Stable policy ordering resolves exact ties after Brier.
            selected[a][str(q)] = max(POLICIES, key=lambda p: (
                results[a][p]['models'][str(q)]['validation']['pr_auc'],
                -results[a][p]['models'][str(q)]['validation']['brier']))
    best_algorithm = {
        str(q): max(ALGORITHMS, key=lambda a: (
            results[a][selected[a][str(q)]]['models'][str(q)]['validation']['pr_auc'],
            -results[a][selected[a][str(q)]]['models'][str(q)]['validation']['brier']))
        for q in THRESHOLDS_MM}
    model_metadata = {}
    for a in ALGORITHMS:
        model_metadata[a] = {}
        for p in POLICIES:
            model_metadata[a][p] = {
                str(q): {k: v for k, v in results[a][p]['models'][str(q)].items()
                         if k not in ['train', 'validation']}
                for q in THRESHOLDS_MM}
    manifest = {'version': VERSION, 'selected_policy_by_algorithm_and_threshold': selected,
                'selected_algorithm_by_threshold': best_algorithm, 'model_metadata': model_metadata,
                'training_data': {'years': list(range(2017, 2023)), 'policy': 'all_eligible_rows',
                                    'rows': prepared['train_rows'], 'threshold_positive_counts': prepared['train_positives']},
                'smote_metadata': {str(q): {**json.loads((WORK/'smote'/f'gt_{q}mm'/'generation.json').read_text()),
                                             'rng_seed': 5208+q} for q in THRESHOLDS_MM},
                'selection_metric': '2023 H2 PR-AUC; Brier as tie-breaker', 'test_not_used_for_selection': True,
                'thresholds_mm': list(THRESHOLDS_MM), 'algorithms': ALGORITHMS,
                'calibration': 'Monotone Platt fit on natural 2023 H1; all base models fit all eligible 2017--2022 rows',
                'feature_definition': 'task2_features.NUMERIC_FEATURES and station_id',
                'runtime': {'spark': '3.5.8', 'xgboost': '3.2.0',
                            'master': f"local[{int(os.environ.get('RAINFALL_CPU_THREADS', '2'))}]",
                            'resource_configuration': json.loads((ROOT/'work/task2/full_run/resource_limits.json').read_text())
                            if (ROOT/'work/task2/full_run/resource_limits.json').exists()
                            else {'cpu_threads': 2, 'driver_heap_gib': 2,
                                  'spark_phase_rss_limit_gib': 4, 'xgb_phase_rss_limit_gib': 4}}}
    save_json(MODELS/'manifest.json', manifest)
    save_json(WORK/'selection.json', manifest)
    log('VALIDATION_SELECTION_COMPLETE', selected=selected)


def test_algorithm(spark, algorithm):
    target = WORK/'test'/f'{algorithm}.json'
    if target.exists():
        log('TEST_ALGORITHM_RESUMED', algorithm=algorithm)
        return
    kind = representation(algorithm)
    cached = vector_path('test', kind)
    if not (cached/'_SUCCESS').exists():
        front, _ = front_model(kind)
        front.transform(read_years(spark, [2024])).select('features', *[f'label_gt_{q}mm' for q in THRESHOLDS_MM]).write.mode('overwrite').parquet(str(cached))
    frame = spark.read.parquet(str(cached))
    selected = json.loads((WORK/'selection.json').read_text())['selected_policy_by_algorithm_and_threshold'][algorithm]
    model_states = {}
    for q in THRESHOLDS_MM:
        policy = selected[str(q)]
        pipeline = PipelineModel.load(str(MODELS/algorithm/policy/f'gt_{q}mm'))
        # Prepared vectors already include the frozen fitted front end.
        for stage in pipeline.stages[-3:]:
            frame = stage.transform(frame)
        model_states[str(q)] = json.loads((WORK/'results'/f'{algorithm}_{policy}_gt_{q}mm.json').read_text())
    columns = [F.col(f'label_gt_{q}mm') for q in THRESHOLDS_MM]+[
        vector_to_array(f'prob_{q}')[1].alias(f'p_{q}') for q in THRESHOLDS_MM]
    if algorithm != 'svm':
        columns += [vector_to_array(f'base_prob_{q}')[1].alias(f'uncal_{q}') for q in THRESHOLDS_MM]
    scored = frame.select(*columns).persist(StorageLevel.DISK_ONLY)
    try:
        results = {}
        prepared = json.loads((WORK/'preparation.json').read_text())
        for q in THRESHOLDS_MM:
            checkpoint = WORK/'test'/f'{algorithm}_gt_{q}mm.json'
            if checkpoint.exists():
                results[str(q)] = json.loads(checkpoint.read_text())
                continue
            rate = prepared['train_positives'][str(q)]/prepared['train_rows']
            result = metrics(scored.select(F.col(f'p_{q}').alias('p'), F.col(f'label_gt_{q}mm').alias('y')),
                             rate, model_states[str(q)]['validation']['probability_cutoff'])
            result['selected_policy'] = selected[str(q)]
            if algorithm != 'svm':
                result['uncalibrated_probability_quality'] = scored.agg(
                    F.avg(F.pow(F.col(f'uncal_{q}')-F.col(f'label_gt_{q}mm'), 2)).alias('brier'),
                    F.avg(f'uncal_{q}').alias('mean_probability')).first().asDict()
            if result['n'] != 6374476:
                raise RuntimeError('Full test count mismatch')
            results[str(q)] = result
            save_json(checkpoint, result)
            log('FULL_TEST_METRICS_COMPLETE', algorithm=algorithm, threshold=q,
                **{k: v for k, v in result.items() if k != 'calibration_curve'})
        violations = scored.agg(F.sum(F.when((F.col('p_25')>F.col('p_5')) | (F.col('p_5')>F.col('p_0')), 1).otherwise(0))).first()[0]
        save_json(target, {'algorithm': algorithm, 'selected_policies': selected, 'test': results,
                           'test_probability_order_violations': violations})
    finally:
        scored.unpersist(blocking=True)


def train_diagnostics(spark):
    """Evaluate all real training rows for the three final selected models."""
    manifest = json.loads((WORK/'selection.json').read_text())
    prepared = json.loads((WORK/'preparation.json').read_text())
    for q in THRESHOLDS_MM:
        target = WORK/'train_final'/f'gt_{q}mm.json'
        if target.exists():
            continue
        algorithm = manifest['selected_algorithm_by_threshold'][str(q)]
        policy = manifest['selected_policy_by_algorithm_and_threshold'][algorithm][str(q)]
        frame = spark.read.parquet(str(vector_path('train', representation(algorithm))))
        pipeline = PipelineModel.load(str(MODELS/algorithm/policy/f'gt_{q}mm'))
        for stage in pipeline.stages[-3:]:
            frame = stage.transform(frame)
        score = frame.select(vector_to_array(f'prob_{q}')[1].alias('p'), F.col(f'label_gt_{q}mm').alias('y'))
        choice = json.loads((WORK/'results'/f'{algorithm}_{policy}_gt_{q}mm.json').read_text())
        result = metrics(score, prepared['train_positives'][str(q)]/prepared['train_rows'],
                         choice['validation']['probability_cutoff'])
        assert result['n'] == prepared['train_rows'] == 34542285
        result.update(algorithm=algorithm, policy=policy, threshold_mm=q,
                      scope='All real eligible 2017--2022 training rows; no synthetic observations')
        save_json(target, result)
        log('FULL_TRAIN_DIAGNOSTIC_COMPLETE', threshold=q, algorithm=algorithm,
            roc_auc=result['roc_auc'], pr_auc=result['pr_auc'], rows=result['n'])


def check(spark):
    # Real Spark estimators, categorical station metadata, monotone Platt and persistence.
    from pyspark.ml.feature import StringIndexer
    source = spark.createDataFrame([(float(i%20), str(i%3), float((i%20)+(i%3)*2>10)) for i in range(300)],
                                   ['rain_feature', 'station_id', 'label_gt_0mm'])
    indexed = StringIndexer(inputCol='station_id', outputCol='station_index', handleInvalid='keep').fit(source).transform(source)
    tiny = VectorAssembler(inputCols=['rain_feature', 'station_index'], outputCol='features').transform(indexed).withColumn('weight', F.lit(1.0))
    for algorithm in ALGORITHMS:
        est = estimator(algorithm, 0, 2)
        if algorithm == 'xgb':
            est.setParams(n_estimators=3)
        base = est.fit(tiny)
        sliced = VectorSlicer(inputCol='raw_0', outputCol='cal_features_0', indices=[1])
        cal = platt_estimator(0).fit(sliced.transform(base.transform(tiny)))
        assert cal.coefficients[0] >= 0
        pipe = PipelineModel(stages=[base, sliced, cal])
        dest = WORK/'checks'/algorithm
        pipe.write().overwrite().save(str(dest))
        a = pipe.transform(tiny.drop('label_gt_0mm')).select(vector_to_array('prob_0')[1].alias('p')).collect()
        b = PipelineModel.load(str(dest)).transform(tiny.drop('label_gt_0mm')).select(vector_to_array('prob_0')[1].alias('p')).collect()
        assert a == b and all(0 <= r['p'] <= 1 for r in a)
        log('SPARK_MODEL_FIXTURE_PASSED', algorithm=algorithm)


def check_artifacts(spark):
    """Check complete saved pipelines on real label-free rows and unseen station IDs."""
    manifest = json.loads((MODELS/'manifest.json').read_text())
    raw = spark.read.parquet(str(ROOT/'data/task2/features/2024/03')).select(
        'station_id', 'slot_epoch', *NUMERIC_FEATURES).orderBy('slot_epoch', 'station_id').limit(12)
    raw = raw.unionByName(raw.limit(3).withColumn('station_id', F.concat(F.lit('__unseen__'), F.col('station_id'))))
    raw = raw.persist(StorageLevel.DISK_ONLY)
    results = []
    try:
        for algorithm in ALGORITHMS:
            if algorithm == 'xgb':
                import xgboost.spark
            front, _ = front_model(representation(algorithm))
            for q in THRESHOLDS_MM:
                policy = manifest['selected_policy_by_algorithm_and_threshold'][algorithm][str(q)]
                pipeline = PipelineModel.load(str(MODELS/algorithm/policy/f'gt_{q}mm'))
                reference = front.transform(raw)
                for stage in pipeline.stages[-3:]:
                    reference = stage.transform(reference)
                columns = ['station_id', 'slot_epoch', vector_to_array(f'prob_{q}')[1].alias('p')]
                expected = reference.select(*columns).orderBy('station_id', 'slot_epoch').collect()
                actual = pipeline.transform(raw).select(*columns).orderBy('station_id', 'slot_epoch').collect()
                assert len(actual) == 15 and len(expected) == len(actual)
                delta = 0.0
                for a, b in zip(actual, expected):
                    assert a['station_id'] == b['station_id'] and a['slot_epoch'] == b['slot_epoch']
                    assert math.isfinite(a['p']) and 0 <= a['p'] <= 1
                    delta = max(delta, abs(a['p']-b['p']))
                assert delta <= 1e-12
                results.append({'algorithm': algorithm, 'threshold_mm': q, 'selected_policy': policy,
                                'label_free_rows': 15, 'unseen_station_rows': 3, 'max_probability_difference': delta})
                log('COMPLETE_PIPELINE_CHECK_PASSED', **results[-1])
        save_json(WORK/'artifact_checks.json', {'passed': True, 'checks': results})
    finally:
        raw.unpersist(blocking=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'smote-vectors', 'fit', 'select', 'test', 'check', 'artifacts', 'train-diagnostics'], required=True)
    parser.add_argument('--algorithm', choices=ALGORITHMS)
    parser.add_argument('--policy', choices=POLICIES)
    parser.add_argument('--job', default='manual')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.phase == 'select':
        select_models(); return
    spark = start_spark('SingaporeRainfallFullFeatureBenchmark')
    spark.conf.set('spark.sql.execution.arrow.maxRecordsPerBatch', '4096')
    spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
    try:
        if args.phase == 'prepare': prepare(spark)
        elif args.phase == 'smote-vectors': prepare_smote_vectors(spark)
        elif args.phase == 'fit': fit_group(spark, args.algorithm, args.policy)
        elif args.phase == 'test': test_algorithm(spark, args.algorithm)
        elif args.phase == 'train-diagnostics': train_diagnostics(spark)
        elif args.phase == 'check': check(spark)
        elif args.phase == 'artifacts': check_artifacts(spark)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
