"""Spark MLlib L2 logistic feature ablations with chronological evaluation.

Working samples, scores and metrics are not submission artifacts. Saved models
include fitted preprocessing. No validation/test data fits preprocessing or LR.
"""
import argparse
import json
import time

from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from task1_spark import ROOT, save_json, start_spark
from task2_features import FEATURE_VERSION, NUMERIC_FEATURES, THRESHOLDS_MM

VERSION = 'lr_l2_ablation_v1'
WORK = ROOT / 'work/task2/logistic'
MODELS = ROOT / 'models/logistic_baseline'
GROUPS = {
    'calendar_station': NUMERIC_FEATURES[:11],
    'own_history': [c for c in NUMERIC_FEATURES if not c.startswith('neighbor_')],
    'neighbors': NUMERIC_FEATURES,
}
LABELS = [f'label_gt_{q}mm' for q in THRESHOLDS_MM]
SAMPLE_MODULUS = 1000000
HASH_SEED = 5208


def log(event, **details):
    print(event, json.dumps(details, default=str), flush=True)


def read_years(spark, years):
    paths = [str(ROOT / f'data/task2/features/{y}/{m:02d}') for y in years for m in range(1, 13)]
    return spark.read.parquet(*paths).select('station_id', 'slot_epoch', *NUMERIC_FEATURES, *LABELS)


def settings(args):
    return {'version': VERSION, 'feature_version': FEATURE_VERSION,
            'train_fraction': args.train_fraction, 'hash_seed': HASH_SEED,
            'reg_param': args.reg_param, 'max_iter': args.max_iter,
            'tol': args.tol, 'thresholds_mm': list(THRESHOLDS_MM)}


def prepare(spark, args):
    config = settings(args)
    marker = WORK / 'preparation.json'
    if marker.exists():
        stored = json.loads(marker.read_text())
        if stored['config'] != config:
            raise RuntimeError('Existing experiment has different settings; use a separate work/model directory')
        log('PREPARATION_RESUMED', rows=stored['train_rows'])
        return stored
    started = time.monotonic()
    raw = read_years(spark, range(2017, 2023))
    # Stable inclusion independent of labels, file ordering and partitioning.
    sample = raw.filter(F.pmod(F.xxhash64('station_id', 'slot_epoch', F.lit(HASH_SEED)),
                              F.lit(SAMPLE_MODULUS)) < int(args.train_fraction * SAMPLE_MODULUS))
    dest = WORK / 'training_sample'
    sample.write.mode('overwrite').parquet(str(dest))
    sample = spark.read.parquet(str(dest))
    expressions = [F.count('*').alias('n')] + [F.sum(c).alias(c) for c in LABELS]
    for i, c in enumerate(NUMERIC_FEATURES):
        expressions += [F.count(c).alias(f'n{i}'), F.min(c).alias(f'lo{i}'), F.max(c).alias(f'hi{i}')]
    stats = sample.agg(*expressions).first().asDict()
    if any(not 0 < stats[c] < stats['n'] for c in LABELS):
        raise RuntimeError('Every classifier needs both classes in training')
    active = [c for i, c in enumerate(NUMERIC_FEATURES)
              if stats[f'n{i}'] and stats[f'lo{i}'] != stats[f'hi{i}']]
    missing = [c for i, c in enumerate(NUMERIC_FEATURES) if 0 < stats[f'n{i}'] < stats['n']]
    # Preserve missingness even if all observed values of a column are constant.
    missing_stage = SQLTransformer(statement='SELECT *, ' + ', '.join(
        f'CASE WHEN `{c}` IS NULL OR isnan(`{c}`) THEN 1.0 ELSE 0.0 END AS `{c}__missing`'
        for c in missing) + ' FROM __THIS__') if missing else None
    stages = ([missing_stage] if missing_stage else []) + [
        Imputer(strategy='mean', inputCols=active, outputCols=[f'{c}__filled' for c in active]),
        StringIndexer(inputCol='station_id', outputCol='station_index', handleInvalid='keep',
                      stringOrderType='alphabetAsc'),
        OneHotEncoder(inputCol='station_index', outputCol='station_vector',
                      handleInvalid='keep', dropLast=False),
    ]
    preprocessing = Pipeline(stages=stages).fit(sample)
    preprocessing.write().overwrite().save(str(MODELS / 'preprocessing'))
    result = {'config': config, 'train_rows': stats['n'],
              'train_positives': {str(q): stats[f'label_gt_{q}mm'] for q in THRESHOLDS_MM},
              'active_numeric': active, 'missing_indicators': missing,
              'dropped_numeric': [c for c in NUMERIC_FEATURES if c not in active],
              'train_years': list(range(2017, 2023)), 'validation_year': 2023, 'test_year': 2024,
              'sampling_policy': 'label-independent fixed station/time xxhash64 inclusion, same rows for all classifiers',
              'class_weighting': 'none; natural training proportions retained',
              'preparation_seconds': time.monotonic() - started}
    save_json(marker, result)
    log('PREPARATION_COMPLETE', **result)
    return result


def aucs(scores):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='p', labelCol='y', numBins=0)
    # Spark 3.5.8's public Scala getMetrics shares the sorted cumulative counts
    # across both areas. The Python evaluator otherwise repeats that sort.
    evaluator._transfer_params_to_java()
    native = evaluator._java_obj.getMetrics(scores._jdf)
    try:
        return {'roc_auc': native.areaUnderROC(), 'pr_auc': native.areaUnderPR()}
    finally:
        native.unpersist()


def histogram(scores):
    # Fine logarithmic bins support rare-event cutoff selection; AUC remains exact.
    # At most 1,201 aggregate records leave Spark, never the full score rows.
    bucket = F.floor(F.log10(F.greatest(F.col('p'), F.lit(1e-6))) * 200).cast('int')
    return [r.asDict() for r in scores.withColumn('bin', bucket).groupBy('bin').agg(
        F.count('*').alias('n'), F.sum('y').alias('positives'), F.sum('p').alias('sum_p'),
        F.sum(F.pow(F.col('p') - F.col('y'), 2)).alias('squared_error'),
        F.sum(-F.col('y') * F.log(F.greatest('p', F.lit(1e-15)))
              - (1-F.col('y')) * F.log(F.greatest(1-F.col('p'), F.lit(1e-15)))).alias('logloss'),
    ).collect()]


def choose_cutoff(hist):
    positive = sum(r['positives'] for r in hist)
    tp = fp = 0
    best = {'f1': -1, 'cutoff': 1.0}
    for r in sorted(hist, key=lambda r: r['bin'], reverse=True):
        tp += r['positives']; fp += r['n'] - r['positives']
        f1 = 2*tp / (tp+fp+positive) if tp+fp+positive else 0.0
        if f1 > best['f1']:
            best = {'f1': f1, 'cutoff': 0.0 if r['bin'] == -1200 else 10**(r['bin']/200)}
    return best['cutoff']


def metrics(scores, train_rate, cutoff=None):
    begun = time.monotonic()
    scores = scores.select('p', 'y').persist(StorageLevel.DISK_ONLY)
    try:
        invalid = scores.filter(F.col('p').isNull() | F.isnan('p') |
                                ~F.col('p').between(0, 1) | F.col('y').isNull() | ~F.col('y').isin(0.0, 1.0)).limit(1).count()
        if invalid:
            raise RuntimeError('Invalid score or label')
        hist = histogram(scores)
        n = sum(r['n'] for r in hist); positive = sum(r['positives'] for r in hist)
        cutoff = choose_cutoff(hist) if cutoff is None else cutoff
        conf = scores.agg(
            F.sum(F.when((F.col('p') >= cutoff) & (F.col('y') == 1), 1).otherwise(0)).alias('tp'),
            F.sum(F.when((F.col('p') >= cutoff) & (F.col('y') == 0), 1).otherwise(0)).alias('fp')).first()
        tp, fp = conf['tp'], conf['fp']; fn = positive-tp; tn = n-positive-fp
        result = {'n': n, 'positives': positive, 'prevalence': positive/n,
                  **aucs(scores), 'brier': sum(r['squared_error'] for r in hist)/n,
                  'log_loss': sum(r['logloss'] for r in hist)/n,
                  'mean_probability': sum(r['sum_p'] for r in hist)/n,
                  'constant_baseline_roc_auc': 0.5, 'constant_baseline_pr_auc': positive/n,
                  'constant_baseline_brier': ((1-train_rate)**2*positive+train_rate**2*(n-positive))/n,
                  'probability_cutoff': cutoff, 'precision': tp/(tp+fp) if tp+fp else 0,
                  'recall': tp/positive if positive else 0,
                  'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                  'accuracy': (tp+tn)/n,
                  'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}}
        # Equal-count reliability groups made from the fine log histogram.
        curve, count, pos, pred = [], 0, 0, 0.0
        for r in sorted(hist, key=lambda r: r['bin']):
            count += r['n']; pos += r['positives']; pred += r['sum_p']
            if count >= n/20:
                curve.append({'n': count, 'mean_probability': pred/count, 'observed_frequency': pos/count})
                count, pos, pred = 0, 0, 0.0
        if count:
            curve.append({'n': count, 'mean_probability': pred/count, 'observed_frequency': pos/count})
        result['calibration_curve'] = curve
        result['evaluation_seconds'] = time.monotonic()-begun
        return result
    finally:
        scores.unpersist(blocking=True)


def fit_group(spark, args):
    prepared = json.loads((WORK / 'preparation.json').read_text())
    if prepared['config'] != settings(args):
        raise RuntimeError('Settings mismatch with prepared experiment')
    marker = WORK / f'{args.group}.json'
    if marker.exists():
        log('GROUP_RESUMED', group=args.group)
        return
    start = time.monotonic()
    preprocess = PipelineModel.load(str(MODELS / 'preprocessing'))
    allowed = GROUPS[args.group]
    assembled = [f'{c}__filled' for c in prepared['active_numeric'] if c in allowed]
    assembled += [f'{c}__missing' for c in prepared['missing_indicators'] if c in allowed]
    assembler = VectorAssembler(inputCols=assembled+['station_vector'], outputCol='features', handleInvalid='error')
    front = PipelineModel(stages=preprocess.stages+[assembler])
    train = front.transform(spark.read.parquet(str(WORK/'training_sample'))).select(
        'features', *LABELS).persist(StorageLevel.DISK_ONLY)
    log('GROUP_START', group=args.group, rows=train.count(), assembled_numeric=len(assembled))
    models, results = {}, {}
    try:
        for q in THRESHOLDS_MM:
            name = f'gt_{q}mm'
            state = WORK / f'{args.group}_{name}.json'
            dest = MODELS / args.group / name
            if state.exists():
                old = json.loads(state.read_text())
                models[q] = PipelineModel.load(str(dest)).stages[-1]
                results[str(q)] = old
                log('FIT_RESUMED', group=args.group, threshold=q)
                continue
            begun = time.monotonic()
            lr = LogisticRegression(labelCol=f'label_{name}', featuresCol='features',
                    probabilityCol=f'prob_{q}', rawPredictionCol=f'raw_{q}', predictionCol=f'pred_{q}',
                    regParam=args.reg_param, elasticNetParam=0.0, standardization=True,
                    maxIter=args.max_iter, tol=args.tol, aggregationDepth=2)
            fitted = lr.fit(train)
            elapsed = time.monotonic()-begun
            loss = list(fitted.summary.objectiveHistory)
            # Save a complete deployable Spark pipeline, not coefficients alone.
            pipeline = PipelineModel(stages=front.stages+[fitted])
            pipeline.write().overwrite().save(str(dest))
            # Round-trip scores prove serialization includes fitted preprocessing.
            probe = read_years(spark, [2024]).limit(5)
            saved = PipelineModel.load(str(dest))
            a = pipeline.transform(probe).select(vector_to_array(f'prob_{q}')[1].alias('p')).collect()
            b = saved.transform(probe).select(vector_to_array(f'prob_{q}')[1].alias('p')).collect()
            if a != b:
                raise RuntimeError('Saved-model score round trip failed')
            result = {'threshold_mm': q, 'group': args.group, 'fit_seconds': elapsed,
                      'iterations': fitted.summary.totalIterations, 'objective_history': loss,
                      'iteration_cap_reached': fitted.summary.totalIterations >= args.max_iter,
                      'numeric_and_missing_features': assembled,
                      'vector_dimension': fitted.numFeatures, 'model_relative_path': str(dest.relative_to(ROOT)),
                      'serialization_roundtrip_passed': True}
            save_json(state, result)
            models[q] = fitted; results[str(q)] = result
            log('FIT_COMPLETE', group=args.group, threshold=q, fit_seconds=elapsed,
                iterations=result['iterations'], final_loss=loss[-1], cap=result['iteration_cap_reached'])
        for split, frame in [('train', train), ('validation', read_years(spark, [2023])),
                             ('test', read_years(spark, [2024]))]:
            scored = frame if split == 'train' else front.transform(frame)
            for q, fitted in models.items():
                scored = fitted.transform(scored)
            scored = scored.select(*LABELS, *[vector_to_array(f'prob_{q}')[1].alias(f'p_{q}')
                                              for q in THRESHOLDS_MM]).persist(StorageLevel.DISK_ONLY)
            try:
                for q in THRESHOLDS_MM:
                    state = WORK / f'{args.group}_gt_{q}mm.json'
                    result = results[str(q)]
                    if split in result:
                        continue
                    rate = prepared['train_positives'][str(q)]/prepared['train_rows']
                    cutoff = result['validation']['probability_cutoff'] if split == 'test' else (
                        0.5 if split == 'train' else None)
                    result[split] = metrics(scored.select(F.col(f'p_{q}').alias('p'),
                                           F.col(f'label_gt_{q}mm').alias('y')), rate, cutoff)
                    save_json(state, result)
                    log('METRICS_COMPLETE', group=args.group, threshold=q, split=split,
                        **{k: v for k, v in result[split].items() if k not in ('calibration_curve',)})
                if split == 'test':
                    monotonic = scored.agg(F.sum(F.when((F.col('p_25')>F.col('p_5')) |
                                                        (F.col('p_5')>F.col('p_0')), 1).otherwise(0)).alias('violations')).first()[0]
            finally:
                scored.unpersist(blocking=True)
        result = {'group': args.group, 'config': settings(args), 'models': results,
                  'total_seconds': time.monotonic()-start, 'test_probability_order_violations': monotonic,
                  'auc_implementation': 'Spark BinaryClassificationEvaluator, numBins=0 (exact scores); trapezoidal PR area, not average precision',
                  'calibration_policy': 'raw natural-proportion logistic probabilities; reliability diagnostics, no post-hoc calibrator fitted',
                  'cutoff_policy': 'maximize F1 on full 2023 validation logarithmic score histogram; fixed for 2024',
                  'selection_policy': 'fixed regularization; compare feature groups by 2023 PR-AUC; all 2024 results are descriptive ablations'}
        save_json(marker, result)
        log('GROUP_COMPLETE', group=args.group, total_seconds=result['total_seconds'])
    finally:
        train.unpersist(blocking=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--train-fraction', type=float, default=0.05)
    parser.add_argument('--reg-param', type=float, default=0.001)
    parser.add_argument('--max-iter', type=int, default=80)
    parser.add_argument('--tol', type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1 or args.reg_param < 0 or args.max_iter < 1:
        parser.error('Invalid sampling/optimization parameters')
    if not args.prepare and not args.group:
        parser.error('Choose --prepare or --group')
    WORK.mkdir(parents=True, exist_ok=True)
    spark = start_spark('SingaporeRainfallLogisticBaseline')
    try:
        if args.prepare:
            prepare(spark, args)
        else:
            fit_group(spark, args)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
