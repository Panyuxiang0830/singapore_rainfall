"""Recover ROC/PR curves for the validation-selected final models; no refitting.

Exact AUCs use every eligible 2024 row. Only bounded curve aggregates leave
Spark. Working JSON is excluded from the deliverable; report coordinates are
embedded directly in the bilingual LaTeX sources.
"""
import argparse
import fcntl
import json
import math
import os
import sys
import time

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from task1_spark import ROOT, save_json, start_spark

WORK = ROOT / 'work/task2/model_comparison'


def area(points):
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(points, points[1:]))


def curve_frame(spark, rdd):
    enc = spark.sparkContext._jvm.org.apache.spark.sql.Encoders
    encoder = enc.tuple(enc.DOUBLE(), enc.DOUBLE())
    ds = spark._jsparkSession.createDataset(rdd, encoder)
    return DataFrame(ds.toDF(), spark).select(F.col('_1').alias('x'), F.col('_2').alias('y'))


def bounded_points(frame, kind, bins=4096):
    # Preserve the first/last point and vertical extrema in each x interval.
    # Neither ROC/PR AUC computation nor test rows are downsampled.
    # Each returned coordinate is an actual point, never a fabricated score.
    xy = F.struct('x', 'y')
    yx = F.struct('y', 'x')
    groups = frame.withColumn('bucket', F.floor(F.col('x')*bins)).groupBy('bucket').agg(
        F.min(xy).alias('first'), F.max(xy).alias('last'),
        F.min(yx).alias('low'), F.max(yx).alias('high')).collect()
    values = set()
    for row in groups:
        for key in ['first', 'last', 'low', 'high']:
            v = row[key]
            values.add((float(v['x']), float(v['y'])))
    return sorted(values, key=lambda p: (p[0], p[1] if kind == 'roc' else -p[1]))


def simplify(points, epsilon):
    """Iterative Douglas-Peucker simplification in the unit square."""
    keep = {0, len(points)-1}
    stack = [(0, len(points)-1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo+1:
            continue
        x0, y0 = points[lo]; x1, y1 = points[hi]
        dx, dy = x1-x0, y1-y0
        length2 = dx*dx+dy*dy
        best, index = -1., None
        for i in range(lo+1, hi):
            x, y = points[i]
            t = max(0., min(1., ((x-x0)*dx+(y-y0)*dy)/length2)) if length2 else 0.
            distance2 = (x-x0-t*dx)**2+(y-y0-t*dy)**2
            if distance2 > best:
                best, index = distance2, i
        if best > epsilon*epsilon:
            keep.add(index); stack += [(lo, index), (index, hi)]
    return [points[i] for i in sorted(keep)]


def curves(spark, scores):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='p', labelCol='y', numBins=0)
    evaluator._transfer_params_to_java()
    native = evaluator._java_obj.getMetrics(scores._jdf)
    try:
        result = {'roc_auc': float(native.areaUnderROC()), 'pr_auc': float(native.areaUnderPR())}
        for kind in ['roc', 'pr']:
            frame = curve_frame(spark, native.roc() if kind == 'roc' else native.pr())
            exact = result[kind+'_auc']
            full = bounded_points(frame, kind)
            tolerance = min(1e-4, exact*0.002)
            if abs(area(full)-exact) > tolerance:
                full = bounded_points(frame, kind, bins=32768)
            if abs(area(full)-exact) > tolerance:
                raise RuntimeError(f'{kind}: bounded curve area differs from exact AUC')
            reduced = full
            for epsilon in [0.0005, 0.0002, 0.0001, 0.00005]:
                candidate = simplify(full, epsilon)
                if abs(area(candidate)-exact) <= tolerance:
                    reduced = candidate
                    break
            result[kind] = reduced
            result[kind+'_plot_area'] = area(reduced)
            result[kind+'_plot_points'] = len(reduced)
        return result
    finally:
        native.unpersist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not args.worker:
        import shutil
        import subprocess
        from run_task1 import run
        WORK.mkdir(parents=True, exist_ok=True)
        lock = (WORK/'run.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if hasattr(os, 'sched_getaffinity'):
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        os.setpriority(os.PRIO_PROCESS, 0, 19)
        if shutil.which('ionice'):
            subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
        env = os.environ.copy()
        env.update({'JAVA_HOME': str(ROOT/'.runtime/java'), 'PYTHONPATH': str(ROOT/'.runtime/python'),
                    'JAVA_TOOL_OPTIONS': '-XX:ActiveProcessorCount=2', 'OPENBLAS_NUM_THREADS': '1',
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '',
                    'SPARK_LOCAL_IP': '127.0.0.1', 'PYSPARK_PYTHON': sys.executable,
                    'TMPDIR': str(ROOT/'work/tmp')})
        env['PATH'] = env['JAVA_HOME']+'/bin:'+env['PATH']
        run([sys.executable, '-u', 'scripts/task2_curves.py', '--worker'], env)
        return
    spark = start_spark('SingaporeRainfallFinalROCPR')
    try:
        # Check the Scala RDD -> Dataset bridge and curve reduction on known ranks.
        fixture = spark.createDataFrame([(0.9, 1.), (0.8, 0.), (0.7, 1.), (0.1, 0.)], ['p', 'y'])
        checked = curves(spark, fixture)
        assert math.isclose(checked['roc_auc'], .75)
        print('CURVE_FIXTURE_PASSED', flush=True)
        manifest = json.loads((WORK/'selection.json').read_text())
        for q in [0, 5, 25]:
            dest = WORK/'curves'/f'final_gt_{q}mm.json'
            algorithm = manifest['selected_algorithm_by_threshold'][str(q)]
            policy = manifest['selected_policy_by_algorithm_and_threshold'][algorithm][str(q)]
            started = time.monotonic()
            kind = 'linear' if algorithm in ['logistic', 'svm'] else 'tree'
            frame = spark.read.parquet(str(WORK/'vectors'/'test'/kind))
            pipeline = PipelineModel.load(str(ROOT/'models/model_comparison'/algorithm/policy/f'gt_{q}mm'))
            for stage in pipeline.stages[-3:]:
                frame = stage.transform(frame)
            scores = frame.select(vector_to_array(f'prob_{q}')[1].alias('p'),
                                  F.col(f'label_gt_{q}mm').alias('y')).persist(StorageLevel.DISK_ONLY)
            try:
                stats = scores.agg(F.count('*').alias('n'), F.sum('y').alias('positives')).first().asDict()
                result = curves(spark, scores)
                expected = json.loads((WORK/'test'/f'{algorithm}_gt_{q}mm.json').read_text())
                assert stats['n'] == expected['n'] == 6374476
                for metric in ['roc_auc', 'pr_auc']:
                    if not math.isclose(result[metric], expected[metric], abs_tol=1e-8):
                        raise RuntimeError(f'{q}: existing {metric} does not match recovered scores')
                result.update(stats, threshold_mm=q, algorithm=algorithm, policy=policy,
                              prevalence=stats['positives']/stats['n'], seconds=time.monotonic()-started)
                save_json(dest, result)
                print('FINAL_CURVES_COMPLETE', q, algorithm, policy, result['roc_auc'], result['pr_auc'],
                      result['roc_plot_points'], result['pr_plot_points'], flush=True)
            finally:
                scores.unpersist(blocking=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
