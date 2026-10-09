"""Small meaningful checks for metric definitions and predictor leakage."""
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from task2_logistic import GROUPS, MODELS, WORK, metrics, start_spark, read_years, save_json
from task2_features import NUMERIC_FEATURES


def main():
    for inputs in GROUPS.values():
        assert set(inputs) <= set(NUMERIC_FEATURES)
        assert not any(c.startswith(('label_', 'future_')) for c in inputs)
    spark = start_spark('RainfallLogisticMetricChecks')
    try:
        scores = spark.createDataFrame([(0.1, 0.0), (0.35, 0.0), (0.4, 1.0), (0.8, 1.0)], ['p', 'y'])
        result = metrics(scores, 0.5)
        assert abs(result['roc_auc']-1) < 1e-12
        assert abs(result['pr_auc']-1) < 1e-12
        direct = BinaryClassificationEvaluator(rawPredictionCol='p', labelCol='y', numBins=0)
        assert result['roc_auc'] == direct.setMetricName('areaUnderROC').evaluate(scores)
        assert result['pr_auc'] == direct.setMetricName('areaUnderPR').evaluate(scores)
        assert abs(result['brier']-0.133125) < 1e-12
        assert result['f1'] == 1 and result['confusion'] == {'tp': 2, 'fp': 0, 'fn': 0, 'tn': 2}
        assert sum(r['n'] for r in result['calibration_curve']) == 4
        tied = metrics(scores.withColumn('p', F.lit(0.5)), 0.5, 0.5)
        assert abs(tied['roc_auc']-0.5) < 1e-12 and abs(tied['brier']-0.25) < 1e-12
        assert abs(tied['pr_auc']-0.5) < 1e-12
        try:
            metrics(scores.withColumn('p', F.lit(-0.1)), 0.5)
        except RuntimeError as error:
            assert 'Invalid score' in str(error)
        else:
            raise AssertionError('Invalid probabilities must be rejected')
        try:
            metrics(spark.createDataFrame([(0.2, None)], 'p double, y double'), 0.5)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing labels must be rejected')
        if (MODELS/'manifest.json').exists():
            manifest = json.loads((MODELS/'manifest.json').read_text())
            probe = read_years(spark, [2024]).select('station_id', 'slot_epoch', *NUMERIC_FEATURES).limit(1).cache()
            key = probe.select('station_id', 'slot_epoch').first()
            smoke = {'station_id': key['station_id'], 'prediction_time': datetime.fromtimestamp(
                key['slot_epoch'], ZoneInfo('Asia/Singapore')).isoformat(), 'probabilities': {}}
            for q in [0, 5, 25]:
                group = manifest['selected_group_by_validation_pr_auc'][str(q)]
                model = PipelineModel.load(str(MODELS/group/f'gt_{q}mm'))
                p = model.transform(probe).select(vector_to_array(f'prob_{q}')[1]).first()[0]
                unknown = model.transform(probe.withColumn('station_id', F.lit('__UNSEEN_STATION__'))).select(
                    vector_to_array(f'prob_{q}')[1]).first()[0]
                assert math.isfinite(p) and 0 <= p <= 1
                assert math.isfinite(unknown) and 0 <= unknown <= 1
                smoke['probabilities'][str(q)] = p
            probe.unpersist()
            save_json(WORK/'prediction_smoke.json', smoke)
            print('LABEL_FREE_AND_UNKNOWN_STATION_MODEL_CHECKS_PASSED', json.dumps(smoke), flush=True)
        print('LOGISTIC_METRIC_CHECKS_PASSED', flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
