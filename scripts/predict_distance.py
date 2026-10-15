"""Replay one archived five-minute slot with the final distance-aware model."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from task1_spark import ROOT, start_spark
from task2_features import NUMERIC_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', required=True)
    parser.add_argument('--time', required=True, help='ISO timestamp; naive time means Singapore local time')
    parser.add_argument('--threshold', required=True, type=int, choices=(0, 5, 25))
    args = parser.parse_args()
    moment = datetime.fromisoformat(args.time)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo('Asia/Singapore'))
    moment = moment.astimezone(ZoneInfo('Asia/Singapore'))
    epoch = int(moment.timestamp())
    if moment.microsecond or epoch % 300 or moment.year not in range(2017, 2025):
        parser.error('Time must be an archived 2017--2024 five-minute slot')

    import xgboost.spark  # Register the Spark XGBoost model class for deserialization.
    manifest = json.loads((ROOT/'models/full_variants/distance/manifest.json').read_text())
    state = json.loads((ROOT/'results/distance_full/feature_preprocessing.json').read_text())
    prep = json.loads((ROOT/'results/distance_full/preparation.json').read_text())
    q = str(args.threshold)
    algorithm = manifest['selected_algorithm_by_threshold'][q]
    policy = manifest['selected_policy_by_algorithm_and_threshold'][algorithm][q]
    selected = ROOT/'models/full_variants/distance'/algorithm/policy/f'gt_{q}mm'
    spatial = ROOT/'work/cloud/experiments/spatial/features'/str(moment.year)/f'{moment.month:02d}'
    distance = ROOT/'work/cloud/experiments/distance/features'/str(moment.year)/f'{moment.month:02d}'
    raw = ROOT/'data/task2/features'/str(moment.year)/f'{moment.month:02d}'
    for path in (raw, spatial, distance):
        if not (path/'_SUCCESS').exists():
            raise FileNotFoundError(f'Prepared causal feature month is missing: {path}')

    spark = start_spark('RainfallDistancePrediction')
    try:
        key = (F.col('station_id') == args.station) & (F.col('slot_epoch') == epoch)
        row = spark.read.parquet(str(raw)).filter(key).select('station_id', 'slot_epoch', *NUMERIC_FEATURES)
        if row.count() != 1:
            raise ValueError('Exactly one archived station/time feature row is required')
        names = state['feature_names']
        spatial_names, distance_names = names[:14], names[14:]
        side1 = spark.read.parquet(str(spatial)).filter(key).select('station_id', 'slot_epoch', *spatial_names)
        side2 = spark.read.parquet(str(distance)).filter(key).select('station_id', 'slot_epoch', *distance_names)
        row = row.join(side1, ['station_id', 'slot_epoch'], 'left').join(side2, ['station_id', 'slot_epoch'], 'left')
        if row.filter(F.col(names[0]).isNull()).count():
            raise ValueError('Causal spatial sidecar row is missing')
        base = PipelineModel.load(str(ROOT/'models/logistic_baseline_full/preprocessing'))
        columns = [f'{c}__filled' for c in prep['active_numeric']]
        columns += [f'{c}__missing' for c in prep['missing_indicators']]
        columns += ['station_index']
        row = base.transform(row)
        row = VectorAssembler(inputCols=columns, outputCol='features', handleInvalid='error').transform(row)
        for name in names:
            missing = F.col(name).isNull() | F.isnan(F.col(name))
            row = row.withColumn(name+'__missing', F.when(missing, F.lit(1.0)).otherwise(F.lit(0.0)))
            row = row.withColumn(name+'__filled', F.when(missing, F.lit(float(state['mean_by_feature'][name]))).otherwise(F.col(name)))
        row = VectorAssembler(inputCols=['features']+[x for name in names for x in (name+'__filled', name+'__missing')],
                              outputCol='enhanced_features', handleInvalid='error').transform(row)
        row = row.drop('features').withColumnRenamed('enhanced_features', 'features')
        if row.select('features').first()[0].size != 117:
            raise RuntimeError('Final tree feature vector must have 117 dimensions')
        model = PipelineModel.load(str(selected))
        for stage in model.stages[-3:]:
            row = stage.transform(row)
        probability = float(row.select(vector_to_array(f'prob_{q}')[1]).first()[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError('Invalid calibrated probability')
        print(json.dumps({'station_id': args.station, 'prediction_time': moment.isoformat(),
                          'threshold_mm': args.threshold, 'horizon_minutes': 30,
                          'probability': probability, 'algorithm': algorithm,
                          'training_policy': policy, 'feature_variant': 'distance'}, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
