"""Predict an archived station/time row using a frozen final grid model."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from task1_spark import ROOT, start_spark
from task2_features import NUMERIC_FEATURES
from prediction_routes import resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', required=True)
    parser.add_argument('--time', required=True, help='ISO timestamp; naive times use Asia/Singapore')
    parser.add_argument('--threshold', type=int, choices=[0, 5, 25], required=True)
    parser.add_argument('--algorithm', choices=['logistic', 'svm', 'xgb'])
    parser.add_argument('--original-model', action='store_true', help='Use original comparison models instead of final grid models')
    parser.add_argument('--features-root', type=Path, default=ROOT/'data/task2/features')
    parser.add_argument('--model-root', type=Path, default=ROOT/'models/model_comparison_full')
    args = parser.parse_args()
    moment = datetime.fromisoformat(args.time)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo('Asia/Singapore'))
    moment = moment.astimezone(ZoneInfo('Asia/Singapore'))
    epoch = int(moment.timestamp())
    if moment.microsecond or epoch % 300:
        parser.error('Prediction time must be on the five-minute calendar grid')
    route = resolve(ROOT, args.model_root, args.threshold, args.algorithm, args.original_model)
    algorithm, policy = route['algorithm'], route['policy']
    if algorithm == 'xgb':
        import xgboost.spark  # Register the external Spark ML model class for load().
    spark = start_spark('SingaporeRainfallCalibratedPredict')
    try:
        rows = spark.read.parquet(str(args.features_root/f'{moment.year}/{moment.month:02d}')).filter(
            (F.col('station_id') == args.station) & (F.col('slot_epoch') == epoch))
        rows = rows.select('station_id', 'slot_epoch', *NUMERIC_FEATURES)
        if rows.count() != 1:
            raise ValueError('Exactly one prepared as-of feature row is required')
        model = PipelineModel.load(str(route['path']))
        probability = float(model.transform(rows).select(vector_to_array(route['probability_column'])[1]).first()[0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError('Invalid saved-model probability')
        print(json.dumps({'station_id': args.station, 'prediction_time': moment.isoformat(),
                          'threshold_mm': args.threshold, 'horizon_minutes': 30,
                          'probability': probability, 'algorithm': algorithm, 'training_policy': policy,
                          'probability_calibration': route['calibration'],
                          'model_version': route['version'],
                          'availability_policy': 'conservative retained-archive as-of replay'}, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
