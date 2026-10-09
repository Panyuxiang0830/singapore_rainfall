"""Score one prepared, causal historical feature row with a saved Spark model.

This entry point reuses archived as-of features; live history retrieval is a
separate deployment concern and is not claimed to be implemented here.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from task1_spark import ROOT, start_spark
from task2_features import NUMERIC_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', required=True)
    parser.add_argument('--time', required=True, help='ISO timestamp; without timezone interpreted as Asia/Singapore')
    parser.add_argument('--threshold', type=int, choices=[0, 5, 25], required=True)
    parser.add_argument('--group', choices=['calendar_station', 'own_history', 'neighbors'])
    parser.add_argument('--features-root', type=Path, default=ROOT/'data/task2/features')
    parser.add_argument('--model-root', type=Path, default=ROOT/'models/logistic_baseline')
    args = parser.parse_args()
    moment = datetime.fromisoformat(args.time)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo('Asia/Singapore'))
    moment = moment.astimezone(ZoneInfo('Asia/Singapore'))
    epoch = int(moment.timestamp())
    if moment.microsecond or epoch % 300:
        parser.error('Prediction time must be on the five-minute calendar grid')
    if args.group is None:
        manifest = json.loads((args.model_root/'manifest.json').read_text())
        group = manifest['selected_group_by_validation_pr_auc'][str(args.threshold)]
    else:
        group = args.group
    spark = start_spark('SingaporeRainfallLogisticPredict')
    try:
        rows = spark.read.parquet(str(args.features_root/f'{moment.year}/{moment.month:02d}')).filter(
            (F.col('station_id') == args.station) & (F.col('slot_epoch') == epoch))
        rows = rows.select('station_id', 'slot_epoch', *NUMERIC_FEATURES)
        if rows.count() != 1:
            raise ValueError('Exactly one prepared station/time row is required; generate as-of features first')
        model = PipelineModel.load(str(args.model_root/group/f'gt_{args.threshold}mm'))
        probability = model.transform(rows).select(vector_to_array(f'prob_{args.threshold}')[1]).first()[0]
        print(json.dumps({'station_id': args.station, 'prediction_time': moment.isoformat(),
                          'threshold_mm': args.threshold, 'horizon_minutes': 30,
                          'probability': probability, 'feature_group': group,
                          'availability_policy': 'conservative retained-archive as-of replay'}, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
