"""Distance-aware extensions of the causal five-neighbour feature experiment."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'work/cloud'))

from pyspark.sql import functions as F

from task1_spark import save_json, start_spark
from task2_features import month_start, year_start
from experiment_spatial_features import DEST as SPATIAL_DEST, edges_for

DEST = ROOT / 'work/cloud/experiments/distance'
DISTANCE_NAMES = [
    'neighbor_rank1_distance_km', 'neighbor_rank2_distance_km',
    'neighbor_rank3_distance_km', 'neighbor_rank4_distance_km',
    'neighbor_rank5_distance_km', 'nearest_reporting_distance_km',
    'neighbor5_idw_rain', 'neighbor5_exp5km_rain',
]


def make_month(spark, year, month, metadata):
    output = DEST / f'features/{year}/{month:02d}'
    marker = DEST / f'summaries/{year}_{month:02d}.json'
    if (output / '_SUCCESS').exists() and marker.exists():
        print('DISTANCE_MONTH_RESUMED', year, month, flush=True)
        return
    start = month_start(year, month)
    end = month_start(year, month + 1) if month < 12 else year_start(year + 1)
    spatial = spark.read.parquet(str(SPATIAL_DEST / f'features/{year}/{month:02d}'))
    anchors = spatial.select('station_id', 'slot_epoch', *[
        f'neighbor_rank{rank}_rain' for rank in range(1, 6)])
    edges = [row for row in edges_for(metadata) if row[1] < end and row[2] > start]
    graph = spark.createDataFrame(edges,
        'target_station string, graph_start long, graph_end long, neighbor_station string, '
        'distance_km double, neighbor_rank int, dx_km double, dy_km double')
    pairs = anchors.alias('a').join(F.broadcast(graph).alias('g'),
        (F.col('a.station_id') == F.col('g.target_station')) &
        (F.col('a.slot_epoch') >= F.col('g.graph_start')) &
        (F.col('a.slot_epoch') < F.col('g.graph_end')), 'left')
    rank_distances = [F.max(F.when(F.col('neighbor_rank') == rank, F.col('distance_km'))).
                      alias(f'neighbor_rank{rank}_distance_km') for rank in range(1, 6)]
    distances = pairs.groupBy(F.col('a.station_id').alias('station_id'),
                              F.col('a.slot_epoch').alias('slot_epoch')).agg(*rank_distances)
    frame = anchors.join(distances, ['station_id', 'slot_epoch'], 'left')
    valid, idw_numerator, idw_denominator, exp_numerator, exp_denominator = [], [], [], [], []
    for rank in range(1, 6):
        rain = F.col(f'neighbor_rank{rank}_rain')
        distance = F.col(f'neighbor_rank{rank}_distance_km')
        available = rain.isNotNull() & distance.isNotNull()
        weight = 1.0 / F.greatest(distance, F.lit(0.1))
        decay = F.exp(-distance / F.lit(5.0))
        valid.append(F.when(available, distance))
        idw_numerator.append(F.when(available, rain * weight).otherwise(F.lit(0.0)))
        idw_denominator.append(F.when(available, weight).otherwise(F.lit(0.0)))
        exp_numerator.append(F.when(available, rain * decay).otherwise(F.lit(0.0)))
        exp_denominator.append(F.when(available, decay).otherwise(F.lit(0.0)))
    idw_denom = sum(idw_denominator)
    exp_denom = sum(exp_denominator)
    frame = frame.withColumn('nearest_reporting_distance_km', F.least(*valid))
    frame = frame.withColumn('neighbor5_idw_rain', F.when(idw_denom > 0, sum(idw_numerator) / idw_denom))
    frame = frame.withColumn('neighbor5_exp5km_rain', F.when(exp_denom > 0, sum(exp_numerator) / exp_denom))
    result = frame.select('station_id', 'slot_epoch', *DISTANCE_NAMES)
    result.write.mode('overwrite').parquet(str(output))
    checked = spark.read.parquet(str(output))
    actual = checked.agg(F.count('*').alias('rows'),
        F.sum(F.when(F.col('neighbor_rank1_distance_km') < 0, 1).otherwise(0)).alias('negative_distance')).first()
    expected = anchors.count()
    if actual.rows != expected or actual.negative_distance:
        raise RuntimeError(f'Distance feature integrity failure: {year}/{month}: {actual}, expected={expected}')
    save_json(marker, {'status': 'COMPLETE', 'year': year, 'month': month,
        'rows': actual.rows, 'negative_distances': actual.negative_distance,
        'features': DISTANCE_NAMES,
        'policy': 'Graph intervals use only station coordinates available at prediction time; '
                  'rain weights use only nonmissing causal five-neighbour readings.'})
    print('DISTANCE_MONTH_COMPLETE', year, month, actual.rows, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=range(2017, 2025), required=True)
    parser.add_argument('--months', nargs='+', type=int, default=list(range(1, 13)))
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'work/task2/feature_metadata.json').read_text())['stations']
    spark = start_spark(f'RainfallDistance_{args.year}')
    try:
        for month in args.months:
            make_month(spark, args.year, month, metadata)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
