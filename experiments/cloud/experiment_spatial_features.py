"""Causal five-neighbor features in an isolated experiment data set."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from pyspark.sql import functions as F

from task1_spark import save_json, start_spark
from task2_features import (LOOKBACK_SECONDS, build_snapshots, haversine_km,
                            metadata_frame, month_start, year_start)

DEST = ROOT / 'work/cloud/experiments/spatial'
SPATIAL_NAMES = [
    'neighbor5_recent_count', 'neighbor4_5_recent_count',
    'neighbor4_5_rain_max', 'neighbor4_5_wet_count',
    'neighbor5_rain_change_max', 'neighbor5_rising_count',
    'neighbor5_rain30_max', 'neighbor5_east_signal', 'neighbor5_north_signal',
    'neighbor_rank1_rain', 'neighbor_rank2_rain', 'neighbor_rank3_rain',
    'neighbor_rank4_rain', 'neighbor_rank5_rain',
]


def edges_for(stations):
    events = sorted({((s['available_epoch'] + 299) // 300) * 300 for s in stations})
    edges = []
    for i, start in enumerate(events):
        end = events[i + 1] if i + 1 < len(events) else year_start(2025)
        known = [s for s in stations if s['available_epoch'] <= start]
        for target in known:
            neighbors = sorted((s for s in known if s['station_id'] != target['station_id']),
                               key=lambda s: (haversine_km(target, s), s['station_id']))[:5]
            for rank, neighbor in enumerate(neighbors, 1):
                distance = haversine_km(target, neighbor)
                dx = 111.32 * math.cos(math.radians(target['lat'])) * (neighbor['lon']-target['lon'])
                dy = 111.32 * (neighbor['lat']-target['lat'])
                edges.append((target['station_id'], start, end, neighbor['station_id'],
                              float(distance), rank, float(dx), float(dy)))
    return edges


def make_month(spark, year, month, metadata):
    output = DEST / f'features/{year}/{month:02d}'
    marker = DEST / f'summaries/{year}_{month:02d}.json'
    if (output / '_SUCCESS').exists() and marker.exists():
        print('SPATIAL_MONTH_RESUMED', year, month, flush=True)
        return
    start = month_start(year, month)
    end = month_start(year, month + 1) if month < 12 else year_start(year + 1)
    source = ROOT / f'data/task2/features/{year}/{month:02d}'
    anchors = spark.read.parquet(str(source)).select('station_id', 'slot_epoch', 'geometry_changed')
    ids = {r.station_id for r in anchors.select('station_id').distinct().collect()}
    visible = [s for s in metadata if s['available_epoch'] < end or s['station_id'] in ids]
    station_frame = metadata_frame(spark, visible)
    grid = spark.range(start - LOOKBACK_SECONDS, end, 300).select(F.col('id').alias('slot_epoch')).crossJoin(F.broadcast(station_frame))
    inputs = [str(ROOT / f'work/task2/feature_inputs/{year}/input_month={month}')]
    if month > 1:
        inputs.append(str(ROOT / f'work/task2/feature_inputs/{year}/input_month={month-1}'))
    inputs = [x for x in inputs if Path(x).exists()]
    if not inputs:
        inputs = [str(ROOT / f'data/annual/{year}')]
        if month == 1 and year > 2017:
            inputs.append(str(ROOT / f'data/annual/{year-1}'))
    observations = spark.read.parquet(*inputs).filter(
        (F.col('slot_epoch') >= start - LOOKBACK_SECONDS) & (F.col('slot_epoch') < end)).select(
        'station_id', 'slot_epoch', 'obs_epoch', 'update_epoch', 'rain_mm', 'lat', 'lon')
    temporary = DEST / f'snapshots/{year}_{month:02d}'
    build_snapshots(observations, grid).filter(F.col('slot_epoch') >= start).write.mode('overwrite').parquet(str(temporary))
    snapshot = spark.read.parquet(str(temporary))
    edges = [e for e in edges_for(metadata) if e[1] < end and e[2] > start]
    graph = spark.createDataFrame(edges,
        'target_station string, graph_start long, graph_end long, neighbor_station string, '
        'distance_km double, neighbor_rank int, dx_km double, dy_km double')
    pair = anchors.alias('a').join(F.broadcast(graph).alias('g'),
        (F.col('a.station_id') == F.col('g.target_station')) &
        (F.col('a.slot_epoch') >= F.col('g.graph_start')) &
        (F.col('a.slot_epoch') < F.col('g.graph_end')), 'left').select(
        F.col('a.station_id'), F.col('a.slot_epoch'), F.col('a.geometry_changed').alias('target_geometry_changed'),
        F.col('g.neighbor_station'), F.col('g.neighbor_rank'), F.col('g.distance_km'),
        F.col('g.dx_km'), F.col('g.dy_km'))
    neighbor = snapshot.select(F.col('station_id').alias('neighbor_station'), 'slot_epoch',
        F.col('geometry_changed').alias('neighbor_geometry_changed'),
        F.col('has_recent_reading').alias('neighbor_recent'),
        F.col('latest_rain_mm').alias('neighbor_rain'),
        F.col('rain_change_5m').alias('neighbor_change'),
        F.col('rain_30m').alias('neighbor_rain30'),
        F.col('max_used_update_epoch').alias('neighbor_max_update'),
        F.col('latest_obs_epoch').alias('neighbor_latest_obs'))
    pair = pair.join(neighbor, ['neighbor_station', 'slot_epoch'], 'left')
    good = ((F.col('target_geometry_changed') == 0) &
            (F.col('neighbor_geometry_changed') == 0) &
            (F.col('neighbor_recent') == 1) & F.col('neighbor_rain').isNotNull())
    change_good = good & F.col('neighbor_change').isNotNull()
    safe_distance = F.greatest(F.col('distance_km'), F.lit(0.1))
    agg = [
        F.sum(F.when(good, 1).otherwise(0)).cast('double').alias('neighbor5_recent_count'),
        F.sum(F.when(good & (F.col('neighbor_rank') >= 4), 1).otherwise(0)).cast('double').alias('neighbor4_5_recent_count'),
        F.max(F.when(good & (F.col('neighbor_rank') >= 4), F.col('neighbor_rain'))).alias('neighbor4_5_rain_max'),
        F.sum(F.when(good & (F.col('neighbor_rank') >= 4) & (F.col('neighbor_rain') > 0), 1).otherwise(0)).cast('double').alias('neighbor4_5_wet_count'),
        F.max(F.when(change_good, F.col('neighbor_change'))).alias('neighbor5_rain_change_max'),
        F.sum(F.when(change_good & (F.col('neighbor_change') > 0), 1).otherwise(0)).cast('double').alias('neighbor5_rising_count'),
        F.max(F.when(good, F.col('neighbor_rain30'))).alias('neighbor5_rain30_max'),
        F.avg(F.when(good, F.col('neighbor_rain') * F.col('dx_km') / safe_distance)).alias('neighbor5_east_signal'),
        F.avg(F.when(good, F.col('neighbor_rain') * F.col('dy_km') / safe_distance)).alias('neighbor5_north_signal'),
    ]
    for rank in range(1, 6):
        agg.append(F.max(F.when(good & (F.col('neighbor_rank') == rank), F.col('neighbor_rain'))).alias(f'neighbor_rank{rank}_rain'))
    agg += [F.max(F.when(good, F.col('neighbor_max_update'))).alias('max_neighbor_update'),
            F.max(F.when(good, F.col('neighbor_latest_obs'))).alias('max_neighbor_obs')]
    grouped = pair.groupBy('station_id', 'slot_epoch').agg(*agg)
    result = anchors.select('station_id', 'slot_epoch').join(grouped, ['station_id', 'slot_epoch'], 'left')
    result.write.mode('overwrite').parquet(str(output))
    check = spark.read.parquet(str(output))
    counts = check.agg(F.count('*').alias('rows'),
        F.sum(F.when((F.col('max_neighbor_update') > F.col('slot_epoch')) |
                     (F.col('max_neighbor_obs') > F.col('slot_epoch')) |
                     (F.col('neighbor5_recent_count') > 5), 1).otherwise(0)).alias('bad')).first()
    expected = anchors.count()
    if counts.rows != expected or counts.bad != 0:
        raise RuntimeError(f'Spatial feature integrity failure: {year}/{month} {counts} expected={expected}')
    save_json(marker, {'status': 'COMPLETE', 'year': year, 'month': month, 'rows': counts.rows,
                       'asof_violations': counts.bad, 'features': SPATIAL_NAMES})
    shutil.rmtree(temporary)
    print('SPATIAL_MONTH_COMPLETE', year, month, counts.rows, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=range(2017, 2025), required=True)
    parser.add_argument('--months', nargs='+', type=int, default=list(range(1, 13)))
    args = parser.parse_args()
    if not set(args.months) <= set(range(1, 13)):
        parser.error('months must be 1..12')
    metadata = json.loads((ROOT / 'work/task2/feature_metadata.json').read_text())['stations']
    spark = start_spark(f'RainfallSpatial_{args.year}')
    try:
        for month in args.months:
            make_month(spark, args.year, month, metadata)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
