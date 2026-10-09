"""Causal rainfall features in Spark; generated data stays in server work storage.

Availability policy: the retained archive version must have observation AND
update timestamps no later than prediction time. This is a conservative archive
simulation: update_timestamp does not establish the first publication time.
"""
import argparse
import hashlib
import inspect
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.sql import Window, functions as F
from task1_spark import ROOT, save_json, start_spark
from task2_labels import LABEL_VERSION, year_start

FEATURE_VERSION = 'asof_archive_latest90_neighbor3_v1'
THRESHOLDS_MM = (0, 5, 25)
LOOKBACK_SECONDS = 5400
MAX_RECENT_AGE_SECONDS = 1800
NUMERIC_FEATURES = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'hour_sin_month_sin', 'hour_sin_month_cos',
    'hour_cos_month_sin', 'hour_cos_month_cos',
    'station_lat', 'station_lon', 'geometry_changed',
    'has_recent_reading', 'latest_age_min', 'latest_rain_mm', 'latest_wet',
    'rain_lag_5m', 'rain_lag_10m', 'rain_lag_15m', 'rain_lag_30m', 'rain_lag_60m',
    'rain_lag_5m_available', 'rain_lag_10m_available', 'rain_lag_15m_available',
    'rain_lag_30m_available', 'rain_lag_60m_available',
    'rain_15m', 'rain_30m', 'rain_60m',
    'history_count_15m', 'history_count_30m', 'history_count_60m',
    'wet_fraction_30m', 'max_rain_30m', 'rain_change_5m', 'rain_trend_15m',
    'wet_streak_slots', 'wet_streak_censored',
    'neighbor_available_count', 'neighbor_wet_fraction',
    'neighbor_latest_mean_mm', 'neighbor_latest_max_mm',
    'neighbor_30m_available_count', 'neighbor_30m_mean_mm', 'neighbor_30m_max_mm',
    'neighbor_mean_age_min', 'neighbor_mean_distance_km',
]
CATEGORICAL_FEATURES = ['station_id']


def month_start(year, month):
    return int(datetime(year, month, 1, tzinfo=ZoneInfo('Asia/Singapore')).timestamp())


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a['lat'], a['lon'], b['lat'], b['lon']))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, value)))


def graph_intervals(metadata):
    """A small metadata-only graph; future stations cannot affect past rankings."""
    events = sorted({((s['available_epoch'] + 299) // 300) * 300 for s in metadata})
    result = []
    for i, start in enumerate(events):
        end = events[i + 1] if i + 1 < len(events) else year_start(2025)
        known = [s for s in metadata if s['available_epoch'] <= start]
        for target in known:
            nearest = sorted((s for s in known if s['station_id'] != target['station_id']),
                             key=lambda s: (haversine_km(target, s), s['station_id']))[:3]
            for neighbor in nearest:
                result.append((target['station_id'], start, end, neighbor['station_id'],
                               float(haversine_km(target, neighbor))))
    return result


def prepare_metadata(spark):
    out = ROOT / 'work/task2/feature_metadata.json'
    if out.exists():
        stored = json.loads(out.read_text())
        if stored.get('feature_version') == FEATURE_VERSION:
            return stored
        raise RuntimeError('Feature metadata version changed: rebuild working metadata explicitly')
    annual = spark.read.parquet(*[str(ROOT / f'data/annual/{y}') for y in range(2017, 2025)])
    positioned = annual.filter(F.col('lat').between(-90, 90) & F.col('lon').between(-180, 180))
    first = F.struct(F.greatest('update_epoch', 'obs_epoch', 'slot_epoch').alias('available_epoch'),
                     F.col('slot_epoch'), F.col('lat'), F.col('lon'))
    small = positioned.groupBy('station_id').agg(
        F.min(first).alias('first'), F.min('lat').alias('lat_min'), F.max('lat').alias('lat_max'),
        F.min('lon').alias('lon_min'), F.max('lon').alias('lon_max')).collect()
    metadata = []
    for row in small:
        record = row.asDict(recursive=True)
        metadata.append({'station_id': record['station_id'], **record['first'],
                         'latitude_span': record['lat_max'] - record['lat_min'],
                         'longitude_span': record['lon_max'] - record['lon_min']})
    metadata.sort(key=lambda r: r['station_id'])
    stored = {'feature_version': FEATURE_VERSION, 'stations': metadata,
              'spatial_policy': 'First known coordinates; graph membership only after availability. '
                                'Recent coordinate displacement >0.005 degree disables spatial contributions.',
              'metadata_not_rainfall_selection': True}
    save_json(out, stored)
    print('FEATURE_METADATA_DONE', len(metadata), 'coordinate_span_gt_0.005',
          sum(s['latitude_span'] > .005 or s['longitude_span'] > .005 for s in metadata), flush=True)
    return stored


def metadata_frame(spark, metadata):
    return spark.createDataFrame([(s['station_id'], float(s['lat']), float(s['lon']), int(s['available_epoch']))
                                  for s in metadata],
                                 'station_id string, meta_lat double, meta_lon double, meta_available_epoch long')


def build_snapshots(observations, grid):
    """Same transform for training and future inference, with no target access.

    Grid must include a 90-minute warm-up before requested prediction anchors.
    Regular calendar slots prevent a missing observation from shifting a lag.
    Rolling totals end at the latest available reading, and carry its age.
    """
    joined = grid.join(observations, ['station_id', 'slot_epoch'], 'left')
    window = Window.partitionBy('station_id').orderBy('slot_epoch').rangeBetween(-LOOKBACK_SECONDS, 0)
    reading = F.when(F.col('rain_mm').isNotNull(), F.struct(
        F.col('slot_epoch').alias('s'), F.col('obs_epoch').alias('o'),
        F.col('update_epoch').alias('u'), F.col('rain_mm').alias('r'),
        F.col('lat').alias('lat'), F.col('lon').alias('lon')))
    frame = joined.withColumn('_history', F.collect_list(reading).over(window))
    frame = frame.withColumn('_available', F.expr(
        'array_sort(filter(_history, x -> x.o <= slot_epoch AND x.u <= slot_epoch AND x.s <= slot_epoch))'))
    frame = frame.withColumn('_latest', F.element_at('_available', -1))
    frame = frame.withColumn('latest_obs_epoch', F.col('_latest.o'))
    frame = frame.withColumn('latest_update_epoch', F.col('_latest.u'))
    frame = frame.withColumn('max_used_update_epoch', F.expr('array_max(transform(_available, x -> x.u))'))
    frame = frame.withColumn('latest_age_min', (F.col('slot_epoch') - F.col('_latest.s')) / 60.0)
    recent = F.col('_latest.s').isNotNull() & (F.col('slot_epoch') - F.col('_latest.s') <= MAX_RECENT_AGE_SECONDS)
    frame = frame.withColumn('has_recent_reading', recent.cast('double'))
    frame = frame.withColumn('latest_rain_mm', F.when(recent, F.col('_latest.r')))
    frame = frame.withColumn('latest_wet', F.when(recent, (F.col('_latest.r') > 0).cast('double')))
    known_position = F.col('meta_available_epoch') <= F.col('slot_epoch')
    frame = frame.withColumn('station_lat', F.when(known_position, F.col('meta_lat')))
    frame = frame.withColumn('station_lon', F.when(known_position, F.col('meta_lon')))
    moved = ((F.abs(F.col('_latest.lat') - F.col('meta_lat')) > .005) |
             (F.abs(F.col('_latest.lon') - F.col('meta_lon')) > .005))
    frame = frame.withColumn('geometry_changed', F.when(known_position, F.coalesce(moved.cast('double'), F.lit(0.0))))
    for minutes in [5, 10, 15, 30, 60]:
        frame = frame.withColumn(f'rain_lag_{minutes}m', F.expr(
            f'element_at(filter(_available, x -> x.s = slot_epoch - {minutes * 60}), 1).r'))
        frame = frame.withColumn(f'rain_lag_{minutes}m_available',
                                 F.col(f'rain_lag_{minutes}m').isNotNull().cast('double'))
    for minutes in [15, 30, 60]:
        col = f'_rolling_{minutes}'
        frame = frame.withColumn(col, F.expr(
            f'filter(_available, x -> x.s > _latest.s - {minutes * 60} AND x.s <= _latest.s)'))
        frame = frame.withColumn(f'history_count_{minutes}m', F.when(recent, F.size(col)).otherwise(0).cast('double'))
        complete = recent & (F.size(col) == minutes // 5)
        frame = frame.withColumn(f'rain_{minutes}m', F.when(complete, F.round(F.expr(
            f'aggregate({col}, cast(0 as double), (a,x) -> a + x.r)'), 9)))
    frame = frame.withColumn('max_rain_30m', F.when(recent, F.expr('array_max(transform(_rolling_30, x -> x.r))')))
    frame = frame.withColumn('wet_fraction_30m', F.when(recent, F.expr(
        'cast(size(filter(_rolling_30, x -> x.r > 0)) as double) / size(_rolling_30)')))
    frame = frame.withColumn('rain_change_5m', F.when(recent, F.col('_latest.r') - F.expr(
        'element_at(filter(_available, x -> x.s = _latest.s - 300), 1).r')))
    frame = frame.withColumn('rain_trend_15m', 2 * F.col('rain_15m') - F.col('rain_30m'))
    # Stop the wet run at the first dry OR missing interval. cap=12 (one hour).
    # Fold once through the ordered available records rather than repeatedly
    # search the array for every prefix of a 12-slot run.
    frame = frame.withColumn('_streak', F.expr('''aggregate(reverse(_available),
        named_struct('n', 0, 'previous', cast(null as bigint), 'keep', true),
        (a,x) -> named_struct(
            'n', if(a.keep AND a.n < 12 AND x.r > 0 AND
                    (a.previous IS NULL OR a.previous - x.s = 300), a.n + 1, a.n),
            'previous', x.s,
            'keep', a.keep AND a.n < 12 AND x.r > 0 AND
                    (a.previous IS NULL OR a.previous - x.s = 300)))'''))
    frame = frame.withColumn('wet_streak_slots', F.when(recent, F.col('_streak.n').cast('double')))
    frame = frame.withColumn('wet_streak_censored', F.when(recent, (F.col('_streak.n') == 12).cast('double')))
    stamp = F.to_timestamp(F.from_unixtime('slot_epoch'))
    hour = F.hour(stamp) + F.minute(stamp) / 60.0
    month = F.month(stamp) - 1
    frame = frame.withColumn('hour_sin', F.sin(hour * (2 * math.pi / 24)))
    frame = frame.withColumn('hour_cos', F.cos(hour * (2 * math.pi / 24)))
    frame = frame.withColumn('month_sin', F.sin(month * (2 * math.pi / 12)))
    frame = frame.withColumn('month_cos', F.cos(month * (2 * math.pi / 12)))
    for h in ['hour_sin', 'hour_cos']:
        for m in ['month_sin', 'month_cos']:
            frame = frame.withColumn(f'{h}_{m}', F.col(h) * F.col(m))
    columns = ['station_id', 'slot_epoch', 'meta_available_epoch', 'latest_obs_epoch',
               'latest_update_epoch', 'max_used_update_epoch']
    own = [c for c in NUMERIC_FEATURES if not c.startswith('neighbor_')]
    return frame.select(*columns, *own)


def add_neighbors(anchors, snapshots, edges):
    # Left joins preserve anchors when metadata or all nearby readings are absent.
    a, g = anchors.alias('a'), F.broadcast(edges).alias('g')
    pairs = a.join(g, (F.col('a.station_id') == F.col('g.target_station')) &
                   (F.col('a.slot_epoch') >= F.col('g.graph_start')) &
                   (F.col('a.slot_epoch') < F.col('g.graph_end')), 'left').select(
        'a.station_id', 'a.slot_epoch', 'a.geometry_changed', 'g.neighbor_station', 'g.distance_km')
    neighbor = snapshots.select(
        F.col('station_id').alias('neighbor_station'), 'slot_epoch',
        F.col('geometry_changed').alias('neighbor_geometry_changed'),
        F.col('has_recent_reading').alias('neighbor_recent'),
        F.col('latest_rain_mm').alias('neighbor_rain'), F.col('rain_30m').alias('neighbor_total'),
        F.col('latest_age_min').alias('neighbor_age'),
        F.col('max_used_update_epoch').alias('neighbor_update'),
        F.col('latest_obs_epoch').alias('neighbor_observation'))
    pairs = pairs.join(neighbor, ['neighbor_station', 'slot_epoch'], 'left')
    available = ((F.col('geometry_changed') == 0) & (F.col('neighbor_geometry_changed') == 0) &
                 (F.col('neighbor_recent') == 1) & F.col('neighbor_rain').isNotNull())
    total_available = available & F.col('neighbor_total').isNotNull()
    grouped = pairs.groupBy('station_id', 'slot_epoch').agg(
        F.sum(F.when(available, 1).otherwise(0)).cast('double').alias('neighbor_available_count'),
        F.avg(F.when(available, (F.col('neighbor_rain') > 0).cast('double'))).alias('neighbor_wet_fraction'),
        F.avg(F.when(available, F.col('neighbor_rain'))).alias('neighbor_latest_mean_mm'),
        F.max(F.when(available, F.col('neighbor_rain'))).alias('neighbor_latest_max_mm'),
        F.sum(F.when(total_available, 1).otherwise(0)).cast('double').alias('neighbor_30m_available_count'),
        F.avg(F.when(total_available, F.col('neighbor_total'))).alias('neighbor_30m_mean_mm'),
        F.max(F.when(total_available, F.col('neighbor_total'))).alias('neighbor_30m_max_mm'),
        F.avg(F.when(available, F.col('neighbor_age'))).alias('neighbor_mean_age_min'),
        F.avg(F.when(available, F.col('distance_km'))).alias('neighbor_mean_distance_km'),
        F.max(F.when(available, F.col('neighbor_update'))).alias('neighbor_max_used_update_epoch'),
        F.max(F.when(available, F.col('neighbor_observation'))).alias('neighbor_latest_obs_epoch'))
    return anchors.join(grouped, ['station_id', 'slot_epoch'], 'left')


def graph_frame(spark, metadata):
    return spark.createDataFrame(graph_intervals(metadata),
        'target_station string, graph_start long, graph_end long, neighbor_station string, distance_km double')


def prepare_year_inputs(spark, year):
    """Read an annual archive once; monthly files avoid twelve full scans."""
    target = ROOT / f'work/task2/feature_inputs/{year}'
    if (target / '_SUCCESS').exists():
        return target
    cols = ['station_id', 'slot_epoch', 'obs_epoch', 'update_epoch', 'rain_mm', 'lat', 'lon']
    start, end = year_start(year), year_start(year + 1)
    records = spark.read.parquet(str(ROOT / f'data/annual/{year}')).filter(
        (F.col('slot_epoch') >= start) & (F.col('slot_epoch') < end)).select(*cols)
    records = records.withColumn('input_month', F.month(F.to_timestamp(F.from_unixtime('slot_epoch'))))
    if year > 2017:
        padding = spark.read.parquet(str(ROOT / f'data/annual/{year - 1}')).filter(
            (F.col('slot_epoch') >= start - LOOKBACK_SECONDS) & (F.col('slot_epoch') < start)).select(*cols)
        records = records.unionByName(padding.withColumn('input_month', F.lit(1)))
    records.repartition(8, 'input_month').write.mode('overwrite').partitionBy('input_month').parquet(str(target))
    print('FEATURE_INPUTS_READY', year, flush=True)
    return target


def check_semantics(spark):
    t = year_start(2021) + 12 * 3600
    metadata = [{'station_id': 'A', 'lat': 1.3, 'lon': 103.8, 'available_epoch': t - 9000},
                {'station_id': 'B', 'lat': 1.31, 'lon': 103.81, 'available_epoch': t - 9000},
                {'station_id': 'C', 'lat': 1.3001, 'lon': 103.8001, 'available_epoch': t + 300}]
    grid = spark.range(t - 5400, t + 600, 300).select(F.col('id').alias('slot_epoch')).crossJoin(
        F.broadcast(metadata_frame(spark, metadata)))
    rows = []
    for station in ['A', 'B']:
        for i in range(-18, 2):
            s = t + i * 300
            rain = 1.0 if station == 'A' else 2.0
            if i >= -1:
                rain = 9999.0
            update = s + 600
            if station == 'B' and i == -2:
                update = t + 300  # A recent archived correction is unavailable at t.
            if station == 'A' and i == -4:
                continue  # Missing interval must stop a streak and invalidate total.
            lat, lon = (1.3, 103.8) if station == 'A' else (1.31, 103.81)
            rows.append((station, s, s, update, rain, lat, lon))
    obs = spark.createDataFrame(rows,
        'station_id string, slot_epoch long, obs_epoch long, update_epoch long, rain_mm double, lat double, lon double')
    snapshots = build_snapshots(obs, grid).cache()
    try:
        at_t = snapshots.filter(F.col('slot_epoch') == t)
        a = at_t.filter("station_id = 'A'").first()
        assert a.latest_rain_mm == 1 and a.latest_age_min == 10
        assert a.rain_lag_5m is None and a.rain_lag_10m == 1
        assert a.rain_15m is None and a.history_count_15m == 2
        assert a.wet_streak_slots == 2 and a.max_used_update_epoch <= t
        b = at_t.filter("station_id = 'B'").first()
        assert b.latest_age_min == 15 and b.latest_rain_mm == 2
        assert b.rain_lag_10m is None and b.rain_30m == 12
        assert b.wet_streak_slots == 12 and b.wet_streak_censored == 1
        graph = graph_frame(spark, metadata)
        before = [r for r in graph_intervals(metadata) if r[1] <= t < r[2]]
        assert all(r[0] != r[3] for r in before)
        assert not any(r[0] == 'C' or r[3] == 'C' for r in before)
        neighbors = add_neighbors(at_t, snapshots, graph).filter("station_id = 'A'").first()
        assert neighbors.neighbor_available_count == 1
        assert neighbors.neighbor_latest_mean_mm == 2 and neighbors.neighbor_30m_mean_mm == 12
        assert neighbors.neighbor_max_used_update_epoch <= t
        c = at_t.filter('station_id = "C"').first()
        assert c.station_lat is None and c.has_recent_reading == 0 and c.hour_sin is not None
        # Mutate only future and unpublished rain: the prediction snapshot must stay identical.
        changed = obs.withColumn('rain_mm', F.when((F.col('slot_epoch') > t) | (F.col('update_epoch') > t),
                                                  F.lit(888888.0)).otherwise(F.col('rain_mm')))
        again = build_snapshots(changed, grid).filter((F.col('slot_epoch') == t) & (F.col('station_id') == 'A')).first()
        for name in [c for c in NUMERIC_FEATURES if not c.startswith('neighbor_')]:
            assert again[name] == a[name], name
        print('TASK2_FEATURE_CHECKS_OK: calendar lags, missing windows, delayed updates, '
              'future perturbation invariance, graph availability, neighbor as-of joins', flush=True)
    finally:
        snapshots.unpersist()


def build_month(spark, year, month, stored):
    final = ROOT / f'data/task2/features/{year}/{month:02d}'
    summary_path = ROOT / f'work/task2/features_{year}_{month:02d}.json'
    fingerprint = hashlib.sha256(json.dumps(stored, sort_keys=True).encode()).hexdigest()
    if (final / '_SUCCESS').exists() and summary_path.exists():
        status = json.loads(summary_path.read_text())
        if (status.get('feature_version') != FEATURE_VERSION or status.get('metadata_sha256') != fingerprint or
                status.get('label_version') != LABEL_VERSION or
                status.get('thresholds_mm') != list(THRESHOLDS_MM) or
                status.get('numeric_features') != NUMERIC_FEATURES or
                status.get('categorical_features') != CATEGORICAL_FEATURES):
            raise RuntimeError(f'Feature definition changed; rebuild {year}/{month:02d}')
        print('FEATURE_MONTH_ALREADY_DONE', year, month, flush=True)
        return
    start = month_start(year, month)
    end = month_start(year, month + 1) if month < 12 else year_start(year + 1)
    # Station IDs are also supplied as inference inputs. Include requested
    # target stations even if their first retained metadata update is delayed;
    # coordinate values/graph membership are still masked until available.
    target_ids = {r.station_id for r in spark.read.parquet(str(ROOT / f'data/task2/labels/{year}')).filter(
        (F.col('slot_epoch') >= start) & (F.col('slot_epoch') < end)).select('station_id').distinct().collect()}
    metadata = [s for s in stored['stations'] if s['available_epoch'] < end or s['station_id'] in target_ids]
    station_frame = metadata_frame(spark, metadata)
    grid = spark.range(start - LOOKBACK_SECONDS, end, 300).select(F.col('id').alias('slot_epoch')).crossJoin(
        F.broadcast(station_frame))
    base = ROOT / f'work/task2/feature_inputs/{year}'
    inputs = [str(base / f'input_month={month}')]
    if month > 1:
        inputs.append(str(base / f'input_month={month - 1}'))
    inputs = [p for p in inputs if Path(p).exists()]
    if not inputs:
        if not (base / '_SUCCESS').exists():
            raise RuntimeError(f'Missing partitioned feature inputs for {year}/{month}')
        # A genuine full-month source gap has no partition file. Reading the
        # bounded annual input still supplies the typed empty observation frame.
        inputs = [str(ROOT / f'data/annual/{year}')]
    observations = spark.read.parquet(*inputs).filter(
        (F.col('slot_epoch') >= start - LOOKBACK_SECONDS) & (F.col('slot_epoch') < end)).select(
        'station_id', 'slot_epoch', 'obs_epoch', 'update_epoch', 'rain_mm', 'lat', 'lon')
    snapshot_path = ROOT / f'work/task2/snapshots/{year}_{month:02d}'
    build_snapshots(observations, grid).filter(F.col('slot_epoch') >= start).write.mode('overwrite').parquet(str(snapshot_path))
    snapshots = spark.read.parquet(str(snapshot_path))
    labels = spark.read.parquet(str(ROOT / f'data/task2/labels/{year}')).filter(
        F.col('label_valid') & (F.col('slot_epoch') >= start) & (F.col('slot_epoch') < end))
    # Preserve targets for a station whose metadata is absent or not yet available.
    anchors = labels.join(snapshots, ['station_id', 'slot_epoch'], 'left')
    edges = graph_frame(spark, stored['stations']).filter(
        (F.col('graph_start') < end) & (F.col('graph_end') > start))
    features = add_neighbors(anchors, snapshots, edges)
    for q in THRESHOLDS_MM:
        features = features.withColumn(f'label_gt_{q}mm', (F.col('future_30min_mm') > q).cast('double'))
    features.write.mode('overwrite').parquet(str(final))
    saved = spark.read.parquet(str(final))
    bad = ((F.col('max_used_update_epoch') > F.col('slot_epoch')) |
           (F.col('latest_obs_epoch') > F.col('slot_epoch')) |
           (F.col('neighbor_max_used_update_epoch') > F.col('slot_epoch')) |
           (F.col('neighbor_latest_obs_epoch') > F.col('slot_epoch')) |
           (F.col('neighbor_available_count') > 3) |
           ((F.col('station_lat').isNotNull()) & (F.col('meta_available_epoch') > F.col('slot_epoch'))))
    # Keep uniqueness separate from percentile_approx. Mixing DISTINCT with a
    # percentile aggregate makes Spark build quantile buffers per unique key.
    stats = saved.agg(F.count('*').alias('rows'),
                     F.sum(F.when(bad, 1).otherwise(0)).alias('asof_or_graph_violations'),
                     F.sum(F.when(F.col('has_recent_reading') == 1, 1).otherwise(0)).alias('recent_own_rows'),
                     F.sum(F.when(F.col('neighbor_available_count') > 0, 1).otherwise(0)).alias('with_neighbor_rows'),
                     F.sum(F.when(F.col('geometry_changed') == 1, 1).otherwise(0)).alias('geometry_changed_rows'),
                     F.avg('neighbor_available_count').alias('mean_available_neighbors'),
                     F.expr('percentile_approx(latest_age_min, array(0.5,0.95,0.99),10000)').alias('latest_age_quantiles_min'),
                     *[F.sum(F.col(f'label_gt_{q}mm')).cast('long').alias(f'positive_gt_{q}mm') for q in THRESHOLDS_MM],
                     *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias('null_' + c) for c in NUMERIC_FEATURES]
                     ).first().asDict()
    integer_stats = ['rows', 'asof_or_graph_violations', 'recent_own_rows',
                     'with_neighbor_rows', 'geometry_changed_rows']
    integer_stats += [k for k in stats if k.startswith(('positive_gt_', 'null_'))]
    for key in integer_stats:
        stats[key] = stats[key] or 0
    stats['unique_keys'] = saved.select('station_id', 'slot_epoch').distinct().count()
    expected = labels.agg(F.count('*').alias('rows'), *[
        F.sum((F.col('future_30min_mm') > q).cast('long')).alias(f'positive_gt_{q}mm')
        for q in THRESHOLDS_MM]).first().asDict()
    for key in expected:
        expected[key] = expected[key] or 0
    if (stats['rows'] != expected['rows'] or stats['unique_keys'] != expected['rows'] or
            stats['asof_or_graph_violations'] != 0 or
            any(stats[f'positive_gt_{q}mm'] != expected[f'positive_gt_{q}mm'] for q in THRESHOLDS_MM)):
        raise RuntimeError(f'Feature integrity check failed: {year}/{month:02d}: {stats}')
    summary = {'year': year, 'month': month, 'feature_version': FEATURE_VERSION,
               'label_version': LABEL_VERSION, 'metadata_sha256': fingerprint,
               'thresholds_mm': list(THRESHOLDS_MM), 'numeric_features': NUMERIC_FEATURES,
               'categorical_features': CATEGORICAL_FEATURES,
               'availability_policy': 'Retained archive version observation/update times <= prediction time; '
                                      'conservative simulation, not proof of historical first publication.',
               'rolling_policy': '15/30/60-minute totals end at latest available reading; '
                                 'null unless all slots present; latest age and counts retained.',
               'missing_policy': 'Keep nulls; any imputation/scaling/encoding fitted on training only in modelling stage.',
               'future_target_columns_excluded_from_feature_list': True, **stats}
    save_json(summary_path, summary)
    shutil.rmtree(snapshot_path)
    print('FEATURE_MONTH_DONE', json.dumps({k: summary[k] for k in [
        'year', 'month', 'rows', 'asof_or_graph_violations', 'recent_own_rows', 'with_neighbor_rows',
        'mean_available_neighbors', 'latest_age_quantiles_min']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=range(2017, 2025))
    parser.add_argument('--months', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    if not set(args.months) <= set(range(1, 13)):
        parser.error('Months must be 1--12')
    spark = start_spark('SingaporeRainfallTask2Features')
    try:
        if args.check_only:
            check_semantics(spark)
        elif args.metadata_only:
            prepare_metadata(spark)
        elif args.year:
            stored = prepare_metadata(spark)
            needed = any(not (ROOT / f'data/task2/features/{args.year}/{m:02d}/_SUCCESS').exists() or
                         not (ROOT / f'work/task2/features_{args.year}_{m:02d}.json').exists()
                         for m in args.months)
            signature = hashlib.sha256(inspect.getsource(build_snapshots).encode()).hexdigest()[:16]
            checked = ROOT / f'work/task2/features_checked_{signature}.json'
            if needed and not checked.exists():
                check_semantics(spark)
                save_json(checked, {'feature_version': FEATURE_VERSION, 'spark': spark.version,
                                    'whole_stage_codegen': spark.conf.get('spark.sql.codegen.wholeStage'),
                                    'semantic_checks': 'passed'})
            input_path = prepare_year_inputs(spark, args.year) if needed else None
            for month in sorted(set(args.months)):
                build_month(spark, args.year, month, stored)
            if input_path is not None:
                shutil.rmtree(input_path)
        else:
            parser.error('--year, --check-only or --metadata-only required')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
