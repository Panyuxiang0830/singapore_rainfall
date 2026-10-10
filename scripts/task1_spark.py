"""Full-data Task 1 cleaning and aggregation with resource-limited Spark."""
import argparse
import calendar
import json
import os
import platform
import shutil
import time
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import psutil

from pyspark.sql import SparkSession, Window, functions as F, types as T

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['date', 'timestamp', 'update_timestamp', 'station_id', 'station_name',
          'station_device_id', 'location_longitude', 'location_latitude',
          'reading_update_timestamp', 'reading_value', 'reading_type', 'reading_unit']
READING_TYPE = 'TB1 Rainfall 5 Minute Total F'


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')


def export_table(frame, name):
    # Only aggregate tables (<= a few hundred thousand rows) leave Spark.
    path = ROOT / f'reports/tables/{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.toPandas().to_csv(path, index=False)
    print(f'TABLE {name}', flush=True)


def start_spark(app_name='SingaporeRainfallTask1'):
    scratch = ROOT / 'work/spark'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    session = (SparkSession.builder.appName(app_name)
               .master('local[2]')
               .config('spark.driver.memory', '2g')
               .config('spark.driver.maxResultSize', '256m')
               .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
               .config('spark.rdd.compress', 'true')
               .config('spark.sql.shuffle.partitions', '8')
               .config('spark.default.parallelism', '2')
               .config('spark.local.dir', str(scratch))
               .config('spark.sql.session.timeZone', 'Asia/Singapore')
               .config('spark.sql.ansi.enabled', 'false')
               .config('spark.ui.enabled', 'false')
               .config('spark.sql.adaptive.enabled', 'true')
               .config('spark.sql.inMemoryColumnarStorage.batchSize', '1000')
               .getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    packages = {}
    for package in ('matplotlib', 'pandas', 'numpy', 'requests', 'psutil'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'unavailable'
    java = session.sparkContext._jvm.java.lang.System
    save_json(ROOT / 'reports/environment.json', {
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'python': platform.python_version(),
        'logical_cpus': os.cpu_count(),
        'physical_memory_gib': psutil.virtual_memory().total / 1024**3,
        'spark': session.version,
        'java': f"{java.getProperty('java.vendor')} {java.getProperty('java.version')}",
        'spark_master': session.sparkContext.master,
        'driver_heap_gib': 2,
        'rss_stop_gib': 4,
        'gpu_used': False,
        'packages': packages,
    })
    return session


def clean_year(spark, year):
    source = ROOT / f'data/raw/{year}.csv.gz'
    dest = ROOT / f'data/annual/{year}'
    audit_path = ROOT / f'reports/quality/{year}.json'
    if (dest / '_SUCCESS').exists() and audit_path.exists():
        print(f'{year}: already cleaned', flush=True)
        return
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError('Disk reserve below 5 GiB; not starting Spark cleaning')
    schema = T.StructType([T.StructField(name, T.StringType()) for name in FIELDS] +
                          [T.StructField('_corrupt_record', T.StringType())])
    raw = spark.read.schema(schema).option('header', True).option('mode', 'PERMISSIVE').csv(str(source))
    parsed = (raw.withColumn('obs_ts', F.to_timestamp('timestamp'))
              .withColumn('obs_epoch', F.unix_timestamp('obs_ts'))
              .withColumn('rain_mm', F.col('reading_value').cast('double'))
              .withColumn('lat', F.col('location_latitude').cast('double'))
              .withColumn('lon', F.col('location_longitude').cast('double'))
              .withColumn('slot_epoch', (F.round(F.col('obs_epoch') / 300) * 300).cast('long'))
              .withColumn('update_epoch', F.coalesce(
                  F.unix_timestamp(F.to_timestamp('reading_update_timestamp')),
                  F.unix_timestamp(F.to_timestamp('update_timestamp')), F.col('obs_epoch'))))
    finite = (F.col('rain_mm').isNotNull() & ~F.isnan('rain_mm') &
              (F.abs(F.col('rain_mm')) < F.lit(float('inf'))))
    timestamp_valid = F.col('obs_epoch').isNotNull()
    on_grid = F.abs(F.col('obs_epoch') - F.col('slot_epoch')) <= 2
    unit_valid = F.col('reading_unit') == 'mm'
    type_valid = F.col('reading_type') == READING_TYPE
    station_valid = F.col('station_id').isNotNull() & (F.length(F.trim('station_id')) > 0)
    valid = (finite & (F.col('rain_mm') >= 0) & timestamp_valid & on_grid &
             unit_valid & type_valid & station_valid & F.col('_corrupt_record').isNull())
    audit = parsed.agg(
        F.count('*').alias('raw_rows'),
        F.sum(F.when(~finite, 1).otherwise(0)).alias('nonfinite_or_missing_rain'),
        F.sum(F.when(F.col('rain_mm') < 0, 1).otherwise(0)).alias('negative_rain'),
        F.sum(F.when(~timestamp_valid, 1).otherwise(0)).alias('invalid_timestamps'),
        F.sum(F.when(timestamp_valid & ~on_grid, 1).otherwise(0)).alias('off_grid_over_2_seconds'),
        F.sum(F.when(~unit_valid | ~type_valid, 1).otherwise(0)).alias('unexpected_units_or_types'),
        F.sum(F.when(F.col('rain_mm') > 50, 1).otherwise(0)).alias('readings_over_50mm_flagged_not_removed'),
        F.sum(F.when(F.to_date('date') != F.to_date('obs_ts'), 1).otherwise(0)).alias('date_field_differs_from_sgt_date'),
        F.min('obs_ts').alias('min_observation_time'),
        F.max('obs_ts').alias('max_observation_time'),
        F.max('rain_mm').alias('max_raw_rain_mm'),
        F.sum(F.when(valid, 1).otherwise(0)).alias('valid_rows_before_deduplication'),
        F.sum(F.when(F.second('obs_ts') == 59, 1).otherwise(0)).alias('timestamps_with_second_59'),
    ).first().asDict()
    manifest = json.loads((ROOT / f'reports/downloads/{year}.json').read_text())
    if audit['raw_rows'] != manifest['api_reported_rows']:
        raise RuntimeError(f'CSV/API row count mismatch: {audit["raw_rows"]} vs {manifest["api_reported_rows"]}')
    audit['source_year'] = year
    audit['timestamp_policy'] = 'Nearest five-minute grid, only when shift <=2 seconds; otherwise excluded. Original epoch retained.'
    columns = ['update_epoch', 'obs_epoch', 'rain_mm', 'station_name', 'lat', 'lon']
    grouped = parsed.filter(valid).groupBy('station_id', 'slot_epoch').agg(
        F.max(F.struct(*[F.col(c) for c in columns])).alias('chosen'),
        F.count('*').alias('raw_key_rows'),
        F.min('rain_mm').alias('min_key_rain'),
        F.max('rain_mm').alias('max_key_rain'))
    unique = grouped.select('station_id', 'slot_epoch', 'chosen.*', 'raw_key_rows', 'min_key_rain', 'max_key_rain')
    unique.write.mode('overwrite').parquet(str(dest))
    saved = spark.read.parquet(str(dest))
    checks = saved.agg(
        F.count('*').alias('clean_rows'),
        F.sum(F.col('raw_key_rows') - 1).alias('duplicate_rows_removed'),
        F.sum(F.when(F.col('min_key_rain') != F.col('max_key_rain'), 1).otherwise(0)).alias('keys_with_conflicting_rain'),
        F.countDistinct('station_id').alias('stations'),
        F.sum('rain_mm').alias('observed_rain_sum_across_stations_mm'),
    ).first().asDict()
    audit.update(checks)
    assert audit['clean_rows'] + audit['duplicate_rows_removed'] == audit['valid_rows_before_deduplication']
    save_json(audit_path, audit)
    print('CLEANED', year, json.dumps(audit, default=str), flush=True)


def metrics(frame, keys):
    return frame.groupBy(*keys).agg(
        F.count('*').alias('n_observed'),
        F.sum('rain_mm').alias('observed_total_mm'),
        F.sum(F.when(F.col('rain_mm') > 0, 1).otherwise(0)).alias('n_wet'),
        F.sum(F.when(F.col('rain_mm') >= 5, 1).otherwise(0)).alias('n_ge5mm'),
        F.sum(F.when(F.col('rain_mm') >= 10, 1).otherwise(0)).alias('n_ge10mm'),
        F.sum(F.when(F.col('rain_mm') > 50, 1).otherwise(0)).alias('n_flagged_over50'),
        F.sum(F.when(F.col('rain_mm') <= 50, F.col('rain_mm')).otherwise(0)).alias('sum_excluding_over50_mm'),
        F.max('rain_mm').alias('max_5min_mm'),
    ).withColumn('wet_frequency_pct', 100 * F.col('n_wet') / F.col('n_observed')) \
     .withColumn('wet_intensity_mm_5min', F.when(F.col('n_wet') > 0, F.col('observed_total_mm') / F.col('n_wet'))) \
     .withColumn('rain_per_observed_24h_mm', 288 * F.col('observed_total_mm') / F.col('n_observed'))


def aggregate(spark):
    final = ROOT / 'data/clean/all'
    if not (final / '_SUCCESS').exists():
        annual = spark.read.parquet(*[str(ROOT / f'data/annual/{y}') for y in range(2017, 2025)])
        chosen_cols = ['update_epoch', 'obs_epoch', 'rain_mm', 'station_name', 'lat', 'lon']
        # Deduplicate again across annual file boundaries, deterministically.
        unique = annual.groupBy('station_id', 'slot_epoch').agg(
            F.max(F.struct(*[F.col(c) for c in chosen_cols])).alias('chosen'),
            F.count('*').alias('annual_file_copies'))
        clean = unique.select('station_id', 'slot_epoch', 'chosen.*', 'annual_file_copies')
        clean = clean.withColumn('time_sgt', F.from_unixtime('slot_epoch').cast('timestamp'))
        clean = clean.withColumn('year', F.year('time_sgt')).filter(F.col('year').between(2017, 2024))
        clean.write.mode('overwrite').partitionBy('year').parquet(str(final))
    clean = spark.read.parquet(str(final))
    overview = clean.agg(
        F.count('*').alias('clean_rows_2017_2024'),
        F.countDistinct('station_id').alias('station_ids'),
        F.sum(F.col('annual_file_copies') - 1).alias('cross_file_duplicate_rows_removed'),
        F.min('time_sgt').alias('first_time_sgt'), F.max('time_sgt').alias('last_time_sgt'),
        F.sum(F.when(F.col('rain_mm') > 50, 1).otherwise(0)).alias('flagged_over50_rows'),
        F.sum(F.when(~F.coalesce(F.col('lat').between(-90, 90) &
                               F.col('lon').between(-180, 180), F.lit(False)), 1)
              .otherwise(0)).alias('missing_or_invalid_coordinate_rows'),
    ).first().asDict()
    clean = clean.withColumn('day', F.to_date('time_sgt')) \
                 .withColumn('month', F.month('time_sgt')) \
                 .withColumn('hour', F.hour('time_sgt'))
    station_year = metrics(clean, ['station_id', 'year'])
    year_days = F.when(F.col('year').isin(2020, 2024), 366).otherwise(365)
    station_year = station_year.withColumn('coverage', F.col('n_observed') / (288 * year_days))
    metadata = clean.groupBy('station_id', 'year').agg(
        F.max(F.struct('slot_epoch', 'station_name', 'lat', 'lon')).alias('last'),
        F.min('lat').alias('lat_min'), F.max('lat').alias('lat_max'),
        F.min('lon').alias('lon_min'), F.max('lon').alias('lon_max'))
    metadata = metadata.select('station_id', 'year', F.col('last.station_name').alias('station_name'),
                               F.col('last.lat').alias('lat'), F.col('last.lon').alias('lon'),
                               'lat_min', 'lat_max', 'lon_min', 'lon_max')
    station_year = station_year.join(metadata, ['station_id', 'year'])
    export_table(station_year.orderBy('year', 'station_id'), 'station_year')
    panel = station_year.groupBy('station_id').agg(
        F.countDistinct('year').alias('n_years'), F.min('coverage').alias('min_annual_coverage'),
        F.min('lat_min').alias('lat_min'), F.max('lat_max').alias('lat_max'),
        F.min('lon_min').alias('lon_min'), F.max('lon_max').alias('lon_max'))
    panel = panel.filter((F.col('n_years') == 8) & (F.col('min_annual_coverage') >= .8) &
                         ((F.col('lat_max') - F.col('lat_min')) <= .005) &
                         ((F.col('lon_max') - F.col('lon_min')) <= .005))
    panel_ids = [row.station_id for row in panel.collect()]
    overview['balanced_panel_station_ids'] = sorted(panel_ids)
    overview['balanced_panel_definition'] = 'All 8 years, >=80% annual five-minute coverage each year, <=0.005 degree latitude/longitude span.'
    print('BALANCED PANEL', len(panel_ids), sorted(panel_ids), flush=True)
    if not panel_ids:
        raise RuntimeError('No stations qualify for balanced panel; investigate coverage before plotting')
    export_table(panel.orderBy('station_id'), 'balanced_panel')

    monthly = metrics(clean, ['station_id', 'year', 'month'])
    month_start = F.make_date('year', 'month', F.lit(1))
    monthly = monthly.withColumn('coverage', F.col('n_observed') / (288 * F.dayofmonth(F.last_day(month_start))))
    export_table(monthly.orderBy('year', 'month', 'station_id'), 'station_month')
    hourly = metrics(clean.filter(F.col('station_id').isin(panel_ids)), ['station_id', 'year', 'month', 'hour'])
    hourly = hourly.withColumn('coverage', F.col('n_observed') / (12 * F.dayofmonth(F.last_day(month_start))))
    export_table(hourly.orderBy('year', 'month', 'hour', 'station_id'), 'panel_station_month_hour')
    daily = metrics(clean, ['station_id', 'day', 'year', 'month'])
    daily = daily.withColumn('coverage', F.col('n_observed') / 288)
    export_table(daily.orderBy('day', 'station_id'), 'station_day')
    export_table(daily.filter(F.col('coverage') >= .95).orderBy(F.desc('observed_total_mm')).limit(100), 'extreme_station_days')

    coordinates = clean.groupBy('station_id', 'lat', 'lon').agg(
        F.count('*').alias('n_records'), F.min('day').alias('first_day'), F.max('day').alias('last_day'))
    export_table(coordinates.orderBy('station_id', F.desc('n_records')), 'station_coordinates')
    raw_top = clean.orderBy(F.desc('rain_mm')).select('station_id', 'station_name', 'time_sgt', 'rain_mm', 'lat', 'lon').limit(100)
    export_table(raw_top, 'largest_5min_readings')
    bins = (F.when(F.col('rain_mm') == 0, '0: dry')
            .when(F.col('rain_mm') <= .2, '1: (0, 0.2]')
            .when(F.col('rain_mm') <= 1, '2: (0.2, 1]')
            .when(F.col('rain_mm') <= 2, '3: (1, 2]')
            .when(F.col('rain_mm') <= 5, '4: (2, 5]')
            .when(F.col('rain_mm') <= 10, '5: (5, 10]')
            .when(F.col('rain_mm') <= 20, '6: (10, 20]')
            .otherwise('7: >20'))
    export_table(clean.withColumn('rain_bin', bins).groupBy('year', 'rain_bin').count().orderBy('year', 'rain_bin'), 'rain_distribution')

    # Spatial co-movement is based on station-day aggregates, not millions of raw rows.
    panel_daily = daily.filter(F.col('station_id').isin(panel_ids) & (F.col('coverage') >= .95))
    export_table(panel_daily.groupBy('day').pivot('station_id', sorted(panel_ids)).agg(F.first('observed_total_mm')).orderBy('day'), 'panel_daily_pivot')

    # 2024 short-term association connects Task 1 to future ML feature design.
    current = clean.filter(F.col('year') == 2024)
    previous = Window.partitionBy('station_id').orderBy('slot_epoch')
    future = previous.rangeBetween(300, 1800)
    windowed = (current.withColumn('prev_epoch', F.lag('slot_epoch').over(previous))
                .withColumn('prev_rain', F.lag('rain_mm').over(previous))
                .withColumn('future_30min_mm', F.round(F.sum('rain_mm').over(future), 9))
                .withColumn('future_30min_n', F.count('*').over(future)))
    pairs = windowed.filter(F.col('slot_epoch') - F.col('prev_epoch') == 300)
    transitions = pairs.withColumn('previous_wet', F.col('prev_rain') > 0).groupBy('previous_wet').agg(
        F.count('*').alias('n_pairs'), F.sum(F.when(F.col('rain_mm') > 0, 1).otherwise(0)).alias('current_wet_pairs'))
    export_table(transitions, 'rain_persistence_2024')
    forward = windowed.filter(F.col('future_30min_n') == 6)
    halfhour = forward.agg(F.count('*').alias('n_complete_windows'),
                          F.expr('percentile_approx(future_30min_mm, array(0.5, 0.9, 0.95, 0.99, 0.999), 10000)').alias('quantiles_mm'),
                          *[F.mean(F.when(F.col('future_30min_mm') > q, 1).otherwise(0)).alias(f'p_gt_{q}mm') for q in [0, 1, 5, 10, 20]])
    overview['halfhour_2024_descriptive_only'] = halfhour.first().asDict()
    overview['metric_definitions'] = {
        'wet': 'reading_value > 0 mm in an observed five-minute interval',
        'wet_intensity': 'sum rainfall / count wet five-minute intervals; mm per wet 5 minutes',
        'rain_per_observed_24h_mm': 'sum rainfall / observed intervals * 288; observation-duration normalized, not an official daily or annual total',
        'coverage': 'unique valid five-minute slots / full calendar expected slots; absent stations/periods are missing, never zero-filled',
        'extreme_flag': '>50 mm per five minutes is a sensitivity flag, not an official QC threshold; retained in main analysis',
        'correlation': 'Pearson correlation of station-day observed totals, pairwise >=95%-covered days; descriptive only',
    }
    overview['environment'] = {
        'spark_version': spark.version, 'python_version': platform.python_version(),
        'host': platform.node(), 'platform': platform.platform(),
        'mode': 'Spark local[2] on one machine, not a multi-node cluster',
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'visible_cpu_count': os.cpu_count(), 'physical_memory_bytes': __import__('psutil').virtual_memory().total,
        'driver_heap': '2g',
        'shuffle_partitions': 8, 'gpu_used': False, 'sgt_timezone': 'Asia/Singapore',
    }
    save_json(ROOT / 'reports/task1_overview.json', overview)
    print('AGGREGATION COMPLETE', json.dumps(overview, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-year', type=int, choices=range(2017, 2025))
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if (args.clean_year is None) == (not args.aggregate):
        parser.error('Choose exactly one of --clean-year YEAR or --aggregate')
    started = time.monotonic()
    spark = start_spark()
    try:
        if args.clean_year is not None:
            clean_year(spark, args.clean_year)
        else:
            aggregate(spark)
    finally:
        spark.stop()
        print(f'Elapsed seconds: {time.monotonic() - started:.1f}', flush=True)


if __name__ == '__main__':
    main()
