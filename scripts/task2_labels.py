"""Build future-only half-hour targets and training-period threshold diagnostics.

All observation transformations use Spark. Generated Parquet/JSON are working
files, not submission artifacts. No model or feature selection uses test labels.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.sql import Window, functions as F
from task1_spark import ROOT, save_json, start_spark

THRESHOLDS_MM = (0, 5, 25, 50)
LABEL_VERSION = 'future6_round9_strict_splitpurge_v1'


def year_start(year):
    return int(datetime(year, 1, 1, tzinfo=ZoneInfo('Asia/Singapore')).timestamp())


def future_windows(frame):
    # A time range, not the next six rows: gaps must invalidate the target.
    horizon = Window.partitionBy('station_id').orderBy('slot_epoch').rangeBetween(300, 1800)
    return (frame.withColumn('future_slots', F.count('*').over(horizon))
            .withColumn('future_sum', F.round(F.sum('rain_mm').over(horizon), 9)))


def valid_target_count(frame, split_end):
    return ((F.col('future_slots') == 6) &
            (F.col('slot_epoch') + 1800 < F.lit(split_end)))


def check_semantics(spark):
    base = year_start(2024)
    data = spark.createDataFrame([('A', base + i * 300, float(i)) for i in range(8)],
                                 ['station_id', 'slot_epoch', 'rain_mm'])
    first = future_windows(data).filter(F.col('slot_epoch') == base).first()
    assert first.future_slots == 6 and first.future_sum == 21.0
    assert not (first.future_sum > 21.0), 'Exceedance is strict >.'
    gap = future_windows(data.filter(F.col('slot_epoch') != base + 900))
    assert gap.filter(F.col('slot_epoch') == base).first().future_slots == 5
    assert future_windows(data).filter(valid_target_count(data, base + 1800)).count() == 0
    assert future_windows(data).filter(valid_target_count(data, base + 1801)).count() == 1
    decimal_example = spark.createDataFrame(
        [('A', base + i * 300, v) for i, v in enumerate([99.0, .1, .2, 0., 0., 0., 0.])],
        ['station_id', 'slot_epoch', 'rain_mm'])
    rounded = future_windows(decimal_example).filter(F.col('slot_epoch') == base)
    assert rounded.first().future_sum == .3
    assert rounded.filter(F.col('future_sum') > .3).count() == 0
    print('TASK2_CHECKS_OK: excludes current/past, invalidates gaps, strict >, purges split boundary', flush=True)


def build_year(spark, year):
    out = ROOT / f'data/task2/labels/{year}'
    summary_file = ROOT / f'work/task2/thresholds_{year}.json'
    if (out / '_SUCCESS').exists() and summary_file.exists():
        if json.loads(summary_file.read_text()).get('label_version') != LABEL_VERSION:
            raise RuntimeError(f'Label definition changed; rebuild generated year {year}')
        print('TASK2_YEAR_ALREADY_DONE', year, flush=True)
        return
    source = ROOT / f'data/annual/{year}'
    audit = ROOT / f'reports/quality/{year}.json'
    if not (source / '_SUCCESS').exists() or not audit.exists():
        raise RuntimeError(f'Missing verified annual input: {year}')
    start, end = year_start(year), year_start(year + 1)
    split, split_end = ('train', year_start(2023)) if year <= 2022 else (
        ('validation', year_start(2024)) if year == 2023 else ('test', year_start(2025)))
    current = spark.read.parquet(str(source)).filter(
        (F.col('slot_epoch') >= start) & (F.col('slot_epoch') < end))
    cols = ['station_id', 'slot_epoch', 'rain_mm', 'update_epoch']
    records = current.select(*cols)
    if year < 2024:
        following = ROOT / f'data/annual/{year + 1}'
        if not (following / '_SUCCESS').exists():
            raise RuntimeError(f'Missing next-year boundary observations: {year + 1}')
        padding = spark.read.parquet(str(following)).filter(
            (F.col('slot_epoch') >= end) & (F.col('slot_epoch') < end + 1800)).select(*cols)
        records = records.unionByName(padding)
    # Annual inputs are unique by station/time; choose deterministically again
    # for the combined input to guarantee exactly six distinct future slots.
    records = (records.groupBy('station_id', 'slot_epoch')
               .agg(F.max(F.struct('update_epoch', 'rain_mm')).alias('chosen'))
               .select('station_id', 'slot_epoch', F.col('chosen.rain_mm').alias('rain_mm')))
    targets = future_windows(records).filter(
        (F.col('slot_epoch') >= start) & (F.col('slot_epoch') < end))
    targets = targets.select(
        'station_id', 'slot_epoch', 'future_slots',
        F.when(F.col('future_slots') == 6, F.col('future_sum')).alias('future_30min_mm'),
        valid_target_count(targets, split_end).alias('label_valid'),
        F.lit(split).alias('split'))
    # Save compact targets once so multiple summaries do not rerun the sort.
    targets.write.mode('overwrite').parquet(str(out))
    saved = spark.read.parquet(str(out)).withColumn('day_sgt', F.to_date(F.from_unixtime('slot_epoch')))
    counters = saved.agg(
        F.count('*').alias('candidate_windows'),
        F.countDistinct('station_id').alias('stations'),
        F.sum(F.when(F.col('future_slots') != 6, 1).otherwise(0)).alias('incomplete_future_windows'),
        F.sum(F.when((F.col('future_slots') == 6) & ~F.col('label_valid'), 1).otherwise(0))
            .alias('complete_windows_purged_at_split_end'),
        F.sum(F.col('label_valid').cast('long')).alias('valid_windows'),
    ).first().asDict()
    expected = json.loads(audit.read_text())['clean_rows']
    if counters['candidate_windows'] != expected:
        raise RuntimeError(f'Annual candidate/audit mismatch for {year}: {counters} vs {expected}')
    valid = saved.filter('label_valid')
    expressions = [F.max('future_30min_mm').alias('max_30min_mm')]
    for q in THRESHOLDS_MM:
        positive = F.col('future_30min_mm') > q
        expressions.extend([
            F.sum(F.when(positive, 1).otherwise(0)).alias(f'positive_gt_{q}mm'),
            F.countDistinct(F.when(positive, F.col('day_sgt'))).alias(f'positive_days_gt_{q}mm'),
            F.countDistinct(F.when(positive, F.struct('station_id', 'day_sgt')))
                .alias(f'positive_station_days_gt_{q}mm'),
        ])
    results = valid.agg(*expressions).first().asDict()
    for q in THRESHOLDS_MM:
        results[f'positive_fraction_gt_{q}mm'] = results[f'positive_gt_{q}mm'] / counters['valid_windows']
    availability = current.withColumn('update_lag_s', F.col('update_epoch') - F.col('obs_epoch')).agg(
        F.min('update_lag_s').alias('min_update_lag_s'),
        F.expr('percentile_approx(update_lag_s, array(0.5, 0.95, 0.99), 10000)')
            .alias('update_lag_quantiles_s'),
        F.max('update_lag_s').alias('max_update_lag_s'),
        F.sum(F.when(F.col('update_lag_s') > 0, 1).otherwise(0)).alias('updates_after_observation'),
    ).first().asDict()
    summary = {
        'year': year, 'split': split, 'thresholds_mm': list(THRESHOLDS_MM),
        'label_version': LABEL_VERSION, 'sum_rounding_decimal_places': 9,
        'target_definition': 'sum of distinct station observations at t+5,...,t+30 minutes; strict > q',
        'interval_assumption': 'For physical 30-minute accumulation, treat observation labels as interval ends. This is a modelling assumption, not an independently verified timestamp convention.',
        'eligibility': 'existing observation anchors; six future slots; target ends strictly before next split',
        'positive_day_note': 'Days and station-days are counts of coverage, not independent storm events.',
        'availability_note': 'Stored reading update time is not necessarily first publication time. Inspect delays before building real-time features.',
        **counters, **results, 'record_update_diagnostics': availability,
    }
    save_json(summary_file, summary)
    print('TASK2_YEAR_DONE', json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=range(2017, 2025))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--task1-check', action='store_true', help='Reconcile existing report thresholds from prepared 2024 targets')
    args = parser.parse_args()
    if not args.check_only and not args.task1_check and args.year is None:
        parser.error('--year or --check-only is required')
    spark = start_spark('SingaporeRainfallTask2Labels')
    try:
        if args.check_only:
            check_semantics(spark)
        elif args.task1_check:
            targets = spark.read.parquet(str(ROOT / 'data/task2/labels/2024')).filter('label_valid')
            counts = targets.agg(F.count('*').alias('windows'), *[
                F.sum(F.when(F.col('future_30min_mm') > q, 1).otherwise(0)).alias(f'gt_{q}mm')
                for q in [0, 1, 5, 10, 20]
            ]).first().asDict()
            save_json(ROOT / 'work/task2/task1_threshold_check.json', counts)
            print('TASK1_THRESHOLD_RECONCILIATION', json.dumps(counts), flush=True)
        else:
            build_year(spark, args.year)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
