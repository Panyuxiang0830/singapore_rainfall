"""Bounded, same-station SMOTE for mixed rainfall summaries.

Continuous rainfall summaries are interpolated with one lambda. Station,
calendar, discrete flags/counts/fractions, ages, distances and the complete
missingness pattern stay fixed. Isolated profiles fall back to random
oversampling and are counted separately. This is a constrained variant, not
an implementation claim for imbalanced-learn SMOTENC.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors

from task2_features import NUMERIC_FEATURES
from task1_spark import ROOT, save_json

CONTINUOUS = [
    'latest_rain_mm', 'rain_lag_5m', 'rain_lag_10m', 'rain_lag_15m',
    'rain_lag_30m', 'rain_lag_60m', 'rain_15m', 'rain_30m', 'rain_60m',
    'max_rain_30m', 'rain_change_5m', 'rain_trend_15m',
    'neighbor_latest_mean_mm', 'neighbor_latest_max_mm',
    'neighbor_30m_mean_mm', 'neighbor_30m_max_mm',
]
CALENDAR = NUMERIC_FEATURES[:8]
LOCKED = [c for c in NUMERIC_FEATURES if c not in CONTINUOUS + CALENDAR]
WORK = ROOT/'work/task2/model_comparison'
VERSION = 'same_station_profile_zero_pattern_smote_cap10_v2'


def neighbors_for(stations, values, scales):
    """Only a bounded minority pool is processed outside Spark."""
    n = len(values)
    continuous = [NUMERIC_FEATURES.index(c) for c in CONTINUOUS]
    locked = [NUMERIC_FEATURES.index(c) for c in LOCKED]
    distance = continuous + [0, 1, 2, 3]
    rain_amounts = [NUMERIC_FEATURES.index(c) for c in CONTINUOUS
                    if c not in ['rain_change_5m', 'rain_trend_15m']]
    z = np.nan_to_num(values[:, distance], nan=0.0) / scales[distance]
    groups = {}
    for i in range(n):
        key = (str(stations[i]), tuple(np.isnan(values[i])),
               tuple(values[i, rain_amounts] == 0),
               tuple(None if np.isnan(values[i, j]) else float(values[i, j]) for j in locked))
        groups.setdefault(key, []).append(i)
    links = np.repeat(np.arange(n, dtype=np.int32)[:, None], 5, axis=1)
    eligible = 0
    for ids in groups.values():
        if len(ids) < 2:
            continue
        ids = np.asarray(ids, dtype=np.int32)
        nn = NearestNeighbors(n_neighbors=min(6, len(ids)), algorithm='brute', n_jobs=1).fit(z[ids])
        local = nn.kneighbors(z[ids], return_distance=False)
        for k, base in enumerate(ids):
            others = ids[local[k][ids[local[k]] != base]][:5]
            if len(others):
                links[base] = np.resize(others, 5)
                eligible += 1
    return links, eligible, len(groups)


def interpolate(values, links, base, choice, lambdas):
    neighbor = links[base, choice]
    result = values[base].copy()
    columns = [NUMERIC_FEATURES.index(c) for c in CONTINUOUS]
    result[:, columns] += lambdas[:, None] * (values[neighbor][:, columns]-values[base][:, columns])
    return result, neighbor


def validate(values):
    for j, c in enumerate(NUMERIC_FEATURES):
        finite = values[:, j][~np.isnan(values[:, j])]
        if not np.isfinite(finite).all():
            raise RuntimeError(f'Nonfinite synthetic feature: {c}')
        if c in CONTINUOUS and c not in ['rain_change_5m', 'rain_trend_15m'] and (finite < -1e-9).any():
            raise RuntimeError(f'Negative synthetic rainfall: {c}')
    for a, b in [('hour_sin', 'hour_cos'), ('month_sin', 'month_cos')]:
        assert np.allclose(values[:, NUMERIC_FEATURES.index(a)]**2+values[:, NUMERIC_FEATURES.index(b)]**2, 1)
    for a, b in [('rain_15m', 'rain_30m'), ('rain_30m', 'rain_60m')]:
        x, y = values[:, NUMERIC_FEATURES.index(a)], values[:, NUMERIC_FEATURES.index(b)]
        ok = ~np.isnan(x) & ~np.isnan(y)
        if (x[ok] > y[ok]+1e-8).any():
            raise RuntimeError('Complete synthetic rolling totals are inconsistent')
    age = values[:, NUMERIC_FEATURES.index('latest_age_min')]
    count = values[:, NUMERIC_FEATURES.index('history_count_30m')]
    fraction = values[:, NUMERIC_FEATURES.index('wet_fraction_30m')]
    wet = (values[:, NUMERIC_FEATURES.index('latest_rain_mm')] > 0).astype(int)
    for lag in [5, 10, 15, 30, 60]:
        rain = values[:, NUMERIC_FEATURES.index(f'rain_lag_{lag}m')]
        wet += ((age <= lag) & (lag <= age+25) & (lag != age) & (rain > 0))
    known = np.isfinite(age) & np.isfinite(fraction) & (count > 0)
    if (wet[known] > np.rint(count[known]*fraction[known])).any():
        raise RuntimeError('Synthetic visible wet slots exceed stored 30-minute wet count')


def generate(q):
    work = WORK/'smote'/f'gt_{q}mm'
    marker = work/'generation.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if result['version'] != VERSION:
            raise RuntimeError('SMOTE version mismatch')
        print('SMOTE_RESUMED', json.dumps(result), flush=True)
        return
    prepared = json.loads((WORK/'preparation.json').read_text())
    n, positive = prepared['train_rows'], int(prepared['train_positives'][str(q)])
    target_positive = min(10*positive, math.ceil((n-positive)*0.2/0.8))
    extra = max(0, target_positive-positive)
    minority = ds.dataset(str(WORK/'minority'/f'gt_{q}mm'), format='parquet', exclude_invalid_files=True).to_table()
    if len(minority) != positive or positive > 200000:
        raise RuntimeError('Minority count mismatch or bounded pool limit exceeded')
    stations = np.asarray(minority['station_id'].to_pylist(), dtype=object)
    epochs = np.asarray(minority['slot_epoch'].to_numpy(), dtype=np.int64)
    if (epochs >= 1672502400).any():  # 2023-01-01 00:00 Asia/Singapore
        raise RuntimeError('Nontraining minority rows')
    values = np.column_stack([minority[c].to_numpy(zero_copy_only=False) for c in NUMERIC_FEATURES]).astype(float)
    scales = np.asarray([prepared['distance_scales'].get(c, 1.0) or 1.0 for c in NUMERIC_FEATURES])
    links, eligible, group_count = neighbors_for(stations, values, scales)
    schema = pa.schema([('station_id', pa.string()), *[(c, pa.float64()) for c in NUMERIC_FEATURES],
                        ('label', pa.float64()), ('source_slot_a', pa.int64()),
                        ('source_slot_b', pa.int64()), ('interpolation_lambda', pa.float64())])
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5208+q)
    interpolated = changed = fallback = 0
    with pq.ParquetWriter(work/'synthetic.parquet', schema, compression='snappy') as writer:
        for start in range(0, extra, 10000):
            size = min(10000, extra-start)
            base = rng.integers(0, positive, size=size)
            choice = rng.integers(0, 5, size=size)
            lam = rng.random(size)
            generated, other = interpolate(values, links, base, choice, lam)
            validate(generated)
            interpolated += int((other != base).sum())
            fallback += int((other == base).sum())
            changed += int(np.any(np.abs(np.nan_to_num(generated-values[base], nan=0)) > 1e-10, axis=1).sum())
            columns = {'station_id': stations[base], **{c: generated[:, j] for j, c in enumerate(NUMERIC_FEATURES)},
                       'label': np.ones(size), 'source_slot_a': epochs[base], 'source_slot_b': epochs[other],
                       'interpolation_lambda': lam}
            arrays = [pa.array(columns[field.name], type=field.type, from_pandas=True) for field in schema]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    result = {'version': VERSION, 'threshold_mm': q, 'original_rows': n, 'original_positives': positive,
              'added_rows': extra, 'augmented_positives': target_positive,
              'augmented_positive_fraction': target_positive/(n+extra),
              'minority_rows_with_neighbor': eligible, 'profile_groups': group_count,
              'interpolated_rows': interpolated, 'numerically_changed_rows': changed,
              'isolated_profile_duplicate_rows': fallback, 'target_positive_fraction': 0.2,
              'maximum_positive_multiplier': 10, 'neighbors_k': 5,
              'continuous_interpolated_features': CONTINUOUS,
              'fixed_features': LOCKED+CALENDAR, 'training_only': True,
              'scope': 'Same station, identical missingness, rain zero pattern and locked feature profile; calendar copied from base'}
    save_json(marker, result)
    print('SMOTE_COMPLETE', json.dumps(result), flush=True)


def check():
    # Distinct station/profile must never be interpolated; an isolated profile copies itself.
    values = np.zeros((4, len(NUMERIC_FEATURES)))
    for i in range(4):
        values[i, 1] = values[i, 3] = 1
        for c in ['latest_rain_mm', 'rain_15m', 'rain_30m', 'rain_60m', 'max_rain_30m']:
            values[i, NUMERIC_FEATURES.index(c)] = i+1
    values[3, NUMERIC_FEATURES.index('latest_wet')] = 1
    links, eligible, _ = neighbors_for(['S1', 'S1', 'S2', 'S1'], values, np.ones(len(NUMERIC_FEATURES)))
    assert eligible == 2 and np.all(links[0] == 1) and np.all(links[1] == 0)
    assert np.all(links[2] == 2) and np.all(links[3] == 3)
    generated, other = interpolate(values, links, np.array([0, 2]), np.array([0, 0]), np.array([0.25, 0.5]))
    assert generated[0, NUMERIC_FEATURES.index('rain_30m')] == 1.25
    assert np.array_equal(generated[1], values[2])
    validate(generated)
    # Equal wet counts do not suffice: interpolation must not union different wet-slot patterns.
    states = np.zeros((2, len(NUMERIC_FEATURES)))
    states[:, 1] = states[:, 3] = 1
    for c, value in [('latest_age_min', 10), ('history_count_30m', 6),
                     ('wet_fraction_30m', 1/6), ('rain_30m', .2), ('rain_60m', .2), ('max_rain_30m', .2)]:
        states[:, NUMERIC_FEATURES.index(c)] = value
    states[0, NUMERIC_FEATURES.index('rain_lag_15m')] = .2
    states[0, NUMERIC_FEATURES.index('rain_15m')] = .2
    states[1, NUMERIC_FEATURES.index('rain_lag_30m')] = .2
    validate(states)
    try:
        validate((states[:1]+states[1:])/2)
    except RuntimeError as exc:
        assert 'wet slots' in str(exc)
    else:
        raise AssertionError('Contradictory wet-count interpolation was accepted')
    links, eligible, _ = neighbors_for(['S1', 'S1'], states, np.ones(len(NUMERIC_FEATURES)))
    assert eligible == 0 and np.all(links[0] == 0) and np.all(links[1] == 1)
    print('CONSTRAINED_SMOTE_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--threshold', type=int, choices=[0, 5, 25])
    args = parser.parse_args()
    if args.check_only:
        check()
    elif args.threshold is not None:
        generate(args.threshold)
    else:
        parser.error('Choose --check-only or --threshold')
