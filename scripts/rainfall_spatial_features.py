"""Causal interactions derived only from the existing as-of rainfall features.

No future labels, rainfall thresholds or evaluation data enter these formulas.
Null inputs stay null. These describe spatial contrasts, not storm velocity.
"""
EXPRESSIONS = {
    'neighbor_own_latest_gap': 'neighbor_latest_mean_mm - latest_rain_mm',
    'neighbor_own_30m_gap': 'neighbor_30m_mean_mm - rain_30m',
    'neighbor_peak_own_latest_gap': 'neighbor_latest_max_mm - latest_rain_mm',
    'neighbor_peak_mean_gap': 'neighbor_latest_max_mm - neighbor_latest_mean_mm',
    'neighbor_wet_own_dry': '''CASE
        WHEN latest_wet IS NULL OR neighbor_wet_fraction IS NULL THEN NULL
        WHEN latest_wet = 0 THEN neighbor_wet_fraction ELSE 0.0 END''',
    'neighbor_own_age_gap': 'neighbor_mean_age_min - latest_age_min',
    'own_recent_rain_share': 'CASE WHEN rain_30m > 0 THEN rain_15m / rain_30m ELSE NULL END',
    'neighbor_latest_log': 'ln(1.0 + neighbor_latest_mean_mm)',
}
EXTRA_FEATURES = list(EXPRESSIONS)


def stage():
    from pyspark.ml.feature import SQLTransformer
    return SQLTransformer(statement='SELECT *, ' + ', '.join(
        f'({formula}) AS `{name}`' for name, formula in EXPRESSIONS.items()) + ' FROM __THIS__')


def check(spark):
    """Check numeric formulas, missingness and independence from future labels."""
    from pyspark.sql import functions as F
    schema = ('latest_rain_mm double, rain_30m double, latest_wet double, '
              'latest_age_min double, rain_15m double, neighbor_latest_mean_mm double, '
              'neighbor_30m_mean_mm double, neighbor_latest_max_mm double, '
              'neighbor_wet_fraction double, neighbor_mean_age_min double')
    frame = spark.createDataFrame([
        (0., 4., 0., 10., 3., 2., 6., 5., .5, 15.),
        (None, None, None, None, None, None, None, None, None, None),
        (0., 0., 0., 10., 0., 0., 0., 0., 0., 10.),
    ], schema)
    rows = stage().transform(frame).select(*EXTRA_FEATURES).collect()
    assert rows[0].neighbor_own_latest_gap == 2.
    assert rows[0].neighbor_own_30m_gap == 2.
    assert rows[0].neighbor_peak_mean_gap == 3.
    assert rows[0].neighbor_wet_own_dry == .5
    assert rows[0].neighbor_own_age_gap == 5.
    assert rows[0].own_recent_rain_share == .75
    assert all(x is None for x in rows[1])
    assert rows[2].own_recent_rain_share is None
    changed = stage().transform(frame.withColumn('future_30min_mm', F.lit(9999.)))
    assert changed.select(*EXTRA_FEATURES).collect() == rows
    print('SPATIAL_FEATURE_CHECKS_OK', flush=True)
