"""Check provenance reconciliation and identities on exported Task 1 results."""
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / 'reports'
    overview = json.loads((report / 'task1_overview.json').read_text())
    audits = [json.loads((report / f'quality/{y}.json').read_text()) for y in range(2017, 2025)]
    for audit in audits:
        manifest = json.loads((report / f'downloads/{audit["source_year"]}.json').read_text())
        assert audit['raw_rows'] == manifest['api_reported_rows']
        assert audit['valid_rows_before_deduplication'] - audit['duplicate_rows_removed'] == audit['clean_rows']
        assert len(manifest['sha256_uncompressed']) == 64
    sy = pd.read_csv(report / 'tables/station_year.csv')
    assert not sy.duplicated(['station_id', 'year']).any()
    assert sy.n_observed.sum() == overview['clean_rows_2017_2024']
    assert sum(a['clean_rows'] for a in audits) - overview['cross_file_duplicate_rows_removed'] == sy.n_observed.sum()
    assert sy.station_id.nunique() == overview['station_ids']
    assert sy.coverage.between(0, 1).all()
    expected = sy.year.map(lambda y: (366 if calendar.isleap(y) else 365)*288)
    np.testing.assert_allclose(sy.coverage, sy.n_observed/expected)
    np.testing.assert_allclose(sy.wet_frequency_pct, 100*sy.n_wet/sy.n_observed)
    np.testing.assert_allclose(sy.rain_per_observed_24h_mm, 288*sy.observed_total_mm/sy.n_observed)
    wet = sy[sy.n_wet > 0]
    np.testing.assert_allclose(wet.wet_intensity_mm_5min, wet.observed_total_mm/wet.n_wet)
    panel = sy[sy.station_id.isin(overview['balanced_panel_station_ids'])]
    assert panel.groupby('station_id').year.nunique().eq(8).all()
    assert panel.coverage.ge(.8).all()
    membership = pd.read_csv(report / 'tables/monthly_panel_membership.csv')
    assert membership.coverage.ge(.8).all()
    assert membership.station_id.isin(overview['balanced_panel_station_ids']).all()
    monthly = pd.read_csv(report / 'tables/panel_month_year.csv')
    actual = membership.groupby(['year', 'month']).station_id.nunique().sort_index()
    np.testing.assert_array_equal(actual.to_numpy(), monthly.set_index(['year','month']).n_stations.sort_index().to_numpy())
    pairs = pd.read_csv(report / 'tables/station_correlation_pairs.csv')
    assert not pairs.duplicated(['station_a','station_b']).any()
    assert pairs.paired_days.ge(365).all()
    assert pairs.daily_rain_pearson_r.between(-1,1).all()
    assert pairs.distance_km.ge(0).all()
    assert len(list((report / 'figures').glob('*.png'))) == 9
    assert len(list((report / 'figures').glob('*.svg'))) == 9
    result = dict(status='PASS', source_rows=sum(a['raw_rows'] for a in audits),
                  cleaned_rows=int(sy.n_observed.sum()), duplicate_rows_removed=sum(a['duplicate_rows_removed'] for a in audits),
                  station_ids=int(sy.station_id.nunique()), fixed_panel_stations=len(overview['balanced_panel_station_ids']),
                  qualifying_year_months=len(monthly), qualifying_station_month_min=int(monthly.n_stations.min()),
                  correlation_pairs=len(pairs), checks='Source/API counts; dedup count reconciliation; station-year uniqueness; leap-year coverage; metric identities; panel and monthly membership; correlation bounds; figure counts.')
    (report / 'validation_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
