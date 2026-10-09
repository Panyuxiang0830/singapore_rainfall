"""Publication-style Task 1 figures from Spark aggregate tables only."""
import calendar
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'reports/figures'
FIGS.mkdir(parents=True, exist_ok=True)
BLUE, TEAL, GOLD, INK = '#2563a6', '#008b83', '#dc9a31', '#20324a'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 13,
    'axes.labelsize': 10, 'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff', 'axes.edgecolor': '#c4ccd5',
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#e5eaf0', 'grid.linewidth': .7, 'savefig.facecolor': 'white',
})


def table(name):
    return pd.read_csv(ROOT / f'reports/tables/{name}.csv')


def save(fig, name, title, subtitle, footnote):
    fig.suptitle(title, fontsize=20, fontweight='bold', x=.06, ha='left', y=1.015, va='bottom')
    fig.text(.06, .945, subtitle, fontsize=10, color='#66768a', ha='left')
    fig.text(.06, .018, footnote, fontsize=8, color='#66768a', va='bottom', ha='left')
    fig.savefig(FIGS / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIGS / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)
    print('FIGURE', name, flush=True)


def rings(geometry):
    if geometry['type'] == 'Polygon':
        yield from geometry['coordinates'][:1]
    elif geometry['type'] == 'MultiPolygon':
        for poly in geometry['coordinates']:
            yield poly[0]


def basemap(ax, boundary):
    ax.set_facecolor('#edf5fa')
    for feature in boundary['features']:
        for ring in rings(feature['geometry']):
            ax.add_patch(Polygon(ring, closed=True, facecolor='#e2e9dd', edgecolor='#a2b5a3', lw=.7, zorder=1))
    ax.set_xlim(103.59, 104.11)
    ax.set_ylim(1.14, 1.49)
    ax.set_aspect(1 / np.cos(np.deg2rad(1.32)))
    ax.set_xlabel('Longitude (degrees east)')
    ax.set_ylabel('Latitude (degrees north)')
    ax.grid(alpha=.5)
    ax.ticklabel_format(useOffset=False)
    ax.text(103.62, 1.155, '0       5       10 km', fontsize=8, color='#65778a')
    km_deg = 10 / (111.32 * np.cos(np.deg2rad(1.32)))
    ax.plot([103.62, 103.62 + km_deg], [1.15, 1.15], color=INK, lw=2)
    ax.annotate('N', xy=(104.085, 1.465), xytext=(104.085, 1.425), ha='center',
                arrowprops={'arrowstyle': '->', 'color': INK}, fontsize=10)


def balanced_monthly(monthly, panel_ids):
    frame = monthly[monthly.station_id.isin(panel_ids) & (monthly.coverage >= .8)].copy()
    counts = frame.groupby(['station_id', 'month']).year.nunique()
    keys = counts[counts == 8].reset_index()[['station_id', 'month']]
    frame = frame.merge(keys, on=['station_id', 'month'])
    metrics = ['wet_frequency_pct', 'wet_intensity_mm_5min', 'rain_per_observed_24h_mm', 'coverage']
    pooled = frame.groupby(['year', 'month'])[metrics].mean().reset_index()
    pooled['n_stations'] = frame.groupby(['year', 'month']).station_id.nunique().values
    return frame, pooled


def make_figures():
    overview = json.loads((ROOT / 'reports/task1_overview.json').read_text())
    boundary = json.loads((ROOT / 'reports/assets/singapore_boundary.geojson').read_text())
    sy = table('station_year')
    sy[(sy.coverage >= .9) & (sy.n_wet == 0)].to_csv(
        ROOT / 'reports/tables/high_coverage_zero_rain_station_years.csv', index=False)
    sm = table('station_month')
    panel_ids = overview['balanced_panel_station_ids']
    mapped = sy[sy.year == 2024].copy()
    mapped = mapped[mapped.lat.between(1.1, 1.6) & mapped.lon.between(103.4, 104.3)]
    eligible = mapped[mapped.coverage >= .9].copy()
    excluded = mapped[mapped.coverage < .9]
    strict_frame, strict_year = balanced_monthly(sm, panel_ids)
    strict_year.to_csv(ROOT / 'reports/tables/strict_balanced_month_year.csv', index=False)
    month_frame = sm[sm.station_id.isin(panel_ids) & (sm.coverage >= .8)].copy()
    metrics = ['wet_frequency_pct', 'wet_intensity_mm_5min', 'rain_per_observed_24h_mm', 'coverage']
    month_year = month_frame.groupby(['year', 'month'])[metrics].mean().reset_index()
    month_year['n_stations'] = month_frame.groupby(['year', 'month']).station_id.nunique().values
    month_frame[['station_id','year','month','coverage']].to_csv(ROOT / 'reports/tables/monthly_panel_membership.csv', index=False)
    month_year.to_csv(ROOT / 'reports/tables/panel_month_year.csv', index=False)
    sensitivity = sy.copy()
    sensitivity['rain_per_observed_24h_without_over50'] = (
        288 * sensitivity.sum_excluding_over50_mm /
        (sensitivity.n_observed - sensitivity.n_flagged_over50))
    sensitivity['wet_intensity_without_over50'] = (
        sensitivity.sum_excluding_over50_mm /
        (sensitivity.n_wet - sensitivity.n_flagged_over50).replace(0, np.nan))
    sensitivity.to_csv(ROOT / 'reports/tables/extreme_reading_sensitivity.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7.4))
    fig.subplots_adjust(top=.84, bottom=.15, wspace=.28)
    for ax, metric, title, label, cmap in [
        (axs[0], 'wet_frequency_pct', 'How often does it rain?', 'Wet five-minute intervals (%)', 'viridis'),
        (axs[1], 'wet_intensity_mm_5min', 'How intense is rain when it occurs?', 'mm per wet five-minute interval', 'YlOrRd')]:
        basemap(ax, boundary)
        dots = ax.scatter(eligible.lon, eligible.lat, c=eligible[metric], s=54, cmap=cmap,
                          edgecolors='white', linewidths=.7, zorder=3)
        ax.scatter(excluded.lon, excluded.lat, c='#a7b0bc', marker='x', s=28, lw=1, zorder=2,
                   label='<90% annual coverage')
        ax.set_title(title, loc='left', pad=12)
        fig.colorbar(dots, ax=ax, shrink=.75, pad=.03).set_label(label)
        if len(excluded): ax.legend(loc='upper left', fontsize=7, framealpha=.9)
    wettest = eligible.nlargest(1, 'wet_frequency_pct').iloc[0]
    axs[0].annotate(str(wettest.station_id), (wettest.lon, wettest.lat), xytext=(5, 9),
                    textcoords='offset points', fontsize=8, fontweight='bold')
    save(fig, '01_station_map', 'Rainfall across Singapore monitoring stations',
         f'2024 | {len(eligible)} stations with >=90% annual coverage | colour shows measured station statistics',
         'Source: NEA/data.gov.sg. Land outline: geoBoundaries (2016 boundary, ODbL). Grey crosses have low coverage.\n'
         'Points are observations at gauges; no spatial interpolation or area-wide rainfall estimates are implied.')

    fig, axs = plt.subplots(1, 2, figsize=(15.5, 8.8))
    fig.subplots_adjust(top=.85, bottom=.14, left=.07, right=.97, wspace=.75)
    points = axs[0].scatter(eligible.wet_frequency_pct, eligible.wet_intensity_mm_5min,
                   s=35 + 50 * eligible.coverage, c=eligible.rain_per_observed_24h_mm,
                   cmap='viridis', alpha=.9, edgecolors='white', lw=.5)
    labels = pd.concat([eligible.nlargest(3, 'wet_frequency_pct'), eligible.nlargest(3, 'wet_intensity_mm_5min'),
                        eligible.nsmallest(2, 'wet_frequency_pct')]).drop_duplicates('station_id')
    for i, row in enumerate(labels.itertuples()):
        axs[0].annotate(row.station_id, (row.wet_frequency_pct, row.wet_intensity_mm_5min),
                        xytext=(5, 7 if i % 2 == 0 else -10), textcoords='offset points', fontsize=8)
    fig.colorbar(points, ax=axs[0], shrink=.72, pad=.04).set_label('mm per observed 24 hours')
    axs[0].set(xlabel='Wet five-minute intervals (%)', ylabel='mm per wet five-minute interval',
               title='Frequency and intensity\nmeasure different things')
    axs[0].grid(alpha=.7)
    ranked = pd.concat([eligible.nsmallest(10, 'wet_frequency_pct'), eligible.nlargest(10, 'wet_frequency_pct')])
    ranked = ranked.drop_duplicates('station_id').sort_values('wet_frequency_pct')
    labels = [f'{r.station_id}  {r.station_name[:27]}' for r in ranked.itertuples()]
    axs[1].barh(labels, ranked.wet_frequency_pct, color=[TEAL if x >= eligible.wet_frequency_pct.median() else '#b6c8dc' for x in ranked.wet_frequency_pct])
    axs[1].set(xlabel='Wet five-minute intervals (%)', title='Ten highest and ten lowest eligible stations')
    axs[1].tick_params(axis='y', labelsize=8)
    axs[1].grid(axis='x', alpha=.5)
    save(fig, '02_station_comparison', 'Comparing stations fairly',
         '2024 | identical annual period, >=90% coverage | labels identify selected stations',
         'All eligible stations appear in the scatter; the ranking shows extremes only. The interactive dashboard lists every station.\n'
         'Rainfall intensity uses observed positive readings. Measurement issues and spatially uneven missingness remain possible.')

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(top=.84, bottom=.24, wspace=.25)
    palette = plt.get_cmap('viridis')(np.linspace(.05, .9, 8))
    for ax, metric, ylabel, title in [
        (axs[0], 'wet_frequency_pct', 'Wet five-minute intervals (%)', 'Frequency'),
        (axs[1], 'wet_intensity_mm_5min', 'mm per wet five-minute interval', 'Intensity when wet')]:
        for i, year in enumerate(range(2017, 2025)):
            frame = month_year[month_year.year == year].set_index('month').reindex(range(1,13))
            ax.plot(frame.index, frame[metric], lw=1, alpha=.6, c=palette[i], label=str(year))
        average = month_year.groupby('month')[metric].mean()
        ax.plot(average.index, average, c=INK, lw=3, marker='o', ms=5, label='Mean of available years')
        ax.set(title=title, ylabel=ylabel, xlabel='Month (Singapore time)')
        ax.set_xticks(range(1, 13), [calendar.month_abbr[i] for i in range(1, 13)], rotation=35)
        ax.grid(alpha=.7)
    handles, legend_labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .14),
               ncol=5, frameon=False, fontsize=8)
    save(fig, '03_monthly_patterns', 'Seasonal rainfall: frequency versus intensity',
         f'2017-2024 | stable {len(panel_ids)}-station pool; each station-year-month requires >=80% coverage',
         'Station subsets and available years vary by month; each eligible station and year has equal weight. Gaps are not interpolated.\n'
         'Missing intervals are excluded from denominators, never filled with zero. Eight years do not establish a long-term climate trend.')

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(top=.84, bottom=.17, wspace=.35)
    values = month_year.pivot(index='year', columns='month', values='rain_per_observed_24h_mm').reindex(index=range(2017, 2025), columns=range(1, 13))
    coverage = month_frame.groupby(['year', 'month']).coverage.mean().unstack().reindex(index=range(2017, 2025), columns=range(1, 13)) * 100
    for ax, frame, title, cmap, fmt in [
        (axs[0], values, 'Rainfall per observed 24 hours (mm)', 'YlGnBu', '.1f'),
        (axs[1], coverage, 'Observation coverage (%)', 'Greens', '.0f')]:
        colormap = plt.get_cmap(cmap).copy(); colormap.set_bad('#d9dfe6')
        image = ax.imshow(frame, aspect='auto', cmap=colormap)
        ax.set_xticks(range(12), [calendar.month_abbr[i] for i in range(1, 13)], rotation=45)
        ax.set_yticks(range(8), range(2017, 2025))
        ax.set_title(title, loc='left', pad=12)
        for y in range(8):
            for x in range(12):
                v = frame.iloc[y, x]
                if np.isfinite(v):
                    norm = image.norm(v)
                    ax.text(x, y, format(v, fmt), ha='center', va='center', fontsize=7,
                            color='white' if norm > .62 else INK)
        fig.colorbar(image, ax=ax, shrink=.75, pad=.03)
    save(fig, '04_year_month_heatmap', 'Year-to-year changes within each season',
         'Stable station pool; monthly coverage >=80% | grey cells have no eligible observations',
         'Rainfall per observed 24h = observed sum / valid five-minute count x 288; this is not an official daily or annual total.\n'
         'Station subsets vary by year-month. Strict all-eight-year month comparisons are provided in strict_balanced_month_year.csv.')

    fig, axs = plt.subplots(1, 3, figsize=(14, 6.5))
    fig.subplots_adjust(top=.82, bottom=.18, wspace=.38)
    annual = sy[sy.station_id.isin(panel_ids)].groupby('year')[['rain_per_observed_24h_mm', 'wet_frequency_pct', 'wet_intensity_mm_5min', 'coverage']].mean()
    for ax, metric, ylabel, color, title in [
        (axs[0], 'rain_per_observed_24h_mm', 'mm per observed 24 hours', BLUE, 'Rainfall amount'),
        (axs[1], 'wet_frequency_pct', 'Wet five-minute intervals (%)', TEAL, 'Rain frequency'),
        (axs[2], 'wet_intensity_mm_5min', 'mm per wet five-minute interval', GOLD, 'Intensity when wet')]:
        bars = ax.bar(annual.index.astype(str), annual[metric], color=color, width=.65)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.set_ylim(0, annual[metric].max() * 1.18)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=.6)
        ax.set_axisbelow(True)
    save(fig, '05_annual_changes', 'Eight years of rainfall differences',
         f'2017-2024 | the same {len(panel_ids)} stable stations in every year | equal station weights',
         'Each panel measures a different component of rainfall. Coverage is >=80% for every station-year in this panel.\n'
         'Amount is normalized by observed duration; missing intervals and unvalidated gauge readings limit interpretation.')

    hourly = table('panel_station_month_hour')
    hourly = hourly[hourly.coverage >= .8]
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(top=.84, bottom=.16, wspace=.3)
    hourly_curve = hourly.groupby('hour')[['wet_frequency_pct', 'wet_intensity_mm_5min']].mean()
    axs[0].plot(hourly_curve.index, hourly_curve.wet_frequency_pct, c=TEAL, lw=3, marker='o', ms=4)
    axs[0].set(xlabel='Hour (Singapore time)', ylabel='Wet five-minute intervals (%)', title='Daily rhythm in rain frequency')
    axs[0].set_xticks(range(0, 24, 3)); axs[0].grid(alpha=.7)
    heat = hourly.groupby(['month', 'hour']).wet_frequency_pct.mean().unstack().reindex(index=range(1, 13), columns=range(24))
    image = axs[1].imshow(heat, aspect='auto', cmap='YlGnBu')
    axs[1].set_yticks(range(12), [calendar.month_abbr[i] for i in range(1, 13)])
    axs[1].set_xticks(range(0, 24, 3), range(0, 24, 3))
    axs[1].set(xlabel='Hour (Singapore time)', title='How the daily rhythm changes by month')
    fig.colorbar(image, ax=axs[1], shrink=.8).set_label('Wet intervals (%)')
    save(fig, '06_diurnal_patterns', 'When during the day does it rain?',
         f'2017-2024 | {len(panel_ids)}-station panel | station-year-month-hour requires >=80% coverage',
         'Rates are averaged over eligible station-year-month-hour strata; equal weights avoid domination by longer records.\n'
         'Observation timestamps are used for temporal grouping; precise five-minute interval endpoint semantics remain undocumented.')

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(top=.84, bottom=.17, wspace=.3)
    for i, year in enumerate(range(2017, 2025)):
        y = sy[sy.year == year].coverage * 100
        # Deterministic horizontal jitter for station dots.
        jitter = np.linspace(-.2, .2, len(y))
        axs[0].scatter(np.full(len(y), year) + jitter, y, s=14, c=BLUE, alpha=.45)
    axs[0].axhline(80, c=GOLD, lw=1, ls='--', label='80% panel threshold')
    axs[0].set(ylim=(0, 102), ylabel='Annual observation coverage (%)', title='Station coverage varies by year')
    axs[0].set_xticks(range(2017, 2025)); axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(axis='y', alpha=.6); axs[0].legend(fontsize=8)
    years = sy.groupby('year').agg(stations=('station_id', 'nunique'), high=('coverage', lambda x: (x >= .9).sum()))
    x = np.arange(8)
    axs[1].bar(x - .18, years.stations, width=.36, color='#b6c8dc', label='Any valid records')
    axs[1].bar(x + .18, years.high, width=.36, color=TEAL, label='>=90% coverage')
    axs[1].set_xticks(x, years.index, rotation=45)
    axs[1].set(ylabel='Station IDs', title='A changing monitoring network')
    axs[1].grid(axis='y', alpha=.6); axs[1].legend(fontsize=8); axs[1].set_axisbelow(True)
    save(fig, '07_data_coverage', 'Data quality comes before weather conclusions',
         'Coverage = unique valid five-minute observations / expected full-calendar observations',
         'No-record station-years are absent, not zero-rain years. The raw datasets have not undergone official climate quality control.\n'
         'Station additions, retirement, relocation, and missing periods can change apparent annual and spatial patterns.')

    daily = table('panel_daily_pivot').set_index('day')
    corr = daily.corr(min_periods=365)
    overlap = daily.notna().astype(int).T @ daily.notna().astype(int)
    names = sorted(panel_ids)
    corr = corr.reindex(index=names, columns=names)
    latest = sy[sy.station_id.isin(panel_ids)].sort_values('year').drop_duplicates('station_id', keep='last').set_index('station_id')
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if pd.notna(corr.loc[a, b]):
                lat1, lat2 = np.deg2rad([latest.loc[a, 'lat'], latest.loc[b, 'lat']])
                dl = np.deg2rad(latest.loc[b, 'lon'] - latest.loc[a, 'lon'])
                h = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dl / 2)**2
                distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
                pairs.append({'station_a': a, 'station_b': b, 'distance_km': distance,
                              'daily_rain_pearson_r': corr.loc[a, b], 'paired_days': overlap.loc[a, b]})
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(ROOT / 'reports/tables/station_correlation_pairs.csv', index=False)
    corr.to_csv(ROOT / 'reports/tables/station_daily_correlation.csv')
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(top=.82, bottom=.20)
    ax.scatter(pair_frame.distance_km, pair_frame.daily_rain_pearson_r, c=TEAL, s=18, alpha=.4, edgecolors='none')
    pair_frame['distance_bin'] = pd.cut(pair_frame.distance_km, np.arange(0, 61, 5), right=False)
    trend = pair_frame.groupby('distance_bin', observed=True).agg(distance=('distance_km', 'mean'), r=('daily_rain_pearson_r', 'mean'))
    ax.plot(trend.distance, trend.r, c=INK, lw=2, marker='o', ms=4, label='Mean within 5 km bins')
    ax.set(xlabel='Gauge separation (km)', ylabel='Daily rainfall correlation (Pearson r)', title='Nearby stations generally show stronger co-movement')
    ax.grid(alpha=.7); ax.legend(fontsize=8)
    save(fig, '08_spatial_correlation', 'Distance and daily-rainfall correlation',
         f'{len(panel_ids)} stable stations | station-days require >=95% coverage; >=365 paired days per correlation',
         'Each point is a station pair; the line is the mean within a 5 km distance group.\n'
         'Nearby gauges can share rainfall events, and different pairs share stations: the samples are not independent.')

    fig, axs = plt.subplots(1, 3, figsize=(14, 6.5))
    fig.subplots_adjust(top=.82, bottom=.22, wspace=.45)
    distribution = table('rain_distribution').groupby('rain_bin')['count'].sum().sort_index()
    wet_bins = distribution.iloc[1:]
    axs[0].bar(range(len(wet_bins)), wet_bins / wet_bins.sum() * 100, color=BLUE)
    axs[0].set_xticks(range(len(wet_bins)), [x.split(': ', 1)[1] for x in wet_bins.index], rotation=45, ha='right', fontsize=8)
    axs[0].set(ylabel='Share of wet intervals (%)', xlabel='Five-minute rainfall (mm)', title='Rainfall amounts when wet')
    axs[0].grid(axis='y', alpha=.6); axs[0].set_axisbelow(True)
    halfhour = overview['halfhour_2024_descriptive_only']
    qs = [0, 1, 5, 10, 20]
    axs[1].bar([str(q) for q in qs], [100 * halfhour[f'p_gt_{q}mm'] for q in qs], color=TEAL)
    axs[1].set(xlabel='Threshold q (mm)', ylabel='Windows exceeding q (%)', title='Next six observation slots')
    axs[1].grid(axis='y', alpha=.6); axs[1].set_axisbelow(True)
    persistence = table('rain_persistence_2024')
    persistence['previous_wet'] = persistence.previous_wet.astype(str).str.lower() == 'true'
    persistence = persistence.sort_values('previous_wet')
    rates = 100 * persistence.current_wet_pairs / persistence.n_pairs
    axs[2].bar(['Previous interval dry', 'Previous interval wet'], rates, color=['#b6c8dc', GOLD], width=.55)
    axs[2].tick_params(axis='x', rotation=25)
    axs[2].set(ylabel='Current interval wet (%)', title='Short-term rain persistence')
    axs[2].grid(axis='y', alpha=.6); axs[2].set_axisbelow(True)
    save(fig, '09_distribution_and_persistence', 'Why rainfall prediction needs more than accuracy',
         'Distribution: 2017-2024 | half-hour windows and adjacent-interval associations: 2024',
         'Six future label slots span 30 minutes; physical interval endpoints are unconfirmed. These are not model predictions.\n'
         'Overlapping windows are dependent. Task 2 thresholds/features must be selected using training/validation years only.')

    result = {
        'panel_stations': len(panel_ids), 'map_2024_eligible_stations': len(eligible),
        'wettest_frequency_2024': eligible.nlargest(1, 'wet_frequency_pct')[['station_id', 'station_name', 'wet_frequency_pct', 'coverage']].to_dict('records')[0],
        'monthly_available_years': {int(m): int(n) for m,n in month_year.groupby('month').year.nunique().items()},
        'strict_balanced_months': sorted(int(x) for x in strict_year.month.unique()),
        'monthly_max_frequency_month': int(month_year.groupby('month').wet_frequency_pct.mean().idxmax()),
        'monthly_min_frequency_month': int(month_year.groupby('month').wet_frequency_pct.mean().idxmin()),
        'hour_max_frequency': int(hourly_curve.wet_frequency_pct.idxmax()),
        'annual_panel_metrics': annual.reset_index().to_dict('records'),
        'rain_persistence_2024': persistence.assign(current_wet_pct=rates).to_dict('records'),
        'overall_dry_interval_pct': float(100 * distribution.iloc[0] / distribution.sum()),
        'high_coverage_all_zero_station_years': int(((sy.coverage >= .9) & (sy.n_wet == 0)).sum()),
        'over50_sensitivity': {
            'flagged_intervals': int(sy.n_flagged_over50.sum()),
            'affected_station_years': int((sy.n_flagged_over50 > 0).sum()),
            'max_station_year_amount_change_pct': float((
                100 * (sy.rain_per_observed_24h_mm - sensitivity.rain_per_observed_24h_without_over50) /
                sy.rain_per_observed_24h_mm.replace(0, np.nan)).max()),
        },
        'pair_correlation_summary': {'pairs': len(pair_frame), 'median_r': float(pair_frame.daily_rain_pearson_r.median())},
    }
    (ROOT / 'reports/task1_findings.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == '__main__':
    make_figures()
