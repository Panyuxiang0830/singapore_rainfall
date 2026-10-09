# Task 1 methods and interpretation

## Data and provenance

Use all eight NEA annual datasets from 2017 through 2024. API previews are used
only to discover schema and dataset IDs. Each complete CSV is downloaded from
the official API and compressed without altering its bytes. Download manifests
record dataset ID, API row count, byte count, retrieval time, and SHA256.

## Cleaning

All full-data transformations use PySpark 3.5.8. Parse ISO-8601 observations
and update timestamps with the Spark session timezone set to Asia/Singapore.
Use observation timestamps for calendar grouping; audit disagreements with
the separate `date` field. Retain only finite nonnegative rainfall readings
with millimetre units, the expected five-minute-total reading type, a station
ID, a parseable timestamp, and a valid CSV record.

Round observation epochs to the nearest five-minute grid only when the shift
is at most two seconds. This accommodates both `xx:04:59` and `xx:05:00` forms
without silently resampling irregular observations. Audit excluded off-grid
rows. Retain original epochs. Aligning the timestamp does not resolve whether
the original label denotes the start or end of its measurement interval.

Deduplicate station/grid-time keys by latest reading-update timestamp, then
original observation epoch, followed by deterministic row-field ordering.
Record duplicate counts and keys with conflicting rain values. Repeat
deduplication across annual file boundaries, then keep aligned observation
years 2017–2024. Do not fill missing readings with zero. Retain readings over
50 mm per five minutes, but flag them and export the largest readings; 50 mm
is a sensitivity threshold, not an official QC rule.

## Coverage and comparability

Coverage is the unique valid observation count divided by full-calendar
expected five-minute slots (288 per day, with leap years handled explicitly).
Partial station lifetimes therefore have low coverage; coverage is not a
claim about completeness during only active periods.

Annual comparisons use a fixed panel of station IDs present in all eight
years, with >=80% coverage in every year and <=0.005 degree latitude and
longitude spans. Coordinate span is a conservative screening heuristic,
not confirmation that a station never moved. The main 2024 spatial map
colours only stations with >=90% annual coverage; other locations appear as
grey crosses.

Monthly comparisons use >=80%-covered station-year-months within the fixed
annual station pool. Retained stations can vary by year-month. Main figures
show gaps explicitly rather than connecting across them; the mean of each
calendar month gives equal weight to eligible stations within a year and
then equal weight to the available years. The number of qualifying years
is recorded in task1_findings.json, and monthly_panel_membership.csv records
exact station membership. This is not a fully balanced seasonal comparison.

A strict sensitivity comparison requires each station-calendar-month to have
>=80% coverage in all eight years. Only six calendar months remain: May,
June, July, September, October and December. Those comparisons are retained
in strict_balanced_month_year.csv, but cannot establish a full annual cycle.
The diurnal figure uses >=80%-covered station-year-month-hour strata within
the fixed panel; its eligible strata can vary as well.

## Metrics

- Wet frequency: count of positive rainfall intervals / valid observed
  intervals, expressed as percent.
- Wet intensity: observed rainfall sum / wet-interval count, in mm per wet
  five-minute interval. It is not an hourly intensity.
- Rainfall per observed 24h: observed rainfall sum / valid interval count x
  288. This normalizes by measured duration; it is not an observed daily
  total or official annual rainfall, and it does not recover missing storms.
- Daily correlation: Pearson correlation of station-day observed totals,
  using >=95%-covered days and at least 365 paired days. Pairwise available
  periods can differ. Distances use latest coordinates and a haversine
  approximation. Spatial correlation does not establish storm movement.
- Future 30-minute sum: sum readings with aligned times in `(t, t+1800s]`,
  retaining only windows containing exactly six slots. Overlapping windows
  are dependent. This assumes the aligned labels can serve as interval
  endpoints; formal endpoint semantics need confirmation for Task 2.
- Persistence: for adjacent observations exactly five minutes apart,
  compare the observed chance of a wet current interval after a wet versus
  dry previous interval. This is association, not forecast performance.

## Interpretation limits

These data are explicitly not official quality-controlled climate records.
Coverage normalization cannot remove weather-dependent missingness,
sensor faults, undocumented timestamp semantics, station changes, or
uneven spatial sampling. Gauge locations are not area weights; their
rainfall sums are not a national rainfall volume. Do not interpolate
coloured points into a rainfall surface without a validated spatial method.
Eight years support descriptive annual comparisons, not a robust claim
about long-term climate change.

The dashboard uses a geoBoundaries Singapore boundary representing 2016,
derived from URA data and licensed under ODbL 1.0. It is a cartographic
reference, not a current coastline survey. Source metadata is preserved.

## Reproducibility and AI use

Download manifests, annual quality audits, Spark aggregate tables and
environment records are generated during execution. These generated products
were removed at the user's request and are excluded from Git. Source code,
visualization builders, static inputs and the PDF/LaTeX reports are retained.
Codex assisted with methods, code, execution, validation, and figure
interpretation. Weather findings must come from the computed tables; no
preview or synthetic data are used as full-dataset results. Final group
members must review the choices, limitations, and outputs and include a
specific AI-use declaration in the submitted report.
