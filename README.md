# Harmonized Fire Calendar

**More fires, or just a better satellite?** Splice the VIIRS fire record onto the MODIS record without
conversion and Greece appears to burn almost three times as much after 2018. MODIS itself shows a decrease.

NASA Space Apps 2026, challenge "Harmonization of MODIS and VIIRS Hot Spots".

## What it does

A web app that turns MODIS and VIIRS active-fire hotspots into one consistent monthly burning-activity
calendar for a chosen area.

- Pick Greece, California, Europe, USA or a custom latitude/longitude box, and a range of years.
- Compare three series: the naive splice (MODIS, then raw VIIRS), the harmonized series, and actual MODIS.
- Get a plain-language "Fire season check" for fire planners: peak months, last season against normal,
  and the current month against normal.
- See "Where it's unusual" on a 0.25° map: this month so far, the last full season, or the shift in
  season timing, each against the square's own normal, with a toggle to show the uncorrected version.
- Draw a rectangle on the map to check a smaller area.
- Download a one-page "Fire profile" for the selected area (HTML, printable to PDF).
- Browse a year-by-month calendar heatmap, raw per sensor or harmonized, with anomalous months outlined.
- See the validation results next to the data, and download the harmonized monthly series as CSV.

Why it matters now: VIIRS (375 m) detects far more hotspots than MODIS (1 km), so a merged record shows a
false jump when VIIRS starts. MODIS is near the end of its life, so the long record has to continue on
VIIRS. This tool keeps the two comparable.

## Headline result

Hotspots per year in 2019–2025 compared with MODIS in 2001–2017:

| Region | Series | Per year on average | Median year |
|---|---|---|---|
| Greece | Naive splice (raw VIIRS vs earlier MODIS) | 2.94× | 2.43× |
| Greece | Actual MODIS | 0.83× | 0.68× |
| Greece | Harmonized | 0.85× | 0.71× |
| California | Naive splice (raw VIIRS vs earlier MODIS) | 9.00× | 3.53× |
| California | Actual MODIS | 2.04× | 0.85× |
| California | Harmonized | 2.07× | 0.86× |

The naive splice overstates post-2018 activity by a factor of 3.56 in Greece and 4.40 in California.
California's average is pulled up by 2020 and 2021; its median year went down.

The harmonized series matches MODIS closely in these years partly by construction, because the factors
are fitted on the same period. The independent test is the validation below.

## Data

| Source | Product | Period used |
|---|---|---|
| NASA FIRMS | MODIS Collection 6.1 active fires, Terra + Aqua, 1 km | 2000-11 to 2026-11 |
| NASA FIRMS | VIIRS NOAA-20 Collection 2 active fires, 375 m | 2018-04 to 2026-11 |

Downloaded from the [FIRMS archive download](https://firms.modaps.eosdis.nasa.gov/download/) as CSV for
California, Greece, the USA and Europe, on 2026-11-15. After filtering, 1,020,953 hotspots are used for
California and Greece and 10,795,593 for the USA and Europe. Raw files are not in
this repository (too large); the small aggregates in `data/agg/` are.

NASA POWER is listed in our plan as a climate context layer; it is not used in the current version.

## Method

1. **Filter.** Keep vegetation fires only (`type == 0`). Drop MODIS detections with confidence below 30
   and VIIRS detections with low confidence. Near-real-time files (from 2026-08) have no type column, so
   points are dropped at 0.01° locations where the archive has a non-vegetation detection.
2. **Aggregate.** Count hotspots per 0.5° cell, month and sensor.
3. **Convert.** For each 5° tile and calendar month, factor = MODIS hotspots / VIIRS hotspots over
   2018-04 to 2025-12. A tile-month with fewer than 50 MODIS hotspots uses the tile's pooled factor; a
   tile with fewer than 200 uses the region's pooled factor.
4. **Harmonize.** MODIS before 2018-04; VIIRS × factor from 2018-04 on. Actual MODIS is kept for comparison.
5. **Flag anomalies.** A month is anomalous if log(1 + count) exceeds the median + 2 × scaled MAD of the
   same calendar month, and it has at least 10 hotspots.
6. **Fire season.** The shortest run of consecutive months holding at least 75% of a typical year
   (sum of calendar-month medians).

Regions are bounding boxes in `regions.json`. Adding one needs the CSVs in `data/raw/` and one line there.

## Validation

Converted VIIRS is compared with actual MODIS on data the factors never saw. Fit period 2018-04 to
2025-12. Monthly error is the median absolute percentage error over months with at least 10 MODIS hotspots.

Three models were tested: one pooled factor per tile, a factor per tile and calendar month, and a power
law (MODIS = a × VIIRS^b). The calendar-month factor has the lowest leave-one-year-out monthly error
averaged over regions (16.8%, against 20.1% pooled and 19.6% power law) and is the one used.

| Region | Test | Model | Monthly error | Median annual error | Worst annual error | Bias | Correlation |
|---|---|---|---|---|---|---|---|
| California | Leave one year out | Pooled | 17.8% | 11.9% | 19.6% | +2.8% | 0.993 |
| California | Leave one year out | **Calendar-month** | **12.0%** | 3.2% | 9.8% | +2.8% | 0.995 |
| California | Leave one year out | Power law | 14.3% | 4.8% | 20.1% | +5.6% | 0.990 |
| California | Forward | Pooled | 16.5% | 8.1% | 12.0% | −8.6% | 0.985 |
| California | Forward | **Calendar-month** | **11.1%** | 0.2% | 3.7% | +1.0% | 0.991 |
| California | Forward | Power law | 11.8% | 5.5% | 10.6% | +3.3% | 0.980 |
| Greece | Leave one year out | Pooled | 22.3% | 8.1% | 25.6% | −0.1% | 0.966 |
| Greece | Leave one year out | **Calendar-month** | **21.5%** | 9.7% | 36.7% | +4.7% | 0.957 |
| Greece | Leave one year out | Power law | 24.8% | 18.0% | 114.5% | +3.8% | 0.784 |
| Greece | Forward | Pooled | 19.6% | 20.6% | 20.8% | −17.0% | 0.974 |
| Greece | Forward | **Calendar-month** | **19.1%** | 16.6% | 18.7% | −12.6% | 0.984 |
| Greece | Forward | Power law | 25.5% | 28.9% | 46.9% | −34.9% | 0.981 |

The forward test fits on 2018-04 to 2022-12 and predicts 2023-01 to 2025-12. At the level of single 5°
tiles the median monthly error of the model used is 19.8% in California and 31.0% in Greece.

### USA and Europe

The same method and tests were run on the whole USA and on Europe (box 35–71°N, 11°W–40°E), built as a
separate dataset (`data/agg_wide/`) so the country-clipped California and Greece numbers stay unchanged.

| Region | Test | Model | Monthly error | Median annual error | Worst annual error | Bias | Correlation |
|---|---|---|---|---|---|---|---|
| USA | Leave one year out | Pooled | 8.4% | 2.9% | 9.6% | +1.9% | 0.984 |
| USA | Leave one year out | **Calendar-month** | **7.0%** | 2.6% | 12.5% | +2.7% | 0.988 |
| USA | Leave one year out | Power law | 10.2% | 4.3% | 10.8% | +2.1% | 0.981 |
| USA | Forward | Pooled | 9.9% | 7.8% | 10.8% | +6.9% | 0.983 |
| USA | Forward | **Calendar-month** | **9.7%** | 12.6% | 14.1% | +9.9% | 0.978 |
| USA | Forward | Power law | 9.5% | 8.4% | 11.4% | +7.4% | 0.972 |
| Europe | Leave one year out | Pooled | 11.5% | 4.3% | 14.4% | −0.1% | 0.982 |
| Europe | Leave one year out | **Calendar-month** | **7.1%** | 3.2% | 11.0% | +0.9% | 0.992 |
| Europe | Leave one year out | Power law | 8.4% | 4.5% | 13.6% | −2.6% | 0.983 |
| Europe | Forward | Pooled | 10.8% | 4.5% | 15.2% | +7.3% | 0.990 |
| Europe | Forward | **Calendar-month** | **8.8%** | 4.1% | 11.1% | +5.5% | 0.990 |
| Europe | Forward | Power law | 9.4% | 8.2% | 13.6% | +4.0% | 0.982 |

Tile error spread (leave one year out, calendar-month factor): USA 49 of 81 tiles had
enough data to test, median 21.6%, range 10.0–58.0%, 8 above 30%. Europe 49 of
70 tiles, median 23.5%, range 11.6–50.7%, 8 above 30%. On the map, squares in tiles
above 30% or without enough data are left blank, and their number is stated under the map.

### MODIS drift check, 2026

2026 is left out of the factor fit. For January to July 2026, actual MODIS is below the harmonized
prediction in both regions:

| Region | Actual MODIS | Harmonized prediction | MODIS below prediction | Largest annual error in held-out years 2018–2025 |
|---|---|---|---|---|
| California | 1,622 | 2,152 | 24.6% | 9.8% |
| Greece | 308 | 428 | 28.1% | 36.7% |

In California this is well outside the normal error. In Greece it is within the normal year-to-year
spread and rests on few hotspots, so it is only suggestive. The cause is unverified.

### ML forecast test (negative result)

We tested whether gradient boosting (scikit-learn `HistGradientBoostingRegressor`) predicts harmonized
monthly hotspots 1 to 3 months ahead better than the seasonal baseline (median of the same calendar month
in earlier years). Features: calendar month, the last three months' anomalies against the same-month
median, and the same month one year earlier. Test: rolling forward, trained only on data up to month t,
predicting t+1 to t+3, for target months 2010-12 to 2026-10. Settings were fixed in advance, not tuned.

| Region | Months ahead | ML monthly error | Seasonal baseline monthly error |
|---|---|---|---|
| California | 1 | 41.2% | 39.7% |
| California | 2 | 45.7% | 38.5% |
| California | 3 | 45.6% | 39.7% |
| Greece | 1 | 60.9% | 52.9% |
| Greece | 2 | 55.3% | 52.9% |
| Greece | 3 | 54.3% | 52.9% |

The baseline was as good or better at every horizon in both regions, so the app shows no ML forecast.
The comparison is in the "Method & validation" tab. Run it with `python forecast.py`
(needs `requirements-pipeline.txt`).

## Limits

- Hotspots are pixels, not fire events. One large fire produces many pixels, so a few extreme months
  dominate totals.
- Factors are 5° tile averages validated at region and month scale. Small boxes have larger errors.
- The series is "Terra + Aqua equivalent" as observed in 2018–2025. Cloud cover, overpass times, orbit
  drift and the larger night-time share of VIIRS detections are absorbed into the factor, not modelled.
- Harmonized 2026 values show what MODIS would have reported at its 2018–2025 behaviour.
- Near-real-time data have no fire-type classification. The static-source mask removes most but not all
  non-vegetation detections, and on archive data it also removes 0.4–3.9% of real vegetation fires.
- Regions mixing forest and cropland burning may need finer tiles. Tiles that tested poorly are left
  blank on the map; some contain gas flaring that the fire-type filter does not fully remove.
- Conflict-related fires in Ukraine since 2022 appear as anomalies; they are real detections, not wildfire
  or farm burning.
- A preset region is its download area (country outline for California and Greece, a box for USA and
  Europe); a custom box counts everything inside it.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The app reads only `data/agg/`. To rebuild the aggregates, put FIRMS CSVs (archive and NRT) in `data/raw/`
and run:

```bash
.venv/bin/python pipeline.py all
```

The USA/Europe dataset is built from its own downloads in `data/raw_big/`:

```bash
FIRE_RAW_DIRS=data/raw_big FIRE_AGG_DIR=data/agg_wide FIRE_REGIONS=regions_wide.json .venv/bin/python pipeline.py all
```

| File | Purpose |
|---|---|
| `pipeline.py` | Load, filter, aggregate, fit factors, validate, write `data/agg/` |
| `app.py` | Streamlit app |
| `forecast.py` | ML forecast test against the seasonal baseline; writes `data/agg/forecast.json` |
| `regions.json` | Region bounding boxes |
| `data/geo/` | Region outlines from Natural Earth (public domain), built by `scripts/make_boundaries.py` |
| `data/agg/` | Aggregates, factors, `validation.json`, `headline.json`, `meta.json` |
| `AI_USAGE.md` | Which AI tool did what |

## AI usage

Code, method review and this README were written with Claude Code (Anthropic), using Claude Opus 5.5 and
Claude Fable 5.2. The team wrote the specification, chose the method and fixes, and reviewed results at
each step. Details per step are in [AI_USAGE.md](AI_USAGE.md). Every number shown comes from the data.
