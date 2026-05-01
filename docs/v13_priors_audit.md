# V13 Step 2 — Frozen Priors Audit

Produced by `src/v13_priors.py`. Train window: **2019-09-01 → 2022-12-31** (post-cliff regime, 1218 days).

## 1. What was built

| File | Shape | Purpose |
|---|---|---|
| `data/processed/v13_priors.parquet` | 2314 × 6 | daily-grain table for date-join into feature assembly |
| `data/processed/v13_priors_meta.json` | — | underlying lookup tables for audit |

Daily-grain columns:

- `date`
- `prior_rev_by_month_dow` — mean Revenue across train-window observations sharing (month, dow)
- `prior_cogs_by_month_dow` — same for COGS
- `aov_prior_by_month` — mean of (Revenue / n_orders) across train-window days in each month
- `prior_rev_event_uplift` — multiplier ≥ 1.0 on event days (max across overlapping flags); 1.0 otherwise
- `prior_cogs_event_uplift` — same for COGS

Every column is deterministic from the date alone — passes the V13 §0 contract.

## 2. The (month, dow) revenue prior

Mean daily Revenue (₫M) per (month, dow). dow=0 is Monday.

| month | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.89 | 1.71 | 1.65 | 1.28 | 1.55 | 1.71 | 1.79 |
| 2 | 2.86 | 2.49 | 2.43 | 2.19 | 2.10 | 2.35 | 2.68 |
| 3 | 3.82 | 4.21 | 4.83 | 4.39 | 3.42 | 3.17 | 3.25 |
| 4 | 3.67 | 4.02 | 5.46 | 5.30 | 5.02 | 4.62 | 4.04 |
| 5 | 4.23 | 3.95 | 4.15 | 5.08 | 4.69 | 4.32 | 4.82 |
| 6 | 4.03 | 4.50 | 5.16 | 4.67 | 3.69 | 3.29 | 3.57 |
| 7 | 2.64 | 2.62 | 3.12 | 3.89 | 3.71 | 2.90 | 2.72 |
| 8 | 3.88 | 3.05 | 3.24 | 3.21 | 3.06 | 2.79 | 3.53 |
| 9 | 3.09 | 2.95 | 2.57 | 2.89 | 2.22 | 2.33 | 2.75 |
| 10 | 2.37 | 2.49 | 2.22 | 2.18 | 2.21 | 2.30 | 2.10 |
| 11 | 1.74 | 1.90 | 1.46 | 1.63 | 1.59 | 1.60 | 1.93 |
| 12 | 1.82 | 1.68 | 1.70 | 1.56 | 1.42 | 1.37 | 1.45 |

**Headline finding:** the brand's seasonality is INVERTED relative to a typical western retailer. **March–June is peak**, **November–December is trough**. This explains why my first-pass event uplift (vs global baseline) showed Black Friday and Christmas as "depressed" — they fall in the brand's worst months.

## 3. Monthly AOV prior

| month | aov_prior_by_month (₫) | n_obs |
|---:|---:|---:|
| 1 | 33,419 | 93 |
| 2 | 34,365 | 85 |
| 3 | 31,355 | 93 |
| 4 | 30,848 | 90 |
| 5 | 32,954 | 93 |
| 6 | 32,621 | 90 |
| 7 | 30,138 | 93 |
| 8 | 29,598 | 93 |
| 9 | 30,762 | 120 |
| 10 | 34,208 | 124 |
| 11 | 30,224 | 120 |
| 12 | 20,664 | 124 |

December AOV (~20.6k) is ~33% below the year average — consistent with end-of-year clearance dragging basket size down.

## 4. Event uplifts — season-matched

**Definition:** for each event day, ratio = actual_rev / prior_rev_by_month_dow. The reported uplift is the **mean of those ratios** across train-window event days (n=3 to 9). This isolates the event signal from the calendar effect.

| Event | uplift_rev | uplift_cogs | n |
|---|---:|---:|---:|
| `is_reunification_day` (Apr 30) | **1.415** | 1.324 | 3 |
| `is_intl_labour_day` (May 1) | **1.342** | 1.344 | 3 |
| `is_new_year_day` (Jan 1) | **1.284** | 1.560 | 3 |
| `is_womens_day_vn` (Oct 20) | 1.181 | 1.175 | 4 |
| `is_tet` (3 days, lunar) | 1.129 | 1.121 | 9 |
| `is_singles_day` (Nov 11) | 1.081 | 0.997 | 4 |
| `is_cyber_monday` | 1.023 | 1.171 | 4 |
| `is_black_friday` | 0.986 | 1.133 | 4 |
| `is_valentines` (Feb 14) | 0.977 | 0.997 | 3 |
| `is_teachers_day_vn` (Nov 20) | 0.891 | 0.920 | 4 |
| `is_nine_nine` (Sep 9) | 0.886 | 0.875 | 4 |
| `is_boxing_day` (Dec 26) | 0.871 | 0.862 | 4 |
| `is_ten_ten` (Oct 10) | 0.860 | 0.859 | 4 |
| `is_twelve_twelve` (Dec 12) | 0.846 | 0.858 | 4 |
| `is_independence_day_vn` (Sep 2) | 0.816 | 0.932 | 4 |
| `is_christmas` (Dec 25) | 0.803 | 0.803 | 4 |
| `is_intl_women_day` (Mar 8) | 0.787 | 0.716 | 3 |

### Reading

- **Three real uplift events (>1.25x):** Reunification + Labour Day cluster (late April / early May), and New Year's Day. These are this brand's "Black Friday equivalents".
- **Mild positive uplift (1.05–1.20x):** Women's Day VN, Tet, Singles Day. These are still revenue lifters but smaller.
- **Flat (~1.0x):** Cyber Monday, Black Friday, Valentine's. The brand does not get a meaningful spike from these.
- **Negative ("cooling") events (<0.9x):** the Western retail anchors (Christmas, Boxing Day) and the double-digit sale clusters other than 11/11. **Important:** these are STILL flagged as "events" in the calendar, but the multiplier honestly reports that they suppress vs. a non-event day in the same month/dow. The model is free to use this signal either way.

### Sanity checks

- Reunification + Labour Day uplift (1.41x, 1.34x) lines up with the (month, dow) grid showing April-May as the high-revenue season — these are the days WITHIN that high season that further outperform.
- Tet positive uplift (1.13x) is small but real, contradicting my naive global-baseline result that showed Tet as 0.84x. The shift is purely from correcting the seasonality bias.
- All events have n ≥ 3 train-window observations. Tet has n=9 (3 days × 3 years that fit cleanly inside the train window: 2020, 2021, 2022).

## 5. Inference logic — restated

For any test date d:

```
priors[d] = {
    prior_rev_by_month_dow:   PRIOR_REV[d.month, d.dow],     # 84-cell lookup
    prior_cogs_by_month_dow:  PRIOR_COGS[d.month, d.dow],
    aov_prior_by_month:       AOV_PRIOR[d.month],            # 12-cell lookup
    prior_rev_event_uplift:   pick_most_informative( UPLIFT_REV[flag]  for flag in EVENT_FLAGS if flag(d) ),
    prior_cogs_event_uplift:  pick_most_informative( UPLIFT_COGS[flag] for flag in EVENT_FLAGS if flag(d) ),
}
# pick_most_informative = the multiplier with largest |log(mult)| across active flags;
# defaults to 1.0 if no event flag is set on the date.  Allows both boost (>1)
# and suppression (<1) values to pass through -- e.g. Christmas = 0.803.
```

**Spot-check (test horizon dates):**

| Date | Event | rev_uplift | cogs_uplift |
|---|---|---:|---:|
| 2024-04-30 | Reunification Day | 1.415 | 1.324 |
| 2024-05-01 | Labour Day | 1.342 | 1.344 |
| 2024-02-10 | Tet day-1 | 1.129 | 1.121 |
| 2024-11-11 | Singles Day | 1.081 | 0.997 |
| 2024-11-25 | Cyber Monday | 1.023 | 1.171 |
| 2024-12-25 | Christmas | 0.803 | 0.803 |
| 2024-12-12 | 12-12 | 0.846 | 0.858 |
| 2024-03-08 | Intl Women's Day | 0.787 | 0.716 |
| 2024-03-15 | (no event) | 1.000 | 1.000 |

**Coverage:** 105 of 1949 train+test days carry a non-neutral multiplier (~5.4 %).

Already materialized to disk in `v13_priors.parquet` for the entire calendar window (2018-09-01 → 2024-12-31), so inference is just:

```python
features = calendar.merge(priors, on='date')
```

## 6. What's NOT in this step

- Long-horizon lag features (rev_same_dow_prev_year, rev_yoy_lag_364) — that's Step 3.
- Trend features fitted on the post-cliff window (aov_trend_proj, etc.) — that's Step 4.
- Final feature assembly joining calendar + priors + lags + trends — that's Step 5.

## 7. Anything to revise in the V13 blueprint?

The blueprint's §2.7 specified `aov_prior_by_month` as a "12-row lookup of the train-window monthly mean of revenue/orders, joined on month." That's exactly what was built.

The blueprint's §2.10 mentioned `prior_rev_event_uplift` only briefly in the §3 feature roster. The implementation uses the **season-matched** definition (ratio to month-dow prior), which is the only correct definition. Earlier draft note in §5.3 says "uplift[event] = mean( actual / stage1_pred ) for dates where event flag = 1" — that's a *prediction-time* recalibration multiplier (Layer 3 of peak-recovery), conceptually similar but distinct from the *feature* `prior_rev_event_uplift`. Both are valid; both are now defined.

No blueprint changes needed.
