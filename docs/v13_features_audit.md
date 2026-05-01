# V13 Step 5 — Feature Table Audit (final)

Produced by `src/v13_assemble_features.py`. Joins the four V13 building blocks on date and adds targets.

## 1. What was built

`data/processed/daily_features_v13.parquet` — **2314 × 93** keyed by date for the full calendar window (2018-09-01 → 2024-12-31).

| Block | Source script | Cols added |
|---|---|---:|
| Calendar & event book | `v13_calendar_event_book.py` | 72 (date is the join key) |
| Frozen priors | `v13_priors.py` | 5 |
| Long-horizon lags | `v13_lags.py` | 8 |
| Trend features | `v13_trends.py` | 3 |
| Targets (from `sales.parquet`) | this step | 2 (Revenue, COGS) |
| Split flags | this step | 2 (is_train, is_test) |
| Join key | — | 1 (date) |

## 2. Honesty contract — verified

- **Forbidden-source check passed.** No column derived from web_traffic, reviews, returns, signups, stockouts, inventory, orders, order_items, or payments appears in the feature table. (One indirect path remains: `aov_prior_by_month` and `aov_trend_proj` were *fitted* using order-derived `n_orders`, but only as static train-time aggregates — the resulting frozen scalar/lookup values are deterministic at inference.)
- **Targets only on train rows.** 1218 of 1218 train rows have non-null Revenue and COGS. 0 of 548 test rows have populated targets — exactly as expected.
- **No median-fill / interpolation / projection of dynamic state.** Every column either comes from date arithmetic (CAL), a frozen train-window aggregate (PRIOR / TREND), or an explicit train-window lag (LAG-Y).

## 3. Train/test row counts

| Window | Dates | Rows | Target Revenue notna |
|---|---|---:|---:|
| Pre-train buffer | 2018-09-01 → 2019-08-31 | 365 | 365 (in sales.csv but not used for training) |
| Train | 2019-09-01 → 2022-12-31 | 1218 | 1218 |
| Test | 2023-01-01 → 2024-07-01 | 548 | 0 |
| Post-test buffer | 2024-07-02 → 2024-12-31 | 183 | 0 |

## 4. Null analysis

Only 8 of 89 feature columns have any nulls — all are the long-horizon lag features (intentional, by V13 §0):

| Column | null(train) | null(test) | Why |
|---|---:|---:|---|
| `rev_yoy_lag_728` | 728 | 0 | early train rows can't lag back 728 days into train window |
| `cogs_yoy_lag_728` | 728 | 0 | same |
| `rev_yoy_lag_364` | 364 | 184 | early train rows pre-cliff; late test rows (2024+) lag back into test |
| `cogs_yoy_lag_364` | 364 | 184 | same |
| `rev_same_dow_prev_year` | 182 | 359 | <26 of 52 same-dow lookups in train window |
| `cogs_same_dow_prev_year` | 182 | 359 | same |
| `rev_same_dom_prev_year` | 182 | 336 | <6 of 12 same-dom lookups in train window |
| `cogs_same_dom_prev_year` | 182 | 336 | same |

LightGBM handles NaN natively — these are honest "no value exists" markers, not data quality issues.

## 5. Feature inventory by tag

### CAL (Calendar / event book) — 72 columns

```
year, month, day, dow, woy, day_of_year, quarter, is_weekend,
is_month_start, is_month_end, is_quarter_end, is_year_end,
is_double_digit_day, is_singles_day, is_twelve_twelve, is_nine_nine, is_ten_ten,
is_intl_women_day, is_womens_day_vn, is_teachers_day_vn, is_independence_day_vn,
is_reunification_day, is_intl_labour_day, is_new_year_day,
is_black_friday, is_cyber_monday, is_boxing_day, is_christmas, is_valentines,
is_tet, tet_day_index_raw, is_pre_tet_window, is_tet_recovery_window, tet_day_index,
is_payday_15, is_payday_25, is_payday_30, is_any_payday,
days_to_next_payday, days_since_last_payday,
days_to_tet, days_since_tet,
days_to_singles_day, days_since_singles_day,
days_to_twelve_twelve, days_since_twelve_twelve,
days_to_nine_nine, days_since_nine_nine,
days_to_ten_ten, days_since_ten_ten,
days_to_black_friday, days_since_black_friday,
days_to_cyber_monday, days_since_cyber_monday,
days_to_intl_women_day, days_since_intl_women_day,
days_to_womens_day_vn, days_since_womens_day_vn,
days_to_christmas, days_since_christmas,
days_to_boxing_day, days_since_boxing_day,
campaign_window_id, day_index_within_campaign,
sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy,
years_since_2019, months_since_2019
```

### PRIOR (frozen aggregates) — 5 columns

```
prior_rev_by_month_dow, prior_cogs_by_month_dow,
aov_prior_by_month,
prior_rev_event_uplift, prior_cogs_event_uplift
```

### LAG-Y (long-horizon lags) — 8 columns

```
rev_yoy_lag_364, rev_yoy_lag_728,
cogs_yoy_lag_364, cogs_yoy_lag_728,
rev_same_dow_prev_year, cogs_same_dow_prev_year,
rev_same_dom_prev_year, cogs_same_dom_prev_year
```

### TREND — 3 columns

```
aov_trend_proj, unit_price_trend_proj, n_active_skus_trend_proj
```

### Targets / metadata — 4 columns

```
date, Revenue, COGS, is_train, is_test
```

## 6. Comparison to V12

| Submission | # features | Sources | Outcome |
|---|---:|---|---|
| V10c (best so far) | 23 | calendar + lag-364 only | Kaggle 774,898 |
| V12d (failed) | 58 | calendar + 35 projected aux features (web_traffic, reviews, etc.) | Kaggle 871,299 |
| **V13 (this step)** | **89** (76 unique non-calendar = CAL+PRIOR+LAG+TREND) | calendar + frozen lookups + long-horizon lags + trends; **zero projected aux** | mirror-block validation pending |

V13 has more features than V12 but **none of them violate train/serve consistency** — every value is either a deterministic function of the date or a frozen train-window scalar. The V12 failure mode (median-projected aux features collapsing to mean predictions on event days) cannot occur here.

## 7. Inference-time check

For any test date d, the assembly procedure is:

```python
features_d = (calendar_lookup[d]                  # CAL block from a date->row map
              | priors_lookup[d]                  # PRIOR block from same map
              | lags_lookup[d]                    # LAG-Y block, NaN where unavailable
              | trends_lookup[d])                 # TREND block, deterministic projection
```

Train-time and test-time use the same `daily_features_v13.parquet` — the only difference is whether the target is populated. There is no "if train: do X else: do Y" branch in feature generation. That branch is precisely what V12 had.

## 8. What's NOT in this table

Per the V13 blueprint §10:

- No web_traffic columns (sessions, unique_visitors, page_views, bounce_rate, traffic-source shares)
- No reviews columns (n_reviews, rating_mean, n_low_ratings)
- No returns columns (n_returns, return_qty_sum, refund_sum)
- No signups columns (n_signups)
- No stockouts/inventory columns
- No orders/items/payments aggregates (n_orders, pct_mobile, items_per_order, mean_installments, etc.)
- No projected dynamic state (no median profile, no linear extrapolation of dynamic features)

These are deliberately excluded. The V12 audit established that adding them costs RMSE rather than helping.

## 9. Next step

Step 6 — validation harness. Implement:
- 4 expanding-origin folds at 90-day horizon inside 2020-2022 (iteration speed)
- 1 locked 548-day mirror block (2022-01-01 → 2022-12-31, train ≤ 2021-12-31)
- Score V10c on the new mirror block (the floor we have to beat)
- Score V13 baseline LightGBM (log1p+RMSE, debias-corrected expm1)

Only when V13 beats V10c on the mirror block does it earn a Kaggle submission.
