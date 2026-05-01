# V13 Step 3 — Long-Horizon Lag Features Audit

Produced by `src/v13_lags.py`. Lookup window: **2019-09-01 → 2022-12-31** (post-cliff regime). Any lookup falling outside that window returns NaN.

## 1. What was built

`data/processed/v13_lags.parquet` — 2314 × 9 (date + 8 lag columns), keyed by date for the full calendar window 2018-09-01 → 2024-12-31.

The 8 columns:

| Column | Definition |
|---|---|
| `rev_yoy_lag_364` | `Revenue[d - 364]` (single point lag, 52 weeks) |
| `rev_yoy_lag_728` | `Revenue[d - 728]` (104 weeks) |
| `cogs_yoy_lag_364` | `COGS[d - 364]` |
| `cogs_yoy_lag_728` | `COGS[d - 728]` |
| `rev_same_dow_prev_year` | mean of `Revenue[d - 7k]` for k=1..52 (≥26 valid lookups required) |
| `cogs_same_dow_prev_year` | analogous for COGS |
| `rev_same_dom_prev_year` | mean of `Revenue[d - DateOffset(months=k)]` for k=1..12 (≥6 valid lookups required) |
| `cogs_same_dom_prev_year` | analogous for COGS |

The thresholds (≥26 of 52 same-dow, ≥6 of 12 same-dom) are intentional — they refuse to compute a feature value when more than half of the lookback window would be NaN. This trades some test-horizon coverage for reliable feature semantics.

## 2. Coverage table

| Column | Train coverage | Test coverage |
|---|---:|---:|
| `rev_yoy_lag_364` | 70.1 % | 66.4 % |
| `rev_yoy_lag_728` | 40.2 % | **100.0 %** |
| `cogs_yoy_lag_364` | 70.1 % | 66.4 % |
| `cogs_yoy_lag_728` | 40.2 % | **100.0 %** |
| `rev_same_dow_prev_year` | 85.1 % | 34.5 % |
| `cogs_same_dow_prev_year` | 85.1 % | 34.5 % |
| `rev_same_dom_prev_year` | 85.1 % | 38.7 % |
| `cogs_same_dom_prev_year` | 85.1 % | 38.7 % |

Reading:
- **`lag_728` is the workhorse for test** — covers the entire test horizon (every test date 2023-01-01..2024-07-01 maps to a train date 2021-01-03..2022-07-03). 100 % coverage.
- **`lag_364` covers 2023 fully but NaN for 2024** — for 2024 test dates, d-364 falls inside the test horizon itself (which we don't have).
- **Same-dow / same-dom averages cover ~34-38 % of test** — mostly the early test horizon (2023-Q1..Q2) where enough same-dow lookups still fall inside train. Beyond that, too many lookups overlap with test → NaN by threshold.
- **For training, lag_364 has 70 % coverage** because early train rows (2019-09 → 2020-08) lag back into the pre-cliff regime, which is intentionally excluded.

LightGBM handles NaN natively — the model learns a "missing" branch per split. So partial coverage is acceptable; the lag features carry information where available, and other features (priors, calendar, trend) carry the rest.

## 3. Spot-checks

| Date | lag-364 (computed) | sales[d-364] (truth) | Status |
|---|---:|---:|---|
| 2023-03-15 | 4,519,934.96 | 4,519,934.96 (2022-03-16) | OK |
| 2022-12-31 | 2,932,155.47 | 2,932,155.47 (2022-01-01) | OK |
| 2024-03-15 | NaN | (target 2023-03-17, in TEST) | NaN(expected) |
| 2024-07-01 | NaN | (target 2023-07-03, in TEST) | NaN(expected) |
| 2024-01-01 | NaN | (target 2023-01-02, in TEST) | NaN(expected) |

| Date | rev_same_dow | rev_same_dom | rev_lag_364 | rev_lag_728 |
|---|---:|---:|---:|---:|
| 2023-03-01 | 3,749,574 | 4,856,602 | 5,221,766 | 1,202,803 |
| 2023-09-01 | NaN | NaN | 1,902,818 | 1,070,973 |
| 2024-01-01 | NaN | NaN | NaN | 545,564 |
| 2024-07-01 | NaN | NaN | NaN | 1,208,546 |

Sample reads sensibly:
- 2023-03-01 same-dow average (~$3.75 M) is in the same neighborhood as the (month=3, dow=Wed) prior ($4.83 M) — within the same band, slightly lower.
- 2023-03-01 lag-364 ($5.22 M) and lag-728 ($1.20 M) span the train regime: 2022-03-02 was a high-revenue Wed, 2021-03-03 was a low-revenue Wed — the model gets both as separate features.
- 2024-07-01 has only lag-728 = $1.21 M (looking back to 2022-07-04). All other lag features NaN.

## 4. Honesty rule restated

For every row, every lag feature is computed by an explicit date arithmetic on the train-window sales table. There is no projection, no fill-in, no interpolation. NaN means "no honest value exists" — the model decides what to do.

This satisfies V13 §0:
- Lookup window restricted to closed train history (2019-09-01..2022-12-31).
- Lookup target derived purely from the date being predicted (d - 364, d - 728, d - 7k, d - DateOffset(months=k)).
- Train-time and test-time produce identical values for the same date.

## 5. Caveats

1. **Pre-cliff data is intentionally excluded from the lookup window.** Early train rows (2019-09 to 2020-08) suffer reduced coverage as a result. This is by design — including pre-cliff data would re-introduce the exact regime mixing we're trying to avoid.

2. **Same-dow features have low test coverage (34 %).** For the ~360 test dates beyond 2023-Q2, the same-dow lookback overlaps with the test horizon itself. The MIN_DOW_OBS=26 threshold makes those NaN. Lowering the threshold would give partial averages but with varying reliability across test dates.

3. **Lag-728 is the only feature with 100 % test coverage.** It's the most stable temporal feature for the model — every test date can rely on it.

## 6. Next step

Step 4 — trend features (`years_since_2019`, `aov_trend_proj`, `unit_price_trend_proj`, `n_active_skus_trend_proj`) — fits linear trends on the post-cliff window, written to disk so inference uses frozen slope+intercept.
