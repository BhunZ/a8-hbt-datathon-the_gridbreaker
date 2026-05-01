# V13 Step 4 — Trend Features Audit

Produced by `src/v13_trends.py`. Train window: **2019-09-01 → 2022-12-31** (40 months, post-cliff).

## 1. What was built

| File | Shape | Purpose |
|---|---|---|
| `data/processed/v13_trends.json` | — | slope, intercept, seasonal means per metric (frozen at train time) |
| `data/processed/v13_trends.parquet` | 2314 × 4 | daily-grain projections for the full calendar window |

Daily-grain columns:

- `date`
- `aov_trend_proj` — projected AOV at the date
- `unit_price_trend_proj` — projected mean unit price
- `n_active_skus_trend_proj` — projected count of active SKUs

## 2. Method — de-seasonalized linear trend

The first attempt was raw OLS of monthly metric vs `months_since_2019`. R² was 0.06 because December AOV dips (~$20k vs annual mean ~$31k) dragged the regression line.

The fix: subtract the month-of-year mean from each observation, then fit OLS on the residual. The R² jumps to 0.34-0.50 because the trend is no longer competing with seasonality.

**Projection formula at any date d:**
```
value_proj(d) = seasonal_mean[month(d)]
              + slope * months_since_2019(d)
              + intercept_resid
```

`seasonal_mean[m]` is the train-window average of the metric across all years for month m. `slope` and `intercept_resid` come from OLS on the de-seasonalized residual. All three are frozen scalars/lookups in `v13_trends.json`.

## 3. Fit quality

| Metric | Slope (per month) | Slope (per year) | R² (residual) | Raw OLS slope (per year) |
|---|---:|---:|---:|---:|
| `aov_trend` | +$115.07 | **+$1,381** | 0.340 | +$1,091 |
| `unit_price_trend` | +$26.87 | **+$322** | 0.383 | +$266 |
| `n_active_skus_trend` | -1.27 | **-15.2** | 0.496 | -12.7 |

Comparison to the timeline audit (which used annual means):

| Metric | Trends fit (this step) | Timeline audit (annual deltas) |
|---|---|---|
| AOV per year | +$1,381 | ~+$1,654 |
| unit_price per year | +$322 | ~+$405 |
| n_active_skus per year | -15.2 | ~-16 |

The fits are slightly conservative on AOV and unit_price (annual means tilt up because of the spike-prone late-year months), but very close on SKU count. Conservative is fine here — the model uses these as features, not as direct predictions, and pairs them with `aov_prior_by_month` from Step 2 which carries the absolute level.

## 4. Projections at key dates

| Date | AOV | unit_price | n_active_skus |
|---|---:|---:|---:|
| 2019-09-01 (regime start) | 28,196 | 5,918 | 459.7 |
| 2022-12-01 (last train month) | 22,784 | 4,556 | 382.3 |
| 2023-07-01 (mid-test horizon) | 32,932 | 7,075 | 395.2 |
| 2024-01-01 (early-2024 test) | 36,952 | 7,717 | 314.9 |
| 2024-07-01 (test horizon end) | 34,316 | 7,399 | 380.0 |

The projections track seasonality (Dec dip, Jan/Feb high) AND drift (each year's January is higher than the last). For 2024-07-01 the trend says AOV ≈ $34.3k — about 14 % above the train-window average ($30.8k) — exactly the kind of regime drift v10c failed to capture.

## 5. Honesty rule restated

- `slope`, `intercept_resid`, `seasonal_means` are all computed from train-window data (2019-09-01 → 2022-12-31) and frozen.
- For any test date d, the projection is a deterministic function of d.month and `months_since_2019(d)` — no inputs from the test horizon, no projection of dynamic state.
- Train-time and test-time produce identical values for the same date.

This satisfies V13 §0 — every value is a frozen scalar/lookup keyed by date components.

## 6. How the model uses these features

The model gets two complementary scalars per metric:

1. **`aov_prior_by_month`** (from Step 2): the train-window average AOV for that month (12-row lookup, no trend).
2. **`aov_trend_proj`** (from Step 4): the level-corrected projection that accounts for drift.

The model can compose them. A simple consumption: predict revenue using `aov_prior * (1 + aov_trend_proj_drift)` — the tree learns this implicitly via splits. The trend feature alone tells the model "we're far from the train-window mean here," which is precisely what test rows in late 2024 need.

## 7. Caveats

1. **R² is moderate (0.34-0.50).** Even after de-seasonalization, monthly noise still dominates the trend within a 40-month window. The slopes are directionally correct but should not be over-relied on as point predictions.
2. **Linear extrapolation assumption.** The trend is fit on 2019-09 → 2022-12 and extrapolated through 2024-12. If the brand's trajectory bends (e.g., a new business cycle), the extrapolation will diverge. Mirror-block validation against 2022 data is the only real check.
3. **Seasonal means use unequal year counts.** September-December months have 4 observations each (2019-2022); January-August months have only 3 (2020-2022). The seasonal_mean for January is therefore noisier than for September.

## 8. Next step

Step 5 — feature assembly: join calendar + priors + lags + trends into a single `daily_features_v13.parquet`. Verify zero columns derived from web_traffic, reviews, returns, signups, stockouts, inventory, orders, items, payments. That's the lockable feature table for the model.
