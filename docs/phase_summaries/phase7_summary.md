# Phase 7 — Forecasting Model (Part 3): Summary

**Objective:** predict daily `Revenue` and `COGS` for 2023-01-01 → 2024-07-01 (548 days).
**Output:** `outputs/submission.csv` (Date, Revenue, COGS) — 548 rows, aligned to `sample_submission.csv`.
**Training horizon:** 2013-01-01 → 2022-12-31 (3,652 days, excluding partial 2012 and future window).

## 1. Validation Fold Results (549-day horizon-matched holdout)

Holdout: 2021-07-01 → 2022-12-31 (549 days), using only pre-2021-07-01 history for training — same structural shape as the real test problem.

| Target   | Model                | MAE       | RMSE      | R²     |
|----------|----------------------|----------:|----------:|-------:|
| Revenue  | baseline (seasonal) | 1,407,166 | 1,734,038 | -0.245 |
| Revenue  | LightGBM (full)     |   626,626 |   868,056 |  0.688 |
| Revenue  | CatBoost (full)     |   557,202 |   777,718 |  0.750 |
| Revenue  | CatBoost (recent)   |   685,661 |   974,776 |  0.607 |
| **Revenue**  | **Blend (0.10 LGBM + 0.90 CB-full)** | **560,885** | **782,046** | **0.747** |
| COGS     | baseline (seasonal) | 1,132,888 | 1,407,455 | -0.056 |
| COGS     | LightGBM (full)     |   549,389 |   750,388 |  0.700 |
| COGS     | CatBoost (full)     |   491,109 |   686,173 |  0.749 |
| COGS     | CatBoost (recent)   |   637,038 |   896,624 |  0.572 |
| **COGS**     | **Blend (0.10 LGBM + 0.90 CB-full)** | **493,933** | **688,493** | **0.747** |

**Reduction vs baseline:** Revenue RMSE -55%, COGS RMSE -51%. The baseline's negative R² reflects its systematic mis-level (it uses the competition's geo-mean YoY growth that is *negative* because 2012-2022 overall is flat/declining).

## 2. Design Decisions

1. **Target:** `log1p(Revenue)`, `log1p(COGS)` — stabilises variance, avoids negative predictions after `expm1`.
2. **Feature set (leak-free):**
    - Calendar: year, month, day, dow, doy, woy, `is_weekend/month_*/quarter_*`.
    - Trend: `t` (day index), `t²/1e6`.
    - Fourier: annual (K=5), weekly (K=3), monthly (K=2).
    - Static lags (fully observed on test window): **730, 1095, 1460** days for both targets.
    - Recursive lag: **365** days — filled day-by-day in 2024 from the model's own 2023 predictions.
    - Rolling same-(month,day) seasonal means: expanding and 3-year rolling.
    - Trailing 365-d level anchors: `level_548`, `level_730`, `level_913` (mean Revenue over 1yr windows ending 1.5-, 2-, 2.5-yr ago — always observed for test dates).
    - `rev_yoy_ratio`, `cogs_yoy_ratio` — derived growth proxies.
    - **Tet (Vietnamese Lunar New Year) indicators**: ±10-day window, day-of vs pre vs post.
3. **Models:**
    - **LightGBM** (n_estim 4000, lr 0.02, num_leaves 47, early-stop 150).
    - **CatBoost full** (iter 4000, lr 0.025, depth 7, early-stop 150).
    - **CatBoost recent** (2019+ uniform weights) — kept as a diversity hedge.
4. **Sample weighting:** geometric decay with **2-year half-life** on full-history fits. The most recent year has weight 1.0, 2012 has weight ≈0.03.
5. **Blend weights:** found by grid search on validation RMSE, constrained to keep LGBM ≥ 0.10 (variance-reduction floor). Both targets landed on **0.10 LGBM + 0.90 CatBoost_full + 0.00 CatBoost_recent**.
6. **Final production fit:** trained on *all* rows 2013-01-01 → 2022-12-31 (includes the validation fold). Iteration counts set to 1.10× the early-stopping point from the validation pass.
7. **Recursive 2024 prediction:** for any test date whose `lag_365` lands in 2023 (i.e. 2024-01-01 onwards), the missing lag is filled with the model's own 2023 prediction. This creates an internally consistent 18-month forecast.

## 3. Key Validation Diagnostics

- Overall bias (actual − pred): Revenue **+197k/day**, COGS **+182k/day** on the 2021H2→2022 fold. The model *systematically under-predicts* because 2022 saw a bounce-back (+12% vs 2021) that the fold-pretraining (2013–2021) couldn't anticipate. On the real test set, 2022 *is* in training, so this bias is expected to shrink substantially.
- Worst months: **March 2022** (+$1.01M/day under) — the peak of the bounce-back was ~15% higher than any prior year.
- Best-fitted months: October 2021 (MAE 339k), December 2022 (MAE 294k, 321k) — these are the flatter, more predictable periods.

## 4. Top Feature Importances (LightGBM gain)

| Feature                        | Revenue gain | COGS gain |
|--------------------------------|-------------:|----------:|
| `*_seas_expand_mean`           |       4,497  |    3,517  |
| `t` (linear trend)             |         643  |      307  |
| `Revenue_lag_365`              |         553  |       —   |
| `*_seas_expand_mean` (cross)   |         278  |      636  |
| `t2`                           |         109  |       51  |
| `Revenue_level_913`            |          67  |       —   |
| `rev_yoy_ratio`                |          60  |       —   |
| `Revenue_lag_1460` / `lag_730` |          44  |       —   |
| `dow`                          |          43  |       39  |
| Fourier annual (K=1)           |          39  |       32  |

The **seasonal same-(month,day) mean** and **long-run trend** dominate — consistent with the data's regular annual cycle and modest multi-year drift. SHAP plots saved to `outputs/phase7_shap_revenue.png` and `outputs/phase7_shap_cogs.png`.

## 5. Submission Check-List

| Check | Status |
|---|---|
| Row count | 548 / 548 ✓ |
| Date range | 2023-01-01 → 2024-07-01 ✓ |
| Row order matches `sample_submission.csv` | ✓ |
| No NaN / non-positive values | ✓ |
| Total Revenue | $1,739,025,083 (implied annual ≈ $1.16B, close to 2022's $1.17B) |
| Total COGS    | $1,549,850,293 (implied annual ≈ $1.03B, close to 2022's $1.02B) |
| Seasonal shape | Mar-May peak, Nov-Dec trough — matches historical profile ✓ |

## 6. Files Produced

- `outputs/submission.csv` — final submission (Date, Revenue, COGS) × 548.
- `outputs/phase7_model_v2.py` — end-to-end training + prediction script (re-runnable).
- `outputs/phase7_validation_metrics.csv` — per-model MAE/RMSE/R².
- `outputs/phase7_validation_diagnostic.csv` — per-date val predictions vs actual.
- `outputs/phase7_feature_importance.csv` — LightGBM gain, Revenue + COGS.
- `outputs/phase7_shap_revenue.png`, `outputs/phase7_shap_cogs.png` — SHAP bar plots (top-20 features).

## 7. Competition Constraint Audit

- **No test data used as features:** all lag, level, and rolling features reference `Date ≤ 2022-12-31` only. Verified by construction (we `shift` on the full spine, so the values are NaN for the test window before prediction).
- **No external data:** all features derive from `sales.csv` only (plus hard-coded Tet anchor dates, which are public calendar facts).
- **Reproducibility:** fixed seed (42), deterministic preprocessing, no stochastic imports beyond LightGBM/CatBoost's seeded RNGs. A re-run of `phase7_model_v2.py` will reproduce `submission.csv` exactly.
- **Explainability:** SHAP plots + feature-importance CSV provided.

## 8. Gap to Top-of-Leaderboard

- Current top Kaggle score (per leaderboard screenshot): **658,785.01**.
- Our validation blend Revenue RMSE: **782,046** (pessimistic — see Sec. 3).
- Our validation blend COGS RMSE:    **688,493**.

On the real test (which includes 2022 in training), the +197k Revenue bias observed on the holdout should shrink, likely pushing Revenue RMSE to within ~10% of the leaderboard top. Headroom to close further comes from: (i) more aggressive hyperparameter search, (ii) ensembling 3–5 random-seed CatBoost runs, (iii) replacing `lag_365` recursive fill with a direct-forecasting approach keyed off `t` (eliminates recursion variance), (iv) a small Prophet or SARIMAX model on the residuals to catch long-horizon trend drift.
