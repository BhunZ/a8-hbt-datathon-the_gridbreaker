# V13 Master Data Blueprint

**Status:** Proposed. Not yet implemented. **Must pass local-holdout gate before any Kaggle submission.**

**Motivation.** V12 failed because we projected dynamic aux-file features (sessions, reviews, returns, stockouts, traffic, signups) into the test horizon via a 5D median profile. That created **train/serve skew**: the model learned patterns against real daily values but predicted against smoothed medians. Result: mean Revenue compressed from v10c's 4.23M to v12b's 3.66M, peaks flattened, RMSE rose to 1.25M on Kaggle (vs 775k for v10c).

**V13 non-negotiable rule.** No feature is admitted unless its test-time value is computed by exactly the same code, from exactly the same inputs, as its training-time value. If we need a median/profile to fill test-period values, the feature is rejected.

---

## 1. Research synthesis (Kaggle winning-solution patterns)

### 1.1 Spike handling — Binary Flags + Days-to-Event

Top M5 solutions and Rossmann/Favorita winners converge on the same pattern for event spikes (Christmas, SNAP days, holidays, store promotions):

- **Binary flags** for the event day itself (`is_singles_day`, `is_tet`, `is_black_friday`).
- **Continuous "days to / days since" features** that let the tree learn the *ramp* around an event (people shop 5 days before Singles Day, clearance drops for 3 days after Christmas). The M5 calendar included day-counters for promotion cycles and summer holidays; Rossmann had days-to/from state holiday features.
- Calendar features: `dow`, `month`, `week_of_year`, `is_weekend`, `day_of_month`.
- Sometimes **excluding** the most anomalous month (e.g. December for Christmas) from training — treated as a separate sub-model — if the anomaly is contaminating the mean model.

Why this works: all of these features are *pure functions of the date*. The model sees the same value at train and inference. Trees happily learn non-linear responses to `days_to_singles_day ∈ [0..14]`.

### 1.2 Master-data architecture — Static vs Dynamic

Two legitimate ways to fold in 13-file master data:

- **Static entity features.** Attributes that do not change over time (product category, store region, customer segment). These can be joined at any date without leakage. Rossmann's 3rd place (Cheng Guo, "entity embeddings") turned high-cardinality categoricals into learned dense vectors — works with neural nets; with tree models use **target encoding with smoothing and out-of-fold computation** to avoid leakage.
- **Dynamic features whose future values are known**. Example: a promotion calendar that extends into the test horizon. If the organizer ships future promo dates with the data, they're legal features.

**What is NOT legal:** dynamic per-day aggregates (sessions, reviews, stockouts, signups) whose source file ends before the test horizon. These have no deterministic future value. Our V12 "profile projection" was an attempt to fake them — it produced the train/serve skew that tanked the score.

### 1.3 Validation — Blocked time-series CV mirroring LB

Kaggle LBs typically split test data 20-30% public / 70-80% private. The *shake-up* risk is real — teams that tune to public LB often collapse on private. Defense:

- **Blocked time-series split** (not leave-one-out). Hold out contiguous future blocks that mirror the test-horizon length and structure.
- Build CV so that **improvements on CV score correlate with improvements on LB**. Once that correlation is established, trust CV over public LB.
- For this competition: test horizon is 548 days (2023-01-01 → 2024-07-01, 18 months). The closest we can mirror is hold out **2021-07-01 → 2022-12-31** (also 548 days, also crosses a year boundary, includes Tet 2022, Singles Day 2021 + 2022, Black Friday 2021 + 2022). Train on 2017-01-01 → 2021-06-30.

### 1.4 Target transformation — log1p vs Box-Cox

Research synthesis:

- **log1p** stabilizes variance on right-skewed non-negative targets, preserves ranks, invertible via expm1. Default for sales/revenue.
- **Box-Cox** only wins in ~20% of series vs plain log (Proietti & Lütkepohl 2013, and repeated Kaggle results). Tuning λ is an extra hyperparameter that can itself overfit. Not worth it for our horizon.
- **Raw target** (no transform) is what lets tree models preserve peak magnitudes — log compresses the top of the distribution. Our V12 Stage C ("Peak Catcher") used raw target for this reason and that part was correct.
- **Tweedie objective** (LightGBM, `objective='tweedie', tweedie_variance_power=1.5`) is designed for zero-inflated right-skewed data. Revenue has no zeros here, but spiky daily sales match the distribution shape well. Worth testing as an alternative to log1p.

Conclusion for V13: keep **log1p primary** + **raw-target secondary** ensemble, drop Box-Cox entirely, evaluate **Tweedie LightGBM** as an alternative primary.

---

## 2. V13 feature roster — every feature is deterministic at inference

Format codes:
- **FLAG** = binary 0/1, pure function of date
- **NUM(date)** = numeric, pure function of date
- **LAG** = value from sales history at a fixed historical offset (always ≥ 548 days from any test date)
- **STATIC** = value derived once from master data, constant across time

| # | Feature | Type | Source | Inference computation |
|---|---|---|---|---|
| 1 | `year` | NUM(date) | date | `date.year` |
| 2 | `month` | NUM(date) | date | `date.month` |
| 3 | `day_of_month` | NUM(date) | date | `date.day` |
| 4 | `dow` | NUM(date) | date | `date.weekday()` |
| 5 | `day_of_year` | NUM(date) | date | `date.timetuple().tm_yday` |
| 6 | `week_of_year` | NUM(date) | date | `date.isocalendar().week` |
| 7 | `is_weekend` | FLAG | date | `dow >= 5` |
| 8 | `is_month_start` | FLAG | date | `day == 1` |
| 9 | `is_month_end` | FLAG | date | `day == last_day_of_month` |
| 10 | `t_days` | NUM(date) | date | `(date - 2017-01-01).days` |
| 11 | `t_months` | NUM(date) | date | `year*12 + month` |
| 12 | `is_tet` | FLAG | lunisolar table | lookup (precomputed 2017–2025) |
| 13 | `is_black_friday` | FLAG | date | 4th Friday of November |
| 14 | `is_cyber_monday` | FLAG | date | Monday after BF |
| 15 | `is_singles_day` | FLAG | date | `month=11 & day=11` |
| 16 | `is_twelve_twelve` | FLAG | date | `month=12 & day=12` |
| 17 | `is_nine_nine` | FLAG | date | `month=9 & day=9` |
| 18 | `is_double_day` | FLAG | date | `month == day` |
| 19 | `is_womens_day_vn` | FLAG | date | `month=10 & day=20` |
| 20 | `is_intl_women_day` | FLAG | date | `month=3 & day=8` |
| 21 | `is_mothers_day_vn` | FLAG | date | 2nd Sunday of May |
| 22 | `is_christmas` | FLAG | date | `month=12 & day=25` |
| 23 | `is_new_year` | FLAG | date | `month=1 & day=1` |
| 24 | `is_payday_15` | FLAG | date | `day == 15` |
| 25 | `is_payday_eom` | FLAG | date | `day == last_day_of_month` |
| 26 | `is_tet_window_7d` | FLAG | lunisolar table | 1 if within ±3 days of Tet |
| 27 | `days_to_tet` | NUM(date) | lunisolar table | days until next Tet |
| 28 | `days_from_tet` | NUM(date) | lunisolar table | days since last Tet |
| 29 | `days_to_singles_day` | NUM(date) | date | capped at 365 |
| 30 | `days_from_singles_day` | NUM(date) | date | capped at 365 |
| 31 | `days_to_twelve_twelve` | NUM(date) | date | capped at 365 |
| 32 | `days_from_twelve_twelve` | NUM(date) | date | capped at 365 |
| 33 | `days_to_black_friday` | NUM(date) | date | capped at 365 |
| 34 | `days_from_black_friday` | NUM(date) | date | capped at 365 |
| 35 | `days_to_nine_nine` | NUM(date) | date | capped at 365 |
| 36 | `days_from_nine_nine` | NUM(date) | date | capped at 365 |
| 37 | `days_to_payday_15` | NUM(date) | date | `15 - day` if day<=15 else `30+15-day` |
| 38 | `days_to_payday_eom` | NUM(date) | date | `last_day - day` |
| 39 | `rev_lag_365` | LAG | sales | Revenue 365 days ago (always in train period for our test) |
| 40 | `rev_lag_730` | LAG | sales | Revenue 730 days ago |
| 41 | `rev_lag_1095` | LAG | sales | Revenue 1095 days ago (NaN for early training) |
| 42 | `cogs_lag_365` | LAG | sales | COGS 365 days ago |
| 43 | `cogs_lag_730` | LAG | sales | COGS 730 days ago |
| 44 | `rev_lag_same_dow_365` | LAG | sales | Rev from nearest-same-dow date ~365 days prior |
| 45 | `rev_rolling_mean_365_lag_365` | LAG | sales | mean Revenue over (t-730 … t-365) |
| 46 | `rev_rolling_mean_28_lag_365` | LAG | sales | mean Revenue over (t-393 … t-365) |
| 47 | `rev_same_doy_mean_all_years` | LAG | sales | mean Revenue on same day-of-year across all prior training years |
| 48 | `rev_same_doy_std_all_years` | LAG | sales | std on same day-of-year (uncertainty proxy) |
| 49 | `cogs_same_doy_mean_all_years` | LAG | sales | mean COGS on same day-of-year across all prior training years |
| 50 | `avg_catalog_price` | STATIC | products | mean `products.price_active` — single constant |
| 51 | `avg_catalog_cogs` | STATIC | products | mean `products.cogs` — single constant |
| 52 | `cogs_ratio_catalog` | STATIC | products | `avg_catalog_cogs / avg_catalog_price` — single constant |

**Count: 52 features. Every one produces an identical value at train time and test time.**

### 2.1 Features explicitly REJECTED from the 13 files

| File | Fields considered | Reason rejected |
|---|---|---|
| orders.csv | n_orders/day, avg_order_value/day | Dynamic; ends 2022-12-31; no deterministic future value |
| order_items.csv | item_count/day, gross_revenue/day | Dynamic; same reason |
| sessions.csv | sessions, unique_visitors, bounce_rate | Dynamic; our V12 failure mode lives here |
| traffic_sources.csv | source shares | Dynamic; ends 2022-12-31 |
| reviews.csv | n_reviews, avg_rating | Dynamic |
| returns.csv | n_returns, return_value | Dynamic |
| signups.csv | n_signups | Dynamic |
| stockouts.csv | n_stockouts | Dynamic |
| promotions.csv | n_active_promos | Dynamic and ends 2022 — if the competition shipped future promo dates, this would be legal; it did not |
| campaigns.csv | n_campaigns | Same as promotions |
| customers.csv | per-customer aggregates | Static per customer but cannot be attributed to future days without knowing who shops; for daily-total prediction, useless |
| products.csv | per-product attributes | Same — can't attribute future daily total to specific products; use only for catalog-level constants (features 50-52) |

**If the competition later publishes future-horizon promo/campaign schedules, promotions.csv and campaigns.csv become legal immediately and should be added.**

---

## 3. Inference logic — train/test symmetry proof

One-line guarantee: **`build_features(date)` is a pure function of `date` plus the frozen sales/products history up to 2022-12-31.**

```python
def build_features(date):
    cal  = calendar_features(date)            # 11 features, pure date
    evt  = event_flags(date)                  # 14 features, pure date + Tet table
    d2e  = days_to_from_events(date)          # 12 features, pure date + Tet table
    lag  = sales_lags(date, sales_frozen)     # 11 features, lookup in frozen sales 2017-2022
    stat = static_catalog_constants()         # 3 features, constants
    return concat(cal, evt, d2e, lag, stat)
```

For every test date D ∈ [2023-01-01, 2024-07-01]:
- `D - 365` ∈ [2022-01-01, 2023-07-01] — crosses into test horizon at 2023-07-02! **Problem.**

Fix: use `D - 548` minimum lag for test period. Training rows near the gap will have `D - 365` landing in train (fine for training). But **test-time `rev_lag_365` for D=2024-05-01 would need Revenue from 2023-05-01, which is unobserved.**

**Resolution (V13 rule):** for test-period predictions, compute lags from the *last fully observed date* (2022-12-31). So `rev_lag_365` at D=2024-06-01 becomes `rev_at(2022-12-31 - (2024-06-01 - D'))` where D' is the reference point. In practice: **all test-period lag features are anchored to 2022-12-31, not to D.** This is still deterministic and symmetric (the same rule applies to the last day of training).

Cleaner alternative: **shift lag distances so they always reach pre-2023 data**. Use `rev_lag_N` where `N = max(365, (D - 2022-12-31).days + 30)`. This guarantees the lookup is always in training territory.

The *correctness criterion*: unit-test that `build_features('2020-11-11')` computed during training equals `build_features('2020-11-11')` when called at inference. If they differ, the feature is broken. A single failing test = V13 does not ship.

---

## 4. Peak-recovery strategy

Three layers, each independently measurable on held-out 2022.

### 4.1 Layer 1 — Explicit peak feature injection

`days_to_singles_day`, `days_to_tet`, `days_to_twelve_twelve` etc. let the tree build non-linear ramps. The tree carves out leaves like "days_to_singles_day ∈ [0,2] AND year ≥ 2021 → predict 8M Revenue" — and because `days_to_singles_day` is deterministic, the test-time prediction is pulled into that leaf on 2023-11-11, 2024-11-11.

### 4.2 Layer 2 — Ensemble of three target-transform regimes

- **Model A (Anchor):** LightGBM on log1p(Revenue), trained 2017+. Stabilizes the mean.
- **Model B (Peak-sensitive):** LightGBM with `objective='tweedie', tweedie_variance_power=1.5`, trained on raw Revenue. Tweedie naturally handles right-skew without compressing peaks as hard as log.
- **Model C (Peak Catcher):** ExtraTrees on *raw* Revenue (no transform), trained 2019+ only. Extreme-value preserver.

Final Revenue blend: **0.40 A + 0.40 B + 0.20 C**. Same three-model architecture for COGS.

The key change from V12: **Model B is Tweedie LightGBM, not weighted RF**. Research shows Tweedie specifically shines on Poisson/zero-inflated-ish sales data and avoids the log compression problem while still being variance-stabilized.

### 4.3 Layer 3 — Conditional gold blend (only if it helps locally)

Keep v10c in reserve but do not blend by default. Only blend if:
- V13 alone beats v10c on held-out 2022 RMSE, AND
- A v10c blend at some weight w ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} improves held-out RMSE further.

Choose the weight that minimizes held-out RMSE. If no weight improves over V13 alone, ship V13 alone. If V13 alone loses to v10c, ship v10c — do not blend a worse model into a better one.

---

## 5. Validation protocol — no Kaggle probe without local proof

### 5.1 The holdout

- **Train:** 2017-01-01 → 2021-06-30 (4.5 years)
- **Holdout:** 2021-07-01 → 2022-12-31 (548 days — exact test-horizon length, includes Tet 2022, Singles Day 2021+2022, BF 2021+2022, Xmas/NY)

### 5.2 The pseudo-LB split

Within the 548 holdout days, randomly assign 25% to "pseudo-public" and 75% to "pseudo-private" (matching Kaggle's typical 20-30 / 70-80). Evaluate RMSE separately on each. The public-vs-private gap is our shake-up risk estimate. If the gap > 10%, the model is overfitting to the public subset and we reject the change.

### 5.3 Acceptance gate

Before any V13 submission to Kaggle:

1. Run v10c on the same holdout → record baseline RMSE.
2. Run V13 on the same holdout → record candidate RMSE.
3. Accept V13 iff:
   - candidate_RMSE < baseline_RMSE by ≥ 2% on private subset, AND
   - candidate peak-day RMSE < baseline peak-day RMSE (where peak-day = any day with `is_*` event flag = 1)
4. If accepted: one Kaggle probe. If Kaggle RMSE exceeds v10c's 775k, **stop**, inspect residuals, do not keep probing.

### 5.4 Forbidden practices

- No tuning to public Kaggle LB.
- No building a new variant until the current variant has finished local CV and one Kaggle probe.
- No blending two models locally and submitting the blend to Kaggle without first submitting each component alone to measure the unbiased scores.

---

## 6. Implementation sequence

1. **`v13_calendar_table.py`** — precompute every deterministic feature (features 1-38, 50-52) for 2016-01-01 → 2024-12-31. Save parquet. Unit-test symmetry at 3 dates.
2. **`v13_lag_features.py`** — compute features 39-49. Unit-test that lag lookups are always in pre-2023 territory for any test-horizon row.
3. **`v13_build_features.py`** — assemble full feature matrix. Validate: `nrows == 2739 + 548`, `ncols == 52 + target_cols`, `isna().sum() == 0` on all non-lag features.
4. **`v13_holdout_baseline.py`** — score v10c on 2021-07-01 → 2022-12-31 holdout. Baseline RMSE logged to `docs/v13_holdout_baseline.json`.
5. **`v13_train_ensemble.py`** — train 3 models (A/B/C) on 2017 → 2021-06-30. Score on holdout. Log RMSE + per-event-day RMSE.
6. **`v13_gate.py`** — apply acceptance gate from §5.3. Output pass/fail + recommendation.
7. **Gate passes?** → one Kaggle submission. Otherwise, diagnose and iterate within local holdout only.

---

## 7. What we are giving up

Being honest about the cost of this discipline:

- **50+ potentially-useful aux signals dropped.** Sessions, promotions, reviews carry real information about demand. We lose that lever because we have no way to know their future values.
- **No more "throw features at the wall" phase.** Every feature passes a deterministic-symmetry test or it's rejected.
- **Slower iteration.** Each candidate change requires a holdout run (~15 min) before any Kaggle submission.

The trade: we stop paying Kaggle-submission taxes for locally-blind guesses. V12 burned 4 submissions on a model that was 62% worse than the prior best, and we had no local evidence to predict that. V13 burns 0 Kaggle submissions until the local gate says the change is real.

---

## 8. Decision rules for future iterations

Once V13 is baselined:

- **V13 Kaggle ≤ 760k:** ship as new best, iterate on peak-recovery layer 1 (add more event flags, finer ramp windows).
- **V13 Kaggle 760k–775k:** tie or marginal improvement. Explore Tweedie parameter tuning, larger ensembles.
- **V13 Kaggle > 775k:** V13 is a regression. Do *not* submit variants. Diagnose residuals on holdout vs residuals on Kaggle public slice (if possible). The defect is either in feature symmetry or in ensemble blending weights — find it before building V14.

---

*Prepared 2026-04-24 after V12 post-mortem. Implementation gated on approval.*
