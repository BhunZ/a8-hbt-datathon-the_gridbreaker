# V13 Master Data Blueprint — Deterministic-at-Inference Feature Architecture

**Date:** 2026-04-25
**Status:** Design draft (no code written yet)
**Replaces:** V11, V12 (both confirmed worse than V10c on Kaggle)

---

## REVISIONS — 2026-04-25 (post-timeline-audit)

After running `src/v13_timeline_audit.py` and `src/v13_covid_audit.py` against all 13 raw files, the following sections of this blueprint are **superseded** by the new sections 11–15 below. Read those before relying on the originals.

| Section | Status | Replaced by |
|---|---|---|
| §0 ("Three lawful sources" — train window 2017→2022) | **Window changed** | §12 — train window now **2019-09-01 → 2022-12-31** |
| §2.1 ("LAG-Y from 2017→2022") | **Window changed** | §12 — lag lookups restricted to post-cliff regime |
| §6 ("Validation protocol — train ≤ 2021-06-30") | **Superseded** | §12 — mirror block + folds redrawn inside the post-2019-09 window |
| §9 ("Build order — Day 1 frozen priors on 2017→2022") | **Superseded** | §14 — revised build order; COVID handling deleted, trend features added |
| Implicit assumption that 2019 was a normal pre-pandemic baseline | **Wrong** | §11 — 2019 is itself the new regime; the cliff happened mid-2019, not in 2020 |
| Implicit COVID-as-major-disruption framing | **Wrong** | §12 — empirical audit shows COVID is a non-event for this brand |

The deterministic-at-inference *constraint* (§0) and the per-file *tags* (§2) are unchanged. Only the **time window**, **COVID handling**, and **expected revenue magnitude** moved.

---

## 0. The constraint that drives every decision

> **A feature is allowed in V13 if and only if its value for any test date can be computed from sources that exist at inference time without inventing a number.**

That eliminates: median-profile projections, linear-trend projections of dynamic state, "filled with mean" placeholders, and any feature whose training-time value comes from a source that ends 2022-12-31 unless we deliberately freeze it as a static lookup at the train cutoff.

Three lawful sources of feature value at inference time:

1. **Calendar** — the date itself, plus deterministic event tables (Tet via lunar conversion, named sale days, Vietnamese holidays).
2. **Frozen aggregates** — scalars or small lookup tables computed once on the train window (2017-01-01 → 2022-12-31) and joined by a key that exists at inference time (date components, entity id).
3. **Long-horizon historical lags** — values reaching back ≥ the forecast gap. For our 548-day horizon, only same-day-prior-year and earlier are honest, since lag-1…lag-547 require unknown test data.

Everything else is an illusion that scores well in cross-validation and worse on Kaggle.

---

## 1. Why V12 lost — confirmed by Kaggle scoreboard

| Submission | RMSE | Δ vs v10c |
|---|---:|---:|
| v10c (23-feature baseline)  | 774,898 | — |
| v12a (58-feature pure model) | 1,252,820 | **+477,922** |
| v12b (+ Stage D COGS ratio)  | 1,243,589 | +468,691 |
| v12c (40/60 blend with v10c) | 852,132 | +77,234 |
| v12d (v12c × 0.99 / 0.98)    | 871,299 | +96,401 |

The pure V12 model is the diagnostic: 58 features instead of 23 cost us ~62% in RMSE. The mechanism is **train/serve skew on 35+ projected aux features**. At training time the model saw real daily sessions, returns, stockouts, traffic-source shares; at inference time it saw 5-D median profile values that, by construction, cannot reproduce a Black Friday or a Tet eve. Tree splits learned on live signal collapse to the conditional mean of the projected signal — peaks get crushed.

This matches the canonical [training-serve skew failure mode](https://building.nubank.com/dealing-with-train-serve-skew-in-real-time-ml-models-a-short-guide/) and the [data-leakage rule](https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/): "any feature relying on information unavailable at the prediction moment will appear better in evaluation than in production."

---

## 2. Per-file decision table

For each of the source files we classify every candidate feature as:

| Tag | Meaning | Inference-time source |
|---|---|---|
| **CAL**  | Calendar / event flag, computed from the date | Date arithmetic |
| **STAT** | Static entity attribute or out-of-fold target encoding | Frozen lookup keyed by entity id |
| **PRIOR** | Frozen scalar prior keyed by date components | Lookup table (train-window aggregate) |
| **LAG-Y** | Same-period-prior-year aggregate from the train window | Date-shift join into closed history |
| **DROP** | Dynamic-state feature with no honest test-time value | (forbidden) |

### 2.1 `sales.csv` — TARGETS + lag source
Used only to (a) supply Revenue / COGS labels on training rows and (b) compute prior-year same-DOW/month aggregates as **LAG-Y** features.

| Column | V12 use | V13 use |
|---|---|---|
| Revenue | target | target only |
| COGS    | target | target only |
| Date    | join key | join key |

Derived **LAG-Y** features (computed from sales history only, frozen at train cutoff):
- `rev_same_dow_prev_year` — average revenue on the same dow in the prior 52-week window.
- `rev_same_dom_prev_year` — average revenue on the same day-of-month in the prior 12-month window.
- `cogs_same_dow_prev_year`, `cogs_same_dom_prev_year` — analogues for COGS.
- `rev_yoy_lag_364`, `rev_yoy_lag_728` — point lags at 364 and 728 days.

These are honest because the lookup window is 2017→2022 and the join key is just a calendar shift.

### 2.2 `orders.csv` + `order_items.csv` + `payments.csv` — DYNAMIC, mostly drop
V12 added 16 dynamic columns from these (n_orders, pct_mobile, n_items, items_gross, items_per_order, payment_value_sum, …). All 16 → **DROP** as features. Their values for 2023-01-01 forward do not exist.

Three things from this group are still useful, but not as dynamic features:
1. `items_cogs_recon` and `items_gross` are kept as **diagnostic columns** for the train period only — they let us verify the COGS identity (`r=1.0000` per Step 3) and audit the AOV trend, but they are NOT model features.
2. The full historical order stream feeds the **STAT** target encodings on `products` and `customers` below.
3. The historical AOV time series feeds one **PRIOR** scalar (see §2.7) — but only as a frozen monthly mean, not a per-day projection.

### 2.3 `products.csv` — STATIC, all kept
This file describes entities that exist at inference time. Every column is safe.

| Feature | Tag | Notes |
|---|---|---|
| `product_id`, `category`, `brand`, `gender`, `size`, `color` | STAT | direct attributes |
| `price`, `cogs`, `cogs_ratio = cogs/price` | STAT | static per product |
| **Target encoding**: `enc_rev_per_product` | STAT | out-of-fold mean revenue per product on the train window |
| **Target encoding**: `enc_units_per_product` | STAT | out-of-fold mean units per product |

The target encodings must be computed [out-of-fold](https://brendanhasz.github.io/2019/03/04/target-encoding.html) to avoid label leakage. We never use these directly per row — we aggregate them up to daily features (see §3).

### 2.4 `customers.csv` — STATIC, all kept
| Feature | Tag |
|---|---|
| `customer_id`, `signup_date`, `signup_channel`, `country`, `gender`, `age_band` | STAT |
| `customer_tenure_days = (order_date − signup_date)` | STAT (per order) |
| **Target encoding** `enc_aov_per_cohort` (cohort = signup_year × signup_channel) | STAT |

### 2.5 `web_traffic.csv` — DROP
Every column from this file (sessions, unique_visitors, page_views, bounce_rate, avg_session_duration, traffic-source shares) is dynamic state with no inference-time value. **All 13 columns dropped.** This is the single largest source of V12's failure.

The history is still useful for **offline analysis** (which holidays correlate with traffic spikes, which traffic sources drive peak-day revenue) — that analysis informs the calendar event book in §2.10, but no traffic column ever becomes a model feature.

### 2.6 `reviews.csv`, `returns.csv`, `signups.csv`, `stockouts.csv` — DROP all dynamic features
Same reasoning as web_traffic. Daily counts and rates from these files end 2022-12-31. **DROP** every column.

Two **STAT** survivors:
- `enc_return_rate_per_product` from `returns.csv` joined to `order_items` — frozen per product.
- `enc_avg_rating_per_product` from `reviews.csv` — frozen per product.

These travel forward only because they are tied to a product id, not a date.

### 2.7 AOV — one **PRIOR** scalar, not a per-day projection
V12 did `aov(t) = intercept + slope × t` for the test horizon. That is a guess. **DROP.**

Replacement: `aov_prior_by_month` — a 12-row lookup of the train-window monthly mean of revenue/orders, joined on `month`. One scalar, no future projection, no skew.

### 2.8 `promotions.csv` — EVENT BOOK
Promotions have explicit `start_date` / `end_date`. If the test horizon contains promo records (check this — they may or may not be supplied), expand them and create:
- `is_promo_active` (CAL)
- `n_active_promos` (CAL — count of records overlapping the date)
- `max_discount_active` (CAL)

If promotions for 2023+ are NOT supplied, this collapses to a **PRIOR**: `promo_intensity_by_(month,dow)` from the train window. Verify promotions table extent before building.

### 2.9 `campaigns.csv` — EVENT BOOK (same logic)
Same treatment as promotions. Inspect whether 2023+ campaigns are provided. If yes → CAL; if no → PRIOR by `(month, dow)`.

### 2.10 The dense calendar event book — the most important addition
This is the V13 replacement for the entire 5-D-projected feature block. Built once, joined by date. Every column is **CAL**.

**Base calendar (already in V12, keep):**
year, month, day, dow, woy, day_of_year, is_weekend, quarter, is_month_start, is_month_end.

**Vietnamese fixed-date events (existing in V12):**
- `is_singles_day` (11/11)
- `is_twelve_twelve` (12/12)
- `is_nine_nine` (9/9)
- `is_ten_ten` (10/10)
- `is_double_digit_day` (any month==day)
- `is_intl_women_day` (3/8)
- `is_womens_day_vn` (10/20)
- `is_teachers_day_vn` (11/20)
- `is_independence_day_vn` (9/2)
- `is_reunification_day` (4/30)
- `is_intl_labour_day` (5/1)

**Western retail anchors (existing in V12, keep):**
- `is_black_friday`, `is_cyber_monday` (4th Friday/Monday of November)
- `is_boxing_day` (12/26)

**Lunar / Tet event book (UPGRADE — use lunar library, not hand-coded list):**
Replace V12's 8-year hand-coded `TET = [...]` with [`python-holidays`](https://pypi.org/project/holidays/) Vietnam locale. This auto-handles Tet for any future year.
- `is_tet` (the 3 official Tet days)
- `is_pre_tet_window` (D-30 to D-1) — Vietnamese e-commerce data shows ~33% of consumers begin Tet shopping 40 days out, ~60% within 20 days; the M5 winners' "-15 day ramp" trick generalizes ([Artefact M5 retrospective](https://medium.com/artefact-engineering-and-data-science/sales-forecasting-in-retail-what-we-learned-from-the-m5-competition-445c5911e2f6))
- `is_tet_recovery_window` (D+1 to D+10) — depressed sales after Tet
- `tet_day_index` — signed integer days from nearest Tet (-30 ... +10), clipped to 0 outside the window
- `is_lunar_15_full_moon` — monthly lunar full moon, relevant in Vietnamese culture

**Payday / month-rhythm (existing in V12 as `is_pd`, expand):**
- `is_payday_15`, `is_payday_25`, `is_payday_30`
- `days_to_next_payday` (clipped 0–10)
- `days_since_last_payday` (clipped 0–10)

**Event-distance features (NEW — the key M5 insight):**
For each named event in the book above, two columns:
- `days_to_<event>` — clipped to 30 (negative-then-zero behaviour gives the build-up ramp)
- `days_since_<event>` — clipped to 30 (the post-event tail)

This is what the [Rossmann winner](https://medium.com/kaggle-blog/rossmann-store-sales-winners-interview-1st-place-gert-jacobusse-a14b271659b) calls "relative time indicators" and the M5 1st-place writeup attributes much of the holiday lift to.

**Sale-cluster id (NEW):**
- `campaign_window_id` — integer id per known sale window (Tet, Singles Day cluster Nov 11–12, Year-end cluster Dec 12–25, etc.)
- `day_index_within_campaign` — 0-based index within the active campaign window

---

## 3. The V13 feature roster — final list

After the audit, V13 ships **~38 features** (down from V12's 58, up from V10c's 23):

**Calendar & event book (24 columns, all deterministic):**
year, month, day, dow, woy, day_of_year, is_weekend, quarter, is_month_start, is_month_end, is_double_digit_day, is_singles_day, is_twelve_twelve, is_nine_nine, is_ten_ten, is_intl_women_day, is_womens_day_vn, is_teachers_day_vn, is_black_friday, is_cyber_monday, is_tet, is_pre_tet_window, is_tet_recovery_window, tet_day_index.

**Event-distance (6 columns):**
days_to_tet, days_since_tet, days_to_singles_day, days_to_twelve_twelve, days_to_black_friday, days_since_payday.

**Frozen priors keyed by date components (4 columns):**
prior_rev_by_month_dow, prior_cogs_by_month_dow, prior_rev_event_uplift (multiplicative), aov_prior_by_month.

**Long-horizon lag features (4 columns):**
rev_same_dow_prev_year, rev_yoy_lag_364, cogs_same_dow_prev_year, cogs_yoy_lag_364.

Total: 38 features, every one of which can be evaluated for any test date with no projection.

---

## 4. Inference logic — the proof that train and test are identical

The defining test for V13: take a test date (e.g. 2024-03-08), generate its feature vector, and the procedure must be character-for-character identical to how the same date would be generated if we held it out of training.

```
def features_for(date d):
    f = {}
    # Calendar
    f.update(calendar(d))           # year, month, day, dow, ... is_weekend
    f.update(event_flags(d, BOOK))  # is_singles_day, is_tet, ...
    f.update(distance_to(d, BOOK))  # days_to_<event>, days_since_<event>
    f["tet_day_index"] = signed_distance_to_nearest_tet(d)

    # Frozen priors  -- lookup tables loaded once
    f["prior_rev_by_month_dow"]  = PRIOR_REV[d.month, d.dow]
    f["prior_cogs_by_month_dow"] = PRIOR_COGS[d.month, d.dow]
    f["aov_prior_by_month"]      = AOV_PRIOR[d.month]
    f["prior_rev_event_uplift"]  = EVENT_UPLIFT[active_event(d) or "none"]

    # Long-horizon lags
    f["rev_same_dow_prev_year"]  = SALES_HISTORY.rev_same_dow_prev_year(d)
    f["rev_yoy_lag_364"]         = SALES_HISTORY.lookup(d - 364)
    f["cogs_same_dow_prev_year"] = SALES_HISTORY.cogs_same_dow_prev_year(d)
    f["cogs_yoy_lag_364"]        = SALES_HISTORY.lookup(d - 364)

    return f
```

`PRIOR_REV`, `PRIOR_COGS`, `AOV_PRIOR`, `EVENT_UPLIFT` are computed once on `sales.csv` restricted to dates < `train_cutoff`, written to disk, and reloaded identically at train time and test time. There is no branch in the code that says "if date in test horizon, do something different." That branch is the bug class V12 exhibited.

---

## 5. Peak-recovery strategy

Three layers, applied in order:

### 5.1 Layer 1 — model selection biased toward peak preservation
Switch from RandomForest/ExtraTrees to **LightGBM with a Tweedie objective** (`tweedie_variance_power=1.15`, identity target). The Artefact M5 retrospective notes this objective "forces the model to predict the right amount of zeros and is well suited to retail-style spiky distributions." Run as a head-to-head with the V10c log1p+RMSE baseline; pick by mirror-LB peak-day RMSE.

If we keep log1p + RMSE, apply the [back-transform debias correction](https://arxiv.org/pdf/2208.12264): `y_hat_corrected = expm1(y_log_hat + 0.5 × residual_var_log_space)`. V10c does NOT do this — adding it is a free improvement.

### 5.2 Layer 2 — uplift model on event days only
Train a second LightGBM with `objective='quantile', alpha=0.85` on the subset of training rows where `is_event = 1` (any flag in the event book is set). Target: `(actual − stage1_pred) / stage1_pred`, clipped to a sane range. At inference, multiply stage-1 prediction by `(1 + uplift_pred)` only on event-flagged days.

This is the [hybrid pattern from M5 Uncertainty 2nd place](https://www.sciencedirect.com/science/article/abs/pii/S0169207022000097) (Mamonov et al.) — separate models for the body of the distribution and the tail.

### 5.3 Layer 3 — frozen multiplicative event uplift table
For each named event in the book, fit a multiplier on a held-out year (use 2022 as the calibration set):

```
uplift[event] = mean( actual / stage1_pred )  for dates where event flag = 1
```

Apply at inference. This is the [Walmart Recruiting winner's trick](https://medium.com/analytics-vidhya/walmart-recruiting-store-sales-forecasting-kaggle-competition-856c72c9265a) — when per-event spike correction beats trying to learn the spike inside the model.

---

## 6. Validation protocol — mirror the LB structure exactly

**Single locked mirror-LB block:** hold out **2021-07-01 → 2022-12-31** (548 days, exactly the test horizon length, includes one Tet, one Singles Day, one Black Friday). Train on 2017-01-01 → 2021-06-30. Score this block at most once per substantive iteration. This is our "private LB at home."

**Iteration CV:** 4 expanding-origin folds at 90-day horizon for fast iteration:
- Fold 1: train ≤ 2020-12-31, predict 2021-Q1
- Fold 2: train ≤ 2021-03-31, predict 2021-Q2
- Fold 3: train ≤ 2021-06-30, predict 2021-Q3
- Fold 4: train ≤ 2021-09-30, predict 2021-Q4

Configuration and feature changes get scored on these 4 folds. Only when a candidate beats the current best on iteration CV does it get evaluated on the locked 548-day mirror block. Only when it beats the current best on the mirror does it get a Kaggle submission.

**Two metrics, always reported together:**
- `RMSE_global` — overall RMSE on the held-out window
- `RMSE_peak_days` — RMSE restricted to dates where any event flag is set OR daily revenue ≥ 90th percentile of training

A submission that improves global RMSE by lowering peak predictions toward the mean is exactly the V12 failure. The peak metric exists to catch it locally.

---

## 7. Data-leakage audit checklist (run before every submission)

For each feature column in the final feature table:

1. **Inference availability:** can this column be computed for an arbitrary future date using only the train cutoff data and the date itself? If no → DROP.
2. **Train/test consistency:** if I generated this column for a row that's in training, then re-generated it for the same row treating it as test, are the values identical? If no → DROP.
3. **Encoding leakage:** if this is a target encoding, is it computed out-of-fold? If no → recompute OOF.
4. **Lag horizon:** if this is a lagged feature, is the lag ≥ the gap between train cutoff and the date being predicted? If no → DROP.
5. **Median/mean/mode placeholder:** is any value in the test split a fill-in (median, mean, fwd-fill, profile median)? If yes → DROP the column entirely.

V12's 35+ projected features fail check #5 universally. V10c's 23 features pass all five. V13 is designed to pass all five.

---

## 8. Submission discipline

- Every Kaggle submission must have a written justification: which feature/architecture change, what its iteration-CV delta was, what its mirror-LB delta was, and what it predicts the LB delta will be.
- If mirror-LB regresses but iteration-CV improves, do not submit.
- If iteration-CV improves but peak-day RMSE regresses, do not submit.
- v10c remains our floor. Any submission that doesn't beat 774,898 locally on the mirror block does not get uploaded.

---

## 9. Build order for V13

1. **Day 1 — calendar & event book.** Build the dense event book including lunar Tet via `python-holidays`. Verify event flags hit the right dates for 2017-2024.
2. **Day 1 — frozen priors.** Compute `PRIOR_REV`, `PRIOR_COGS`, `AOV_PRIOR`, `EVENT_UPLIFT` from `sales.csv` train window. Write to `data/processed/v13_priors.parquet`.
3. **Day 1 — long-horizon lags.** Compute same-dow-prev-year and lag-364 columns. Write to `data/processed/v13_lags.parquet`.
4. **Day 1 — feature assembly.** `daily_features_v13.parquet` joining calendar + event book + priors + lags. **Zero columns derived from web_traffic, reviews, returns, signups, stockouts, inventory, orders, items, payments.**
5. **Day 2 — validation harness.** Implement the 4-fold expanding CV + the 548-day locked mirror block. Score V10c on it as a baseline (this is the only number to beat).
6. **Day 2 — V13 baseline model.** LightGBM, log1p + RMSE, debias-corrected `expm1`. Score on the validation harness.
7. **Day 3 — peak-recovery layer.** Add the stage-2 quantile uplift model and the frozen multiplicative uplift table. Re-score.
8. **Day 3 — Tweedie head-to-head.** Train the Tweedie variant. Pick by peak-day RMSE on mirror block.
9. **Day 4 — only if mirror block beats V10c locally — submit to Kaggle.**

---

## 10. What we are explicitly NOT doing

- Not using web_traffic, reviews, returns, signups, stockouts, inventory as feature sources for the model.
- Not projecting any aux value into the test horizon.
- Not blending with v10c in feature space (we may blend in prediction space at the end, but only if V13 beats V10c standalone first).
- Not trusting in-sample correlations or means as proxies for Kaggle RMSE.
- Not submitting to Kaggle without a mirror-LB delta to justify it.

---

## 11. The 2019 regime change — the biggest finding in the data

`src/v13_timeline_audit.py` ran across 13 raw files for 2012-07 → 2022-12. The dominant signal in the entire history is **not** COVID — it is a **mid-2019 step-change** that resets the business onto a permanently smaller (and structurally different) trajectory.

### 11.1 The cliff (2018 → 2019, yearly mean)

| Metric | 2018 → 2019 | Reading |
|---|---|---|
| `n_orders` | 5792.5 → 3466.8 | **−40.2 %** |
| `n_unique_cust` | 5255.7 → 3253.8 | **−38.1 %** |
| `revenue_actual` (12-mo rolling, 2019-10) | 5.22 M → 3.19 M | **−38.9 %** |
| `n_signups` | 1084 → 1255 | +15.7 % |
| `sessions` | 784 590 → 832 512 | +6.1 % |
| `unique_visitors` | 595 291 → 631 714 | +6.1 % |
| `aov` | 25 532 → 25 958 | +1.7 % |
| `items_per_order` | 4.90 → 4.89 | −0.3 % |
| `avg_unit_price` | 5483 → 5621 | +2.5 % |
| `n_active_skus` | 483 → 457 | −5.4 % |
| `bounce_rate` | 0.005 → 0.005 | flat |
| `rating_mean` | 3.93 → 3.93 | flat |

Change-point detector (12-month rolling, threshold ±20 %) flags the same dates for `n_orders`, `n_unique_cust`, and `revenue_actual` — all between **2019-09 and 2020-02**, all between **−39 % and −41 %**. No other regime change of comparable magnitude exists in the entire 10-year history.

### 11.2 What the cliff was — and was not

- **Not a demand crash.** Sessions, unique visitors, and signups all grew through 2019.
- **Not a price/basket crash.** AOV, items-per-order, unit price, ratings all stable.
- **Not COVID.** It happened ~6 months before COVID, and COVID itself shows up as essentially flat in the empirical audit (`docs/v13_covid_audit.md`). Only Aug 2021 showed a meaningful dip (−16 %), nothing at 2020 lockdown levels.
- **Was a conversion crash.** Traffic up, but ~40 % fewer of those visitors became orders. AOV *rose* slightly, consistent with losing the lower-AOV slice of buyers.
- **Was a SKU contraction.** `n_active_skus` started a multi-year decline (483 → 410 by 2022, −15 %).

The most plausible read: a deliberate **business-model pivot in 2019** — likely DTC channel shift, marketplace exit, or audience repositioning — that shrank order volume by ~40 % while leaving traffic and basket economics intact. The brand never recovered the 2017-2018 volume; 2020-2022 stays inside the new, smaller regime and continues to drift downward in volume while AOV rises.

### 11.3 The drift inside the new regime (2019 → 2022)

| Metric | 2019 → 2022 | Reading |
|---|---|---|
| `n_orders` | 3467 → 3000 | −13.5 % |
| `n_unique_cust` | 3254 → 2841 | −12.7 % |
| `aov` | 25 958 → 30 920 | **+19.1 %** |
| `avg_unit_price` | 5621 → 6836 | **+21.6 %** |
| `items_per_order` | 4.89 → 4.77 | −2.4 % |
| `n_signups` | 1255 → 1759 | +40.1 % |
| `n_active_skus` | 457 → 410 | −10.3 % |

Inside the post-cliff regime there is a clean monotonic story: **fewer orders, higher AOV, narrower SKU set, more signups (slow-converting).** The test horizon (2023-01 → 2024-07) lives at the far end of this drift. Models trained on 2017-2018 data will systematically over-predict because they remember the old regime.

### 11.4 Why this matters more than COVID

The empirical COVID audit (`docs/v13_covid_audit.md`) showed the brand was largely insulated from COVID — most COVID months sit inside the NORMAL band relative to the *new-regime* 2019 baseline. The 2019 cliff is roughly **40 % of revenue**. The COVID dip is roughly **0–16 % of revenue, and only in Aug 2021.** Treating COVID as the regime change is mis-attribution; the regime change was 2019.

---

## 12. Locked training decisions

Three decisions are locked before any V13 code is written. They override the original §0 / §2.1 / §6 / §9 wording in this document.

### 12.1 Train window: 2019-09-01 → 2022-12-31 (~3 years 4 months)

- **Why not 2017+:** the 2017-2018 window is a different business (40 % larger volume, different SKU mix). Including it injects bias toward over-prediction.
- **Why 2019-09-01 specifically:** the change-point detector localizes the cliff to 2019-09 → 2020-02. 2019-09-01 is the earliest date that sits cleanly inside the new regime.
- **Why this is enough data:** ~1218 days, includes 3 Tets (2020, 2021, 2022), 4 Singles Days, 4 Black Fridays. Sufficient for the calendar+event book.
- **Frozen priors and lag lookups in §2.1, §2.7, §3 are recomputed on this window** — the 2017-2018 averages are not used. Same procedure, smaller window.

### 12.2 COVID handling: dropped entirely

- **No COVID flag block.** No `is_covid_lockdown`, `is_covid_recovery`, `covid_severity` scalar — none of it.
- **Justification:** the empirical audit shows COVID barely moved this brand's revenue. Adding flags trains the model to chase noise.
- **Mirror block contains COVID dates anyway.** If COVID-era held-out RMSE is much worse than non-COVID held-out RMSE *on the mirror window*, we revisit. Until then, no special handling.

### 12.3 Add explicit trend features (NEW feature group)

Inside the new regime there is a real downward drift in volume and an upward drift in AOV. The model needs scalars that encode "where are we in the post-2019 regime."

| Feature | Definition | Type |
|---|---|---|
| `years_since_2019` | `(date − 2019-09-01).days / 365.25` | CAL (deterministic for any date) |
| `months_since_2019` | `12 × years_since_2019` | CAL |
| `aov_trend_proj` | linear extrapolation of monthly AOV from 2019-09 → 2022-12 fit | PRIOR (frozen slope+intercept, evaluated by date) |
| `unit_price_trend_proj` | analogous for `avg_unit_price` | PRIOR |
| `n_active_skus_trend_proj` | analogous for SKU count | PRIOR |

These are honest because the slope/intercept are frozen at train time and the value at any date is `intercept + slope × t`. They give the tree splits a way to say "by 2024 the AOV regime is X" without needing dynamic state.

**Sample weighting:** linear ramp by month index inside the train window (oldest month weight 1.0, newest month weight 2.0). Weights recent quarters more heavily where the test horizon's drift has progressed further. Optional first iteration; turn on if mirror-LB peak metric improves.

---

## 13. Revised revenue expectation

The pre-timeline plan implicitly anchored on v10c's predicted ~$4.23 M / day. That number is **biased high** because v10c trained on 2017+ and never saw the post-cliff drift play out fully into 2024.

Projecting the 2019 → 2022 trend forward 18-30 months:

| Year | Approx revenue/day | Source |
|---|---|---|
| 2019 (full) | $5.21 M / 365 ≈ ~$3.37 M (rolling 12-mo at 2019-10) | timeline audit |
| 2022 (full) | $3.19 M (12-mo rolling at 2020-02 floor) → drift up via AOV → ~$3.0–3.2 M | timeline audit |
| **2024 projected** | **~$3.0 M / day** (volume drift continues, AOV rise partially compensates) | trend extrapolation |

| Submission | Predicted daily mean | Status |
|---|---|---|
| v10c | $4.23 M | scored 774,898 — best so far, but predictions likely high |
| v12d | $3.66 M | scored 871,299 — worse, but mean is closer to truth (peaks crushed, body OK) |
| **V13 target** | **$2.9–3.2 M** | what trend extrapolation says is honest |

This reframes what "good" looks like. **A V13 mirror-block prediction averaging ~$3 M / day is on-trend, not a regression.** A V13 prediction averaging $4 M / day is the v10c failure mode (over-projecting old-regime volume).

**Implication for blending:** the original §10 left open "may blend with v10c in prediction space." That option is now closed for the *level* of predictions — v10c is biased high. Blending only makes sense for *peak shape*, not for daily mean.

---

## 14. Updated build order

This replaces §9. Steps removed: COVID flag construction. Steps added: trend features, regime audit gate.

1. **Day 1 — calendar & event book.** Same as old §9. Still uses `python-holidays` for Tet. Verify event flags hit 2019-2024.
2. **Day 1 — frozen priors on the post-cliff window.** Compute `PRIOR_REV`, `PRIOR_COGS`, `AOV_PRIOR`, `EVENT_UPLIFT` from `sales.csv` restricted to **2019-09-01 → 2022-12-31**. Write `data/processed/v13_priors.parquet`.
3. **Day 1 — long-horizon lags on the post-cliff window.** Same-dow-prev-year and lag-364 columns, lookup window starts 2019-09-01. Write `data/processed/v13_lags.parquet`.
4. **Day 1 — trend features.** Fit linear slope+intercept for `aov_prior_by_month`, `avg_unit_price_by_month`, `n_active_skus_by_month` on the post-cliff window. Write coefficients to `data/processed/v13_trends.json`. Build `years_since_2019`, `months_since_2019` from date arithmetic at runtime.
5. **Day 1 — feature assembly.** `daily_features_v13.parquet` joining calendar + event book + priors + lags + trend features. Verify zero columns derived from web_traffic, reviews, returns, signups, stockouts, inventory, orders, items, payments. Verify no row has a value computed from data after 2022-12-31 (other than the trend extrapolation, which is by definition deterministic).
6. **Day 2 — validation harness.** Mirror block redrawn: hold out **2022-01-01 → 2022-12-31** (one full post-regime year, includes Tet 2022, Singles Day, Black Friday). Train on 2019-09-01 → 2021-12-31. Iteration folds: 4 expanding-origin folds inside 2020-2022 at 90-day horizon.
7. **Day 2 — V10c re-baseline on the new mirror.** Re-run V10c on the redrawn mirror block (it will likely score worse than its Kaggle 774k because the mirror is harder/closer to test). Whatever number it scores **is the new floor**.
8. **Day 2 — V13 baseline.** LightGBM, log1p + RMSE, debias-corrected `expm1`. Score on the new mirror. Compare to V10c re-baseline.
9. **Day 3 — peak-recovery layer.** Stage-2 quantile uplift + frozen multiplicative event uplift table. Re-score.
10. **Day 3 — Tweedie head-to-head.** Pick by mirror-block peak-day RMSE.
11. **Day 4 — only if mirror beats V10c re-baseline AND predicted daily mean ≈ $2.9–3.2 M — submit to Kaggle.**

---

## 15. Updated decision rules

The original §8 "Submission discipline" stays. Add three sanity gates derived from the timeline:

1. **Daily-mean sanity gate.** If V13's predicted mean over the test horizon falls outside [$2.7 M, $3.4 M]/day, do not submit until investigated. Outside this band means the model is either remembering the old regime (>$3.4 M) or has collapsed into mean-of-trough (<$2.7 M).
2. **Trend-feature ablation gate.** Train V13 with and without the trend features (§12.3). If "with" doesn't help on the mirror block, revisit whether the trends are real — that would mean the post-cliff drift is noise and we can simplify.
3. **2019-09 cut ablation gate.** Train V13 on (a) 2019-09+ only and (b) 2017+ with sample weighting. If (b) wins on the mirror block by a non-trivial margin, the cliff is less destructive than the audit suggests and we re-open the window. Default expectation: (a) wins.

Loose ends parked, not forgotten:
- **Customer turnover audit.** Confirm whether old `customer_id`s stop appearing in mid-2019 (which would prove the DTC-pivot story). Doesn't change V13 features, but informs follow-up modeling.
- **Product turnover audit.** Confirm whether specific high-revenue SKUs disappear in 2019. Same deal — explanatory, not load-bearing.
- **Promotion table extent check.** §2.8 / §2.9 still need verification that 2023+ promo records are NOT in the supplied data; if they are, switch from PRIOR to CAL.

---

## Appendix — V12 features dropped vs V13

**Dropped from V12 (35 columns):**
n_orders, n_unique_cust, pct_mobile, pct_desktop, pct_tablet, pct_cc, pct_cod, pct_paypal, n_items, n_line_items, items_gross, items_cogs_recon, avg_unit_price, items_per_order, payment_value_sum, mean_installments, sessions, unique_visitors, page_views, bounce_rate, avg_sess_dur, sess_direct, sess_email, sess_organic, sess_paid_search, sess_referral, sess_social, sess_unknown, pct_direct, pct_email, pct_organic, pct_paid_search, pct_referral, pct_social, n_reviews, rating_mean, n_low_ratings, n_returns, return_qty_sum, refund_sum, n_signups, n_stockouts, mean_fill_rate, n_active_promos, max_discount_active, n_stackable_active, aov.

**Kept from V12 (24 columns):**
year, month, day, dow, woy, day_of_year, is_weekend, is_dd, is_pd, is_tet, is_singles_day, is_twelve_twelve, is_nine_nine, is_cyber_monday, is_black_friday, is_womens_day_vn, is_intl_women_day plus the calendar mechanics.

**Added in V13 (~14 new columns):**
is_pre_tet_window, is_tet_recovery_window, tet_day_index, days_to_tet, days_since_tet, days_to_singles_day, days_to_twelve_twelve, days_to_black_friday, days_since_payday, prior_rev_by_month_dow, prior_cogs_by_month_dow, aov_prior_by_month, prior_rev_event_uplift, rev_same_dow_prev_year, rev_yoy_lag_364, cogs_same_dow_prev_year, cogs_yoy_lag_364.

---

## Sources

- [M5 Forecasting Accuracy 1st place writeup (YeonJun In)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/yeonjun-in-stu-1st-place-solution)
- [M5 Accuracy results paper (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [M5 Uncertainty hybrid LightGBM + DeepAR (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0169207022000097)
- [Sales forecasting in retail: M5 lessons (Artefact)](https://medium.com/artefact-engineering-and-data-science/sales-forecasting-in-retail-what-we-learned-from-the-m5-competition-445c5911e2f6)
- [Rossmann 1st place winner interview (Gert Jacobusse)](https://medium.com/kaggle-blog/rossmann-store-sales-winners-interview-1st-place-gert-jacobusse-a14b271659b)
- [Walmart Recruiting Store Sales Forecasting writeup](https://medium.com/analytics-vidhya/walmart-recruiting-store-sales-forecasting-kaggle-competition-856c72c9265a)
- [Favorita Grocery Sales review (Ceshine Lee)](https://medium.com/data-science/review-kaggle-corporaci%C3%B3n-favorita-grocery-sales-forecasting-part-i-9330b7350713)
- [TS-10: Validation methods for time series (Konrad Banachewicz)](https://www.kaggle.com/code/konradb/ts-10-validation-methods-for-time-series)
- [Forecasting: Principles and Practice §5.10 (Hyndman)](https://otexts.com/fpp3/tscv.html)
- [Reduce Bias in Time-Series CV with Blocked Split (TDS)](https://towardsdatascience.com/reduce-bias-in-time-series-cross-validation-with-blocked-split-4ecbfc88f5a4/)
- [Avoiding Data Leakage in Timeseries 101 (TDS)](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)
- [Train-serve skew in real-time ML models (Nubank)](https://building.nubank.com/dealing-with-train-serve-skew-in-real-time-ml-models-a-short-guide/)
- [Preventing Data Leakage in Feature Engineering (dotData)](https://dotdata.com/blog/preventing-data-leakage-in-feature-engineering-strategies-and-solutions/)
- [Target encoding leakage and out-of-fold cure (Brendan Hasz)](https://brendanhasz.github.io/2019/03/04/target-encoding.html)
- [Lag features for long-horizon forecasting (MS AutoML docs)](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automl-forecasting-lags?view=azureml-api-2)
- [Identifying and Overcoming Transformation Bias in Forecasting (arXiv)](https://arxiv.org/pdf/2208.12264)
- [LightGBM for Quantile Regression (TDS)](https://towardsdatascience.com/lightgbm-for-quantile-regression-4288d0bb23fd/)
- [LightGBM Parameters (Tweedie)](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [python-holidays library](https://pypi.org/project/holidays/)
- [Item2Vec embeddings (arXiv)](https://arxiv.org/abs/1603.04259)
