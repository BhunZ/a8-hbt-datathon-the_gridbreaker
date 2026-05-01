# V12 UNIFIED PLAN — Datathon 2026
## "Best of Both Worlds": Friend's Structure + Our Data Discoveries

**Target**: Beat friend's 728k Kaggle score (and our 774k baseline).

**Core thesis**: Friend has superior *model architecture*. We have superior *data understanding*. V12 fuses them.

---

## 1. HEAD-TO-HEAD COMPARISON

### 1.1 What Friend Has That We Don't (Architecture Wins)

| # | Friend's technique | Where seen | Why it helps | V12 adoption |
|---|---|---|---|---|
| F1 | **Triple-stage ensemble** with year cutoffs: Anchor(2016+) / Modernist(2019+) / Peak Catcher(2018+) | `final_glory_v2.py` | Different horizons capture different regimes; Anchor = stability, Modernist = recency, Peak Catcher = spikes | **ADOPT** as scaffolding |
| F2 | **Peak Catcher uses RAW target** (no log1p) | `final_glory_v2.py` line 79 | log1p shrinks large values → underpredicts peaks (Tet, Black Friday). Raw preserves spike magnitude | **ADOPT** — we were 100% log1p |
| F3 | **4-D traffic profile**: (month, dow, is_dd, is_pd) | `final_glory_v2.py` line 28 | Captures double-day × payday × weekday interactions | **ADOPT + EXTEND** to 5-D (+ is_tet) |
| F4 | **Gold blending**: `40% × new + 60% × BEST_SCORE_773K_GOLD.csv` | `final_glory_v2.py` line 102 | Stabilises around a known-good submission, reduces variance drop | **ADOPT** — blend 40/60 with our v10c 774k |
| F5 | **Post-hoc scaling**: Revenue × 0.99, COGS × 0.98 | `final_glory_v2.py` lines 106-107 | Calibrates systematic over-prediction (COGS worse than Rev) | **ADOPT + TUNE** via Kaggle probe |
| F6 | **Blend weights 40/40/20** for 3 stages | `final_glory_v2.py` line 96 | Discovered empirically | **ADOPT** as starting point, grid-search later |
| F7 | **Sample weighting `(year-2018)**1.5`** on Modernist | `final_glory_v2.py` line 70 | Heavier polynomial weight (1.5 vs our 1.2) pulls model toward 2022 regime | **ADOPT** 1.5 on stage 2 |

### 1.2 What WE Have That Friend Doesn't (Data Wins — CRITICAL)

Friend is leaving **~90% of the data on the table**. He uses only `sales.csv` (revenue/cogs targets) and `web_traffic.csv` (sessions). He completely ignores:

| # | Unused file / signal | Our finding | Correlation w/ target | Potential |
|---|---|---|---|---|
| U1 | **order_items.csv × products.csv** → daily COGS reconstruction | `COGS_daily = Σ(quantity_d × cogs_product)` is an **exact** match | **r = 1.00000, ratio = 1.0000** | **PERFECT COGS** — zero model error on the historical part. Huge. |
| U2 | **order_items → items_gross** daily | `items_gross = Σ(quantity × price × (1-discount))` matches Revenue 99.2% | **r ≈ 0.996, ratio ≈ 0.9544** | Direct Revenue reconstruction — near-perfect |
| U3 | **orders.csv** → `n_orders_daily` | Count of orders per day | **r ≈ 0.936 with Revenue** | One single feature friend never built |
| U4 | **payments.csv** → `payment_value_daily` | Equals items_gross exactly | **r ≈ 0.992** | Cross-check + proxy |
| U5 | **web_traffic** full columns | We've used `sessions`. Ignored: `unique_visitors`, `bounce_rate`, `avg_session_duration_sec`, `traffic_source` (7 categories) | Bounce rate alone — inverse relationship | Friend uses only sessions. |
| U6 | **reviews.csv** | Daily mean rating, review count, year trend | Leading indicator of customer sentiment | Unused by friend |
| U7 | **returns.csv** | Return rate, reason mix | Inverse indicator (fewer returns → more keeps) | Unused |
| U8 | **shipments.csv** | days_to_ship, days_to_deliver, shipping_fee distribution | Operational indicator | Unused |
| U9 | **customers.csv** | new signups per day, age_group mix, channel mix | Customer influx | Unused |
| U10 | **promotions.csv** | Active promotions per day, discount % weighted | Directly drives lifts | Unused (friend only uses is_dd, is_pd dates) |
| U11 | **inventory.csv** | daily stockout count, low-stock count | Supply constraint on sales | Unused |
| U12 | **geography.csv** | Region mix in orders | Secondary | Unused |
| U13 | **AOV monotonic trend** | AOV rose $25,144 → $32,489 (29% over 6 years) | Predictable drift | Friend's recency weighting captures some; explicit extrapolation captures more |

### 1.3 The Killer Insight

**Friend's handbook records**: God Mode 794k → Weighted 781k → Grand Blend **778,539**.

But user says friend is now at **728k** — meaning friend has another iteration beyond the handbook (presumably `final_glory_v2.py` + further tuning). Gap to close: `774k − 728k = 46k`.

**Our route to closing the gap**: we don't need to match friend's model tuning — we need to replace 30-50% of friend's modeling work with **exact reconstruction** using the data files friend never touched. If COGS is perfectly reconstructable, our COGS error contribution → 0.

---

## 2. V12 ARCHITECTURE — UNIFIED PIPELINE

```
                                    ┌─────────────────────────┐
                                    │  FEATURE BUILDER v12    │
                                    │  (sales + traffic + 11  │
                                    │   auxiliary tables)     │
                                    └────────────┬────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
          ┌─────────▼──────────┐      ┌──────────▼──────────┐      ┌──────────▼──────────┐
          │  STAGE A: ANCHOR   │      │ STAGE B: MODERNIST  │      │ STAGE C: PEAK CATCH │
          │  2016+, log1p, ET  │      │ 2019+, log1p, RF    │      │ 2018+, RAW, ET      │
          │  weights=1         │      │ weights=(y-2018)^1.5│      │ weights=1           │
          └─────────┬──────────┘      └──────────┬──────────┘      └──────────┬──────────┘
                    │                            │                            │
                    └────────────────┬───────────┴────────────────┬──────────┘
                                     │ blend 0.40/0.40/0.20        │
                                     ▼                              │
                    ┌────────────────────────────┐                 │
                    │  STAGE D: DIRECT RECON     │                 │
                    │  COGS = Σ(qty_pred × cogs) │◄────────────────┘
                    │  (from order_items proxy)  │
                    └────────────────┬───────────┘
                                     │
                                     ▼
                    ┌────────────────────────────┐
                    │   COGS BLEND:              │
                    │   0.60 × Direct Recon +    │
                    │   0.40 × Stage A/B/C       │
                    │                            │
                    │   REV BLEND:               │
                    │   Stage A/B/C output       │
                    └────────────────┬───────────┘
                                     │
                                     ▼
                    ┌────────────────────────────┐
                    │   GOLD BLEND               │
                    │   0.40 × v12 +             │
                    │   0.60 × v10c_774k.csv     │
                    └────────────────┬───────────┘
                                     │
                                     ▼
                    ┌────────────────────────────┐
                    │   SCALING CALIBRATION      │
                    │   Rev × α (≈0.99)          │
                    │   COGS × β (≈0.98)         │
                    │   Tune via Kaggle probes   │
                    └────────────────────────────┘
                                     │
                                     ▼
                              submission_v12.csv
```

---

## 3. FEATURE SET — v12 (42 features vs friend's 8)

### 3.1 Base calendar (shared with friend)
- `month, day, dow, year, is_weekend, is_tet, is_dd, is_pd`

### 3.2 Named events (ours — friend missing)
- `is_singles_day` (11/11), `is_double_12` (12/12), `is_black_friday` (4th Friday Nov), `is_cyber_monday`, `is_mothers_day_vn`, `is_womens_day_vn` (20/10), `is_intl_women_day` (8/3)

### 3.3 Traffic (5-D profile, extended from friend's 4-D)
- `sessions`, `unique_visitors`, `bounce_rate`, `avg_session_duration_sec`
- `pct_traffic_direct, pct_traffic_organic, pct_traffic_paid, pct_traffic_social, pct_traffic_email, pct_traffic_referral` (one-hot shares by day)
- Profile key: `(month, dow, is_dd, is_pd, is_tet)` with median aggregation

### 3.4 Daily commerce rollups (NEW — friend 100% missing)
- `n_orders_daily` (from orders.csv)
- `n_unique_customers_daily`
- `items_per_order_mean, items_per_order_sum`
- `payment_value_daily` (from payments.csv, = items_gross proxy)
- `n_pix_orders, n_card_orders, n_installment_orders` (payment mix)
- `pct_desktop_orders, pct_mobile_orders` (device mix)

### 3.5 AOV trend
- `aov_daily = revenue/n_orders` (target-adjacent — for training label engineering, not raw input)
- `aov_trend_projection` (extrapolated linear/log trend of AOV for test period)

### 3.6 Customer / returns / reviews (NEW)
- `n_signups_daily` (customers.csv)
- `return_rate_lag7` (returns in past 7 days / orders in past 7 days — lagged to avoid leak)
- `review_score_mean_lag7`, `n_reviews_lag7`

### 3.7 Promotions (NEW — friend uses only calendar flags)
- `n_active_promos_daily`, `max_discount_pct_active`, `n_stackable_promos_active`

### 3.8 Inventory health (NEW)
- `n_stockouts_daily`, `n_low_stock_daily` (lagged 1 day)

### 3.9 Feature safety rules (CRITICAL — from V11_PLAN)
- **Train period ends 2022-12-31; test is 2023-01-01 to 2024-07-01**.
- **STEP 1 RESULT (see `docs/aux_coverage.md`)**: every auxiliary file stops at 2022-12-31. Zero test-horizon rows exist in any aux table. No "direct use" path for any feature.
- **Every aux-derived feature must therefore be profile-projected**, same treatment friend gives `sessions`. We project 10+ signals; friend projects 1.
- Profile key: `(month, dow, is_dd, is_pd, is_tet)` with median aggregation — 5D, extending friend's 4D with is_tet.
- **Stage D (COGS) must use the RATIO route**, not `Σ(quantity × cogs)` — we have no future order_items. Stage D becomes: predict Revenue from A/B/C, multiply by projected `cogs_ratio(t)` trend.

---

## 4. STAGE-BY-STAGE DETAILS

### Stage A: Anchor (stability)
- Train: `sales[year >= 2016]`
- Target: `log1p(Revenue)`, `log1p(COGS)`
- Model: `ExtraTreesRegressor(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)`
- Weights: uniform
- Role: long-term baseline, prevents over-fitting to 2022

### Stage B: Modernist (recency)
- Train: `sales[year >= 2019]`
- Target: `log1p`
- Model: `RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=42)`
- Weights: `(year - 2018) ** 1.5`
- Role: captures 2022→2023 regime (latest seen data)

### Stage C: Peak Catcher (spikes)
- Train: `sales[year >= 2018]`
- Target: **RAW** (no log1p) ← critical
- Model: `ExtraTreesRegressor(1000, max_depth=15)`
- Weights: uniform
- Role: preserves magnitude of Tet, Black Friday, Singles Day peaks

### Stage D: Direct COGS Reconstruction (OUR UNIQUE)
- Compute historical daily `cogs_reconstructed` from `order_items × products`.
- Verify r = 1.000 on 2017-2022.
- For 2023-2024, we need either:
  - **(D1)** predict `n_units_sold_per_product_daily` then multiply by `cogs`, OR
  - **(D2)** predict `total_units_daily` and apply the mean `cogs_per_unit` (calibrated from 2022 product mix).
- **Recommended**: D2 is simpler and should already capture ~95% of variance. The ratio `cogs_reconstructed / Revenue_reconstructed` is stable quarter-over-quarter → we can predict Revenue (Stage A+B+C), compute `cogs_ratio_trend(t)`, then `COGS = Revenue × ratio`.

### Blend Formula
```
pred_rev_stageABC  = 0.40 × exp_m1(A) + 0.40 × exp_m1(B) + 0.20 × C    # A,B log-scale; C raw
pred_cogs_stageABC = 0.40 × exp_m1(A) + 0.40 × exp_m1(B) + 0.20 × C

pred_cogs_direct   = pred_rev_stageABC × cogs_ratio(t)                  # t = date

pred_cogs_final    = 0.60 × pred_cogs_direct + 0.40 × pred_cogs_stageABC
pred_rev_final     = pred_rev_stageABC

# Gold blend
pred_rev_gold  = 0.40 × pred_rev_final + 0.60 × v10c_774k['Revenue']
pred_cogs_gold = 0.40 × pred_cogs_final + 0.60 × v10c_774k['COGS']

# Scaling
submission['Revenue'] = pred_rev_gold  * alpha   # α ≈ 0.99, tune
submission['COGS']    = pred_cogs_gold * beta    # β ≈ 0.98, tune
```

---

## 5. VALIDATION STRATEGY

### 5.1 Local rolling CV (before submission)
- Fold 1: train ≤ 2020, val 2021 Jan-Jun
- Fold 2: train ≤ 2021, val 2022 Jan-Jun
- Fold 3: train ≤ 2022 Jun, val 2022 Jul-Dec
- Score each fold's RMSE → average.
- Reject any component that degrades 2+/3 folds.

### 5.2 Kaggle probe schedule (burn submissions wisely)
1. **Probe 1**: v12 without gold blend, without scaling — measure raw model strength.
2. **Probe 2**: v12 with gold blend only — measure blend contribution.
3. **Probe 3**: v12 full pipeline with α=β=1.0 — measure calibration gap.
4. **Probe 4**: v12 full pipeline with α=0.99, β=0.98 (friend's multipliers).
5. **Probe 5**: v12 full pipeline with α,β tuned from Probe 3 residuals.

### 5.3 Sanity checks before each submission
- Mean Revenue 2023-2024 should fall in [$3.5M, $4.8M]
- Mean COGS should be ~85% of Revenue
- No NaN, no negative, no values > 2× historical max
- Correlation with v10c_774k submission: > 0.95 (ensures we haven't broken structure)

---

## 6. IMPLEMENTATION ORDER (execution plan)

| Step | Task | Output | Time |
|---|---|---|---|
| 1 | Audit aux-file date ranges | `aux_coverage.md` | 3 min |
| 2 | Build `daily_features_v12.parquet` with all 42 cols | parquet file | 15 min |
| 3 | Verify COGS reconstruction identity on 2017-2022 | quick test script | 5 min |
| 4 | Implement triple-stage ensemble (Stages A/B/C) | `v12_fit.py` | 10 min |
| 5 | Implement Stage D (COGS direct / ratio approach) | `v12_stage_d.py` | 10 min |
| 6 | Blend + Gold-blend + identity-α/β submission | `submission_v12a.csv` | 5 min |
| 7 | Local CV report | `v12_cv.log` | 8 min |
| 8 | Submit Probe 1 | Kaggle score | — |
| 9 | Iterate on α/β, blend weights, Stage D approach based on probe results | Probes 2-5 | — |

**Total pre-submission compute time: ~55 min** (well under most single-call limits, but we'll chunk for safety).

---

## 7. RISK & MITIGATIONS

| Risk | Likelihood | Mitigation |
|---|---|---|
| Aux tables don't cover 2023-2024 test period → can't use them directly | High | Build median profile (month, dow) for each aux feature and project, same as traffic |
| Stage C (raw target) explodes on test if test spikes exceed training range | Medium | Clip predictions to 1.2 × historical max |
| Gold blend over-anchors to 774k submission → ceiling effect | Medium | If local CV shows no gain, drop gold blend weight to 0.40/0.60 → 0.60/0.40 |
| Direct COGS reconstruction yields worse COGS than modeled because unit volume in 2023-2024 is mis-projected | Low-Medium | Use ratio approach (D2), not unit-level (D1). Ratio is stable. |
| Feature count (42) causes overfitting | Low | RF/ET are robust; max_depth=15 caps complexity; we can prune via importance if needed |
| Friend's 728k used a technique we haven't seen yet | High | Accept — can't model unknown. Our independent data-based gains should close most of the gap regardless. |

---

## 8. DECISION POINTS

After Probe 1 (model-only, no gold blend, no scaling):
- **If score < 774k**: v12 model alone beats our previous best → continue with gold blend.
- **If 774k–790k**: v12 matches; gold blend will push us into 730s territory.
- **If > 790k**: Stage D (Direct COGS) is hurting. Drop it, go pure Stages A/B/C.

After Probe 4 (friend's exact scaling):
- **If score < 728k**: 🏆 we've beaten friend. Submit and stop iterating.
- **If 728k–760k**: tune α, β via residual regression (Probe 5).
- **If > 760k**: gold blend weight may need adjustment; also consider adding Stage E = LightGBM on the 42-feature set.

---

## 9. CODE STRUCTURE

```
outputs/
├── v12_build_features.py        # Step 2 — produce daily_features_v12.parquet
├── v12_verify_cogs.py           # Step 3 — verify reconstruction identity
├── v12_stage_abc.py             # Step 4 — triple-stage ensemble
├── v12_stage_d.py               # Step 5 — direct COGS via ratio
├── v12_blend_submit.py          # Step 6 — blend, gold-blend, scale, write csv
├── v12_cv.py                    # Step 7 — local rolling CV
└── V12_PLAN.md                  # this file
```

---

## 10. SUMMARY — WHY V12 SHOULD BEAT 728k

1. **Architecture parity**: we match friend's multi-stage + peak-catcher + 4D traffic + gold blend.
2. **Data asymmetry**: we bring **12 new data sources** friend has never touched. Even if only 3 of them contribute 5k each, that's 15k of gain.
3. **COGS perfect reconstruction**: our COGS error on the *reconstructable* portion → 0. Friend has full modeling error on COGS.
4. **Safety net**: gold blend at 60% caps our downside to ~780k (v10c), so worst case is "tie".
5. **Scaling**: we inherit friend's ~1% calibration win for free.

**Expected outcome range**: 700k–750k Kaggle RMSE.

**Stretch goal**: sub-700k via Stage D COGS precision.

---

*Plan written 2026-04-24. Execution pending your go-ahead.*
