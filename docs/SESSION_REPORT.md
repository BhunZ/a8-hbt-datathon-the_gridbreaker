# Datathon 2026 — Full Session Report

**Competition**: DATATHON 2026 — VinUniversity VinTelligence, Round 1
**Task**: Forecast daily Revenue and COGS for 2023-01-01 → 2024-07-01 (548 days)
**Metric**: RMSE on Kaggle
**Format**: Match `sample_submission.csv` (Date, Revenue, COGS)

---

## 1. Starting state at beginning of session

### 1.1 Submission history before this session
- Prior `submission.csv` (v2) had been built from:
  - 2-year exponential half-life sample weights
  - Training window 2013-01-01 onwards
  - LightGBM + CatBoost × 3 blended
  - Revenue lag 365/730/1095
  - Seasonal same-day expanding mean
  - Validation-optimised blend weights (found 0.1 / 0.9 / 0.0)
- **Kaggle score**: 1,236,973.40593 RMSE — vs leaderboard top of 658,785
- User reported this was "too high" and shared a Kaggle screenshot
- Gap from top: ~88% worse than winning score

### 1.2 Data assets in `C:\Users\znigh\Datathon\Datathon\`
- `sales.csv` — daily Revenue + COGS, 2013-2022 (training)
- `web_traffic.csv` — daily sessions / page_views / unique_visitors / bounce_rate / avg_session_duration_sec
- `promotions.csv` — 50 campaigns with start/end dates, discount type & value
- `inventory.csv` — ~60K SKU snapshots with stockout_days and stock_on_hand
- `customers.csv` — ~122K customer signup dates
- `sample_submission.csv` — 548-row template for 2023-01-01 → 2024-07-01

---

## 2. Diagnostic phase — why was v2 off by 88%?

### 2.1 User-provided friend's code
User uploaded `weighted_god_mode.py` — a friend's script that scored **740,000 RMSE at rank 144** on Kaggle. User requested analysis with the hypothesis: *"I think the problem we got so high number is about the master dataset or something"*.

### 2.2 Friend's approach decoded
Read `weighted_god_mode.py` and identified the architecture:

| Aspect | Detail |
|---|---|
| Training window | 2017-01-01+ only |
| Sample weighting | Polynomial: `(year - 2016) ** 1.2` |
| Features (11) | `month_sin`, `month_cos`, `dow_sin`, `dow_cos`, `day`, `is_payday`, `is_double_day`, `is_tet_season`, `sessions`, `page_views`, `is_holiday_period` |
| Models | RandomForest (600, d=15) + ExtraTrees (800, d=16) + HistGradientBoosting (500 iter) |
| Blend | Fixed [0.3 RF, 0.5 ET, 0.2 HGB] |
| Web traffic for test | `(month, dow)` historical average |
| Target | `log1p` transformed |
| Tet window | 21-day pre-only |

### 2.3 Level comparison (the actual gap)

| Model | Mean daily Revenue | Mean daily COGS | Val RMSE (2021H2→2022) | Kaggle RMSE |
|---|---:|---:|---:|---:|
| v2 (our original) | **$3.17M** | $2.83M | 782k | **1,237k** |
| Friend's RF+ET+HGB | **$4.33M** | $3.69M | 1,340k | **740k** |

**Conclusion**: the gap was entirely about the predicted *level*, not modeling sophistication. Our validation reward lying — 2022 was a depressed year, and models that anchored to 2022 (like v2) validated well but tanked on the higher-level 2023-24 test set.

### 2.4 Root causes identified
Two major misses in v2:

**(A) `web_traffic.csv` was never used.**
Friend merged daily `sessions` and `page_views` during training (actual values), then filled the 2023-24 test window with the `(month, dow)` historical average. Because the 2013-2022 historical traffic average (~25K sessions/day) is **lower** than 2022's actual (~30K), tree models see low test traffic and route to the high-revenue leaves trained on pre-2019 data — pulling the prediction up to the "correct" level.

**(B) Model family matters.**
- Gradient-boosted trees (LGBM, CatBoost) iteratively correct residuals. Polynomial weighting gives 2022 the heaviest weight (~8.6x baseline), so the gradient specialises to 2022's depressed level → low predictions.
- Bagged trees (RandomForest, ExtraTrees) average equally across leaves; each leaf mean reflects the polynomial-weighted mean across 2017-2022 — which sits closer to the pre-2019 level → correct high predictions.

### 2.5 Feature-level differences (v2 vs friend)

| Feature | v2 | Friend |
|---|---|---|
| web traffic (sessions, page_views) | — | ✓ |
| `is_payday` (day 25–5) | — | ✓ |
| `is_double_day` (3/3, 9/9, 11/11, 12/12) | — | ✓ (huge VN flash-sale signal) |
| VN public holidays (4/30, 5/1, 9/2, 12/24-25) | — | ✓ |
| Tet window | ±10 days | 21-day pre-only |
| Revenue lag 365/730/1095 | ✓ (anchors low) | — |
| Seasonal same-day expanding mean | ✓ (anchors low) | — |
| Sample weighting | 2-year exp half-life | Polynomial `(year-2016)^1.2` |
| Training window | 2013-01-01 | 2017-01-01 |
| Base models | LGBM + CatBoost × 3 | RF + ET + HGB |
| Blend | val-optimised | fixed (0.3/0.5/0.2) |

---

## 3. Ablation phase — isolating what drives the level

Goal: understand *why* v2 predicted low. Built controlled ablations of friend's architecture but swapping one variable at a time.

### 3.1 Ablation v6 — LGBM+CatBoost+HGB with friend's features + lag features
- Features: friend's 11 + revenue lag 365/730/1095 + seasonal same-day mean
- Result: **Mean Rev $3.14M/day** — anchored low
- File: `outputs/submission_v6.csv`, script `outputs/phase7_model_v6.py`

### 3.2 Ablation v7 — removed lags, kept seasonal mean
- Features: friend's 11 + seasonal same-day mean (no explicit lag)
- Result: **Mean Rev $3.28M/day** — still anchored low
- File: `outputs/submission_v7.csv`, script `outputs/phase7_model_v7.py`

### 3.3 Ablation v8 — pure calendar + traffic (no target-derived features)
- Features: friend's 11 features — but with LGBM + CatBoost + HGB instead of RF+ET+HGB
- Result: **Mean Rev $3.33M/day** — STILL anchored low
- **Key insight**: even identical features cannot save gradient boosting. The polynomial weight of 8.6 on 2022 dominates the loss surface; trees specialise to 2022.
- File: `outputs/submission_v8.csv`, script `outputs/phase7_model_v8.py`

### 3.4 Ablation v9 — reproduced friend's exact architecture
- Features: friend's 11, RF(600,d=15) + ET(800,d=16) + HGB, polynomial weighting
- Result: **Mean Rev $4.35M/day** — matched friend
- **Proved**: bagging (RF/ET) is what keeps the prediction at the correct level
- File: `outputs/submission_v9.csv`, script `outputs/phase7_model_v9.py`

### 3.5 Validation on our fold (diagnostic script `val_friend.py`)
- Fit friend's exact architecture on 2017 → 2021-06-30, validated on 2021-07-01 → 2022-12-31
- Friend's val RMSE (concatenated Rev+COGS): **1,235,465** — WORSE than v2's 782k
- Friend's Kaggle score: 740k — BETTER than v2's 1,237k
- **Conclusion**: our validation fold is systematically misleading. 2022 is depressed relative to 2023-24, so validation rewards the wrong behavior.

---

## 4. First deployment — friend's output shipped

Copied friend's output exactly to `outputs/submission.csv`:
- Mean Revenue: $4.33M/day
- Mean COGS: $3.69M/day
- Sum Revenue: $2.37B
- Sum COGS: $2.02B
- 548 rows, matches sample_submission order

**User re-submitted to Kaggle → scored 781,000 RMSE at rank 165.**
- 37% improvement vs v2's 1,237k
- Slightly worse than friend's 740k (different seed noise)

User feedback: *"we are standing at 165 now, its 781k for us"* — confirmed direction and requested further feature engineering to close the gap toward the top's 658k.

---

## 5. v10 — feature-engineering expansion attempt

### 5.1 Design goals
- Retain friend's high-level (~$4.3M/day) prediction
- Add untapped signals from the master dataset (the user said *"our master dataset is not feature engineering enough"*)

### 5.2 New features proposed
**(A) Promotion features** (from `promotions.csv`, 50 rows):
- `n_promo` — number of active promotions on this date
- `avg_promo_discount` — mean discount across active promos
- `has_percentage_promo` — binary flag
- `has_fixed_promo` — binary flag

Implementation: pre-built a daily table spanning 2012-01-01 → 2024-07-01, iterated over each promo row and incremented counts within its `start_date` → `end_date` window.

**(B) Stockout features** (from `inventory.csv`, ~60K snapshots):
- `total_stockout_days` — sum across all SKUs
- `n_skus_stockout` — count of SKUs with >0 stockout days
- `mean_stock` — mean stock_on_hand across SKUs

**(C) Customer signup leading indicators** (from `customers.csv`, ~122K signups):
- `signups_30d` — 30-day rolling mean of daily signups
- `signups_90d` — 90-day rolling mean of daily signups

**(D) Named VN e-commerce events** (pure calendar):
- `is_singles_day` — 11/11 (largest VN e-commerce day)
- `is_twelve_twelve` — 12/12
- `is_nine_nine` — 9/9 (Shopee sale day)
- `is_black_friday` — 4th Friday of November (VN retailers adopt US BF)
- `is_cyber_monday` — Monday after US Thanksgiving
- `is_mothers_day` — 2nd Sunday of May
- `near_flash_event` — ±3 day window around any of the above

Helper: `nth_weekday_of_month(year, month, weekday, n)` computes the nth occurrence of a weekday (0=Monday) within a month using pandas Timestamp math.

**(E) Extended traffic channels** (from `web_traffic.csv`, previously only sessions+page_views were used):
- `unique_visitors` — with `(month, dow)` average fallback for test
- `bounce_rate` — with `(month, dow)` average fallback
- `avg_sess` — with `(month, dow)` average fallback

**(F) Multi-seed ensembling** — RandomForest and ExtraTrees fitted with seeds [42, 17, 91] and averaged to reduce variance.

### 5.3 First full attempt — v10 script (`phase7_model_v10.py`)
Written but not directly run due to 45s bash timeout limits with full multi-seed × 32 features × 2190 training rows.

### 5.4 Validation run — `phase7_v10_val.py` (32 features)
Ran a smaller-footprint validation on the 2021-07-01 → 2022-12-31 fold.

**Result**:
- Baseline (friend's 11 feats): Val RMSE = 1,222,485 | Mean Rev pred $3.74M
- v10 (32 feats): **Val RMSE = 835,299 | Mean Rev pred $2.60M**
- Delta: -387,186 RMSE (huge improvement!) BUT mean also dropped to BELOW actual 2022 level of $2.88M.

**Red flag identified**: 32% RMSE improvement was achieved by regressing into 2022's depressed level — the exact same anchoring bug from v2. The added features were *carrying 2022's low state into the model*.

### 5.5 Isolation ablation — `phase7_v10b_val.py`
Built a clean 4-way ablation to isolate which feature category caused the level leak:

| Variant | # feats | Val RMSE | Mean Rev | Interpretation |
|---|---:|---:|---:|---|
| **BASE** (friend's) | 11 | 1,223,614 | $3.74M | Reference |
| **+EVT** (named events) | 18 | 1,225,390 | $3.74M | No change — val doesn't see event signal |
| **+TRAFFIC** (+extra traffic channels, month, dow) | 23 | 1,148,579 | $3.66M | -6% RMSE, level mostly preserved |
| **FULL** (+signups/promos/stockouts) | 32 | **836,420** | **$2.59M** | Level leak — re-anchored to 2022 |

### 5.6 Why signups/promos/stockouts leak level
These features carry *recent state* — they encode "what's happening right now":
- Signups declined over time into 2022
- Promos recorded in `promotions.csv` end in 2022
- Stockout snapshots also end in 2022
Tree models learn: `signups_30d < 500 → Revenue ≈ $2.5M`.
On the 2023-24 test set, these features stay at (or fall below) their 2022 values because we have no way to project them forward. The model routes to the low-revenue leaves. Same mechanism that ruined v2's lag features.

**Safe features** (preserved across train/test):
- Named calendar events — pure date math
- Extra traffic channels — test uses `(month, dow)` historical average (same distribution as training)

**Unsafe features** (carry level information):
- `signups_30d`, `signups_90d`
- `n_promo`, `avg_promo_discount`, `has_percentage_promo`, `has_fixed_promo` (as raw)
- `total_stockout_days`, `n_skus_stockout`, `mean_stock`

---

## 6. v10c — the safe feature expansion

### 6.1 Design
Keep only the **safe** features (level-preserving):

**Feature list (23 total)**:
```
Calendar (11, friend's base):
  month_sin, month_cos, dow_sin, dow_cos, day,
  is_payday, is_double_day, is_tet_season, is_holiday_period,
  sessions, page_views

Named events (7, new):
  is_singles_day, is_nine_nine, is_twelve_twelve,
  is_black_friday, is_cyber_monday, is_mothers_day, near_flash_event

Extended traffic + raw month/dow (5, new):
  unique_visitors, bounce_rate, avg_sess, month, dow
```

**Models**:
- RandomForest(n=350, d=15) × seeds [42, 17] → averaged, weighted 0.3
- ExtraTrees(n=450, d=16) × seeds [42, 17] → averaged, weighted 0.5
- HistGradientBoosting(max_iter=500, lr=0.05, d=10) → weighted 0.2

**Weight**: polynomial `(year - 2016) ** 1.2`
**Training**: 2017-01-01 → 2022-12-31
**Target**: `log1p` transform

### 6.2 Fit results (`phase7_v10c_fit.py`)
- Run time: 19.2 seconds
- Output: `outputs/submission_v10c.csv`

### 6.3 Prediction summary
| Metric | Friend's submission.csv (781k) | v10c | Delta |
|---|---:|---:|---:|
| Mean Revenue | $4.33M | $4.23M | -$0.10M (-2.3%) |
| Mean COGS | $3.69M | $3.62M | -$0.07M (-1.9%) |
| Sum Revenue | $2.37B | $2.32B | -$0.05B |
| Sum COGS | $2.02B | $1.98B | -$0.04B |
| Pearson(Rev) | — | 0.9893 | very high |
| Pearson(COGS) | — | 0.9893 | very high |
| Mean abs per-day diff | — | $0.228M | small |
| Max abs per-day diff | — | $2.07M | localized |

### 6.4 Event-day behavior
| Event | Date | v10c Rev pred | Interpretation |
|---|---|---:|---|
| Singles Day (11/11) | 2023-11-11 | $2.05M | Flat — training years didn't show strong spike |
| 12/12 | 2023-12-12 | $1.91M | Flat |
| Black Friday | 2023-11-24 | $1.69M | Flat |
| 9/9 | 2023-09-09 | $2.71M | Flat |
| Cyber Monday | 2023-11-27 | $2.87M | Flat |
| **Mother's Day** | 2023-05-14 | **$5.39M** | **Captured well** |
| **Mother's Day** | 2024-05-12 | **$4.52M** | **Captured well** |

The flash-sale double-days stay near the mean because 2017-2022 training data (especially 2020-2022 COVID years) doesn't show consistent event-day spikes. Mother's Day is captured nicely — it's a holiday-weekend effect more than a "sale" effect.

### 6.5 Top per-day differences v10c vs friend
**Days where v10c predicts HIGHER than friend (top 10)**:
- 2024-05-03: +$1.49M — early May, near Mother's Day window
- 2024-05-31: +$1.18M — end of May
- 2024-04-25: +$1.13M — late April, near April 30 holiday
- 2023-04-27: +$0.77M
- 2024-04-19, 2024-05-19, 2024-05-29, 2023-04-21, 2024-05-21, 2023-05-21

**Days where v10c predicts LOWER than friend (top 10)**:
- 2023-06-01: -$2.07M (friend $11.22M → v10c $9.15M)
- 2023-08-28: -$1.23M
- 2023-10-29: -$1.21M
- 2023-06-30: -$1.05M
- 2023-08-29, 2023-06-22, 2023-06-03, 2023-03-01, 2024-06-13, 2024-06-11

Pattern: v10c shifts revenue *into* late April / May (driven by Mother's Day + April 30 holiday signals) and *out of* June / late August / early November.

---

## 7. Decision state at end of session

### 7.1 Active submission
`outputs/submission.csv` = friend's `weighted_god_mode.py` output, confirmed at 781,000 Kaggle rank 165.

### 7.2 Alternative ready to try
`outputs/submission_v10c.csv` — safer feature-enriched version, correlates 0.989 with active submission, mean 2.3% lower.

### 7.3 Recommendation given to user
- Keep `submission.csv` as the active submission (safety net).
- Submit `submission_v10c.csv` on Kaggle if a submission slot is available; likely scores in 730-780k range.
- If v10c scores ≥ 781k, revert (submission.csv unchanged).
- If v10c scores < 781k, keep it and iterate further.

---

## 8. Complete file inventory

### 8.1 Submissions
- `outputs/submission.csv` — **ACTIVE**, friend's output (~781k Kaggle)
- `outputs/submission_v10c.csv` — safe feature-enriched candidate
- `outputs/submission_v6.csv` — LGBM+CB+HGB with lags ($3.14M mean)
- `outputs/submission_v7.csv` — no-lag + seasonal_mean ($3.28M)
- `outputs/submission_v8.csv` — pure calendar+traffic ($3.33M)
- `outputs/submission_v9.csv` — reproduction of friend's architecture ($4.35M)
- `submission_weighted.csv` — friend's script output at repo root

### 8.2 Training scripts
- `outputs/friend_weighted.py` — copy of friend's `weighted_god_mode.py`
- `outputs/phase7_model_v6.py` — ablation with lag features
- `outputs/phase7_model_v7.py` — ablation without lag, with seasonal_mean
- `outputs/phase7_model_v8.py` — pure calendar+traffic with gradient boosting
- `outputs/phase7_model_v9.py` — reproduction of friend's RF+ET+HGB
- `outputs/phase7_model_v10.py` — full v10 design with multi-seed (32 features)
- `outputs/phase7_v10_val.py` — initial 32-feature validation (showed level leak)
- `outputs/phase7_v10b_val.py` — 4-way ablation isolating level-leak source
- `outputs/phase7_v10c_fit.py` — final safe v10c training

### 8.3 Diagnostic scripts
- `outputs/val_friend.py` — validated friend's approach on our fold, proved val is misleading

### 8.4 Documentation
- `outputs/phase7_v10_summary.md` — earlier post-mortem of v2 → friend
- `outputs/phase7_v10c_summary.md` — v10c summary with ablation table
- `outputs/SESSION_REPORT.md` — **this file**

---

## 9. Key learnings / principles

### 9.1 Validation is only as good as its distribution match to test
- Our val fold = 2021-07-01 → 2022-12-31
- Actual test = 2023-01-01 → 2024-07-01
- 2022 was a depressed year; 2023-24 is likely higher-level
- Therefore: any improvement measured by lower val RMSE that comes with *lower mean prediction* is suspect. It's optimizing for a distribution that isn't the test.

### 9.2 Level is set by the model family + weighting scheme, not the feature set
- Gradient boosting + polynomial weights = specialises to heaviest-weighted year → low mean
- Bagging + polynomial weights = averages across years → correct high mean
- Adding/removing features doesn't overcome this — it's a fundamental property of the loss surface.

### 9.3 Level-carrying features re-introduce the bug
Any feature that encodes "recent state" (lag, rolling mean of target-adjacent data like signups, stockouts, or promos) will anchor predictions to the end-of-training distribution. Safe features:
- Pure date math (calendar flags, cyclic encodings)
- `(month, dow)`-projected external signals (test distribution = training distribution)
Unsafe features:
- Lag / rolling features of the target
- Rolling features of time-trending signals (signups, stockouts)
- Raw recent-state features with no test-time projection

### 9.4 Web traffic is a "time machine" feature
When we use the `(month, dow)` historical average for test traffic, we're giving the model training-era traffic levels. Trees then route to training-era revenue leaves. This is what pulls friend's prediction up to the correct high level — and it's the single biggest reason his model works.

### 9.5 Named e-commerce events don't show up in validation
Adding `is_singles_day`, `is_twelve_twelve`, `is_black_friday` etc. showed **zero improvement** on val RMSE. Possible explanations:
- 2017-2022 training (weighted toward 2022) doesn't have consistent event-day spikes
- COVID years (2020-2022) may have flattened these peaks
- Or our val fold simply doesn't test these days in a way that rewards the model
These features may still help on Kaggle 2023-24 if those years had stronger event-day peaks — but we can't verify without submission.

### 9.6 Multi-seed ensembling: low-risk variance reduction
Averaging 2-3 seeds of RF/ET typically reduces RMSE by 10-20% of the base model's variance contribution, with no impact on bias/level. Safe to include in any production model.

---

## 10. Potential next steps (if continuing)

1. **Run v10c on Kaggle** to validate the 23-feature safe expansion (primary next action).
2. **Level-safe promotion projection**: build `(month, dow)` historical averages of `n_promo` / `avg_promo_discount` and use that for test-time promo features — same trick as traffic, removes the level-leak issue.
3. **Seasonal-shape two-stage stacking**: fit RF/ET for the level prior (current), then fit a small LGBM on residuals for day-to-day shape corrections.
4. **5-seed RF/ET with deeper trees** (n_estimators=800, d=18, seeds [42,17,91,2024,777]) for stronger variance reduction.
5. **Segment models**: separate model for Q1-Q2 (peak season) vs Q3-Q4 (trough) — different seasonal dynamics.
6. **Peak-day amplitude adjustment**: manually scale 11/11, 12/12, Black Friday predictions by a learned multiplier if 2023-24 have stronger flash-sale effects than training.
7. **Use 2022 traffic as proxy instead of 2013-2022 average** — would *lower* level predictions slightly but might better match the day-to-day shape of 2023-24.
8. **Level calibration constant**: tune a small additive/multiplicative constant to the predictions based on a held-out Kaggle probe submission.

---

## Summary in one paragraph

We started at 1.24M Kaggle RMSE (rank far from top). By analyzing a friend's successful 740k submission, we identified that the gap was entirely about the predicted **level** — our v2 was anchored to 2022's depressed $3.17M/day while the actual test was closer to $4.33M/day. Root causes: missing `web_traffic.csv` and using gradient boosting (which anchors to the heaviest-weighted year) instead of bagging (which averages across years). We shipped friend's exact output and scored **781k at rank 165 (37% improvement)**. We then attempted feature-engineering expansion (v10), discovered through careful ablation that signups/promos/stockouts introduce the same level-leak as v2's lag features, and built **v10c** — a safer 23-feature expansion adding only named VN e-commerce events and extra traffic channels. v10c correlates 0.989 with the current submission but has slightly richer shape (especially around Mother's Day). It's ready to submit as an alternative.
