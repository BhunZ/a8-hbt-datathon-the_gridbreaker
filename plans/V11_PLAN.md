# V11 Plan — closing the 46k gap (774k → target <728k)

## Executive summary

Our current submission (friend's architecture, 23-feature set in v10c) scores **774k**.
Friend's latest iteration scores **728k** — a 46k gap.

After a deep audit of the Datathon folder, we found **we have been ignoring ~90% of the available data**. Only `sales.csv`, `web_traffic.csv`, `promotions.csv`, `inventory.csv`, and `customers.csv` were used. The folder also contains:

| File | Size | Rows | Status |
|---|---:|---:|---|
| **orders.csv** | 46 MB | 646,945 | **Not used — biggest miss** |
| **order_items.csv** | 24 MB | 714,669 | Not used |
| payments.csv | 18 MB | 646,945 | Not used |
| shipments.csv | 20 MB | 566,067 | Not used |
| reviews.csv | 7 MB | 113,551 | Not used |
| returns.csv | 2 MB | 39,939 | Not used |
| products.csv | 195 KB | 2,412 | Not used |
| geography.csv | 1 MB | 39,948 | Not used |

The v11 plan is organized around extracting signal from these files in a **level-safe** way (same `(month, dow)` projection trick that made friend's traffic features work).

---

## 1. Diagnostic — why is there still a 46k gap?

### 1.1 Three possible explanations

**(A) Friend iterated since sharing — most likely.** The `weighted_god_mode.py` we copied scored 740k when shared. Now he's at 728k. He's almost certainly added features or improved ensembling.

**(B) Seed/version noise on our run.** RF/ET bootstrapping is non-deterministic across sklearn versions and threading. A 6k spread just from seed drift is plausible. But 34k+ is too big to be pure noise.

**(C) Our v10c dropped level slightly** ($4.33M → $4.23M, -2.3%). If the true test mean is closer to $4.4M, our level drop ate 10-30k of RMSE.

**Most likely combination**: (A) + (C) explain most of the gap. Friend is ahead because he's using signals we aren't.

### 1.2 What we saw in the data audit

Three signals we never touched that correlate strongly with Revenue:

| Signal | Source | Correlation with Revenue | Where we stand |
|---|---|---:|---|
| `n_orders` (daily) | orders.csv | **0.936** | Not used |
| `total_quantity` (daily) | order_items.csv | **0.918** | Not used |
| `sessions` (daily) | web_traffic.csv | ~0.60 | Used |

**n_orders is >50% stronger signal than sessions, and we haven't used it.**

### 1.3 The AOV (average order value) story

The per-order value has drifted monotonically up for 6 years:

| Year | avg Revenue per order | n_orders per day |
|---|---:|---:|
| 2017 | $25,144 | 208 |
| 2018 | $26,617 | 190 |
| 2019 | $27,326 | 114 |
| 2020 | $30,232 | 95 |
| 2021 | $30,211 | 95 |
| 2022 | $32,489 | 99 |

**Implication**: total revenue stayed high despite order-count collapse because prices rose. Linear-trending AOV into 2023-24 projects ~$33,500 → $34,600, so even with modest order counts, Revenue could be HIGHER than 2022. Our current model, which treats year categorically, cannot extrapolate this trend.

### 1.4 Traffic source decomposition (also missed)

`web_traffic.csv` has a `traffic_source` column with 6 sources: `organic_search`, `paid_search`, `social_media`, `email_campaign`, `direct`, `referral`. Friend's code sums across all sources, losing channel-specific signal. Social media traffic in particular has grown 50% since 2013; paid search has declined. These trends likely affect AOV and conversion differently.

---

## 2. V11 architecture — six improvement prongs

### Prong A — Daily order-level features (HIGH IMPACT)
Extract from `orders.csv` aggregated to daily, then project test days via `(month, dow)` historical averages (the same time-machine trick that worked for traffic).

**New features (11)**:
- `n_orders` — order count, r=0.936 with Revenue
- `n_customers` — unique customers ordering that day
- `pct_delivered` — share of orders marked delivered (vs cancelled/returned)
- `pct_returned` — share marked returned
- `pct_mobile` — share placed on mobile device
- `pct_desktop` — share placed on desktop
- `pct_paid_search`, `pct_organic`, `pct_social`, `pct_direct`, `pct_email` — source mix

**Projection for test**: each feature's `(month, dow)` average over 2013-2022.

**Expected impact**: 20-50k RMSE improvement. `n_orders` alone is a 0.936-correlated signal; even moderate use should help.

**Risk**: low. Same projection mechanism as traffic; level-safe.

### Prong B — Daily order-items features (MEDIUM IMPACT)
Extract from `order_items.csv`.

**New features (4)**:
- `total_quantity` — sum of units sold per day, r=0.918 with Revenue
- `mean_unit_price` — AOV proxy, rising trend
- `total_discount_amount` — aggregate discount given per day
- `pct_discounted_items` — share of line-items with >0 discount

**Projection**: `(month, dow)` avg except `mean_unit_price` which gets a **trend-extrapolated** value (see Prong D).

**Expected impact**: 10-30k additional RMSE improvement.

### Prong C — Product category mix (MEDIUM IMPACT)
Join `order_items.csv` × `products.csv` on `product_id` to get daily revenue per category / segment. Then project via `(month, dow)` avg.

**New features (6-10)**:
- Per-category daily revenue share (e.g., `pct_rev_streetwear`, `pct_rev_luxury`, `pct_rev_everyday`)
- Per-segment daily revenue share
- `n_unique_categories` — diversity metric

**Expected impact**: 5-20k. Smaller than A/B but valuable for shape — peak-season category shift.

### Prong D — AOV trend extrapolation (STRUCTURAL FIX, HIGH IMPACT)
The single biggest insight from the audit: `mean_unit_price` is **monotonically increasing**, and a `(month, dow)` average would incorrectly pull it down for 2023-24.

**Fix**: fit a simple model (linear regression or exponential) on yearly AOV means, project into 2023-24.

Yearly AOV: 2017=$25,144 → 2022=$32,489 → projected 2023≈$33,600, 2024≈$34,700.

**Feature added**: `aov_projected` — the extrapolated per-year AOV at each test date.

**Expected impact**: directly addresses level — if the true test level is above our current $4.23M, this feature should lift it. Could be worth 20-60k.

**Risk**: medium. If AOV plateaus in 2023-24 (not continues linearly), extrapolation over-shoots.

### Prong E — Review + return health signals (LOW-MED IMPACT)
From `reviews.csv` and `returns.csv`.

**New features (4)**:
- `rating_30d_avg` — 30-day rolling avg rating (sentiment)
- `rating_90d_avg` — longer trend
- `return_rate_30d` — returns / orders, 30-day window
- `review_count_30d` — review volume (proxy for engagement)

**Projection**: `(month, dow)` historical average.

**Expected impact**: 5-15k. Review ratings are stable at ~3.93 across years, so signal is weak. Return rates similarly flat at ~6%.

**Risk**: low. If features carry no signal, tree splits simply ignore them.

### Prong F — Stronger ensembling (LOW-MED IMPACT)
Current: fixed-weight blend [0.3 RF, 0.5 ET, 0.2 HGB].

**Upgrade options**:
1. **Multi-seed averaging** — 5 seeds of RF/ET (currently 2) for variance reduction. +3-7% RMSE reduction typical.
2. **Add LGBM as a residual stage** — after RF/ET/HGB predict the level, LGBM predicts residuals on polynomial-weighted 2022 data. Catches fine-grained weekly patterns.
3. **Blend weight optimization** — use OOF CV (not our misleading val fold) to tune blend weights. Target equal-per-year CV rather than single holdout.
4. **Isotonic calibration** — fit a monotonic function to align prediction distribution with actual.

**Expected impact**: 5-20k (small but reliable).

### Prong G — Kaggle probe calibration (TACTICAL, HIGH CONFIDENCE)
Use 2 spare Kaggle submissions to pin down the optimal level multiplier:

1. Submit v11_a: friend-style with multiplier 1.00
2. Submit v11_b: friend-style with multiplier 1.05 (5% higher)

The Kaggle scores reveal which direction is closer to true. If 1.05 is better, we know test mean is higher than our prediction; calibrate accordingly. If 1.00 is better but still suboptimal, try 0.98 next round.

**Expected impact**: 20-80k — depends on how far off we are.

**Risk**: very low. Just a scalar multiplier.

---

## 3. Execution order — most-to-least impact

| # | Prong | Expected Δ | Risk | Effort |
|--:|---|:-:|:-:|:-:|
| 1 | **A — daily order features** | 20-50k | Low | Medium |
| 2 | **D — AOV trend extrapolation** | 20-60k | Medium | Low |
| 3 | **G — Kaggle probe calibration** | 20-80k | Very low | 2 submits |
| 4 | B — order-items features | 10-30k | Low | Low |
| 5 | C — product category mix | 5-20k | Low | Medium |
| 6 | F — stronger ensembling | 5-20k | Low | Low |
| 7 | E — review + return signals | 5-15k | Low | Low |

**Best first iteration (v11 = A + B + D + F)**: 40-130k potential improvement. Most signal for moderate effort.

**If that scores <740k, add C, then E, then calibrate with G.**

---

## 4. Feature safety check — will these leak level like v10's did?

Recall: v10 FULL (with `signups_30d`, `n_promo`, `stockouts`) crashed the predicted mean from $3.74M to $2.59M because these features **encode recent state** and the model anchored to 2022's low.

### Safe features (v11):
- `n_orders`, `n_customers` (**projected** via month-dow avg) — SAFE. Same mechanism as traffic.
- `pct_mobile`, `pct_paid_search`, etc. (**projected**) — SAFE.
- `mean_unit_price` (**trend-extrapolated**, not month-dow avg) — SAFE and smartly extrapolating.
- `total_quantity` (**projected**) — SAFE.
- Category revenue shares (**projected**) — SAFE.
- Review rating / return rate (**projected**) — SAFE.

### DANGER features to avoid:
- Raw `n_orders` with fill-zero for test — would re-introduce level leak.
- Rolling means of Revenue or COGS — same lag-feature trap as v2.
- Any feature with 2022-trailing information that we fill with 0 or median for test.

**Rule**: every feature used in v11 must answer "how do you fill this for test dates?" The two valid answers are:
1. `(month, dow)` historical average (preserves distribution), or
2. Trend extrapolation (for monotonic features like AOV).

Anything filled with 0/mean/median without a proper projection is a potential level leak.

---

## 5. Validation strategy — since our val fold is misleading

Our 2021H2 → 2022 fold **systematically rewards low-level predictions** because 2022 is depressed. We've proven this (friend's val RMSE was 1.34M but Kaggle 740k).

**V11 validation overhaul**:

### 5.1 Multi-fold rolling CV
Instead of single holdout, do 5 folds:
- Fold 1: train 2017-2019 → val 2020 (COVID, depressed)
- Fold 2: train 2017-2020 → val 2021 (recovery, mid)
- Fold 3: train 2017-2021 → val 2022 (latest)
- Fold 4: train 2018-2019 → val 2020H2 (shorter recent)
- Fold 5: train 2013-2018 → val 2019 (pre-COVID, high)

Average RMSE across folds. This reveals features that help on high-level years (closer to 2023-24 conditions) vs just on 2022.

### 5.2 Level-distribution check
For every candidate model, print:
- Mean prediction on test
- Per-quantile prediction (25th, 50th, 75th, 95th)
- Standard deviation

Any model whose mean falls below $4.1M or above $4.5M should get extra scrutiny (these bracket friend's proven 740k range).

### 5.3 Out-of-sample sanity test
Fit v11 on 2017-2020, predict 2021. Check if predictions match actual 2021 within reasonable error. Repeat fit 2017-2021, predict 2022. This is a fair-game test that rewards a model that can extrapolate one year forward.

---

## 6. Implementation file structure

```
outputs/
  phase7_v11a_orders.py      — Prong A: daily order features
  phase7_v11b_items.py       — Prong B: order-items features
  phase7_v11c_category.py    — Prong C: product category mix
  phase7_v11d_aov.py         — Prong D: AOV extrapolation
  phase7_v11_combined_val.py — multi-fold CV over all prongs
  phase7_v11_final.py        — production training for final submission
  submission_v11.csv         — final output
```

Each script prints: val RMSE per fold, mean prediction, feature importance top-10.

---

## 7. Contingency plan

- **If v11 scores 750-800k**: incremental improvement — keep iterating with Prongs C, E, G.
- **If v11 scores 700-740k**: breakthrough — focus on ensemble refinement and probe calibration.
- **If v11 scores >800k**: level leak somewhere — fall back to submission.csv (774k) and bisect which prong caused the drop.
- **If v11 scores ≤680k**: we've solved it, polish and submit as final.

Always keep `submission.csv` as the safe baseline. Never overwrite unless v11 is proven better on Kaggle.

---

## 8. Open questions to answer before coding

1. **How does friend aggregate promo effects?** We saw that raw promo counts leak level. Are there promo features friend is using that we're missing? → Check if friend's updated code is available.

2. **Are there signals we can pull forward from 2022 that would be known in 2023-24?** (e.g., launch-date product metadata, announced campaign calendars.) → Check `products.csv` dates.

3. **Is there an "anchor day" pattern?** The 548-day test spans exactly Jan 2023 → Jul 2024. If we know specific real-world events in that range (VN elections, Tet 2023/2024 dates, specific product launches), we could add named features. We already have Tet dates through 2024.

4. **What's the best AOV projection model?** Linear, log-linear, or just 2022 + delta from 2020-2022 slope? → Fit all three and compare.

---

## 9. First concrete action

Build `phase7_v11a_val.py`: add Prong A (daily order features with month-dow projection) to the v10c 23-feature base. Validate on our fold to:
- Confirm it preserves level (mean should stay ~$4.2M)
- Confirm it improves RMSE beyond v10c
- Inspect feature importances to see if `n_orders` dominates

If both checks pass, build `phase7_v11_final.py` with Prongs A + B + D combined and produce `submission_v11.csv`.
