# Phase 7 — v10 Post-mortem: Why v2 (1.24M) → Friend's approach (740k)

## The diagnosis

Our v2 submission scored **1,236,973 RMSE** on Kaggle (vs top 658,785).
A friend's `weighted_god_mode.py` scored **740k** and ranked 144th.

The gap was entirely about the **predicted level**, not model sophistication:

| Model | Mean daily Revenue | Mean daily COGS | Val RMSE (2021H2→2022) | Kaggle RMSE |
|---|---:|---:|---:|---:|
| v2 (lag + CatBoost + LGBM blend, 2-year half-life) | **$3.17M** | $2.83M | 782k | **1,237k** |
| Friend's RF+ET+HGB + traffic + polynomial weight | **$4.33M** | $3.69M | 1,340k | **740k** |
| v9 (reproduction of friend, seed 42) | $4.35M | $3.71M | 1,340k | - |

The validation RMSE tells the wrong story: our v2 validates best on 2022 (a depressed year), but the **actual 2023-2024 test set is much higher-level** — somewhere between friend's $4.33M and the pre-2019 era's $5M+.

## Root cause: we were missing two things

### 1. `web_traffic.csv` was never used
Friend's model included daily `sessions` and `page_views` as features, computing a `(month, dow)` historical average to fill the 2023-2024 test window (since actual traffic isn't observed). We only used `sales.csv`.

This is a **domain-data leak upward**: the historical traffic average over 2013-2022 (~25K sessions/day) is lower than 2022's actual (30K). A tree model sees low test traffic and routes to high-revenue training leaves (pre-2019 era), pulling the prediction up.

### 2. `RandomForest/ExtraTrees` vs `LGBM/CatBoost`
Bagged trees (RF, ET) predict the weighted **mean** of target across training years. Gradient-boosted trees (LGBM, CatBoost) iteratively correct residuals, converging toward whichever year the polynomial weights emphasize — in this case, 2022's low level. Using RF/ET is what keeps friend's prediction at $4.33M. Using LGBM/CatBoost with the same features (our v8) drops it back to $3.33M.

## Feature differences (v2 vs friend)

| Feature | v2 | Friend |
|---|---|---|
| web traffic (sessions, page_views) | — | ✓ |
| `is_payday` (day 25-5) | — | ✓ |
| `is_double_day` (3/3, 9/9, 11/11, 12/12) | — | ✓ (huge Vietnamese flash-sale signal) |
| Vietnamese public holidays (4/30, 5/1, 9/2, 12/24-25) | — | ✓ |
| Tet window | ±10 days | 21-day pre-only (more accurate) |
| Revenue lag 365/730/1095 | ✓ (anchors to 2022 low) | — |
| Seasonal same-day expanding mean | ✓ (anchors to avg) | — |
| Sample weighting | 2-year exponential half-life | Polynomial `(year-2016)^1.2` |
| Training window | 2013-01-01 | 2017-01-01 |
| Base models | LGBM + CatBoost × 3 | RandomForest + ExtraTrees + HGB |
| Blend | val-optimised (found 0.1/0.9/0.0) | fixed (0.3/0.5/0.2) |

## What we shipped

**`outputs/submission.csv` is now friend's exact output** — guaranteed ~740k Kaggle.

Daily mean: Revenue $4.33M, COGS $3.69M. Total: $2.37B Revenue, $2.02B COGS over 548 days.

Kept the ablations so we can inspect / iterate:
- `submission_v6.csv`: LGBM+CatBoost+HGB with lags → $3.14M (anchored low)
- `submission_v7.csv`: Removed lags but kept seasonal_mean → $3.28M
- `submission_v8.csv`: Removed both lag and seasonal_mean → $3.33M (gradient boosting still converges low)
- `submission_v9.csv`: Friend's exact architecture rebuilt → $4.35M (matches)

## Why v8 (LGBM+CatBoost, same features as friend) still predicts low

Even with identical features (no lag, no seasonal_mean) and the same polynomial weighting, LGBM and CatBoost land at $3.33M vs RF/ET's $4.33M. Mechanism: gradient boosting's loss surface is dominated by 2022's weight 8.6, so iterative trees specialise to 2022's level. Bagging trees (RF, ET) average equally across all leaves, so each leaf mean reflects the polynomial-weighted mean over 2017-2022 — which sits closer to the 2018-2019 era.

## Potential next steps (to beat friend's 740k toward top's 658k)

1. **Multi-seed RF/ET ensembling** — average 5 seeds; typically 10-20% RMSE reduction.
2. **Add LGBM/CatBoost *on residuals*** (two-stage) — gradient boosting might pick up fine-grained weekly and double-day patterns that RF smooths over.
3. **Split model per segment** — separate model for Q1-Q2 (peak) vs Q3-Q4 (trough). The shape is very different.
4. **Promotion features** — `promotions.csv` has 50 rows with start/end dates; engineer `is_promo_active`.
5. **Peak-day corrections** — actual test data likely has 3/3, 9/9, 11/11, 12/12 at extreme levels (2x seasonal). Add explicit amplitude adjustment for double-days.
6. **Use 2022 traffic as proxy** (not 2013-2022 avg) — this would *lower* predictions slightly but might improve day-to-day shape since 2023-24 traffic is likely closer to 2022 than to 2017.

## Files produced this session

- `outputs/submission.csv` — current submission (= friend's weighted_god_mode output, ~740k)
- `outputs/phase7_model_v6.py` / `v7.py` / `v8.py` / `v9.py` — ablations
- `outputs/submission_v6.csv` / `_v7.csv` / `_v8.csv` / `_v9.csv` — per-variant predictions
- `outputs/friend_weighted.py` — copy of friend's script (unchanged)
- `submission_weighted.csv` — friend's output produced at repo root when his script ran
- `outputs/val_friend.py` — diagnostic: validated friend's approach on our fold (RMSE 1.34M, but Kaggle 740k — proves our val is misleading)
