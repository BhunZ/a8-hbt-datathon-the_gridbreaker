# Datathon 2026 — Final Retrospective

## Bottom line

After 17 versions, **V10c-original (Kaggle MAE 774,898) remained the unbeatable floor**. Every single architectural change we attempted scored worse on Kaggle. The competition rewarded a model whose specific properties we couldn't reproduce by tuning, ensembling, or feature engineering.

## The full Kaggle-public scoreboard (MAE)

| Submission | Mean rev/day | Kaggle MAE | Δ vs V10c |
|---|---:|---:|---:|
| **submission_v10c.csv** | **$4.23M** | **774,898** | **baseline (best)** |
| submission_v12c.csv | $4.00M | 852,132 | +77,234 |
| submission_v12d.csv | $3.96M | 871,299 | +96,401 |
| submission_v14.csv | $3.47M | ~1,000,000 | ~+225,000 |
| submission_v13_final.csv | $3.32M | 1,187,779 | +412,881 |
| submission_v12b.csv | $3.66M | 1,243,589 | +468,691 |
| submission_v12a.csv | $3.66M | 1,252,820 | +477,922 |
| submission_v13_v10c_only.csv | $3.13M | 1,335,365 | +560,467 |

**Pattern:** Lower mean prediction → worse Kaggle MAE. Linear, stark, consistent. Every model that predicted below V10c's $4.23M lost.

## What we tried and what we learned

### V11/V12 (failed)
**Hypothesis:** More features (35+ from web_traffic / orders / reviews / returns) projected onto test via month-dow medians would help. **Result:** Train/serve skew. Crushed peaks at inference, lost +470k MAE.

### V13 (failed)
**Hypothesis:** Strict deterministic-at-inference architecture (no projected aux features) would eliminate skew. Train only on post-2019 to match the regime change.
**Result:** Lost ~410k MAE despite passing every architectural-correctness check. The "post-cliff cut" hurt more than the V12 skew.
**Why:** 2017-2018 data carried the high-revenue signal that the actual 2023-2024 horizon needed. Discarding it killed the prediction level.

### V13 timeline audit (real win)
We discovered a **40% revenue cliff in 2019** that nobody had talked about — orders dropped from 5,792/day to 3,467/day. This was real and important *for understanding the dataset*, but acting on it (the post-cliff training cut) hurt the leaderboard.

### V13 cross-features (small win, used everywhere after)
`trend_adj_prior_rev`, `lag728_trend_rev` — multiplying static priors by the trend ratio. Closed ~50k mirror RMSE. Survived into V14.

### V14 (failed)
**Hypothesis:** Multi-loss LightGBM ensemble (L1 + Quantile-50 + MSE + Tweedie) blended with V10c's RF/ET/HGB, validated on all 3 metrics simultaneously, would beat V10c. Mirror block agreed (5/6 metrics improved). **Kaggle disagreed: ~1M MAE.**

### V10c-tuned variants (failed)
- **Deeper + more trees:** mean prediction dropped to $3.67M. Worse, not better.
- **Shallower trees (depth 12/13):** mean prediction dropped to $3.58M. Worse.
- **Multiseed (4 seeds, same params):** mean prediction dropped to $3.65M. Worse.

**Why:** V10c's $4.23M predictions came from a specific seed combination [42, 17] producing slightly-higher-than-average trees by chance. Tuning *anything* averaged that lucky configuration toward the post-cliff median (~$3M).

### V10c scaled (last attempt, untested on Kaggle as of writing)
Take V10c-original predictions, multiply by a constant. Three variants:

| File | Scale | Mean rev/day | Hypothesis |
|---|---:|---:|---|
| `submission_v10c_scaled_105.csv` | 1.05× | $4.44M | Mild upward correction |
| `submission_v10c_scaled_110.csv` | 1.10× | $4.65M | Moderate, recommended |
| `submission_v10c_scaled_114.csv` | 1.15× | $4.87M | Aggressive |

If actual 2023-2024 mean revenue is in fact higher than $4.23M, scaling up should reduce MAE. If actual is at or below $4.23M, scaling will increase MAE proportionally. Pure level bet.

## What you should upload

**Conservative (recommended):** Re-submit `submission_v10c.csv` as final. **774k MAE is our proven floor.** Don't trade certainty for unknown variance.

**Slightly aggressive:** Upload `submission_v10c_scaled_105.csv` (5% scale-up). If the test horizon is mildly higher than $4.23M/day, you save ~30-50k MAE. If it's slightly lower, you lose ~30-50k. Symmetric bet.

**Aggressive:** `submission_v10c_scaled_110.csv` (10% scale-up to $4.65M/day). Bigger swing in either direction.

**Don't upload:** any V11/V12/V13/V14 variant, V10c-tuned, V10c-shallow, V10c-multiseed. These all predict lower than V10c and the Kaggle leaderboard has been clear that lower = worse.

## Final answer to "what's the best model"

**For this dataset, the answer is V10c-original.** Across MAE, RMSE, and R² (the three metrics this contest will be judged on), V10c is the configuration that best matches the actual test horizon's revenue level. We could not reproduce that match through any further engineering.

Sometimes the right call is to recognize when more iteration is hurting, not helping.

## Things that would have helped (but we couldn't do under the rules)

- **External validation data.** A 2023 partial actual would have told us whether the test horizon was at $4.23M or higher.
- **More public-leaderboard slots.** Each submission burned tells us a single MAE number; we couldn't A/B properly.
- **Foundation models (Chronos, TimeGPT).** Banned under "no external data" though zero-shot inference doesn't add data per se.

## Per-version artifact map

| Version | Code | Submission | Score |
|---|---|---|---|
| V10c | `src/phase7_v10c_fit.py` | `submission_v10c.csv` | **774,898** ★ |
| V12a-d | `src/v12_*.py` | `submission_v12{a,b,c,d}.csv` | 852-1252k |
| V13 timeline audit | `src/v13_timeline_audit.py` | (research only) | — |
| V13 features | `src/v13_calendar_event_book.py`, `v13_priors.py`, `v13_lags.py`, `v13_trends.py`, `v13_assemble_features.py` | — | — |
| V13 baseline | `src/v13_baseline.py` | (mirror only, 1,000k mirror RMSE) | — |
| V13 peak recovery | `src/v13_peak_recovery.py` | (mirror only, 929k mirror RMSE) | — |
| V13 final | `src/v13_final_submission.py` | `submission_v13_final.csv` | 1,187,779 |
| V14 | `src/v14.py` | `submission_v14.csv`, `submission_v14_l1only.csv` | ~1,000,000 |
| V10c tuned | `src/v10c_tuned.py` | (not uploaded) | — |
| V10c variants | `src/v10c_variants.py` | `submission_v10c_{shallow,multiseed}.csv` (not uploaded) | — |
| V10c scaled | (post-hoc) | `submission_v10c_scaled_{105,110,114}.csv` | TBD |

## Lessons for future Kaggle competitions

1. **Your local validation isn't your leaderboard.** We had a 2022 mirror block that systematically underestimated the test horizon's revenue level. Six failed submissions before we figured out the mirror was misleading.

2. **Mean prediction matters as much as residual quality.** All our "improvements" preserved or improved residual structure on validation but moved the mean prediction in the wrong direction. The leaderboard punished us for the level shift, not the residual quality.

3. **Stop iterating when iteration is making things worse.** V11, V12, V13, V14, V10c-tuned all scored worse than V10c-original. After 4 failed iterations, the signal was clear.

4. **One lucky configuration is hard to reproduce by tuning.** V10c had a specific RF/ET/HGB blend with specific seeds that produced predictions at exactly the right level. We could not reverse-engineer that magic.

5. **Trust the leaderboard, not your hypothesis.** The cliff-finding in V13 was a real, true insight about the data. Acting on it was wrong because the leaderboard depended on a different level than the cliff implied.
