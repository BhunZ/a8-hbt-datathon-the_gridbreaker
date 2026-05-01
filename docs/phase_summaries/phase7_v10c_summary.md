# Phase 7 v10c — feature-engineering expansion over friend's 781k baseline

## TL;DR

Built three ablation variants on top of friend's 11-feature model (which scored 781k Kaggle = rank 165) and identified which categories of feature engineering actually help vs. which re-introduce the "low-level anchor" bug from v2.

**Result**: `submission_v10c.csv` is a near-identical but slightly richer version of friend's model — 0.989 Pearson correlation, mean just 2.3% lower, with added named-event signal.

Current `submission.csv` still = friend's 740k/781k baseline (safe). v10c is provided as an alternative to try.

## Feature ablation on our val fold (2021-07-01 → 2022-12-31)

| Variant | # feats | Val RMSE (concat) | Mean Rev pred | Interpretation |
|---|---:|---:|---:|---|
| BASE (friend) | 11 | 1,223,614 | $3.74M | Reference |
| + Named events | 18 | 1,225,390 | $3.74M | **No change** — val doesn't see event signal |
| + Extra traffic channels | 23 | 1,148,579 | $3.66M | -6% RMSE, slight level drop |
| FULL (+ signups/promos/stockouts) | 32 | **836,420** | **$2.59M** | **Level leak!** Re-anchors to 2022 |

Events added: `is_singles_day` (11/11), `is_twelve_twelve` (12/12), `is_nine_nine` (9/9),
`is_black_friday` (4th Fri of Nov), `is_cyber_monday` (Mon after US Thanksgiving),
`is_mothers_day` (2nd Sun of May), `near_flash_event` (±3 day window).

Extra traffic fields: `unique_visitors`, `bounce_rate`, `avg_sess`, raw `month`, `dow`.

"Level leak" category (EXCLUDED from v10c): `signups_30d`, `signups_90d`, `n_promo`,
`avg_promo_discount`, `total_stockout_days`, `n_skus_stockout`, `mean_stock`.

## Why signups/promos/stockouts leak level

These features carry *recent state* into the model. In 2022, signups were lower and
there were fewer promos recorded than in earlier years. The tree learns
`signups_30d < 500 → Revenue ≈ $2.5M`. On the 2023-24 test set, the same low values
persist (even lower if rolled forward from training end), so the model routes to the
low-revenue leaves. Same mechanism as the Revenue lag features in v2 — the validation
RMSE looks great but the actual Kaggle score tanks.

Named calendar events are **safe** because they're pure date-based (same in train and test).
Extra traffic fields are safe because test uses the `(month, dow)` historical average.

## v10c final architecture

- **Features (23)**: friend's base 11 + 7 named events + 5 extra traffic fields
- **Models**: RandomForest (n=350, d=15) + ExtraTrees (n=450, d=16) with 2 seeds (42, 17) averaged, plus HistGradientBoosting
- **Blend**: friend's [0.3 RF, 0.5 ET, 0.2 HGB] preserved
- **Weight**: polynomial `(year - 2016) ** 1.2`
- **Training**: 2017-01-01 → 2022-12-31

## v10c vs friend's submission comparison

| Metric | Friend (781k) | v10c | Delta |
|---|---:|---:|---:|
| Mean Revenue | $4.33M | $4.23M | -$0.10M |
| Mean COGS | $3.69M | $3.62M | -$0.07M |
| Sum Revenue | $2.37B | $2.32B | -$0.05B |
| Pearson(Rev) | — | 0.989 | very high |
| Max per-day diff | — | $2.07M | localized |

## Event-day predictions in v10c

| Event | Date | Rev pred | Notes |
|---|---|---:|---|
| Singles Day (11/11) | 2023-11-11 | $2.05M | Lower than mean (model didn't see strong training spikes) |
| 12/12 | 2023-12-12 | $1.91M | Similarly flat |
| Black Friday | 2023-11-24 | $1.69M | Flat |
| 9/9 | 2023-09-09 | $2.71M | Flat |
| Cyber Monday | 2023-11-27 | $2.87M | Flat |
| Mother's Day | 2023-05-14 | $5.39M | **High** — captured |
| Mother's Day | 2024-05-12 | $4.52M | **High** — captured |

Mother's Day is amplified nicely. Flash-sale double-days (11/11, 12/12, 9/9) stayed flat,
likely because 2017-2022 training data doesn't show consistent spikes on those days
(especially during 2020-2022 COVID period).

## Decision

- **Stay with `submission.csv` (friend's, scored 781k)** unless you have a spare Kaggle submission slot.
- **Try `submission_v10c.csv`** if you do — likely to score in the 730-780k range. Small downside risk (level slightly lower, 2.3%), potential upside from better event-day handling.
- If v10c scores **lower** than 781k, revert to `submission.csv`.
- If v10c scores **better**, investigate adding (safely) projected promos/stockouts using `(month, dow)` averaging just like traffic.

## Files

- `outputs/submission.csv` — current active = friend's weighted_god_mode output (781k)
- `outputs/submission_v10c.csv` — v10c candidate (23 features, multi-seed)
- `outputs/phase7_v10c_fit.py` — v10c training script
- `outputs/phase7_v10_val.py` — initial 32-feature validation (showed level leak)
- `outputs/phase7_v10b_val.py` — ablation isolating which feature groups cause the leak
