# V12 Step 3 — COGS Reconstruction Verification

## 1. Identity check

| Identity | Pearson r | Σrecon / Σactual | Interpretation |
|---|---:|---:|---|
| items_cogs_recon ≈ sales.COGS | 1.000000 | 1.000000 | PERFECT — zero modelling error on historical COGS |
| items_gross ≈ sales.Revenue    | 0.992805 | 0.952740 | Near-perfect Revenue proxy |

**MAE of COGS reconstruction**: 0.00   (actual mean: 3,235,691.39, so error is 0.000% of mean)

## 2. cogs_ratio yearly stability

| Year | Mean ratio | Std | Days |
|---:|---:|---:|---:|
| 2017 | 0.90332 | 0.16010 | 365 |
| 2018 | 0.84279 | 0.07255 | 365 |
| 2019 | 0.89756 | 0.16187 | 365 |
| 2020 | 0.84729 | 0.07847 | 366 |
| 2021 | 0.92084 | 0.17690 | 365 |
| 2022 | 0.87910 | 0.07831 | 365 |

**Overall**: mean = 0.88180, std = 0.13253, range [0.71308, 1.57458]

## 3. Projection strategy for Stage D

Because `cogs_ratio` is bounded (COGS can't exceed Revenue) and has drifted only modestly across 2017-2022, we blend two estimators:

- **Flat recent**: last-90-day median = 0.86725
- **Linear trend (2021+)**: slope = 5.951e-05/day, intercept = 0.75538

Final projection: `ratio(t) = 0.6 × flat_recent + 0.4 × linear_trend(t)`, clipped to [0.70, 0.95].

Projected range across test horizon: 0.87465 → 0.88767, mean 0.88116

## 4. How Stage D consumes this

```python
predicted_COGS = predicted_Revenue × cogs_ratio_proj
```

where `predicted_Revenue` comes from Stages A/B/C. This gives Stage D a deterministic, bounded COGS prediction tied to the Revenue model rather than an independent COGS model.

## 5. Artifacts

- `data/processed/cogs_ratio_projection.parquet` — 548-row projection for test dates.
- `figures/v12_cogs_ratio_trend.png` — trend visualization.
