# V12 Step 4 — Triple-Stage Ensemble

**Run-time**: 35.0s  ·  **Trees per model**: 500  ·  **max_depth**: 15
**Features**: 57

## Stage in-sample scores

| Stage | Target transform | Model | Train cutoff | Weights | Rev RMSE | COGS RMSE | Rev test μ | COGS test μ |
|---|---|---|---|---|---:|---:|---:|---:|
| A Anchor | log1p | ExtraTrees | 2017+ | uniform | 1,001 | 438 | $3.66M | $3.16M |
| B Modernist | log1p | RandomForest | 2019+ | (y-2018)^1.5 | 33,410 | 23,354 | $3.67M | $3.16M |
| C Peak Catcher | RAW | ExtraTrees | 2018+ | uniform | 720 | 336 | $3.66M | $3.16M |

## Blend  (0.4·A + 0.4·B + 0.2·C)

- Blend test  mean Revenue = **$3.662M**
- Blend test  mean COGS    = **$3.160M**
- Blend test  cogs/rev     = **0.8629**
- Blend in-sample (2019+ coverage): RMSE rev=13,473  cogs=9,349

## Comparison to prior results

| Version | Mean test Rev | Mean test COGS | Notes |
|---|---:|---:|---|
| v10c (current best) | $4.23M | $3.62M | Friend-style 23-feature blend |
| v12 ABC blend | $3.66M | $3.16M | 57 features, 3-stage |

## Outputs
- `data/processed/v12_stage_abc_preds.parquet` — 548 test-day predictions from each stage + blend.

## Next steps
- Step 5: Stage D (COGS via cogs_ratio projection) — already prepared in Step 3.
- Step 6: Final blend, gold blend with v10c, scaling.
