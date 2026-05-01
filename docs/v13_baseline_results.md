# V13 Step 8 -- Baseline LightGBM Results

- Objective: tweedie, USE_LOG_TARGET=False
- DEFAULT_ITERS: 600, recency-weighted
- Train: 2019-09-01 .. 2021-12-31
- Mirror: 2022-01-01 .. 2022-12-31

## Iteration CV folds

| Fold | rmse_combined | rmse_rev | rmse_peak_rev | rmse_log_rev | mean_pred | mean_actual |
|---|---:|---:|---:|---:|---:|---:|
| fold1_2021Q1 | 878,250 | 962,899 | 1,890,245 | 0.3311 | 2,328,421 | 2,509,164 |
| fold2_2021Q2 | 886,517 | 969,817 | 1,230,330 | 0.2253 | 4,336,071 | 4,465,463 |
| fold3_2021Q3 | 985,390 | 1,130,904 | 1,352,422 | 0.4195 | 3,188,625 | 2,653,515 |
| fold4_2021Q4 | 556,087 | 563,458 | 268,636 | 0.3074 | 1,847,231 | 1,812,332 |

## Mirror block

- **rmse_combined: 1,000,402**
- rmse_rev: 1,154,336   rmse_peak_rev: 1,945,528
- rmse_cogs: 817,995  rmse_peak_cogs: 1,215,281
- mean pred rev: 2,609,792/day vs actual: 3,204,791/day

## vs V10c floor

| | V10c rebaseline | V13 baseline | Delta |
|---|---:|---:|---:|
| rmse_combined | 902,932 | 1,000,402 | +97,470 |

**Verdict: WORSE than V10c**