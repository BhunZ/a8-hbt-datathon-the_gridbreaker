# V13 Step 9 -- Peak Recovery Results

- Trend-derived level scale (honest, train-only): 1.0872
- V10c floor: 902,932
- **Best variant: stage1+trendscale+s3**  rmse_combined=929,677  (WORSE than V10c by 26,745)

## Variant table

| Variant | rmse_combined | rmse_rev | rmse_peak_rev | mean_pred | delta_vs_floor |
|---|---:|---:|---:|---:|---:|
| stage1_only | 1,000,402 | 1,154,336 | 1,945,528 | 2,609,792 | +97,470 |
| stage1+stage3 | 999,572 | 1,153,085 | 1,941,962 | 2,611,580 | +96,640 |
| stage1+stage2 | 1,003,617 | 1,153,509 | 1,943,169 | 2,632,554 | +100,685 |
| stage1+stage2+stage3 | 1,003,580 | 1,153,279 | 1,942,515 | 2,634,967 | +100,648 |
| stage1+trendscale | 930,431 | 1,044,290 | 1,723,205 | 2,837,380 | +27,499 |
| stage1+trendscale+s3 | 929,677 | 1,043,112 | 1,719,774 | 2,839,324 | +26,745 |