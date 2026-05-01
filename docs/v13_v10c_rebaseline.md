# V13 Step 7 -- V10c Rebaseline on Mirror Block

Reproduces V10c's 23-feature setup on the post-cliff train window.
Train: 2019-09-01 .. 2021-12-31  (rows=853)
Mirror: 2022-01-01 .. 2022-12-31  (rows=365)

## Floor scores (V13 must beat these)

- **rmse_combined:** 902,932
- rmse_rev:  954,301    rmse_peak_rev:  1,526,708
- rmse_cogs: 848,459   rmse_peak_cogs: 1,333,601
- rmse_log_rev: 0.3051    rmse_log_cogs: 0.3164
- mean predicted revenue: 2,824,586/day  vs actual: 3,204,791/day
- mean predicted COGS:    2,457,451/day  vs actual: 2,795,672/day
- n_rows: 365    n_peak_rows: 76

## Reading

On Kaggle (full 2017+ training), V10c scored **774,898** RMSE.
On this redrawn mirror block (post-cliff training only), V10c scores 
**954,301** for revenue.  This is harder than Kaggle because:
  1. Mirror block size is 365 days vs Kaggle's 548; volatility per day is similar.
  2. Train window is 2.3 years (post-cliff only) vs Kaggle's full 6 years.
  3. The model is data-starved on early train rows (lag-364 reaches pre-cliff).

**This number is the V13 floor.**  If V13 scores worse than this on the same mirror, we don't submit.  If it scores better, we have a real reason to try Kaggle.