# V13 Step 2 — Full-history Timeline Forensics

Produced by `src/v13_timeline_audit.py`.  Data: 13 raw files, 2012-07 → 2022-12.

## 1. Headline finding

Revenue dropped ~50% between 2018 and 2019. The audit attributes this to **specific dimensions** below; multiply through to confirm whether the cause was volume (orders/customers) or unit economics (AOV/items/price).

## 2. 2018 vs 2019 — what changed (yearly mean)

| Metric | 2018 → 2019 | Direction |
|---|---|---|
| `n_orders` | 5792.500 → 3466.750  (-40.2%) | |
| `n_unique_cust` | 5255.667 → 3253.833  (-38.1%) | |
| `n_signups` | 1084.250 → 1254.833  (+15.7%) | |
| `orders_per_cust` | 1.092 → 1.064  (-2.6%) | |
| `aov` | 25532.425 → 25957.893  (+1.7%) | |
| `items_per_order` | 4.901 → 4.887  (-0.3%) | |
| `avg_unit_price` | 5483.419 → 5620.627  (+2.5%) | |
| `discount_pct` | 0.052 → 0.055  (+5.2%) | |
| `cogs_ratio` | 0.896 → 0.954  (+6.5%) | |
| `sessions` | 784590.417 → 832512.333  (+6.1%) | |
| `unique_visitors` | 595291.333 → 631713.750  (+6.1%) | |
| `bounce_rate` | 0.005 → 0.005  (+0.2%) | |
| `n_active_skus` | 483.000 → 456.917  (-5.4%) | |
| `return_rate` | 0.064 → 0.061  (-4.0%) | |
| `rating_mean` | 3.927 → 3.926  (-0.0%) | |
| `mean_ship_fee` | 4.463 → 4.489  (+0.6%) | |
| `mean_delivery_days` | 4.506 → 4.498  (-0.2%) | |
| `mean_installments` | 3.437 → 3.440  (+0.1%) | |
| `pct_cod` | 0.150 → 0.150  (+0.1%) | |
| `pct_mobile` | 0.452 → 0.448  (-0.9%) | |
| `pct_paid_search` | 0.219 → 0.222  (+1.7%) | |
| `pct_acq_paid` | — | |
| `n_active_promos` | 1.000 → 1.455  (+45.5%) | |
| `mean_discount` | 15.000 → 20.000  (+33.3%) | |

## 3. 2019 vs 2022 — same-regime comparison

| Metric | 2019 → 2022 | |
|---|---|---|
| `n_orders` | 3466.750 → 3000.333  (-13.5%) | |
| `n_unique_cust` | 3253.833 → 2840.500  (-12.7%) | |
| `n_signups` | 1254.833 → 1758.583  (+40.1%) | |
| `orders_per_cust` | 1.064 → 1.054  (-0.9%) | |
| `aov` | 25957.893 → 30920.123  (+19.1%) | |
| `items_per_order` | 4.887 → 4.768  (-2.4%) | |
| `avg_unit_price` | 5620.627 → 6836.222  (+21.6%) | |
| `discount_pct` | 0.055 → 0.053  (-3.7%) | |
| `cogs_ratio` | 0.954 → 0.936  (-1.9%) | |
| `sessions` | 832512.333 → 921971.500  (+10.7%) | |
| `unique_visitors` | 631713.750 → 700773.333  (+10.9%) | |
| `bounce_rate` | 0.005 → 0.004  (-1.1%) | |
| `n_active_skus` | 456.917 → 410.000  (-10.3%) | |
| `return_rate` | 0.061 → 0.062  (+1.3%) | |
| `rating_mean` | 3.926 → 3.962  (+0.9%) | |
| `mean_ship_fee` | 4.489 → 4.115  (-8.3%) | |
| `mean_delivery_days` | 4.498 → 4.506  (+0.2%) | |
| `mean_installments` | 3.440 → 3.458  (+0.5%) | |
| `pct_cod` | 0.150 → 0.150  (-0.5%) | |
| `pct_mobile` | 0.448 → 0.455  (+1.6%) | |
| `pct_paid_search` | 0.222 → 0.216  (-3.0%) | |
| `pct_acq_paid` | — | |
| `n_active_promos` | 1.455 → 1.000  (-31.2%) | |
| `mean_discount` | 20.000 → 15.000  (-25.0%) | |

## 4. Yearly mean — every key metric

| Year | n_orders | n_unique_cust | n_signups | orders_per_cust | aov | items_per_order | avg_unit_price | discount_pct | cogs_ratio | sessions | unique_visitors | bounce_rate | n_active_skus | return_rate | rating_mean | mean_ship_fee | mean_delivery_days | mean_installments | pct_cod | pct_mobile | pct_paid_search | pct_acq_paid | n_active_promos | mean_discount |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2012 | 5341.833 | 4817.667 | 79.750 | 1.107 | 23447.613 | 5.176 | 4558.337 | 0.000 | 0.793 | — | — | — | 409.333 | 0.057 | 3.955 | 5.965 | 4.506 | 3.446 | 0.148 | 0.452 | 0.219 | 0.000 | — | — |
| 2013 | 6404.083 | 5730.667 | 249.083 | 1.116 | 20665.159 | 5.123 | 4263.441 | 0.056 | 0.952 | 566828.333 | 430541.750 | 0.005 | 446.500 | 0.066 | 3.934 | 6.051 | 4.508 | 3.441 | 0.150 | 0.452 | 0.218 | 0.000 | 1.364 | 19.773 |
| 2014 | 6720.417 | 5978.500 | 419.500 | 1.121 | 22046.713 | 5.109 | 4536.232 | 0.052 | 0.899 | 611746.667 | 466354.000 | 0.004 | 463.750 | 0.065 | 3.925 | 5.591 | 4.480 | 3.433 | 0.148 | 0.450 | 0.218 | 0.000 | 1.000 | 15.000 |
| 2015 | 6885.167 | 6132.500 | 594.417 | 1.120 | 21728.298 | 5.049 | 4543.577 | 0.056 | 0.954 | 655161.500 | 496849.333 | 0.004 | 471.417 | 0.065 | 3.939 | 5.517 | 4.496 | 3.448 | 0.150 | 0.448 | 0.219 | 0.000 | 1.455 | 20.000 |
| 2016 | 6853.917 | 6133.167 | 766.833 | 1.114 | 24370.774 | 5.014 | 5104.157 | 0.051 | 0.905 | 700283.250 | 532804.500 | 0.004 | 473.833 | 0.064 | 3.928 | 4.816 | 4.504 | 3.478 | 0.148 | 0.450 | 0.218 | 0.000 | 1.000 | 15.000 |
| 2017 | 6334.167 | 5711.167 | 923.167 | 1.105 | 23787.745 | 4.975 | 5058.305 | 0.056 | 0.961 | 749383.500 | 568197.667 | 0.004 | 485.667 | 0.066 | 3.936 | 4.702 | 4.491 | 3.441 | 0.148 | 0.452 | 0.222 | 0.000 | 1.455 | 20.000 |
| 2018 | 5792.500 | 5255.667 | 1084.250 | 1.092 | 25532.425 | 4.901 | 5483.419 | 0.052 | 0.896 | 784590.417 | 595291.333 | 0.005 | 483.000 | 0.064 | 3.927 | 4.463 | 4.506 | 3.437 | 0.150 | 0.452 | 0.219 | 0.000 | 1.000 | 15.000 |
| 2019 | 3466.750 | 3253.833 | 1254.833 | 1.064 | 25957.893 | 4.887 | 5620.627 | 0.055 | 0.954 | 832512.333 | 631713.750 | 0.005 | 456.917 | 0.061 | 3.926 | 4.489 | 4.498 | 3.440 | 0.150 | 0.448 | 0.222 | 0.000 | 1.455 | 20.000 |
| 2020 | 2906.750 | 2747.000 | 1434.250 | 1.056 | 28732.817 | 4.799 | 6283.952 | 0.053 | 0.902 | 882590.167 | 672113.917 | 0.005 | 441.667 | 0.062 | 3.981 | 4.214 | 4.504 | 3.428 | 0.148 | 0.449 | 0.221 | 0.000 | 1.000 | 15.000 |
| 2021 | 2877.083 | 2722.833 | 1596.167 | 1.054 | 28899.056 | 4.793 | 6357.562 | 0.054 | 0.976 | 915977.083 | 697251.167 | 0.004 | 438.333 | 0.061 | 3.917 | 4.214 | 4.508 | 3.434 | 0.153 | 0.455 | 0.216 | 0.000 | 1.455 | 20.000 |
| 2022 | 3000.333 | 2840.500 | 1758.583 | 1.054 | 30920.123 | 4.768 | 6836.222 | 0.053 | 0.936 | 921971.500 | 700773.333 | 0.004 | 410.000 | 0.062 | 3.962 | 4.115 | 4.506 | 3.458 | 0.150 | 0.455 | 0.216 | 0.000 | 1.000 | 15.000 |

## 5. Detected change points (>20% shift in 12-month rolling mean)

| Metric | Date | Prev avg | New avg | Δ |
|---|---|---:|---:|---:|
| `n_orders` | 2019-10-01 | 6059.167 | 3546.250 | -41.5% |
| `n_orders` | 2019-09-01 | 6169.750 | 3622.750 | -41.3% |
| `n_orders` | 2020-02-01 | 5639.500 | 3313.083 | -41.3% |
| `n_unique_cust` | 2019-10-01 | 5486.083 | 3322.750 | -39.4% |
| `n_unique_cust` | 2019-09-01 | 5581.583 | 3390.750 | -39.3% |
| `n_unique_cust` | 2020-02-01 | 5115.833 | 3115.250 | -39.1% |
| `n_signups` | 2013-06-01 | 38.500 | 164.583 | +327.5% |
| `n_signups` | 2013-07-01 | 45.857 | 177.917 | +288.0% |
| `n_signups` | 2013-08-01 | 52.125 | 191.000 | +266.4% |
| `discount_pct` | 2014-01-01 | 0.000 | 0.057 | +157113.5% |
| `discount_pct` | 2014-02-01 | 0.004 | 0.055 | +1288.1% |
| `discount_pct` | 2014-03-01 | 0.011 | 0.054 | +385.8% |
| `revenue_actual` | 2020-02-01 | 4942392.789 | 2975785.758 | -39.8% |
| `revenue_actual` | 2020-01-01 | 5017555.741 | 3049484.480 | -39.2% |
| `revenue_actual` | 2019-10-01 | 5218174.688 | 3187644.458 | -38.9% |

## 6. Figures

- `figures/v13_timeline_summary.png` — one-page overview of 8 key metrics.
- `figures/v13_timeline_volume.png` — orders, customers, signups, revenue.
- `figures/v13_timeline_basket.png` — AOV, items per order, unit price, discount, COGS ratio.
- `figures/v13_timeline_mix.png` — device, payment, channel, acquisition.
- `figures/v13_timeline_ops.png` — returns, ratings, shipping, delivery, SKUs.
- `figures/v13_timeline_web_promo.png` — sessions, visitors, bounce, promotions.

Orange shading = 2018-09 → 2019-12 (the cliff).  Red shading = 2020-03 → 2021-12 (COVID window).

## 7. How this drives V13

Inspect the change-point table and per-group figures, then decide:
1. **Training cut.** If most key metrics shifted in 2019 and stayed shifted, train on 2019+ only.
2. **COVID flags.** If COVID didn't materially move metrics relative to 2019 baseline, demote the COVID feature block to a single `covid_severity` scalar (or drop it).
3. **Per-feature regime adjustment.** For metrics that shifted (e.g. mean_installments doubling), consider regime indicators rather than treating them as continuous.
