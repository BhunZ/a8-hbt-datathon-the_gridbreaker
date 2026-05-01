# Complete Data Audit — Every File, Every Column

## Headline discoveries (the big ones)

### 🎯 Discovery 1: `sales.COGS` is **PERFECTLY reconstructable** from order_items × products
- `sum(order_items.quantity × products.cogs)` per day ≡ `sales.COGS`
- Correlation = **1.00000** (exact)
- Mean ratio = **1.0000** (exact)
- **Implication**: if we can project order line-items for test days, we get COGS *exactly*.

### 🎯 Discovery 2: `sales.Revenue` is **99.2% reconstructable** from order_items
- `sum(order_items.quantity × unit_price − discount_amount)` per day
- Correlation = 0.99210
- Mean ratio = 0.9544 (our reconstruction is ~5% lower than Revenue)
- **The missing 5%**: likely shipping fees + possibly tax or platform revenue
- On sample day 2018-06-01: sales $19.24M = pay_sum $19.24M = items_gross $19.24M (exact match). The ~5% gap averages over 3,833 days.
- **Implication**: Revenue ≈ items_gross × 1.05 — we can back-compute Revenue from the same projections.

### 🎯 Discovery 3: `payments.payment_value` per day ≡ `items_gross` (both identical, same 0.9544 ratio to Revenue)
- payments.csv is essentially a pre-aggregated `qty × price − discount` per order
- **It's a cleaner, higher-correlation signal** (99.2% r² vs 87.6% for n_orders)

### 🎯 Discovery 4: **All 646,945 orders have exactly 1 payment each** (1:1 order↔payment)
- payments adds `payment_method` (matches orders anyway) and `installments` (1-12, new signal)
- `installments` distribution: 1 (41%), 3 (34%), 6 (17%), 12 (8%), 2 (<1%)
- This is a rough buyer-intent signal — higher installments = luxury buyer / big ticket

### 🎯 Discovery 5: Only 87.5% of orders have shipments
- 87.5% = orders with `delivered` or `shipped` status
- days_to_ship: 0-3 days (median 1)
- days_to_deliver: 2-7 days (median 4)
- These are fulfillment lag — **NOT useful for forecasting**, but useful for estimating 2023-24 lead times

### 🎯 Discovery 6: Only 126 unique inventory snapshot dates (10 years × ~12/year = monthly snapshots)
- Not daily — inventory is updated monthly
- Has 17 columns (we only used 3-4)
- Untapped: `units_received`, `units_sold`, `fill_rate`, `sell_through_rate`, `overstock_flag`
- `reorder_flag` is always 0 (useless)

### 🎯 Discovery 7: `web_traffic.csv` has ONE source per day (not aggregated)
- 3652 rows = 3652 unique dates (one row per day)
- The `traffic_source` column shows the **dominant/sampled source** of that day
- Distribution: organic_search (30%), paid_search (21%), social_media (17%), email (14%), referral (10%), direct (7%)
- Friend's code treats this as if it's all-sources-summed, which may be wrong
- **New feature**: one-hot encode `traffic_source` per day — reveals channel-specific patterns

---

## File-by-file complete audit

### 1. `sales.csv` — the target
| Col | Type | Range/Values |
|---|---|---|
| Date | datetime | 2012-07-04 → 2022-12-31 (3833 days) |
| Revenue | float | $279,814 → $20,905,271, mean $4.29M, std $2.62M |
| COGS | float | $236,576 → $16,535,858, mean $3.70M, std $2.22M |

**Key stat**: Revenue mean over all 10.5 years = **$4.29M/day** — this is why friend's $4.33M prediction scored so well. Our v10c at $4.23M is 1.4% below the historical mean, which may cost us Kaggle points.

---

### 2. `web_traffic.csv` — marketing funnel
| Col | Type | Detail |
|---|---|---|
| date | datetime | 2013-01-01 → 2022-12-31 (3652 days) |
| sessions | int | 7,973 → 50,947, mean 25,042, std 9,423 |
| unique_visitors | int | 6,136 → 40,430, mean 19,031 |
| page_views | int | 30,451 → 275,560, mean 108,615 |
| bounce_rate | float | 0.0 → 0.01 (values 0-1%) |
| avg_session_duration_sec | float | 100 → 320, mean 210 |
| **traffic_source** | string | 6 values, one per day |

**Traffic source counts** (#days marked each): organic_search (1090), paid_search (784), social_media (632), email_campaign (505), referral (375), direct (266)

**We IGNORED**: `traffic_source`, `bounce_rate`, `avg_session_duration_sec`, `unique_visitors`. Friend only used `sessions` and `page_views`.

**Action**: one-hot encode `traffic_source`. Add the 4 unused numeric channels.

---

### 3. `orders.csv` — the heart of the dataset (biggest miss)
| Col | Type | Uniq | Distribution |
|---|---|---|---|
| order_id | int | 646,945 | Unique identifier |
| order_date | datetime | 3,833 | 2012-07-04 → 2022-12-31 |
| customer_id | int | 90,246 | (of 121,930 customers — 74% ever ordered) |
| zip | int | 29,932 | Join to geography.csv |
| **order_status** | string | 6 | delivered (80%), cancelled (9%), returned (6%), shipped (2%), paid (2%), created (1%) |
| **payment_method** | string | 5 | credit_card (55%), paypal (15%), cod (15%), apple_pay (10%), bank_transfer (5%) |
| **device_type** | string | 3 | mobile (45%), desktop (40%), tablet (15%) |
| **order_source** | string | 6 | organic_search (28%), paid_search (22%), social_media (20%), email (12%), referral (10%), direct (8%) |

**Why it's huge**: every one of these categoricals can be aggregated to daily percentages and becomes a feature. Twenty possible new features from this table alone.

**Key signal**: `n_orders` per day has correlation **0.936** with Revenue (vs sessions 0.60).

---

### 4. `order_items.csv` — line-level detail
| Col | Type | Uniq | Detail |
|---|---|---|---|
| order_id | int | 646,945 | Join to orders |
| product_id | int | 1,598 | Join to products (only 1,598/2,412 products ever sold) |
| quantity | int | 8 | 1-8, mean 4.5 |
| unit_price | float | 501,330 | $393 → $43,056, mean $5,115 |
| discount_amount | float | 204,449 | $0 → $35,235, mean $1,049 |
| **promo_id** | string | 50 | 61% null — 39% of items discounted via a promo |
| **promo_id_2** | string | 2 | 99.97% null — secondary promo rare (206 items total, only PROMO-0015 or PROMO-0025) |

**Items per order**: 1 (most), up to 5. Mean 1.10 items per order — mostly single-item baskets.

**Discovery**: `quantity` is NOT 1 — it's 1-8 per line. So total units per order = quantity × n_lines, mean ~4.5 units/order.

**Missing**: we never computed daily revenue by product category, discount rates over time, promo engagement over time.

---

### 5. `payments.csv` — clean payment-level
| Col | Type | Detail |
|---|---|---|
| order_id | int | 1:1 with orders |
| payment_method | string | Same 5 as orders |
| **payment_value** | float | $390 → $331,570, mean $24,238 — CORR 0.9921 with Revenue |
| **installments** | int | 5 values: 1 (41%), 3 (34%), 6 (17%), 12 (8%), 2 (<1%) |

**NEW SIGNAL**: `installments` distribution per day — buyer financing behavior.

**Also**: `payment_value` is a pre-aggregated `items_gross`. Cleaner than reconstructing from items.

---

### 6. `shipments.csv` — fulfillment
| Col | Type | Detail |
|---|---|---|
| order_id | int | 566,067 unique (87.5% of orders) |
| ship_date | datetime | 2012-07-04 → 2022-12-29 |
| delivery_date | datetime | 2012-07-06 → 2022-12-31 |
| shipping_fee | float | $0 → $32, median $1.73 |

**Derived signals** (new):
- `days_to_ship` = ship_date − order_date: 0-3 days, median 1
- `days_to_deliver` = delivery_date − ship_date: 2-7 days, median 4
- Daily mean shipping_fee — proxy for order mix

**Non-shipped orders** (12.5%) = cancelled / created / paid — these contribute 0 shipping data.

---

### 7. `reviews.csv` — customer satisfaction
| Col | Type | Detail |
|---|---|---|
| review_id | string | Unique |
| order_id | int | 111,369 unique (93% of reviews have 1 review per order) |
| product_id | int | 1,412 products reviewed |
| customer_id | int | 48,676 unique reviewers |
| review_date | datetime | 2012-07-10 → 2022-12-31 |
| **rating** | int | 1-5, mean 3.94, distribution: 5★ (45K), 4★ (36K), 3★ (17K), 2★ (9K), 1★ (6K) |
| **review_title** | string | 18 unique templated titles |

**Mean rating by year** (stable ≈ 3.93 across all years). No clear trend signal.

**Review titles are TEMPLATED** (only 18 values) — likely synthetic. Not useful as text features.

**Derivable signals**: 30-day rolling rating average, 30-day review volume trend.

**Review delay**: reviews happen AFTER order → lag feature risk for training dates.

---

### 8. `returns.csv` — returns
| Col | Type | Detail |
|---|---|---|
| return_id | string | Unique |
| order_id | int | 36,062 unique (6.2% of orders returned) |
| product_id | int | 1,286 products |
| return_date | datetime | 2012-07-11 → 2022-12-31 |
| **return_reason** | string | 5 values: wrong_size (35%), defective (20%), not_as_described (18%), changed_mind (17%), late_delivery (10%) |
| return_quantity | int | 1-8, mean 2.7 |
| refund_amount | float | $459 → $160,938, mean $12,784 |

**Return delay**: 5-31 days after order, median 18 days. Returns happen with ~3-week lag.

**Return rate by year** (stable ≈ 6% across all years). No trend.

---

### 9. `products.csv` — catalog
| Col | Type | Detail |
|---|---|---|
| product_id | int | 2,412 unique |
| product_name | string | 2,172 unique |
| **category** | string | 4 values: Streetwear (55%), Outdoor (31%), Casual (8%), GenZ (6%) |
| **segment** | string | 8 values: Activewear, Everyday, Performance, Balanced, Standard, Premium, All-weather, Trendy |
| size | string | S, M, L, XL (evenly distributed 603 each) |
| color | string | 10 colors evenly distributed |
| price | float | $9 → $40,950, median $4,400 |
| cogs | float | $5 → $38,903, mean $3,868 |

**Margin**: (price - cogs) / price ranges 5% → 50%, median 30.6%. This is a fixed attribute per product — no trend over time.

**Big signal**: join order_items.product_id → products.category to compute daily category revenue mix.

---

### 10. `geography.csv` — location mapping
| Col | Type | Detail |
|---|---|---|
| zip | int | 39,948 unique |
| city | string | 42 VN cities (Hanoi, Hai Phong, Hue, Ho Chi Minh etc.) |
| **region** | string | 3 values: East (47%), Central (36%), West (16%) |
| district | string | 39 district values |

**Signal**: daily order count by region — geographic spread. But likely slow-moving, probably not a strong daily signal.

---

### 11. `customers.csv` — customer demographics
| Col | Type | Detail |
|---|---|---|
| customer_id | int | 121,930 |
| zip | int | 31,491 |
| city | string | 42 cities (matches geography) |
| signup_date | datetime | 2012-01-17 → 2022-12-31 |
| **gender** | string | Female (49%), Male (47%), Non-binary (4%) |
| **age_group** | string | 25-34 (30%), 35-44 (26%), 45-54 (19%), 18-24 (14%), 55+ (11%) |
| **acquisition_channel** | string | organic_search (30%), social_media (20%), paid_search (20%), email (12%), referral (10%), direct (8%) |

**Signal**: daily customer demographics (age/gender mix of buyers via customer_id join). Could shift over years.

**Note**: customer_id goes up to 157,563 but only 121,930 records → gaps in IDs (likely deleted accounts?). Not a concern.

---

### 12. `promotions.csv` — campaigns
| Col | Type | Detail |
|---|---|---|
| promo_id | string | 50 unique |
| promo_name | string | 50 unique (descriptive) |
| **promo_type** | string | percentage (90%), fixed (10%) |
| discount_value | float | percentages 10-20 or fixed 50, avg 18.5 |
| start_date / end_date | datetime | 2013-01-31 → 2022-12-31, 29-45 days duration (median 30) |
| **applicable_category** | string | 80% null (all-category), 10% Streetwear-only, 10% Outdoor-only |
| **promo_channel** | string | all_channels (38%), online (26%), email (14%), social_media (12%), in_store (10%) |
| **stackable_flag** | int | 0 (76%), 1 (24%) |
| **min_order_value** | int | 5 values: 0, 50000, 100000, 150000, 200000 |

**Friend missed**: `promo_channel`, `stackable_flag`, `min_order_value`, `applicable_category` — potential signals.

**Caveat**: promo_id coverage in test is unknown. If new promos run in 2023-24 (not in this file), features may not reflect them.

---

### 13. `inventory.csv` — 17 columns, only 4 used!
| Col | Type | Detail | Our usage |
|---|---|---|---|
| snapshot_date | datetime | 126 unique dates (~monthly) | Used |
| product_id | int | 1,624 unique | Used for join |
| stock_on_hand | int | 3-2673, mean 189 | Used |
| **units_received** | int | 1-817, mean 18 | **NOT USED** — monthly supply |
| **units_sold** | int | 1-670, mean 15 | **NOT USED** — monthly demand |
| stockout_days | int | 0-28, mean 1.2 | Used |
| **days_of_supply** | float | 5-68,100, mean 913 | **NOT USED** |
| **fill_rate** | float | 0.07-1.00, mean 0.96 | **NOT USED** — operational health |
| stockout_flag | int | 0/1, mean 0.67 | Used |
| **overstock_flag** | int | 0/1, mean 0.76 | **NOT USED** |
| reorder_flag | int | always 0 | Useless |
| **sell_through_rate** | float | 0-0.85, mean 0.15 | **NOT USED** |
| product_name | string | Lookup |  |
| **category** | string | Already joinable via products | **NOT USED** |
| **segment** | string | Already joinable | **NOT USED** |
| year, month | int | Derivable from date |  |

**Implication**: monthly units_sold per category is a LEADING operational signal. If units_sold dropped in November 2022 for Streetwear, December might follow. Inventory snapshots can be merged to daily via forward-fill within the month.

---

## Cross-file signals we can now construct

### Join keys discovered
```
orders.order_id     ↔ items.order_id, payments.order_id, shipments.order_id, reviews.order_id, returns.order_id
orders.customer_id  ↔ customers.customer_id, reviews.customer_id
orders.zip          ↔ customers.zip ↔ geography.zip
items.product_id    ↔ products.product_id, reviews.product_id, returns.product_id, inventory.product_id
items.promo_id      ↔ promotions.promo_id
```

### Complete list of daily features we can compute (80+ new signals)

**From orders.csv (20 features)**:
- `n_orders`, `n_customers`, `n_unique_zips`
- pct of each order_status: `pct_delivered`, `pct_cancelled`, `pct_returned_status`, `pct_shipped`, `pct_paid`, `pct_created`
- pct of each payment_method: `pct_cc`, `pct_paypal`, `pct_cod`, `pct_applepay`, `pct_banktransfer`
- pct of each device: `pct_mobile`, `pct_desktop`, `pct_tablet`
- pct of each order_source: `pct_organic`, `pct_paid`, `pct_social`, `pct_email`, `pct_referral`, `pct_direct`

**From order_items.csv (8 features)**:
- `n_items`, `total_quantity`, `mean_quantity_per_order`
- `mean_unit_price`, `median_unit_price`
- `total_discount`, `mean_discount_amount`, `pct_discounted`

**From payments.csv (5 features)**:
- `pay_total_sum` (near-perfect revenue proxy, corr 0.9921)
- `pay_mean`, `pay_median`
- `pct_installments_1`, `pct_installments_3`, `pct_installments_6+`

**From products-items join (12 features)**:
- pct of revenue per category: `pct_rev_streetwear`, `pct_rev_outdoor`, `pct_rev_casual`, `pct_rev_genz`
- pct of quantity per category: 4 more
- pct of revenue per segment: 8 features (activewear, everyday, performance, balanced, standard, premium, all-weather, trendy)
- `mean_product_margin`, `median_product_margin`

**From shipments.csv (4 features)**:
- `mean_ship_days`, `mean_deliver_days`
- `mean_ship_fee`
- `pct_orders_shipped`

**From reviews.csv (4 features)** (lag-safe, use `(month, dow)` avg):
- `reviews_per_day`, `mean_rating_30d`, `pct_5star_30d`, `pct_1star_30d`

**From returns.csv (4 features)** (lag-safe):
- `return_count_30d`, `return_rate_30d` (returns/orders)
- Return reason distribution: `pct_wrong_size`, `pct_defective`

**From inventory.csv (8 features)** (forward-fill monthly):
- `monthly_units_sold_total`, `monthly_units_sold_per_category` (4)
- `mean_fill_rate`, `mean_sell_through`, `pct_overstock`, `pct_stockout`

**From geography join (3 features)**:
- pct of daily orders per region: `pct_east`, `pct_central`, `pct_west`

**From customers demographics (6 features)** (via active buyers):
- `pct_buyers_female`, `pct_buyers_male`
- `pct_buyers_18-24`, `pct_buyers_25-34`, `pct_buyers_55+`
- `pct_new_buyers` (first order this day)

**From web_traffic (5 features, adds to existing)**:
- `bounce_rate`, `avg_session_duration`, `unique_visitors`
- One-hot of `traffic_source`: 6 features

**From promotions (5 features)**:
- `n_promo_active` (we had this), `has_promo_stackable`, `promo_min_order_value_mean`, `has_streetwear_promo`, `has_outdoor_promo`

**TOTAL new possible features: 80+**

All must be projected for test via `(month, dow)` average (most) or trend extrapolation (AOV-like monotonic).

---

## Revised V11 Plan (using all findings)

### Prong 1 — The "Direct Reconstruction" strategy (BIGGEST IMPACT)
Since COGS is *exactly* reconstructable and Revenue is *99.2%* reconstructable, we can:
1. Predict daily aggregates: `n_orders`, `items_per_order`, `mean_unit_price`, `category_mix`
2. Compute Revenue = n_orders × items_per_order × mean_unit_price × (1 − discount_rate) × 1.05 (shipping/tax gross-up)
3. Compute COGS = n_orders × items_per_order × (cogs_weighted_by_category_mix)
4. Use tree models to predict each aggregate separately, then multiply

This is fundamentally different from treating Revenue as a black-box target. **Potential impact: 50-150k RMSE reduction** if the aggregate predictions are more stable than direct Revenue prediction.

### Prong 2 — Feature firehose
Add all 80+ new features to friend's architecture (RF+ET+HGB with polynomial weights). Use `(month, dow)` projection for level-safety.

**Priority ordering within the firehose** (based on expected signal):
1. `n_orders`, `n_customers` (r=0.936)
2. `pay_total_sum` (r=0.992)
3. `total_quantity` (r=0.918)
4. Category revenue mix (4 features)
5. Order-source mix per day (6 features)
6. Device type mix per day (3 features)
7. Payment installments distribution (3 features)
8. Traffic decomposition (6 features)
9. Inventory sell-through per category (4 features)
10. Promotion metadata (5 features)
11. Review 30d rolling rating
12. Shipping lead time

### Prong 3 — AOV trend extrapolation
Fit a simple linear trend on yearly `mean_unit_price` (2013: $4,677 → 2022: $6,901, slope ≈ +$250/year). Extrapolate:
- 2023 mean = ~$7,150
- 2024 mean = ~$7,400

Use as a SEPARATE feature that doesn't get zeroed out in test.

### Prong 4 — Two-stage model
Stage 1: predict `pay_total_sum` (cleaner than Revenue). Call this `revenue_proxy`.
Stage 2: learn the Revenue/pay_sum ratio from training (mean 1.0478) and apply.

This decouples the "what's the order aggregate" question from "what fees does the company add on top."

### Prong 5 — Multi-fold CV (replacing misleading single holdout)
Rolling 5-fold validation as described in earlier plan, averaged RMSE.

---

## Concrete first action for v11a

Build `phase7_v11a_val.py`:
1. Pre-compute all 80 daily features from the master dataset
2. Project each for test via `(month, dow)` avg (or trend for AOV)
3. Add to friend's 23-feature base → 100+ feature set
4. Run 5-fold rolling CV with RF+ET+HGB
5. Report: CV RMSE, feature importance top 20, test-day level distribution

If level stays ≥$4.2M AND CV improves over v10c — ship. If level drops below $4.1M, bisect which features leak.

**Expected time for v11a script**: ~15 min design, 15-20s runtime per fold × 5 folds = ~2 min total.

---

## Summary

Before this audit, we were using 11-23 features drawn from 4 files. The Datathon folder actually has **13 files with 80+ derivable daily signals**, including `n_orders` (r=0.936 with Revenue), `pay_total_sum` (r=0.992), and the monotonically-rising AOV that the winning prediction level implicitly depends on.

The single most important structural finding: **`sales.COGS = sum(order_items.quantity × products.cogs)` EXACTLY.** This means if we can predict the daily order volume and category mix, we get COGS without any modeling error at all.

This is the dataset-level lift the user was looking for.
