# V12 Step 1 — Auxiliary File Date Coverage

**Test horizon**: 2023-01-01 → 2024-07-01  (548 days)
**Train cutoff**: ≤ 2022-12-31

## Coverage per file

| Table | Rows | Date col | Min | Max | % train | % test | Verdict |
|---|---:|---|---|---|---:|---:|---|
| sales | 3,833 | Date | 2012-07-04 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| web_traffic | 3,652 | date | 2013-01-01 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| orders | 646,945 | order_date | 2012-07-04 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| order_items | 714,669 | - | - | - | - | - | VIA-JOIN(order_id) |
| payments | 646,945 | - | - | - | - | - | VIA-JOIN(order_id) |
| shipments | 566,067 | ship_date | 2012-07-04 | 2022-12-29 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
|   shipments (via delivery_date) | 566,067 | delivery_date | 2012-07-06 | 2022-12-31 | 100.0% |   0.0% | (secondary date) |
| reviews | 113,551 | review_date | 2012-07-10 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| returns | 39,939 | return_date | 2012-07-11 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| customers | 121,930 | signup_date | 2012-01-17 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| promotions | 50 | start_date | 2013-01-31 | 2022-11-18 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
|   promotions (via end_date) | 50 | end_date | 2013-03-01 | 2022-12-31 | 100.0% |   0.0% | (secondary date) |
| inventory | 60,247 | snapshot_date | 2012-07-31 | 2022-12-31 | 100.0% |   0.0% | PROJECT (no test-horizon coverage) |
| products | 2,412 | - | - | - | - | - | STATIC / via-join |
| geography | 39,948 | - | - | - | - | - | STATIC / via-join |

## Order-id join coverage (for date-less tables)

`order_items.csv` and `payments.csv` inherit their date from `orders.order_date` via `order_id`.

| Aux file | Rows | Min (via orders) | Max (via orders) | % in test horizon |
|---|---:|---|---|---:|
| order_items.csv | 714,669 | 2012-07-04 | 2022-12-31 | 0.0% |
| payments.csv | 646,945 | 2012-07-04 | 2022-12-31 | 0.0% |

## KEY FINDING — every aux table ends 2022-12-31

**All 12 auxiliary tables have ZERO coverage of the test horizon.** The competition has given us no "future" data beyond the training cutoff. Concretely:

- Orders, items, payments, shipments, reviews, returns, customers, inventory, promotions — all stop at 2022-12-31 (promotions stop at 2022-11-18 for start_date).
- `order_items` and `payments` join back to `orders.order_date`, which also ends 2022-12-31 → they too are train-only.

### What this means for V12

**The "Direct COGS Reconstruction" idea in V12_PLAN.md §4 Stage D needs revision.**

We *can* perfectly reconstruct historical COGS as `Σ(quantity × products.cogs)` (verified r = 1.00000 on 2017-2022), but we **cannot** apply this formula to 2023-2024 — we don't have future order_items. So Stage D must use the ratio route:
1. Predict Revenue for 2023-2024 (via Stages A/B/C).
2. Compute historical `cogs_ratio(t) = COGS/Revenue`, fit a trend.
3. Project `cogs_ratio` to 2023-2024.
4. `predicted_COGS = predicted_Revenue × projected_cogs_ratio`.

This keeps the "exact" part: the ratio is bounded, stable, and free of units-volume error.

### The real asymmetry vs friend

Friend uses **1** projected signal (`sessions` via 4D profile). We have **10+** projectable signals he's leaving on the table:

| Source | Profile-projectable signals for V12 |
|---|---|
| `orders.csv` | n_orders, n_unique_customers, pct_desktop, pct_mobile, pct_pix, pct_card, pct_installment |
| `order_items.csv` + `products.csv` | avg_price_per_item, avg_margin_per_item, items_per_order, category_mix (apparel vs accessories vs other) |
| `payments.csv` | mean_installments, pct_high_installment |
| `shipments.csv` | mean_days_to_ship, mean_shipping_fee |
| `reviews.csv` | daily_mean_rating, n_reviews, n_low_ratings |
| `returns.csv` | return_rate (7d trailing), top_return_reason_share |
| `customers.csv` | n_signups |
| `promotions.csv` | n_active_promos, max_discount_active, mean_discount_active |
| `inventory.csv` | n_stockouts, n_low_stock |
| `web_traffic.csv` (full) | unique_visitors, bounce_rate, avg_session_duration, 7 traffic_source shares |

Each of these gets the same treatment: build a `(month, dow, is_dd, is_pd, is_tet)` median profile from 2017-2022, apply to the 548 test days.

### Strategy per file (final verdict)

| File | Strategy | Why |
|---|---|---|
| sales | TARGET | predict directly (Stages A/B/C) |
| web_traffic | PROJECT → feature | friend's approach, extend to more columns & 5D |
| orders | PROJECT daily rollups → feature | n_orders is most-correlated variable |
| order_items | PROJECT daily rollups → feature | avg price, item mix, margin |
| payments | PROJECT daily rollups → feature | mix is modestly predictive |
| shipments | PROJECT daily rollups → feature | likely weak signal, test during Step 7 |
| reviews | PROJECT daily rollups → feature | sentiment leading indicator |
| returns | PROJECT daily rollups → feature | inverse indicator, small n |
| customers | PROJECT daily rollups → feature | signup pulse |
| promotions | JOIN + PROJECT → flags | "any active promo on date D" |
| inventory | PROJECT daily rollups → feature | stockout impact |
| products | STATIC LOOKUP | cogs_ratio anchor, category mix |
| geography | STATIC LOOKUP | region mix only (weak) |

### Correction to V12 plan

V12_PLAN.md §3.4 talked about "Daily commerce rollups (NEW — friend 100% missing)" and listed them as ordinary features. That framing is correct for **training**, but for the **test set** they must all be profile-projected, exactly like `sessions`. I'll note this inline when writing the v12 feature builder in Step 2.

See V12_PLAN.md §3 for the full 42-feature roster that builds on this coverage map.
