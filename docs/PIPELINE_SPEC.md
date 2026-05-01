# DATATHON 2026 — Master Data Pipeline Technical Specification

**Project:** GridBreaker (VinTelligence DATATHON 2026 — Round 1)
**Document type:** Single source of truth for data architecture and ETL pipeline
**Status:** Draft v1.0 — awaiting decisions flagged in Section 7
**Author role:** Senior Data Architect

---

## 1. Executive Summary

### 1.1 Business context
The dataset simulates a Vietnamese fashion e-commerce operator from 2012-07-04 to 2022-12-31 (train) with a forecast horizon of 2023-01-01 to 2024-07-01 (test). The competition has three scored components:

| Part | Weight | Output |
|---|---|---|
| Part 1 — Multiple choice (MCQ) | 20% | 10 deterministic answers derived from the data |
| Part 2 — EDA and visualization | 60% | Insight narrative across four analytical tiers (descriptive, diagnostic, predictive, prescriptive) |
| Part 3 — Revenue forecasting | 20% | Daily `Revenue` prediction scored on MAE, RMSE, R² on a Kaggle leaderboard |

### 1.2 Engineering objective
Build a reproducible pipeline (Jupyter Notebook) that ingests 14 raw CSV tables, audits integrity, resolves granularity mismatches, and produces **two** analytical master datasets:

- **`df_txn`** — line-item granularity, used for MCQ computation and Part 2 EDA.
- **`df_daily`** — daily granularity aligned with the `sales.csv` target, used for Part 3 forecasting feature engineering.

A single master dataset is deliberately rejected (see Section 3.3).

### 1.3 Non-goals
- This document does not cover model selection, hyperparameter tuning, or Kaggle submission logistics.
- The pipeline does not support incremental loads. All data is assumed to be loaded in full per execution.

---

## 2. Schema Review and Gap Analysis

### 2.1 Entity inventory

The dataset contains 14 source tables (excluding `sample_submission.csv`) organized in four logical layers:

| Layer | Tables | Nature |
|---|---|---|
| Master (reference) | `products`, `customers`, `promotions`, `geography` | Slowly changing dimensions, static per snapshot |
| Transaction (events) | `orders`, `order_items`, `payments`, `shipments`, `returns`, `reviews` | Event-sourced, append-only in principle |
| Analytical (target) | `sales.csv` | Aggregate daily KPI, treated as label |
| Operational (signal) | `inventory`, `web_traffic` | Exogenous demand drivers with mixed granularity |

### 2.2 Normalization assessment

The schema is **not fully normalized**. Four violations of Third Normal Form (3NF) were identified and must be resolved in the audit phase to prevent silent divergence:

**Violation N1 — Redundant `payment_method`.** The column exists in both `orders` (transactional header) and `payments` (transactional detail). Both are described as "payment method used." Unless the schema explicitly models a discrepancy between intended and executed payment method, one source must be designated canonical. **Recommendation:** treat `payments.payment_method` as canonical; retain `orders.payment_method` for cancelled orders where no `payments` record exists. Detect and log cases where the two disagree for non-cancelled orders.

**Violation N2 — Redundant `city` in `customers`.** The column duplicates information derivable from `customers.zip → geography.city`. The existence of a standalone column implies either (a) denormalization for query performance, or (b) historical divergence (a customer's recorded city no longer matches the zip's canonical city after a geography update). **Recommendation:** audit equality, flag divergence as a data quality metric, and use `geography.city` via join for analytical purposes.

**Violation N3 — Denormalized columns in `inventory`.** `product_name`, `category`, and `segment` are copied from `products`. This is a classic star-schema denormalization and should not be relied upon. **Recommendation:** drop these columns after load and re-derive by joining `products` on `product_id`. Validate equality on a sample before dropping.

**Violation N4 — Mixed stock and flow metrics in `inventory`.** `stock_on_hand` is a point-in-time measurement at `snapshot_date` (month-end), but `units_sold`, `units_received`, and `stockout_days` are flow metrics spanning the calendar month. These are semantically different and require distinct lag strategies when used as forecasting features (see Section 5.6).

### 2.3 Identified gaps and design issues

Gaps flagged during schema audit, each with a proposed mitigation:

**G1 — `order_items` lacks an explicit primary key.** The natural candidate `(order_id, product_id)` is **not guaranteed unique**; the same SKU may appear on two separate lines of the same order (e.g., different promotions applied, or front-end duplication). Before any join, introduce a surrogate key `line_item_id` via `reset_index()`. All downstream fan-out checks must use this key.

**G2 — Stacking rule is not enforced at the schema level.** `order_items.promo_id_2` is allowed to be non-null, but the business constraint that both `promo_id` and `promo_id_2` must satisfy `promotions.stackable_flag = 1` is implicit. Must be validated explicitly (audit rule B7 in Section 4.3).

**G3 — Promotion temporal validity is not enforced.** An `order_items` row can reference a `promo_id` whose `order_date` lies outside `[promotions.start_date, promotions.end_date]`. Must be validated.

**G4 — Promotion category applicability is not enforced.** `promotions.applicable_category` is nullable. When non-null, the referenced `order_items.product_id` must belong to a product whose `products.category` matches. Must be validated.

**G5 — `returns` allows partial refunds without a clear rule.** `refund_amount` can be any value ≤ `quantity × unit_price`. There is no column indicating partial vs. full refund. This complicates the Revenue reconciliation: if a refund is partial, the Revenue impact is not simply the full line value. Pipeline must treat `refund_amount` as the authoritative net reduction.

**G6 — `web_traffic.traffic_source` cardinality is undocumented.** The table may be (a) one row per day (with the dominant or latest source recorded), or (b) multiple rows per day (one per traffic source). This is **critical for aggregation strategy**: option (b) requires weighted aggregation before joining to `df_daily`. Must be determined empirically in Phase 1 profiling.

**G7 — No audit trail of order state transitions.** `orders.order_status` captures the terminal state only. It is impossible to reconstruct when an order transitioned from `pending → shipped → delivered → returned`. Shipping and delivery dates are available in `shipments`; return dates are in `returns`. The "pending" and "cancelled" transition timestamps are lost. **Impact:** limited; cancellation rate analysis is coarser than possible.

**G8 — No session-level attribution between `web_traffic` and `orders`.** Web traffic is aggregated; orders carry `order_source` as a categorical. There is no session ID to link a specific session to a specific order. **Impact:** Part 2 attribution analysis is limited to aggregate correlation.

**G9 — `sample_submission.csv` schema is assumed, not verified.** The exam document implies columns `(Date, Revenue, COGS)` but the file has not been inspected in this planning phase. Must inspect in Phase 1.

**G10 — `sales_train.csv` vs `sales.csv` naming.** The exam document refers to `sales_train.csv` but the folder contains `sales.csv`. Assumed identical; must verify date range (`2012-07-04 → 2022-12-31`) matches the documented training split.

**G11 — Currency and timezone are unspecified.** Monetary columns are assumed to be in VND (Vietnamese Dong). Dates are assumed to be in Asia/Ho_Chi_Minh (UTC+7). If the data is ever joined with external data at sub-day resolution (e.g., web session timestamps), timezone will matter. Within this dataset everything is day-truncated, so impact is limited.

**G12 — No explicit Lunar New Year (Tet) calendar.** For a Vietnamese fashion retailer, Tet is the single largest annual revenue event and its date varies by year (solar calendar). The dataset provides no holiday flag. **Mitigation:** hardcode Tet dates for 2012–2024 as a derived feature (does not violate the "no external data" constraint as calendar constants are not external data).

### 2.4 Scalability assessment

The current dataset fits in memory on a standard workstation (15 CSVs, likely sub-gigabyte total). For this competition, no distributed compute is required. However, the pipeline design should:

- Use **Parquet** for intermediate exports (10–50× faster read, preserves dtypes).
- Partition `df_txn` by `order_date` year when exporting, in case the dataset grows.
- Use **categorical dtypes** for low-cardinality string columns (`category`, `segment`, `region`, `payment_method`, etc.) to reduce memory by 70–90%.
- Avoid `pd.DataFrame.apply` on large frames; use vectorized operations or `merge_asof`.

---

## 3. Detailed Data Schema and Relationships

### 3.1 Reference diagram
The Entity-Relationship diagram is maintained in the companion file `schema_diagram.mermaid`. This section formalizes cardinality and join semantics in tabular form.

### 3.2 Cardinality matrix

| Parent | Child | Parent key | Child FK | Cardinality | Optionality on child | Notes |
|---|---|---|---|---|---|---|
| `geography` | `customers` | `zip` | `zip` | 1 : N | required | Registered address |
| `geography` | `orders` | `zip` | `zip` | 1 : N | required | Shipping address; may differ from customer's |
| `customers` | `orders` | `customer_id` | `customer_id` | 1 : N | required | Customer lifecycle backbone |
| `customers` | `reviews` | `customer_id` | `customer_id` | 1 : N | required | Reviews also FK to order and product |
| `orders` | `order_items` | `order_id` | `order_id` | 1 : N (≥ 1) | required | Spine of line-item analysis |
| `orders` | `payments` | `order_id` | `order_id` | 1 : 1 | conditional | May not exist for cancelled orders (see G3 note) |
| `orders` | `shipments` | `order_id` | `order_id` | 1 : (0 or 1) | conditional | Only for status ∈ {shipped, delivered, returned} |
| `orders` | `returns` | `order_id` | `order_id` | 1 : (0..N) | conditional | Only for status = returned |
| `orders` | `reviews` | `order_id` | `order_id` | 1 : (0..N) | conditional | Approximately 20% of delivered orders |
| `products` | `order_items` | `product_id` | `product_id` | 1 : N | required | |
| `products` | `returns` | `product_id` | `product_id` | 1 : N | required | Return is per-product |
| `products` | `reviews` | `product_id` | `product_id` | 1 : N | required | Review is per-product |
| `products` | `inventory` | `product_id` | `product_id` | 1 : N | required | One row per product per month |
| `promotions` | `order_items` | `promo_id` | `promo_id` | (0..1) : N | optional | Primary promotion |
| `promotions` | `order_items` | `promo_id` | `promo_id_2` | (0..1) : N | optional | Stacking promotion; requires `stackable_flag = 1` on both |

### 3.3 Dual-master architecture rationale

A single monolithic master dataset is rejected. `sales.csv` is aggregated at the day-company granularity and carries no product, customer, or region dimension. Forcing it into a line-item master would require either (a) broadcasting the daily total across all line-items (creating spurious correlation and violating granularity semantics), or (b) aggregating line-items up to the daily grain (eliminating the dimensional features that make Part 2 analysis possible). The dual-master design isolates these concerns:

**`df_txn` — line-item master** is the canonical source for Part 2 EDA and all MCQ computation. Spine: `order_items` augmented with surrogate `line_item_id`.

**`df_daily` — daily master** is the canonical source for Part 3 forecasting features. Spine: a complete calendar `date_spine` from 2012-07-04 to 2024-07-01, with daily aggregates and lag/rolling features.

A derived table `df_order` (one row per `order_id`) may be materialized for order-level analytics, but is not a blocker for Phase 4.

### 3.4 Join strategy for `df_txn`

Join order matters for fan-out detection. Each join must preserve the count of `line_item_id` values.

```
order_items  (spine; add line_item_id)
 ├── LEFT JOIN orders          on order_id
 │    ├── LEFT JOIN customers  on customer_id        -> suffix _cust
 │    │    └── LEFT JOIN geography as geo_home on zip -> suffix _home
 │    └── LEFT JOIN geography as geo_ship on zip      -> suffix _ship
 ├── LEFT JOIN products        on product_id
 ├── LEFT JOIN promotions p1   on promo_id            -> suffix _p1
 ├── LEFT JOIN promotions p2   on promo_id_2          -> suffix _p2
 ├── LEFT JOIN payments        on order_id            (1:1, safe)
 ├── LEFT JOIN shipments       on order_id            (1:0..1, safe)
 ├── LEFT JOIN returns_agg     on (order_id, product_id)   (pre-aggregated)
 └── LEFT JOIN reviews_agg     on (order_id, product_id)   (pre-aggregated)
```

Two geography aliases are explicit: `geo_ship` (via `orders.zip`, used for revenue-by-region) and `geo_home` (via `customers.zip`, used for demographic segmentation). A derived boolean `is_shipping_to_home = (orders.zip == customers.zip)` is created.

### 3.5 Join strategy for `df_daily`

```
date_spine (2012-07-04 to 2024-07-01, daily)
 ├── LEFT JOIN orders_daily           on date (from orders.order_date)
 ├── LEFT JOIN items_daily            on date (from orders.order_date via join)
 ├── LEFT JOIN returns_daily          on date (from returns.return_date)
 ├── LEFT JOIN reviews_daily          on date (from reviews.review_date)
 ├── LEFT JOIN web_traffic_daily      on date
 ├── LEFT JOIN inventory_monthly_lag1 on month (lag-1 month; forward-filled to daily)
 └── LEFT JOIN sales_train            on Date (target variable; train only)
```

Inventory alignment uses the **monthly aggregate plus one-month lag strategy** (see Section 5.6). This eliminates any risk of using same-month inventory signals to predict same-month revenue.

### 3.6 Promotion join: two-alias strategy

`promotions` is joined twice with aliases `p1` and `p2`. Derived discount features:

```
discount_p1 = f(p1.promo_type, p1.discount_value, quantity, unit_price)
discount_p2 = f(p2.promo_type, p2.discount_value, quantity, unit_price)
discount_total = coalesce(discount_p1, 0) + coalesce(discount_p2, 0)
```

Where `f(percentage, v, q, u) = q × u × v / 100` and `f(fixed, v, q, u) = q × v`.

Note: this derived `discount_total` is for auditing against the `order_items.discount_amount` column provided by the source. Large discrepancies indicate either (a) a different business rule applied at order time, or (b) data corruption.

---

## 4. Data Validation Checklist

Each check produces a numeric metric and a pass/fail verdict against a stated threshold. All results are logged to a dedicated audit section of the notebook.

### 4.1 Primary key integrity (PK)

| ID | Check | Threshold |
|---|---|---|
| PK1 | `products.product_id` is unique | 0 duplicates |
| PK2 | `customers.customer_id` is unique | 0 duplicates |
| PK3 | `promotions.promo_id` is unique | 0 duplicates |
| PK4 | `geography.zip` is unique | 0 duplicates |
| PK5 | `orders.order_id` is unique | 0 duplicates |
| PK6 | `payments.order_id` is unique (1:1 with orders) | 0 duplicates |
| PK7 | `shipments.order_id` is unique (1:0..1 with orders) | 0 duplicates |
| PK8 | `returns.return_id` is unique | 0 duplicates |
| PK9 | `reviews.review_id` is unique | 0 duplicates |
| PK10 | `(inventory.product_id, snapshot_date)` is unique | 0 duplicates |

### 4.2 Referential integrity (FK)

All FKs are checked with a standard helper `audit_fk(child, child_key, parent, parent_key)` that returns the count and percentage of orphan keys in the child relative to the parent.

| ID | Child.FK | → Parent.PK | Acceptable orphan rate |
|---|---|---|---|
| FK1 | `customers.zip` | `geography.zip` | 0% |
| FK2 | `orders.customer_id` | `customers.customer_id` | 0% |
| FK3 | `orders.zip` | `geography.zip` | 0% |
| FK4 | `order_items.order_id` | `orders.order_id` | 0% |
| FK5 | `order_items.product_id` | `products.product_id` | 0% |
| FK6 | `order_items.promo_id` | `promotions.promo_id` | 0% (null is allowed, not orphan) |
| FK7 | `order_items.promo_id_2` | `promotions.promo_id` | 0% |
| FK8 | `payments.order_id` | `orders.order_id` | 0% |
| FK9 | `shipments.order_id` | `orders.order_id` | 0% |
| FK10 | `returns.order_id` | `orders.order_id` | 0% |
| FK11 | `returns.product_id` | `products.product_id` | 0% |
| FK12 | `reviews.order_id` | `orders.order_id` | 0% |
| FK13 | `reviews.product_id` | `products.product_id` | 0% |
| FK14 | `reviews.customer_id` | `customers.customer_id` | 0% |
| FK15 | `inventory.product_id` | `products.product_id` | 0% |

Non-zero orphan rates do not necessarily block the pipeline but must be logged. Decision policy is captured in Section 7.

### 4.3 Business rule compliance

| ID | Rule | Rationale |
|---|---|---|
| B1 | `products.cogs < products.price` for every row | Exam-stated constraint |
| B2 | `order_items.quantity > 0` and `unit_price > 0` | Transactional sanity |
| B3 | `reviews.rating ∈ {1, 2, 3, 4, 5}` | Stated domain |
| B4 | `shipments.ship_date >= orders.order_date` | Temporal logic |
| B5 | `shipments.delivery_date >= shipments.ship_date` | Temporal logic |
| B6 | `returns.return_date >= shipments.delivery_date` (when delivery exists) | Cannot return before delivery |
| B7 | If `order_items.promo_id_2 IS NOT NULL` then both referenced promotions have `stackable_flag = 1` | G2 |
| B8 | `orders.order_date ∈ [promotions.start_date, promotions.end_date]` for each applied promo | G3 |
| B9 | If `promotions.applicable_category IS NOT NULL`, then `products.category = promotions.applicable_category` for the associated `product_id` | G4 |
| B10 | Orders with `order_status ∈ {shipped, delivered, returned}` have exactly one `shipments` row; others have none | Spec |
| B11 | Orders with `order_status = returned` have at least one `returns` row | Spec |
| B12 | `reviews.review_date >= orders.order_date` | Temporal logic |
| B13 | `customers.signup_date <= min(orders.order_date)` per customer | Temporal logic |
| B14 | `returns.refund_amount <= (return_quantity × order_items.unit_price)` for the matched line | Bounds check |
| B15 | `order_items.discount_amount` matches formula from `promotions` within rounding tolerance | Audit only |

### 4.4 Fan-out detection

Before and after each join in Section 3.4, record `n_rows` and `n_unique_line_item_id`. Assert `n_unique_line_item_id` is constant. Log `n_rows` divergence as a warning.

### 4.5 Financial reconciliation

Revenue in `sales.csv` must be reconciled against line-item data. Because the exact aggregation rule is not specified, test five candidate formulas against all dates in the train split and select the formula with the lowest MAE versus `sales.Revenue`.

| Formula | Definition |
|---|---|
| F1 | `sum(quantity × unit_price)` grouped by `orders.order_date`, all orders |
| F2 | F1 minus `sum(refund_amount)` grouped by `returns.return_date` |
| F3 | F1 minus `sum(refund_amount)` attributed to `orders.order_date` of the original order |
| F4 | F1 restricted to `order_status ∈ {delivered, shipped}` |
| F5 | F4 minus refunds attributed to `orders.order_date` |

The winning formula becomes the canonical `REVENUE_DEFINITION` constant used throughout feature engineering. Analogous reconciliation is performed for COGS using `quantity × products.cogs`.

**Acceptance criterion:** at least one formula must achieve MAE ≤ 1% of mean daily revenue. Otherwise the reconciliation is considered failed and the target is treated as externally defined.

### 4.6 Temporal leakage guards (forecasting only)

| ID | Check |
|---|---|
| L1 | No feature column in `df_daily` at date `D` uses any value observed at date `D' > D` |
| L2 | Inventory features at date `D` use snapshots with `snapshot_date < first_day_of_month(D)` only |
| L3 | Rolling features use `shift(horizon)` where `horizon ≥ 1 day` |
| L4 | Target encoding (if used) is computed out-of-fold using `TimeSeriesSplit` |
| L5 | Cross-validation is temporal (expanding or sliding window), never random K-fold |
| L6 | `sales_test.csv` Revenue and COGS columns are **never** used as features (exam-stated disqualification condition) |

### 4.7 Granularity alignment

| ID | Check |
|---|---|
| G_CHK1 | `web_traffic` row count per date is constant (either 1 or N for fixed N). If variable, long-format with multiple `traffic_source` rows is confirmed |
| G_CHK2 | `inventory.snapshot_date` is always the last calendar day of its month |
| G_CHK3 | `date_spine` has no missing days in the advertised train range |
| G_CHK4 | `sales.csv` has exactly one row per date in the train range; no duplicates |

---

## 5. Implementation Roadmap

The notebook is structured into ten phases. Each phase has explicit inputs, outputs, and exit criteria. A phase does not start until its predecessor's exit criteria pass.

### Phase 0 — Environment and configuration
**Inputs:** None.
**Actions:** Import libraries (pandas, numpy, pyarrow, matplotlib). Set `random_seed = 42`. Define path constants. Define helper functions: `load_csv_typed()`, `audit_fk()`, `assert_no_fanout()`, `reconcile_revenue()`, `check_leakage()`.
**Outputs:** Helper module loaded.
**Exit criterion:** All helpers importable; seed set.

### Phase 1 — Ingest and profile
**Inputs:** 14 CSVs plus `sample_submission.csv`.
**Actions:** Load each CSV with explicit dtype specification. Parse date columns. Profile shape, dtypes, null counts, memory, cardinality. Inspect `sample_submission.csv` schema (resolves G9). Verify `sales.csv` date range (resolves G10). Check `web_traffic` row-per-date cardinality (resolves G6).
**Outputs:** `raw` dict of DataFrames; `profile_report` DataFrame.
**Exit criterion:** All files loaded; profiling summary printed; open questions G6/G9/G10 resolved.

### Phase 2 — Audit
**Inputs:** `raw`.
**Actions:** Run all checks in Sections 4.1, 4.2, 4.3, 4.7. Produce audit report.
**Outputs:** `audit_results` DataFrame with rule ID, metric, verdict.
**Exit criterion:** No PK violation; FK orphan rate ≤ policy threshold; business rule failures logged and triaged.

### Phase 3 — Pre-join aggregation of satellite tables
**Inputs:** `raw`.
**Actions:** Build `returns_agg` grouped by `(order_id, product_id)`: `return_quantity_sum`, `refund_amount_sum`, `n_return_events`, `first_return_date`, `dominant_return_reason`. Build `reviews_agg` grouped by `(order_id, product_id)`: `rating_mean`, `rating_last`, `n_reviews`, `first_review_date`, `has_review_flag`.
**Outputs:** `returns_agg`, `reviews_agg`.
**Exit criterion:** Both aggregates have unique keys on the stated grain.

### Phase 4 — Build `df_txn`
**Inputs:** `raw`, `returns_agg`, `reviews_agg`.
**Actions:** Assign `line_item_id = arange(len(order_items))`. Execute joins per Section 3.4. After each join, record `n_rows` and `n_unique_line_item_id`; assert no fan-out. Resolve column name collisions using explicit suffixes.
**Outputs:** `df_txn` with approximately 30 columns.
**Exit criterion:** Row count equals initial `len(order_items)`; all spine keys retained.

### Phase 5 — Financial reconciliation
**Inputs:** `df_txn`, `raw['sales']`.
**Actions:** Compute five Revenue formulas (Section 4.5). Compare against `sales.Revenue` by date. Select winning formula. Repeat for COGS. Persist selected formulas as module-level constants.
**Outputs:** `REVENUE_DEFINITION`, `COGS_DEFINITION`, reconciliation report.
**Exit criterion:** At least one formula meets the 1% MAE acceptance criterion, or the discrepancy is explicitly documented and accepted.

### Phase 6 — Build `df_daily`
**Inputs:** `raw`, `df_txn`.
**Actions:** Construct `date_spine`. Aggregate `orders`, `order_items`, `returns`, `reviews`, `web_traffic` to daily grain. Align `inventory` using the monthly-aggregate-plus-lag-1 strategy: group by `year_month`, compute aggregate statistics (`total_stock_on_hand`, `mean_fill_rate`, `pct_stockout_products`, `total_units_sold`), shift by one month, forward-fill to daily. Merge everything onto `date_spine`. Attach `sales.Revenue` and `sales.COGS` as the target (train range only).
**Outputs:** `df_daily` with daily features and target.
**Exit criterion:** No missing dates in `date_spine`; inventory features are null for the first month of data (expected due to shift).

### Phase 7 — Feature engineering
**Inputs:** `df_daily`.
**Actions:** Add calendar features (day of week, month, quarter, is_weekend, is_tet). Hardcode Lunar New Year dates for 2012–2024 (resolves G12). Add lag features (`revenue_lag_1`, `revenue_lag_7`, `revenue_lag_28`, `revenue_lag_365`). Add rolling features (`revenue_rolling_7_mean`, `revenue_rolling_28_std`). Add year-over-year and month-over-month growth features. Every lag and rolling is computed with `shift(horizon)` where `horizon ≥ 1`.
**Outputs:** `df_daily_feat`.
**Exit criterion:** Feature matrix has no look-ahead columns (verified by L1–L3 checks).

### Phase 8 — Leakage guard and validation split
**Inputs:** `df_daily_feat`.
**Actions:** Run `check_leakage()`. Define train/validation temporal split: train = 2012-07-04 to 2021-12-31; validation = 2022-01-01 to 2022-12-31; test = 2023-01-01 to 2024-07-01 (features only, no target). Confirm no row from validation or test appears in any lag or rolling feature computed for the train subset.
**Outputs:** `train_df`, `valid_df`, `test_df`.
**Exit criterion:** L1–L6 checks all pass.

### Phase 9 — Export
**Inputs:** `df_txn`, `df_daily_feat`, splits.
**Actions:** Export `df_txn.parquet` (partitioned by `order_year`), `df_daily_feat.parquet`, `train/valid/test` splits. Export `audit_results.csv` and `reconciliation_report.csv`.
**Outputs:** Files in `Datathon/outputs/`.
**Exit criterion:** Re-importing a Parquet file yields a frame identical to the in-memory version (checksum equality on numeric columns).

---

## 6. Pipeline Risk Register (MCQ-driven)

Each exam question introduces a specific computational risk. The pipeline must produce answers consistent with the following interpretations:

| Qn | Primary risk | Resolution in pipeline |
|---|---|---|
| Q1 | Median inter-order gap per customer, not median of per-customer means | Compute gaps within each customer_id, concatenate, take global median; filter customers with ≥ 2 orders |
| Q2 | Margin is per-product, then averaged within segment | `(price - cogs) / price` per product, then `groupby(segment).mean()` |
| Q3 | Case-sensitive category match, join on product_id | Confirm `category` values via `.unique()` before filtering |
| Q4 | Multiple rows per date per source likely; use simple mean of `bounce_rate` | Confirmed in Phase 1 via G_CHK1 |
| Q5 | Denominator is all order_items rows; numerator uses `promo_id IS NOT NULL` only (not `promo_id_2`) | Explicit per the question wording |
| Q6 | Non-null age_group filter; orders per customer = orders / distinct customers in group | |
| Q7 | `sales.csv` lacks region; use `orders.zip → geography.region` path for region revenue | Use shipping zip (orders.zip), not customer home zip |
| Q8 | Cancelled orders may not have `payments` rows; use `orders.payment_method` | |
| Q9 | Return rate denominator is line-item count at that size, not order count | Join both `returns` and `order_items` to `products` filtered to S/M/L/XL |
| Q10 | Group `payments` by `installments`, take mean `payment_value` | Straight groupby |

---

## 7. Open Decisions Required Before Code Implementation

The following product-level decisions must be resolved before Phase 2 begins. Default recommendations are in bold.

**D1 — Orphan key policy.** When a child row has an FK value absent from the parent, what is the handling? Options: (a) drop the child row, (b) retain with null parent attributes, (c) fail loudly. **Recommendation: (b) retain with null.** Rationale: dropping biases revenue reconciliation; failing loudly blocks the pipeline if any single orphan exists.

**D2 — Cancelled order inclusion in `df_txn`.** Cancelled orders have no revenue impact but carry useful signal for fraud detection and Q1/Q8. **Recommendation: include with `is_cancelled` flag.**

**D3 — Inventory alignment strategy.** Strategy A uses `merge_asof(direction='backward')` to match each date with the most recent snapshot. Strategy B aggregates monthly, shifts by one month, and forward-fills. **Recommendation: B** because it guarantees no same-month leakage and produces interpretable monthly features for Part 3 SHAP analysis.

**D4 — `df_daily` date coverage.** Options: (a) train range only (2012-07-04 to 2022-12-31), (b) extended to test range (to 2024-07-01) with target null in test rows. **Recommendation: (b)** with an `is_test` flag, so feature engineering is performed end-to-end consistently.

**D5 — Canonical `payment_method` (N1).** **Recommendation:** `payments.payment_method` when present, else `orders.payment_method`. Log disagreements.

**D6 — `inventory` denormalized columns (N3).** **Recommendation:** drop `product_name`, `category`, `segment` from `inventory` after load; re-derive via join to `products` when needed.

**D7 — Revenue reconciliation failure.** If no formula in Phase 5 meets the 1% MAE threshold, how is the pipeline to proceed? **Recommendation:** select the lowest-MAE formula, document the residual, and proceed. Do not block the pipeline; the unexplained residual becomes a known modeling limitation.

---

## 8. Deliverables from This Specification

Upon approval of Section 7 decisions, the following artifacts will be produced:

1. `pipeline.ipynb` — Jupyter notebook implementing Phases 0 through 9.
2. `audit_results.csv` — machine-readable record of all Section 4 checks.
3. `reconciliation_report.csv` — Revenue/COGS formula comparison.
4. `df_txn.parquet` — line-item master.
5. `df_daily_feat.parquet` — daily master with features and target.
6. `train.parquet`, `valid.parquet`, `test.parquet` — temporal splits.
7. `schema_diagram.mermaid` — visual ER diagram (already produced).

---

## 9. Revision History

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-04-21 | Senior Data Architect | Initial consolidated specification, translated from planning discussion, hardened with gap analysis (G1–G12) and normalization review (N1–N4). |
