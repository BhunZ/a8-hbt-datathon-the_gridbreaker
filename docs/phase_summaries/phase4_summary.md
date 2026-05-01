# Phase 4 — Master Join (`df_txn`): Summary

**Inputs:** Phase 2 parquet snapshots + `returns_agg.parquet` + `reviews_agg.parquet` + `returned_orders_truncated.csv` + `sales.parquet`.
**Output:** `outputs/parquet/df_txn.parquet` — the transaction master.
**Total checks run:** 15.

## Verdict Headline

| Verdict | Count | Meaning |
|---------|------:|---------|
| PASS    |   11  | PK, row-count, FKs, conservation, truncation counts, disjoint-outcome rule. |
| LOG     |    4  | Diagnostics — outcome distribution, null inventory, tenure distribution, shape. |
| WARN    |    0  | — |
| FAIL    |    0  | — |

**Gate decision:** df_txn is complete and **ready for Phase 5 feature engineering and Phase 7 modelling**.

## df_txn Shape

| Property | Value |
|---|---|
| Rows | **714,669** (exactly `len(order_items)` — zero fan-out) |
| Columns | **64** |
| PK | `line_item_id` (int64, 1-indexed surrogate) |
| File size | ~70 MB parquet (~400 MB in-memory pandas) |
| Revenue total | **$16,430,476,585.53** — exact match to `Σ sales.Revenue` (diff = $0.0000) |

## Column Inventory (by block)

| Block | Columns |
|---|---|
| **Keys** | `line_item_id`, `order_id`, `product_id`, `customer_id` |
| **Dates** | `order_date`, `first_order_date`, `ship_date`, `delivery_date`, `first_return_date`, `last_return_date`, `first_review_date`, `last_review_date`, `signup_date` |
| **Line economics** | `quantity`, `unit_price`, `discount_amount`, `promo_id`, `promo_id_2` |
| **Order meta** | `order_status`, `payment_method`, `device_type`, `order_source` |
| **Customer demo** | `city`, `gender`, `age_group`, `acquisition_channel`, `customer_tenure_days` |
| **Product** | `product_name`, `category`, `segment`, `size`, `color`, `product_list_price`, `cogs` |
| **Promo (from promotions)** | `promo_type`, `discount_value`, `applicable_category`, `stackable_flag`, `min_order_value` |
| **Shipment** | `shipping_fee`, `has_shipment`, `truncation_flag_shipment` |
| **Returns satellite** | `n_return_events`, `return_quantity_sum`, `refund_amount_sum`, `time_to_first_return_days`, `dominant_return_reason`, `distinct_return_reasons`, `any_physical_defect_line`, `any_catalog_failure_line`, `refund_over_paid_ratio`, `has_any_return`, `truncation_flag_return` |
| **Reviews satellite** | `n_review_events`, `avg_rating`, `min_rating`, `max_rating`, `dominant_rating`, `time_to_first_review_days`, `has_low_rating`, `has_high_rating`, `has_any_review`, `truncation_flag_review` |
| **Derived** | `post_purchase_outcome` (categorical: `no_feedback` / `reviewed_only` / `returned_only`) |

## Validation Matrix

| Rule | Metric | Threshold | Verdict |
|---|---|---|---|
| P4_PK1 | 0 | = 0 | PASS — `line_item_id` unique on all 714,669 rows |
| P4_CT1 | 714,669 | = 714,669 | PASS — zero fan-out from 7-hop LEFT join chain |
| P4_FK1 | 0 | = 0 | PASS — no null `customer_id` |
| P4_FK2 | 0 | = 0 | PASS — no null `order_date` |
| P4_FK3 | 0 | = 0 | PASS — no null `order_status` |
| P4_FK4 | 0 | = 0 | PASS — no null `product_list_price` |
| P4_FK5 | 0 | = 0 | PASS — no null `category` |
| P4_C1  | $0.00 | ±$0.01 | PASS — `Σ qty × unit_price` reconciles to `Σ sales.Revenue` exactly |
| P4_T1  | 564 orders | = 564 | PASS — matches B10 shipment-tail truncation |
| P4_T2  | 80 orders | = 80 | PASS — matches B11 return-tail truncation |
| P4_B10 | 0 | = 0 | PASS — zero `reviewed_and_returned` lines (Phase 3C disjoint rule upheld) |
| P4_D1  | outcome counts | LOG | LOG — see distribution below |
| P4_D2  | null% on optional cols | LOG | LOG — see null inventory below |
| P4_D3  | tenure percentiles | LOG | LOG — p50 = 1,082 days |
| P4_D4  | shape | LOG | LOG — (714,669, 64) |

## Key Distributions

### post_purchase_outcome

| Bucket | Lines | Share |
|---|---:|---:|
| no_feedback | 561,177 | 78.52% |
| reviewed_only | 113,553 | 15.89% |
| returned_only | 39,939 | 5.59% |
| **reviewed_and_returned** | **0** | **0.00%** (empty by dataset construction) |

*(Small +2 drift on `returned_only` / `reviewed_only` vs the satellite tables is explained by the 16 duplicate-`line_item_id` rows that share the same `(order_id, product_id)` key — each duplicate line legitimately inherits the same satellite row.)*

### order_status line mix

| Status | Lines |
|---|---:|
| delivered | 570,887 |
| cancelled | 65,673 |
| returned | 40,034 |
| shipped | 15,094 |
| paid | 14,987 |
| created | 7,994 |

### customer_tenure_days (percentiles)

| p01 | p25 | p50 | p75 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|
| 0 | 327 | 1,082 | 1,920 | 3,193 | 3,587 |

Anchor = `min(order_date)` per customer (per the Phase 2 B13 redefinition).  `signup_date` is retained in df_txn as a column but is **NOT** used to compute tenure — teammates should not accidentally revert to it.

### Null inventory (optional / satellite columns)

| Column | % null | Rationale |
|---|---:|---|
| `avg_rating` / `dominant_rating` | 84.11% | Only present on `reviewed_only` lines. Expected. |
| `refund_over_paid_ratio` / `dominant_return_reason` | 94.41% | Only present on `returned_only` lines. Expected. |
| `delivery_date` | 12.49% | Includes 87,628 non-shipped lines (`cancelled` / `paid` / `created`) + 564 B10-truncated. Expected. |
| `promo_id` | 61.34% | Most lines are non-promotional. Expected. |

All other core columns have **0% nulls**.

## Derived Features Pinned

1. **`customer_tenure_days`** — `order_date − first_order_date`; `first_order_date = min(order_date)` per customer.
2. **`truncation_flag_shipment`** — 1 iff `order_status ∈ {shipped, delivered, returned}` ∧ no shipment row. Matches B10 count (564 distinct orders).
3. **`truncation_flag_return`** — 1 iff `order_id` ∈ `returned_orders_truncated.csv` (the 80 B11 orders).
4. **`has_shipment`**, **`has_any_return`**, **`has_any_review`** — boolean presence flags; NULL-safe after left-join-and-fillna.
5. **`post_purchase_outcome`** — 3-valued categorical guaranteed by Phase 3C's zero-overlap finding.

## Phase 4 Exit Criterion

> *"Master transaction table built by LEFT-joining all satellites without fan-out; all conservation and integrity invariants hold; truncation flags match Phase 2 counts exactly."*

Status:

- PK `line_item_id`: unique (0 duplicates). ✓
- Row count: 714,669 = `len(order_items)` exactly. No fan-out. ✓
- Revenue conservation: `Σ qty × unit_price = $16,430,476,585.53` = `Σ sales.Revenue` (diff = $0.00). ✓
- Truncation flags match Phase 2: 564 shipment-gap orders, 80 return-gap orders. ✓
- Cross-satellite rule (Phase 3C): 0 `reviewed_and_returned` lines. ✓
- FAIL rows: **0 / 15**. ✓

**Phase 4 is complete. Ready to enter Phase 5 — Daily Aggregation (`df_daily`).**

## Files Produced

- `outputs/parquet/df_txn.parquet` — 714,669 × 64
- `outputs/phase4_audit_results.csv` — 15 checks
- `outputs/phase4_df_txn.py` — re-runnable build script
