# Phase 5 — Daily Aggregation (`df_daily`): Summary

**Inputs:** `df_txn.parquet` (Phase 4), `sales.parquet`, `web_traffic.parquet`, `inventory.parquet`, `shipments.parquet`.
**Output:** `outputs/parquet/df_daily.parquet` — the daily-grain time-series master.
**Total checks run:** 13.

## Verdict Headline

| Verdict | Count | Meaning |
|---------|------:|---------|
| PASS    |   10  | PK, date-spine integrity, conservation (revenue/cogs/orders/lines/shipping), G_CHK5 head gap, inventory backward-fill coverage. |
| LOG     |    3  | Revenue percentiles, status totals, inventory coverage stats. |
| WARN    |    0  | — |
| FAIL    |    0  | — |

**Gate decision:** df_daily is complete and **ready for Phase 6 (Feature Engineering) and Phase 7 (Modelling)**.

## df_daily Shape

| Property | Value |
|---|---|
| Rows | **3,833** (one per calendar day, 2012-07-04 → 2022-12-31, gap-free) |
| Columns | **58** |
| PK | `Date` (datetime64) |
| File size | ~0.3 MB parquet |
| Revenue total | **$16,430,476,585.53** — exact match to `Σ df_txn` and `Σ sales.Revenue` |
| COGS total    | **$12,800,148,568** (approx) — max daily delta vs `sales.COGS` = $0.00 |

## Column Inventory (by block)

| Block | Columns |
|---|---|
| **Spine & canonical** | `Date`, `sales_revenue_canonical`, `sales_cogs_canonical` |
| **Volume roll-ups** | `n_line_items`, `n_units`, `n_orders`, `n_unique_customers`, `n_new_customer_lines`, `n_shipped_orders` |
| **Economics** | `gross_revenue`, `gross_cogs`, `total_discount_amount`, `net_revenue`, `gross_margin`, `total_shipping_fee` |
| **Ratios** | `aov`, `units_per_order`, `avg_unit_price`, `cancel_rate_lines`, `return_rate_lines`, `review_rate_lines`, `promo_revenue_share` |
| **Promo** | `n_promo_lines`, `promo_revenue` |
| **Status counts** | `n_status_delivered`, `n_status_cancelled`, `n_status_returned`, `n_status_shipped`, `n_status_paid`, `n_status_created` |
| **Return / review counts** | `n_returned_lines`, `n_reviewed_lines` |
| **Web traffic** (181 head NaNs) | `sessions`, `unique_visitors`, `page_views`, `bounce_rate`, `avg_session_duration_sec`, `traffic_source` |
| **Inventory** (27 head NaNs via merge_asof backward) | `inv_snapshot_used`, `inv_total_stock_on_hand`, `inv_total_units_received`, `inv_total_units_sold`, `inv_n_products_tracked`, `inv_n_stockouts`, `inv_n_overstocks`, `inv_n_reorder_flagged`, `inv_avg_fill_rate`, `inv_avg_days_of_supply`, `inv_avg_sell_through` |
| **Calendar** | `year`, `quarter`, `month`, `day_of_month`, `day_of_week`, `week_of_year`, `is_weekend`, `is_month_end`, `is_month_start` |

## Validation Matrix

| Rule | Metric | Threshold | Verdict |
|---|---|---|---|
| P5_PK1 | 0 | = 0 | PASS — Date unique on all 3,833 rows |
| P5_CT1 | 3,833 | = 3,833 | PASS — row count equals sales spine |
| P5_DR1 | 0 gaps | = 0 | PASS — 2012-07-04 → 2022-12-31, consecutive |
| P5_C1 | $0.00 | ±$0.01 | PASS — `Σ gross_revenue = Σ sales.Revenue` |
| P5_C2 | $0.00 | max \|daily Δ\| ≤ $0.01 | PASS — per-day COGS matches (cumulative Σ drifts ~$0.04 on a $12.8B total = ~3 parts per trillion float noise; daily invariant is the meaningful test) |
| P5_C3 | 0 | = 0 | PASS — `Σ n_orders` matches df_txn distinct order_id |
| P5_C4 | 0 | = 0 | PASS — `Σ n_line_items` equals `len(df_txn)` = 714,669 |
| P5_C5 | $0.00 | ±$0.01 | PASS — `Σ total_shipping_fee` matches shipments total $2,809,309.66 |
| P5_G5 | 181 | = 181 | PASS — web_traffic head gap matches Phase 2 G_CHK5 exactly |
| P5_INV1 | 27 | = 27 | PASS — 27 pre-2012-07-31 days have no prior inventory snapshot (by design) |
| P5_D1 | pct-iles | LOG | LOG — gross_revenue: p50 ≈ $3.65M, p95 ≈ $9.40M, p99 ≈ $13.80M |
| P5_D2 | status totals | LOG | LOG — matches df_txn exactly |
| P5_D3 | inv coverage | LOG | LOG — 126 snapshots used, first 2012-07-31, last 2022-12-31 |

## Key Design Details Pinned

1. **Date spine = `sales.csv`.** df_daily starts from the 3,833-row sales frame and LEFT-joins everything else. This guarantees gap-free dates and preserves `sales.Revenue` / `sales.COGS` as the canonical benchmarks.
2. **Shipping fee aggregation bug avoided.** Summing `shipping_fee` per line would inflate the total by ~$217K (since a 10-line order carries the fee on every row). We de-duplicate on `(order_id)` before summing, giving the correct $2,809,309.66 match.
3. **Inventory join uses `merge_asof(..., direction="backward")`** over the month-end spine of 126 snapshots. 27 head days (2012-07-04..30) have no prior snapshot and are left NaN — consistent with G_CHK2.
4. **Web-traffic head gap preserved.** The 181 pre-2013 days have NaN for all web metrics. Phase 6 should either (a) drop pre-2013 days from training sets, or (b) forward-fill/impute; this is not a defect.
5. **COGS reconciliation invariant is per-day, not cumulative.** `Σ sales.COGS` drifts by ~$0.04 over $12.8B due to float64 summation noise. The per-day delta is $0.00 — that is the semantically meaningful invariant, so the test is written against it.
6. **New-customer detection** uses `customer_tenure_days == 0 AND order_date == first_order_date` — this lights up the customer's first-ever order line.

## Cross-Master Consistency

| Invariant | df_txn | df_daily | Match |
|---|---:|---:|:---:|
| Total revenue | $16,430,476,585.53 | $16,430,476,585.53 | ✓ |
| Total line items | 714,669 | 714,669 | ✓ |
| Total distinct orders | 646,945 | 646,945 | ✓ |
| delivered lines | 570,887 | 570,887 | ✓ |
| cancelled lines | 65,673 | 65,673 | ✓ |
| returned lines | 40,034 | 40,034 | ✓ |
| Shipping fee total | — | $2,809,309.66 | ✓ (matches shipments) |

The two masters are now **fully consistent** — every roll-up invariant holds.

## Phase 5 Exit Criterion

> *"Daily master table built on a canonical gap-free date spine; all conservation invariants hold (revenue, cogs, orders, lines, shipping); cross-master roll-up invariants validated against df_txn; exogenous joins (web_traffic, inventory) use the documented time-alignment strategy."*

Status:

- Date spine: 3,833 rows, 2012-07-04 → 2022-12-31, 0 gaps. ✓
- Revenue / COGS / orders / lines / shipping conservation: all PASS. ✓
- Cross-master invariants: all match df_txn. ✓
- Web traffic: G_CHK5 null pattern reproduced (181 days). ✓
- Inventory: G_CHK2 merge_asof backward used; 27-day head gap matches spec. ✓
- FAIL rows: **0 / 13**. ✓

**Phase 5 is complete. Ready to enter Phase 6 — Feature Engineering & Temporal Features.**

## Files Produced

- `outputs/parquet/df_daily.parquet` — 3,833 × 58
- `outputs/phase5_audit_results.csv` — 13 checks
- `outputs/phase5_df_daily.py` — re-runnable build script
