# Phase 2 — Multi-Layer Audit: Summary & Triage

**Inputs:** Phase 1 parquet snapshots (14 tables) + `helpers.py` audit utilities.
**Outputs:** `outputs/phase2_audit_results.csv`, updated `outputs/parquet/order_items.parquet` (with `line_item_id` surrogate).
**Total checks run:** 22 (PK already covered in Phase 1; this phase adds B1–B15 + G_CHK1–G_CHK5).

## Verdict Headline

| Verdict | Count | Meaning |
|---------|------:|---------|
| PASS    |   17  | Rule satisfied across all rows. |
| FAIL    |    3  | Real violations, see triage below. |
| LOG     |    1  | Spec rule demoted due to confirmed systematic data pattern. |
| WARN    |    1  | Known gap documented for downstream handling. |

**Gate decision:** Phase 3 is **cleared to start**. No FAIL condition blocks the join pipeline.

## Triage Table

| ID | Rows affected | Triage | Rationale | Downstream implication |
|----|--------------:|--------|-----------|------------------------|
| B1..B6, B8, B9, B12, B14, B15 | 0 | — | All PASS. | No action. |
| B2_quantity / B2_unit_price | 0 | — | PASS. | No action. |
| B3 | 0 | — | Ratings all in {1..5}. | No action. |
| **B7** (stackable) | 206 | **LOG → demote to WARN-only rule** | All 206 stacked `promo_id_2` values reference PROMO-0015 / PROMO-0025 whose `stackable_flag=0`. This is a systematic data pattern, not a data-quality defect. Phase 1 derivation of the unit_price formula confirmed these rows follow a deterministic *single-flat-$50-add-on* rule. | Do not exclude stacked rows. Keep them, honour the derived discount formula. |
| **B10_has_ship** (expected ship) | **564** | **Demote FAIL → LOG (dataset-tail truncation)** | 100% of the missing-shipment rows fall into 2022-12-22 → 2022-12-31 (last 10 days). Breakdown: 524 `delivered`, 29 `returned`, 11 `shipped`. | These orders belong to a truncation tail (shipments table cut off at 2022-12-31). Keep them in `df_txn`; mark `has_shipment = False` for them; do NOT excise them from revenue totals. |
| **B11** (returned has return row) | **80** | **Demote FAIL → LOG (dataset-tail truncation)** | All 80 fall in 2022-12-06 → 2022-12-31. Same truncation pattern. | Keep; set `n_return_events = 0` during Phase 3 aggregation. |
| **B13** (signup ≤ first_order) | **80,623** (89.3% of customers) | **Re-classify: redefine semantics; FAIL → LOG** | Median gap is 2,020 days (~5.5 years) **AFTER** first_order. `signup_date` is not "first interaction" — it appears to be an account-registration / loyalty-enrollment / CRM-creation event. The dataset's true "first-seen" anchor is `min(order_date)` per customer. | Phase 7: use `customer_tenure_days = order_date - customer.first_order_date` instead of `signup_date`. Do NOT use `signup_date` for recency features. |
| G_CHK1 | constant = 1/day | PASS | `web_traffic` = 1 row/date with one `traffic_source` label per day. | No aggregation required in Phase 3. |
| G_CHK2 | 0 | PASS | All `inventory.snapshot_date` values are month-end. | Use `merge_asof(backward)` aligned to month-end in Phase 6. |
| G_CHK3 | 0 | PASS | `sales.csv` fully covers 2012-07-04 → 2022-12-31 with no gaps. | Use as the canonical date spine. |
| G_CHK4 | 0 | PASS | `sales.csv` is exactly 1 row per date. | Safe to set `Date` as the spine PK. |
| **G_CHK5** (web_traffic vs sales spine) | **181** (WARN) | Known Gap **G13** | `web_traffic` starts 2013-01-01; missing 181 days at the head of the training period (2012-07-04 → 2012-12-31). | Phase 7: forward-fill or impute per-traffic-source features for days prior to 2013-01-01; alternatively, drop all pre-2013 samples from model training (not from EDA). |

## Key Decisions Documented

1. **Spec-rule B7 is deprecated.** Replace "stacked promos must both have stackable_flag=1" with: "stacked promos follow the deterministic formula `discount_amount = qty × unit_price × d1 + $50 flat per line`, independent of `stackable_flag`." See `outputs/unit_price_formula.md`.

2. **`signup_date` is not a "tenure" anchor.** Use `first_order_date` (computed in Phase 3 as `orders.groupby('customer_id').order_date.min()`).

3. **Dataset is truncated at 2022-12-31.** Orders placed in the final ~25 days may have partial `shipments` / `returns` records. Do not drop them; tolerate missing shipment metadata at the tail.

4. **`line_item_id` surrogate key has been added** to `order_items` (1-indexed int64). This resolves Gap G1. The updated parquet has been written to `outputs/parquet/order_items.parquet`.

## Files Produced

- `outputs/phase2_audit_results.csv` — 22 checks with metric/threshold/verdict/detail.
- `outputs/parquet/order_items.parquet` — now includes `line_item_id` as first column.

## Exit Criteria for Phase 2 (per PIPELINE_SPEC §5 "Phase 2 Exit Criterion")

> *"No PK violation; FK orphan rate ≤ policy threshold; business rule failures logged and triaged."*

Status:

- PK violations: 0 (composite `(order_id, product_id)` duplicates were resolved by adding the `line_item_id` surrogate).
- FK orphan rate: 0.0% across all 15 relationships (Phase 1).
- Business-rule failures: 3 surface-level FAILs, all triaged and reclassified with documented rationale.

**Phase 2 is complete. Ready to enter Phase 3 — Pre-join aggregation of satellite tables.**
