# Phase 3 — Pre-Join Aggregation: Summary & Triage

**Inputs:** Phase 2 parquet snapshots (14 tables, with `line_item_id` surrogate key on `order_items`).
**Outputs:** `outputs/parquet/returns_agg.parquet`, `outputs/parquet/reviews_agg.parquet`, `outputs/phase3_audit_results.csv`, `outputs/returned_orders_truncated.csv`, `outputs/phase3_crosscheck.json`.
**Total checks run:** 18 (9 for Phase 3A, 9 for Phase 3B).

## Verdict Headline

| Verdict | Count | Meaning |
|---------|------:|---------|
| PASS    |   13  | Integrity / conservation / business rule satisfied. |
| LOG     |    5  | Diagnostic-only (distributions, coverage, truncation). |
| WARN    |    0  | — |
| FAIL    |    0  | — |

**Gate decision:** Phase 4 (Master Join) is **cleared to start**. Zero FAIL rows. All conservation invariants hold to the cent.

## Deliverables Produced

| Artifact | Rows | Cols | Grain / PK | Notes |
|---|---:|---:|---|---|
| `returns_agg.parquet`  | 39,937  | 15 | `(order_id, product_id)` | 9 aggregate features + 2 tiered defect flags + 2 meta flags |
| `reviews_agg.parquet`  | 113,551 | 14 | `(order_id, product_id)` | 9 aggregate features + 2 rating flags + 2 meta flags |
| `returned_orders_truncated.csv` | 80 | 1 | `order_id` | B11 dataset-tail list for Phase 4 left-join |
| `phase3_audit_results.csv` | 18 | 6 | audit trail | cumulative across 3A + 3B |

## Phase 3A — returns_agg (9 checks)

| Rule | Metric | Verdict |
|---|---|---|
| P3_PK1 | 0 duplicates on `(order_id, product_id)` | PASS |
| P3_FK1 | 0 orphans vs `order_items` keys | PASS |
| P3_C1  | `Σ return_quantity_sum == Σ returns.return_quantity` (0 diff) | PASS |
| P3_C2  | `Σ refund_amount_sum` reconciles to ±$0.01 | PASS |
| P3_B14 | `refund_amount_sum ≤ return_quantity_sum × unit_price` (0 violations) | PASS |
| P3_B6  | `first_return_date ≥ delivery_date` (0 violations) | PASS |
| P3_D1  | Coverage = 5.588% of distinct order_items lines | LOG |
| P3_D4  | `n_return_events` distribution = {1: 39,935, 2: 2} | LOG |
| P3_D5  | 80 `returned`-status orders with no return row (matches B11) | PASS |

### Tiered defect classification — headline counts

| Flag | Reason set | Line count | Share |
|---|---|---:|---:|
| `any_physical_defect_line` | `{"defective", "damaged"}` | 8,020 | 20.08% |
| `any_catalog_failure_line` | `{"not_as_described", "wrong_item"}` | 7,035 | 17.62% |
| Neither flagged (customer preference / logistics: `wrong_size`, `changed_mind`, `late_delivery`) | — | 24,882 | 62.30% |

The two flags are mutually exclusive by construction. ~38% of return lines are quality/catalog-driven; the remaining ~62% are customer-side or logistics-side reasons that are not defects of the product itself.

## Phase 3B — reviews_agg (9 checks)

| Rule | Metric | Verdict |
|---|---|---|
| P3B_PK1 | 0 duplicates on `(order_id, product_id)` | PASS |
| P3B_FK1 | 0 orphans vs `order_items` keys | PASS |
| P3B_C1  | `Σ n_review_events == len(reviews)` (0 diff) | PASS |
| P3B_B3  | `min_rating ≥ 1`, `max_rating ≤ 5` across all lines | PASS |
| P3B_B6  | `first_review_date ≥ order_date` (0 violations; min latency = 5 days) | PASS |
| P3B_D1  | Coverage = 15.889% of distinct order_items lines | LOG |
| P3B_D2  | Dominant-rating distribution — {1:5,772 · 2:9,095 · 3:17,016 · 4:36,412 · 5:45,256} | LOG |
| P3B_D3  | Latency percentiles — p01=5, p50=21, p99=38 days | LOG |
| P3B_D4  | `n_review_events ≡ 1` on every line (source-PK collapse) | PASS |

### Sentiment flags — headline counts

| Flag | Rule | Line count | Share | Mean rating |
|---|---|---:|---:|---:|
| `has_high_rating` | `rating ≥ 4` | 81,668 | 71.92% | — |
| `has_low_rating`  | `rating ≤ 2` | 14,867 | 13.09% | — |
| All reviewed lines | — | 113,551 | 100% | 3.936 / 5 |

## Cross-Check — Returns vs. Reviews Overlap

**Finding:** `returns_agg` and `reviews_agg` are **structurally disjoint** in this dataset — the intersection is **empty**, not just sparse.

| Set | Size |
|---|---:|
| `returns_agg` lines | 39,937 |
| `reviews_agg` lines | 113,551 |
| **Intersection on `(order_id, product_id)`** | **0** |
| Common `order_id` alone (ignoring product) | 0 |
| Union | 153,488 |

Verified at three independent layers (post-agg parquets, source `returns.parquet` + `reviews.parquet`, and the raw merge), with identical int32 dtypes on both keys. The same order_id never appears in both tables — customers who return an item do not review it, and reviewers do not return.

**Implications for Phase 4 & 7:**

1. **No "negative-review + return" segment exists in this data.** A model treating `avg_rating` as a return-risk signal will find zero direct signal at the line-level. Any "low-rating → return" inference must be done via transfer through product-level aggregates (`product_id` rating history), not the line-level join.

2. **Left-joining both satellites to `order_items` is safe and non-conflicting.** Because the key-sets are disjoint, there is no row where both `has_any_review == 1` and `has_any_return == 1`. Downstream feature engineering should encode this as an explicit categorical: `post_purchase_outcome ∈ {"no_feedback", "reviewed_only", "returned_only"}` — a 4-th bucket (reviewed AND returned) will always be empty.

3. **Coverage math:**
   - `reviewed_only` = 113,551 lines (15.89% of order_items)
   - `returned_only` = 39,937 lines (5.59% of order_items)
   - `no_feedback`   = 714,653 − 153,488 = **561,165 lines (78.52%)**

4. **Dataset-generator hint (logged, not acted on):** the zero-overlap pattern is too perfect to be organic customer behaviour; it is very likely a property of the synthetic generator (a line was assigned *either* a review *or* a return, never both). Worth flagging in the final presentation but does not affect join correctness.

## Key Decisions Pinned in Phase 3

1. **Tiered defect classification is the standard.** Going forward, business-rule text should refer to `any_physical_defect_line` and `any_catalog_failure_line` separately, not a combined "defect" flag.
2. **`has_any_return` and `has_any_review` are the canonical "this line has a satellite row" indicators.** They are constant 1 on the agg tables and fill to 0 after the LEFT join in Phase 4. Do not compute them post-join by null-checking, use the explicit flag.
3. **`truncation_flag_return` is populated in Phase 4, not Phase 3.** The 80 B11 orders have no rows in `returns_agg`; Phase 4 sets the flag when left-joining the master list of `order_id` from `orders`. `returned_orders_truncated.csv` is the source of truth for that list.
4. **`truncation_flag_review` is a schema placeholder (constant 0).** Reviews are voluntary — missing ≠ truncated. Kept only for symmetry with `returns_agg`.
5. **`refund_over_paid_ratio`** is the preferred return-severity feature (float64, scale [0,1] for normal cases), computed as `refund_amount_sum / (return_quantity_sum × unit_price)`. Values > 1 indicate a refund exceeding the line's paid amount and are rare but valid (freight / goodwill).

## Exit Criteria for Phase 3 (per PIPELINE_SPEC §5 "Phase 3 Exit Criterion")

> *"All satellite tables aggregated to match master-table grain; aggregate-level conservation holds; aggregate-level FK is a proper subset of master-table PK; zero FAIL rows in consolidated audit."*

Status:

- Grain alignment: both agg tables keyed on `(order_id, product_id)` — matches `order_items` grain. PASS.
- Conservation: `Σ return_quantity` and `Σ refund_amount` reconcile to zero diff (±$0.01). `Σ n_review_events == len(reviews)`. PASS.
- FK subset: both agg tables are strict subsets of `order_items` keys (0 orphans). PASS.
- FAIL rows in consolidated audit: **0 / 18**. PASS.

**Phase 3 is complete. Ready to enter Phase 4 — Master Join & df_txn construction.**

## Files Produced

- `outputs/parquet/returns_agg.parquet` — 39,937 × 15
- `outputs/parquet/reviews_agg.parquet` — 113,551 × 14
- `outputs/phase3_audit_results.csv` — 18 checks, all PASS/LOG
- `outputs/returned_orders_truncated.csv` — 80 order_ids for Phase 4 truncation_flag_return
- `outputs/phase3_crosscheck.json` — machine-readable overlap stats
- `outputs/phase3a_returns_agg.py` / `outputs/phase3b_reviews_agg.py` — re-runnable build scripts
