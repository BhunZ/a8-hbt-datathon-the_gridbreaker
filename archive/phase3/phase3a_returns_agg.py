"""
Phase 3A — Build and validate returns_agg.

Contract (per Phase 3 preview):
  Grain / PK   : (order_id, product_id)
  Output       : outputs/parquet/returns_agg.parquet
  Validations  : P3_PK1, P3_FK1, P3_C1, P3_C2, P3_B14, P3_B6,
                 P3_D1, P3_D4, P3_D5

Pinned design decisions:
  - defect-set = {"defective"}  (only reason code matching "defective" semantics
                                 in the source; others — wrong_size / not_as_described
                                 / changed_mind / late_delivery — are non-defect).
  - truncation_flag_return = 1 iff order_id ∈ (orders.order_status=='returned' ∧
                                               order_id NOT IN returns.order_id)
                             else 0. Expected count = 80 (matches Phase 2 B11).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")
import helpers as H

ROOT   = Path("/sessions/lucid-relaxed-edison/mnt/Datathon")
SNAP   = ROOT / "outputs" / "parquet"
OUT    = ROOT / "outputs"
AUDIT  = OUT / "phase3_audit_results.csv"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

# ---------------------------------------------------------------------------
# Tiered defect classification (design-decision v2)
#   is_physical_defect  : product quality / condition failures.
#   is_catalog_failure  : listing-vs-received mismatch (description / SKU).
# Other reasons ("wrong_size", "changed_mind", "late_delivery") are neither;
# they fall into a third implicit bucket of customer-preference / logistics.
# The sets below are kept in variables so future data with new codes
# ("damaged", "wrong_item", etc.) can be honoured without code changes.
# ---------------------------------------------------------------------------
PHYSICAL_DEFECT_REASONS: set[str] = {"defective", "damaged"}
CATALOG_FAILURE_REASONS: set[str] = {"not_as_described", "wrong_item"}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
returns = pd.read_parquet(SNAP / "returns.parquet")
orders  = pd.read_parquet(SNAP / "orders.parquet")[["order_id","order_date","order_status"]]
oi      = pd.read_parquet(SNAP / "order_items.parquet")
ship    = pd.read_parquet(SNAP / "shipments.parquet")[["order_id","delivery_date"]]

print(f"Loaded  returns={len(returns):,}  orders={len(orders):,}  "
      f"order_items={len(oi):,}  shipments={len(ship):,}")

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
gkey = ["order_id", "product_id"]

# dominant_return_reason (mode, tie-break: first observed)
def _mode(s: pd.Series) -> str:
    vc = s.value_counts(dropna=True)
    return vc.index[0] if len(vc) else np.nan

agg = (returns
       .groupby(gkey, as_index=False, observed=True)
       .agg(
            n_return_events=("return_id", "size"),
            return_quantity_sum=("return_quantity", "sum"),
            refund_amount_sum=("refund_amount", "sum"),
            first_return_date=("return_date", "min"),
            last_return_date=("return_date", "max"),
            dominant_return_reason=("return_reason", _mode),
            distinct_return_reasons=("return_reason", lambda s: s.nunique(dropna=True)),
       ))

# ---------------------------------------------------------------------------
# Tiered defect aggregation
#   - per-event: is_physical_defect, is_catalog_failure  (robust to nulls:
#     reason NaN → False → contributes 0 to the max; aggregate stays 0)
#   - per-line : any_physical_defect_line = 1 iff ANY event on that
#                (order_id, product_id) fell in PHYSICAL_DEFECT_REASONS; same for catalog.
# ---------------------------------------------------------------------------
returns_evt = returns.assign(
    is_physical_defect = returns["return_reason"].isin(PHYSICAL_DEFECT_REASONS).astype("int8"),
    is_catalog_failure = returns["return_reason"].isin(CATALOG_FAILURE_REASONS).astype("int8"),
)
defect_flags = (returns_evt
                .groupby(gkey, as_index=False, observed=True)
                .agg(any_physical_defect_line=("is_physical_defect", "max"),
                     any_catalog_failure_line=("is_catalog_failure", "max")))
agg = agg.merge(defect_flags, on=gkey, how="left")

# time_to_first_return_days (needs orders.order_date)
agg = agg.merge(orders[["order_id","order_date"]], on="order_id", how="left")
agg["time_to_first_return_days"] = (
    (agg["first_return_date"] - agg["order_date"]).dt.days.astype("Int32")
)
agg = agg.drop(columns=["order_date"])

# refund_over_paid_ratio (needs unit_price from order_items)
# If a pair has duplicate line_items (16 cases), take mean unit_price.
unit = (oi.groupby(gkey, as_index=False, observed=True)
          .agg(unit_price=("unit_price","mean"),
               line_qty=("quantity","sum")))
agg = agg.merge(unit, on=gkey, how="left")
agg["refund_over_paid_ratio"] = np.where(
    (agg["return_quantity_sum"] > 0) & (agg["unit_price"].notna()),
    agg["refund_amount_sum"] / (agg["return_quantity_sum"] * agg["unit_price"]),
    np.nan,
)

# has_any_return — always 1 on this table; fills to 0 on LEFT-join downstream.
agg["has_any_return"] = np.int8(1)

# truncation_flag_return — 0 here. The 80 truncated orders have NO rows in returns,
# so they are absent from returns_agg. Phase 4 will mark them when joining.
agg["truncation_flag_return"] = np.int8(0)

# Dtype compaction
agg["n_return_events"]          = agg["n_return_events"].astype("int16")
agg["return_quantity_sum"]      = agg["return_quantity_sum"].astype("int32")
agg["distinct_return_reasons"]  = agg["distinct_return_reasons"].astype("int8")
agg["any_physical_defect_line"] = agg["any_physical_defect_line"].astype("int8")
agg["any_catalog_failure_line"] = agg["any_catalog_failure_line"].astype("int8")
agg["dominant_return_reason"]   = agg["dominant_return_reason"].astype("category")
# keep refund_over_paid_ratio / unit_price as float64 for precision

# Reorder & drop helper columns
cols = [
    "order_id","product_id",
    "n_return_events","return_quantity_sum","refund_amount_sum",
    "first_return_date","last_return_date","time_to_first_return_days",
    "dominant_return_reason","distinct_return_reasons",
    "any_physical_defect_line","any_catalog_failure_line",
    "refund_over_paid_ratio",
    "has_any_return","truncation_flag_return",
]
returns_agg = agg[cols].copy()

OUT_PATH = SNAP / "returns_agg.parquet"
returns_agg.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH}   rows={len(returns_agg):,}   cols={len(returns_agg.columns)}")

# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------
results: list[dict] = []
def record(rule_id: str, metric, threshold: str, verdict: str, detail: str = ""):
    results.append(dict(rule_id=rule_id, metric=metric, threshold=threshold,
                        verdict=verdict, detail=detail))

# ---- P3_PK1 : PK uniqueness on (order_id, product_id)
dup = int(returns_agg.duplicated(subset=gkey).sum())
record("P3_PK1", dup, "= 0", "PASS" if dup == 0 else "FAIL",
       "returns_agg is unique on (order_id, product_id)")

# ---- P3_FK1 : subset of order_items (order_id, product_id)
oi_keys = set(map(tuple, oi[gkey].drop_duplicates().itertuples(index=False, name=None)))
ra_keys = set(map(tuple, returns_agg[gkey].itertuples(index=False, name=None)))
orphans = ra_keys - oi_keys
record("P3_FK1", len(orphans), "= 0", "PASS" if not orphans else "FAIL",
       "returns_agg keys must be a subset of order_items keys")

# ---- P3_C1 : conservation of return_quantity
diff_q = int(returns_agg["return_quantity_sum"].sum()) - int(returns["return_quantity"].sum())
record("P3_C1", diff_q, "= 0", "PASS" if diff_q == 0 else "FAIL",
       "Σ returns_agg.return_quantity_sum == Σ returns.return_quantity")

# ---- P3_C2 : conservation of refund_amount
diff_r = round(returns_agg["refund_amount_sum"].sum()
               - returns["refund_amount"].sum(), 2)
record("P3_C2", diff_r, "= 0 (±$0.01)", "PASS" if abs(diff_r) <= 0.01 else "FAIL",
       "Σ returns_agg.refund_amount_sum == Σ returns.refund_amount")

# ---- P3_B14 : refund_amount_sum <= return_quantity_sum × unit_price
# (stricter re-check on agg level)
ra_chk = returns_agg.merge(unit[["order_id","product_id","unit_price"]]
                           .rename(columns={"unit_price":"up"}),
                           on=gkey, how="left")
cap = ra_chk["return_quantity_sum"] * ra_chk["up"]
bad = ra_chk[(cap.notna()) & (ra_chk["refund_amount_sum"] > cap + 0.01)]
record("P3_B14", len(bad), "= 0", "PASS" if len(bad) == 0 else "FAIL",
       "refund_amount_sum must not exceed return_quantity_sum × unit_price")

# ---- P3_B6 : first_return_date >= delivery_date (when delivery exists)
ra_del = returns_agg.merge(ship, on="order_id", how="left")
mask = ra_del["delivery_date"].notna()
bad = ra_del[mask & (ra_del["first_return_date"] < ra_del["delivery_date"])]
record("P3_B6", len(bad), "= 0", "PASS" if len(bad) == 0 else "FAIL",
       "first_return_date must be >= shipments.delivery_date")

# ---- P3_D1 : coverage vs order_items (diagnostic, LOG)
pct = round(len(returns_agg) / oi[gkey].drop_duplicates().shape[0] * 100, 3)
record("P3_D1", pct, "LOG", "LOG",
       f"returns_agg touches {pct}% of distinct order_items line keys")

# ---- P3_D4 : distribution of n_return_events (diagnostic)
vc = returns_agg["n_return_events"].value_counts().sort_index()
vc_dict = returns_agg["n_return_events"].value_counts().sort_index().to_dict()
record("P3_D4", vc_dict, "LOG", "LOG",
       "n_return_events distribution (expect heavy modal=1)")

# ---- P3_D5 : truncation_flag_return count
returned_orders = set(orders.loc[orders["order_status"].eq("returned"), "order_id"])
with_returns    = set(returns["order_id"].unique())
truncated       = returned_orders - with_returns
n_trunc = len(truncated)
record("P3_D5", n_trunc, "= 80 (matches Phase 2 B11)",
       "PASS" if n_trunc == 80 else "FAIL",
       "Count of 'returned'-status orders missing from returns table")

# persist the truncated order_id list for Phase 4 reference
trunc_path = OUT / "returned_orders_truncated.csv"
pd.DataFrame({"order_id": sorted(truncated)}).to_csv(trunc_path, index=False)
print(f"Wrote {trunc_path}  ({n_trunc} order_ids)")

# ---------------------------------------------------------------------------
# Save audit results
# ---------------------------------------------------------------------------
res = pd.DataFrame(results)
res["phase"] = "3A"
# merge with existing phase3_audit_results if present
if AUDIT.exists():
    prev = pd.read_csv(AUDIT)
    prev = prev[prev["phase"] != "3A"]  # idempotent replace
    res = pd.concat([prev, res], ignore_index=True)
res.to_csv(AUDIT, index=False)
print(f"\nWrote {AUDIT}  ({len(res)} rows cumulative)")

print("\n" + "=" * 78)
print("Phase 3A verdict summary")
print("=" * 78)
summary_cols = ["rule_id","metric","threshold","verdict"]
print(res[res["phase"].eq("3A")][summary_cols].to_string(index=False))
print("\nHead of returns_agg:")
print(returns_agg.head(5).to_string(index=False))

# ---- Row-count summary for tiered defect flags (per user request) ----
n_phys = int(returns_agg["any_physical_defect_line"].sum())
n_cat  = int(returns_agg["any_catalog_failure_line"].sum())
n_tot  = len(returns_agg)
print(f"\nTiered defect counts:")
print(f"  any_physical_defect_line==1 : {n_phys:,}  ({n_phys/n_tot*100:.2f}% of {n_tot:,} lines)")
print(f"  any_catalog_failure_line==1 : {n_cat:,}  ({n_cat/n_tot*100:.2f}% of {n_tot:,} lines)")
 