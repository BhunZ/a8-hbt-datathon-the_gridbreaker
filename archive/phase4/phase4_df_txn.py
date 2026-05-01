"""
Phase 4 — Master Join: build df_txn.

Contract:
  Grain / PK   : line_item_id  (1-indexed int64 surrogate over order_items)
  Row count    : 714,669   (equal to order_items; LEFT joins must NOT fan-out)
  Output       : outputs/parquet/df_txn.parquet
  Validations  : P4_PK1, P4_CT1, P4_FK1..FK5, P4_C1, P4_T1, P4_T2, P4_B10,
                 P4_D1..D4

Design decisions (sealed in earlier phases):
  - line_item_id is the transaction-master PK (resolves Gap G1).
  - customer_tenure_days uses first_order_date per customer (B13 redefinition),
    NOT signup_date.
  - truncation_flag_shipment = 1 iff order_status ∈ {shipped, delivered, returned}
    ∧ no shipment row (the 564-order B10 dataset-tail gap).
  - truncation_flag_return   = 1 iff order_id ∈ returned_orders_truncated.csv
    (the 80-order B11 gap).
  - post_purchase_outcome ∈ {"no_feedback","reviewed_only","returned_only"}.
    (The 4th cell "reviewed_and_returned" is empty by dataset construction —
     confirmed in Phase 3C cross-check.)
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")

ROOT   = Path("/sessions/lucid-relaxed-edison/mnt/Datathon")
SNAP   = ROOT / "outputs" / "parquet"
OUT    = ROOT / "outputs"
AUDIT  = OUT / "phase4_audit_results.csv"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
oi       = pd.read_parquet(SNAP / "order_items.parquet")
orders   = pd.read_parquet(SNAP / "orders.parquet")
cust     = pd.read_parquet(SNAP / "customers.parquet")
prods    = pd.read_parquet(SNAP / "products.parquet")
promos   = pd.read_parquet(SNAP / "promotions.parquet")
ship     = pd.read_parquet(SNAP / "shipments.parquet")
ret_agg  = pd.read_parquet(SNAP / "returns_agg.parquet")
rev_agg  = pd.read_parquet(SNAP / "reviews_agg.parquet")
sales    = pd.read_parquet(SNAP / "sales.parquet")
trunc_80 = pd.read_csv(OUT / "returned_orders_truncated.csv")["order_id"].tolist()

print(f"Loaded:")
print(f"  order_items = {len(oi):,}  orders = {len(orders):,}  customers = {len(cust):,}")
print(f"  products    = {len(prods):,}  promotions = {len(promos):,}  shipments = {len(ship):,}")
print(f"  returns_agg = {len(ret_agg):,}  reviews_agg = {len(rev_agg):,}  sales = {len(sales):,}")
print(f"  B11 truncated order_ids = {len(trunc_80):,}")

# ---------------------------------------------------------------------------
# 1. Regenerate line_item_id surrogate key (1-indexed int64)
# ---------------------------------------------------------------------------
if "line_item_id" not in oi.columns:
    oi = oi.reset_index(drop=True)
    oi.insert(0, "line_item_id", np.arange(1, len(oi) + 1, dtype="int64"))
    oi.to_parquet(SNAP / "order_items.parquet", index=False)
    print(f"  → line_item_id surrogate regenerated on order_items.parquet")
else:
    print(f"  → line_item_id already present")

# ---------------------------------------------------------------------------
# 2. Base table + customer tenure anchor
# ---------------------------------------------------------------------------
# first_order_date per customer — the canonical tenure anchor per B13 redefinition
first_order = (orders.groupby("customer_id", as_index=False, observed=True)
                     .agg(first_order_date=("order_date", "min")))

base = oi.copy()
n_base = len(base)

# ---------------------------------------------------------------------------
# 3. LEFT JOIN chain — all joins are strictly 1:1 from the base side
# ---------------------------------------------------------------------------
# 3a. orders → adds customer_id, order_date, order_status, payment, device, source
base = base.merge(orders.drop(columns=["zip"]), on="order_id", how="left", validate="m:1")

# 3b. customers → demographics
base = base.merge(cust.drop(columns=["zip"]), on="customer_id", how="left", validate="m:1")

# 3c. first_order_date for tenure
base = base.merge(first_order, on="customer_id", how="left", validate="m:1")
base["customer_tenure_days"] = (
    (base["order_date"] - base["first_order_date"]).dt.days.astype("Int32")
)

# 3d. products → product attributes
base = base.merge(prods.rename(columns={"price":"product_list_price"}),
                  on="product_id", how="left", validate="m:1")

# 3e. promotions (primary) → promo_type, discount_value, etc.  (promo_id_2 intentionally not joined — it's always +$50 flat per Phase 2 B15 derivation)
promo_cols = ["promo_id","promo_type","discount_value","applicable_category",
              "stackable_flag","min_order_value"]
base = base.merge(promos[promo_cols], on="promo_id", how="left")

# 3f. shipments (1:1 on order_id, but with 564 B10 gaps)
# Validate 1:1 first
ship_pk = ship["order_id"].value_counts()
if (ship_pk > 1).any():
    raise RuntimeError(f"shipments not unique on order_id ({(ship_pk>1).sum()} dupes)")
base = base.merge(ship, on="order_id", how="left", validate="m:1")
base["has_shipment"] = base["ship_date"].notna().astype("int8")

# 3g. returns_agg → satellite (coverage 5.59% of lines)
ret_cols = [c for c in ret_agg.columns if c not in ("order_id","product_id")]
base = base.merge(ret_agg, on=["order_id","product_id"], how="left", validate="m:1")
base["has_any_return"] = base["has_any_return"].fillna(0).astype("int8")

# 3h. reviews_agg → satellite (coverage 15.89% of lines)
rev_cols = [c for c in rev_agg.columns if c not in ("order_id","product_id")]
base = base.merge(rev_agg, on=["order_id","product_id"], how="left", validate="m:1")
base["has_any_review"] = base["has_any_review"].fillna(0).astype("int8")

# Sanity: no fan-out
assert len(base) == n_base, f"Fan-out! rows={len(base)} vs expected {n_base}"

# ---------------------------------------------------------------------------
# 4. Augmentation flags
# ---------------------------------------------------------------------------
# truncation_flag_return — 1 for the 80 B11 orders
base["truncation_flag_return"] = base["order_id"].isin(trunc_80).astype("int8")

# truncation_flag_shipment — 1 for B10 gaps (expected ~564)
ship_required = base["order_status"].astype(str).isin({"shipped","delivered","returned"})
base["truncation_flag_shipment"] = (ship_required & (base["has_shipment"] == 0)).astype("int8")

# post_purchase_outcome (3-valued, guaranteed by Phase 3C disjoint finding)
base["post_purchase_outcome"] = np.select(
    condlist = [
        (base["has_any_review"] == 1) & (base["has_any_return"] == 1),  # should be empty
        (base["has_any_review"] == 1) & (base["has_any_return"] == 0),
        (base["has_any_review"] == 0) & (base["has_any_return"] == 1),
    ],
    choicelist = [
        "reviewed_and_returned",
        "reviewed_only",
        "returned_only",
    ],
    default = "no_feedback",
)
base["post_purchase_outcome"] = base["post_purchase_outcome"].astype("category")

# refund_over_paid_ratio is NaN on non-returned lines; ratio is correct there.

# ---------------------------------------------------------------------------
# 5. Compact dtypes, reorder columns
# ---------------------------------------------------------------------------
# Keep line_item_id first as the PK
pk_cols = ["line_item_id", "order_id", "product_id", "customer_id"]
date_cols = ["order_date","first_order_date","ship_date","delivery_date",
             "first_return_date","last_return_date","first_review_date","last_review_date",
             "signup_date"]
other = [c for c in base.columns if c not in (pk_cols + date_cols)]
df_txn = base[pk_cols + date_cols + other].copy()

OUT_PATH = SNAP / "df_txn.parquet"
df_txn.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH}  rows={len(df_txn):,}  cols={len(df_txn.columns)}")

# ---------------------------------------------------------------------------
# 6. Validations
# ---------------------------------------------------------------------------
results: list[dict] = []
def record(rule_id: str, metric, threshold: str, verdict: str, detail: str = ""):
    results.append(dict(rule_id=rule_id, metric=metric, threshold=threshold,
                        verdict=verdict, detail=detail))

# ---- P4_PK1 : line_item_id uniqueness
dup = int(df_txn.duplicated(subset=["line_item_id"]).sum())
record("P4_PK1", dup, "= 0", "PASS" if dup == 0 else "FAIL",
       "df_txn.line_item_id must be unique (1 row per order_items line)")

# ---- P4_CT1 : row count equals len(order_items)
n_expected = len(oi)
diff_ct = len(df_txn) - n_expected
record("P4_CT1", len(df_txn), f"= {n_expected:,}", "PASS" if diff_ct == 0 else "FAIL",
       "No fan-out from LEFT joins — df_txn row count must equal order_items row count")

# ---- P4_FK1..5 : critical base-path columns must have zero nulls
for rid, col in [("P4_FK1","customer_id"),
                 ("P4_FK2","order_date"),
                 ("P4_FK3","order_status"),
                 ("P4_FK4","product_list_price"),
                 ("P4_FK5","category")]:
    n_null = int(df_txn[col].isna().sum())
    record(rid, n_null, "= 0",
           "PASS" if n_null == 0 else "FAIL",
           f"{col} must not be null after LEFT JOIN chain")

# ---- P4_C1 : revenue conservation against sales.Revenue
rev_txn   = float((df_txn["quantity"] * df_txn["unit_price"]).sum())
rev_sales = float(sales["Revenue"].sum())
diff_rev  = round(rev_txn - rev_sales, 2)
record("P4_C1", diff_rev, "= 0 (±$0.01)",
       "PASS" if abs(diff_rev) <= 0.01 else "FAIL",
       f"Σ quantity × unit_price across df_txn must match Σ sales.Revenue")

# ---- P4_T1 : truncation_flag_shipment count matches B10 (564)
n_t1 = int(df_txn["truncation_flag_shipment"].sum())
record("P4_T1", n_t1, "= 564 (matches Phase 2 B10)",
       "PASS" if n_t1 == 564 else "FAIL",
       "Count of order_items lines that needed a shipment row but had none")

# Wait — B10 was 564 at ORDER level; at line level there may be multiple lines per order.
# So this is diagnostic, and P4_T1 should check the ORDER-distinct count.
n_t1_orders = int(df_txn.loc[df_txn["truncation_flag_shipment"].eq(1), "order_id"].nunique())
# Rewrite the last record entry: replace metric with distinct-order count
results[-1] = dict(rule_id="P4_T1", metric=n_t1_orders,
                   threshold="= 564 (matches Phase 2 B10, order-distinct)",
                   verdict="PASS" if n_t1_orders == 564 else "FAIL",
                   detail=f"Distinct order_ids with truncation_flag_shipment=1 ({n_t1} line rows)")

# ---- P4_T2 : truncation_flag_return distinct-order count matches B11 (80)
n_t2_orders = int(df_txn.loc[df_txn["truncation_flag_return"].eq(1), "order_id"].nunique())
record("P4_T2", n_t2_orders, "= 80 (matches Phase 2 B11, order-distinct)",
       "PASS" if n_t2_orders == 80 else "FAIL",
       "Distinct order_ids with truncation_flag_return=1")

# ---- P4_B10 : post_purchase_outcome 4th cell must be empty
n_both = int((df_txn["post_purchase_outcome"] == "reviewed_and_returned").sum())
record("P4_B10", n_both, "= 0 (Phase 3C cross-check)",
       "PASS" if n_both == 0 else "FAIL",
       "reviews and returns are disjoint in this dataset — no line can be both")

# ---- P4_D1 : post_purchase_outcome distribution (LOG)
outcome_vc = df_txn["post_purchase_outcome"].value_counts().to_dict()
record("P4_D1", outcome_vc, "LOG", "LOG",
       "distribution of post_purchase_outcome over df_txn")

# ---- P4_D2 : column-null inventory (LOG) — focus on satellite columns
null_pct = {}
for c in ["avg_rating","dominant_rating","refund_over_paid_ratio",
          "dominant_return_reason","delivery_date","promo_id"]:
    if c in df_txn.columns:
        null_pct[c] = round(df_txn[c].isna().mean()*100, 2)
record("P4_D2", null_pct, "LOG", "LOG",
       "% nulls on satellite / optional columns (context for Phase 5 imputation plan)")

# ---- P4_D3 : customer_tenure_days distribution (LOG)
t = df_txn["customer_tenure_days"].dropna()
q = t.quantile([.01,.25,.5,.75,.95,.99]).round(0).to_dict()
record("P4_D3", q, "LOG", "LOG",
       "customer_tenure_days percentiles (anchor = min(order_date) per customer)")

# ---- P4_D4 : final shape
record("P4_D4", {"rows": len(df_txn), "cols": len(df_txn.columns)},
       "LOG", "LOG", "final df_txn shape")

# ---------------------------------------------------------------------------
# 7. Persist audit results
# ---------------------------------------------------------------------------
res = pd.DataFrame(results)
res["phase"] = "4"
res.to_csv(AUDIT, index=False)
print(f"\nWrote {AUDIT}  ({len(res)} rows)")

print("\n" + "=" * 78)
print("Phase 4 verdict summary")
print("=" * 78)
summary_cols = ["rule_id","metric","threshold","verdict"]
print(res[summary_cols].to_string(index=False))

print("\n--- df_txn head ---")
print(df_txn[["line_item_id","order_id","product_id","customer_id","order_date",
              "quantity","unit_price","order_status","has_shipment",
              "has_any_return","has_any_review","post_purchase_outcome"]].head(5).to_string(index=False))

print("\n--- post_purchase_outcome distribution ---")
print(df_txn["post_purchase_outcome"].value_counts())
print(f"\n--- total df_txn columns: {len(df_txn.columns)} ---")
print(list(df_txn.columns))
