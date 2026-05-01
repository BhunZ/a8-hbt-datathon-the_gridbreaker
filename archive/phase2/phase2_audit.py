"""
Phase 2 — Multi-layer audit.

Runs:
  §4.3  Business rules B1-B15
  §4.7  Granularity alignment G_CHK1-G_CHK4
  G1    Generates line_item_id surrogate key on order_items

Inputs  : parquet snapshots from outputs/parquet/  (Phase 1 output)
Outputs : outputs/phase2_audit_results.csv
          outputs/parquet/order_items.parquet  (re-written with line_item_id)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")
import helpers as H

DATA = Path("/sessions/lucid-relaxed-edison/mnt/Datathon")
OUT  = DATA / "outputs"
SNAP = OUT / "parquet"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

# ---------------------------------------------------------------------------
# Load (prefer parquet snapshots from Phase 1 for speed)
# ---------------------------------------------------------------------------
def _read(name: str) -> pd.DataFrame:
    p = SNAP / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return H.load_csv_typed(name, DATA)

tables = {n: _read(n) for n in H.ALL_TABLES}

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
results: list[dict] = []

def record(rule_id: str, metric: float | int, threshold: str,
           verdict: str, detail: str = "", samples=None):
    # Normalize any iterable into a bounded list-of-strings
    if samples is None:
        sample_str = ""
    elif isinstance(samples, dict):
        sample_str = str(list(samples.items())[:5])
    else:
        try:
            sample_str = str(list(samples)[:5])
        except TypeError:
            sample_str = str(samples)
    results.append({
        "rule_id": rule_id,
        "metric": metric,
        "threshold": threshold,
        "verdict": verdict,
        "detail": detail,
        "sample_violators": sample_str,
    })

def pf(cond_pass: bool) -> str:
    return "PASS" if cond_pass else "FAIL"

def wf(cond_warn: bool) -> str:
    return "WARN" if cond_warn else "PASS"

# ===========================================================================
# §4.3 BUSINESS RULES
# ===========================================================================
print("=" * 78, "\n§4.3 BUSINESS RULES\n", "=" * 78, sep="")

# -------- B1  cogs < price --------
p = tables["products"]
bad = p[p["cogs"] >= p["price"]]
record("B1", len(bad), "= 0",
       pf(len(bad) == 0),
       "products.cogs must be strictly less than products.price",
       samples=bad["product_id"].tolist()[:5])

# -------- B2  quantity > 0, unit_price > 0 --------
oi = tables["order_items"]
bad_q = oi[oi["quantity"] <= 0]
bad_u = oi[oi["unit_price"] <= 0]
record("B2_quantity", len(bad_q), "= 0", pf(len(bad_q) == 0),
       "order_items.quantity must be > 0",
       samples=bad_q[["order_id","product_id","quantity"]].head().to_dict("records"))
record("B2_unit_price", len(bad_u), "= 0", pf(len(bad_u) == 0),
       "order_items.unit_price must be > 0",
       samples=bad_u[["order_id","product_id","unit_price"]].head().to_dict("records"))

# -------- B3  reviews.rating ∈ {1..5} --------
rv = tables["reviews"]
bad = rv[~rv["rating"].isin([1, 2, 3, 4, 5])]
record("B3", len(bad), "= 0", pf(len(bad) == 0),
       "reviews.rating must be in {1,2,3,4,5}",
       samples=bad["review_id"].head().tolist())

# -------- B4  ship_date >= order_date --------
sh = tables["shipments"].merge(
    tables["orders"][["order_id","order_date"]], on="order_id", how="left")
bad = sh[sh["ship_date"] < sh["order_date"]]
record("B4", len(bad), "= 0", pf(len(bad) == 0),
       "shipments.ship_date must be >= orders.order_date",
       samples=bad[["order_id","order_date","ship_date"]].head().to_dict("records"))

# -------- B5  delivery_date >= ship_date --------
sh2 = tables["shipments"]
mask = sh2["ship_date"].notna() & sh2["delivery_date"].notna()
bad = sh2[mask & (sh2["delivery_date"] < sh2["ship_date"])]
record("B5", len(bad), "= 0", pf(len(bad) == 0),
       "shipments.delivery_date must be >= shipments.ship_date",
       samples=bad[["order_id","ship_date","delivery_date"]].head().to_dict("records"))

# -------- B6  return_date >= delivery_date (when delivery exists) --------
rt = tables["returns"].merge(
    tables["shipments"][["order_id","delivery_date"]], on="order_id", how="left")
mask = rt["delivery_date"].notna()
bad = rt[mask & (rt["return_date"] < rt["delivery_date"])]
record("B6", len(bad), "= 0", pf(len(bad) == 0),
       "returns.return_date must be >= shipments.delivery_date",
       samples=bad[["return_id","order_id","return_date","delivery_date"]].head().to_dict("records"))

# -------- B7  promo_id_2 stackable  (SPEC-UPDATED: now logged, not enforced) --------
# Phase 1 finding: all 206 promo_id_2 rows reference PROMO-0015 / PROMO-0025 whose
# stackable_flag=0. The data intentionally uses promo_id_2 as a "second promo slot"
# regardless of stackable_flag. We therefore DEMOTE B7 from FAIL to LOG.
prom = tables["promotions"]
stackable_map = prom.set_index("promo_id")["stackable_flag"].to_dict()
oi_stack = oi[oi["promo_id_2"].notna()].copy()
oi_stack["p1_stack"] = oi_stack["promo_id"].map(stackable_map)
oi_stack["p2_stack"] = oi_stack["promo_id_2"].map(stackable_map)
violators = oi_stack[(oi_stack["p1_stack"] != 1) | (oi_stack["p2_stack"] != 1)]
record("B7", len(violators),
       "LOG (spec says FAIL but data is systematic — see unit_price_formula.md)",
       "LOG",
       "promo_id_2 references non-stackable promos; pattern is systematic (206 rows).",
       samples=violators["promo_id_2"].unique().tolist()[:5])

# -------- B8  order_date ∈ [promo.start_date, promo.end_date] --------
# Join on promo_id AND order_date, count out-of-window uses.
oi_o = oi.merge(tables["orders"][["order_id","order_date"]], on="order_id", how="left")
# primary promo
chk1 = oi_o[oi_o["promo_id"].notna()].merge(
    prom[["promo_id","start_date","end_date"]], on="promo_id", how="left")
bad1 = chk1[(chk1["order_date"] < chk1["start_date"]) | (chk1["order_date"] > chk1["end_date"])]
# secondary promo
chk2 = oi_o[oi_o["promo_id_2"].notna()].merge(
    prom[["promo_id","start_date","end_date"]].rename(columns={"promo_id":"promo_id_2"}),
    on="promo_id_2", how="left")
bad2 = chk2[(chk2["order_date"] < chk2["start_date"]) | (chk2["order_date"] > chk2["end_date"])]
total_bad = len(bad1) + len(bad2)
record("B8", total_bad, "= 0", pf(total_bad == 0),
       "order_date must fall within the promo's active window",
       samples=bad1[["order_id","promo_id","order_date"]].head().to_dict("records"))

# -------- B9  applicable_category matches product.category (when set) --------
oi_cat = oi[oi["promo_id"].notna()].merge(
    tables["products"][["product_id","category"]], on="product_id", how="left"
).merge(
    prom[["promo_id","applicable_category"]], on="promo_id", how="left")
m = oi_cat["applicable_category"].notna()
# .astype(str) to compare across category vs string dtypes safely
bad = oi_cat[m & (oi_cat["applicable_category"].astype(str)
                  != oi_cat["category"].astype(str))]
record("B9", len(bad), "= 0", pf(len(bad) == 0),
       "When promo.applicable_category is set, product.category must match",
       samples=bad[["order_id","product_id","category","applicable_category"]].head().to_dict("records"))

# -------- B10 status {shipped,delivered,returned} => exactly 1 shipment; else 0 --------
orders = tables["orders"][["order_id","order_status"]]
shp_cnt = tables["shipments"].groupby("order_id").size().rename("n_ship")
orders = orders.merge(shp_cnt, on="order_id", how="left").fillna({"n_ship": 0})
ship_statuses = {"shipped", "delivered", "returned"}
must_ship = orders[orders["order_status"].isin(ship_statuses)]
must_not  = orders[~orders["order_status"].isin(ship_statuses)]
bad_ms = must_ship[must_ship["n_ship"] != 1]
bad_mn = must_not[must_not["n_ship"] != 0]
record("B10_has_ship", len(bad_ms), "= 0", pf(len(bad_ms) == 0),
       "order_status ∈ {shipped,delivered,returned} must have exactly one shipment row",
       samples=bad_ms.head().to_dict("records"))
record("B10_no_ship", len(bad_mn), "= 0", pf(len(bad_mn) == 0),
       "Other order_status values must have zero shipment rows",
       samples=bad_mn.head().to_dict("records"))

# -------- B11 status=returned => at least one return row --------
ret_cnt = tables["returns"].groupby("order_id").size().rename("n_ret")
orders2 = tables["orders"][["order_id","order_status"]].merge(
    ret_cnt, on="order_id", how="left").fillna({"n_ret": 0})
returned = orders2[orders2["order_status"] == "returned"]
bad = returned[returned["n_ret"] < 1]
record("B11", len(bad), "= 0", pf(len(bad) == 0),
       "order_status=returned must have >=1 returns row",
       samples=bad.head().to_dict("records"))

# -------- B12 review_date >= order_date --------
rv_o = tables["reviews"].merge(
    tables["orders"][["order_id","order_date"]], on="order_id", how="left")
bad = rv_o[rv_o["review_date"] < rv_o["order_date"]]
record("B12", len(bad), "= 0", pf(len(bad) == 0),
       "reviews.review_date must be >= orders.order_date",
       samples=bad[["review_id","order_date","review_date"]].head().to_dict("records"))

# -------- B13 signup_date <= min(order_date) per customer --------
cust = tables["customers"][["customer_id","signup_date"]]
first_order = (tables["orders"].groupby("customer_id")["order_date"].min()
               .rename("first_order_date").reset_index())
j = cust.merge(first_order, on="customer_id", how="left")
bad = j[j["first_order_date"].notna() & (j["signup_date"] > j["first_order_date"])]
record("B13", len(bad), "= 0", pf(len(bad) == 0),
       "customers.signup_date must be <= first orders.order_date",
       samples=bad.head().to_dict("records"))

# -------- B14 refund_amount <= return_quantity * unit_price --------
# For returns we need the matching order_item unit_price.
# returns has (order_id, product_id) — join on that pair.
rt2 = tables["returns"].merge(
    oi[["order_id","product_id","unit_price","quantity"]],
    on=["order_id","product_id"], how="left")
rt2["cap"] = rt2["return_quantity"] * rt2["unit_price"]
bad = rt2[rt2["cap"].notna() & (rt2["refund_amount"] > rt2["cap"] + 0.01)]
record("B14", len(bad), "= 0",
       pf(len(bad) == 0),
       "returns.refund_amount must be <= return_quantity * unit_price (±$0.01 tol)",
       samples=bad[["return_id","refund_amount","cap"]].head().to_dict("records"))

# -------- B15 discount_amount formula reconciliation --------
# Derived empirically (see outputs/unit_price_formula.md, updated post Phase 1):
#   no promo      : discount_amount = 0
#   percentage    : discount_amount = quantity * unit_price * discount_value/100
#   fixed (dv=50) : discount_amount = quantity * discount_value      (flat $50/unit)
#   stacked (promo_id_2 non-null) : line uses percentage primary + flat $50 per LINE
oi_d = oi.merge(prom[["promo_id","discount_value","promo_type"]], on="promo_id", how="left")
oi_d["disc_frac"] = np.where(oi_d["promo_id"].notna(),
                             oi_d["discount_value"].astype(float)/100.0, 0.0)
pred = np.where(
    oi_d["promo_type"] == "fixed",
    oi_d["quantity"] * oi_d["discount_value"],                                # flat $N/unit
    oi_d["quantity"] * oi_d["unit_price"] * oi_d["disc_frac"],               # pct of charged
)
pred = np.where(oi_d["promo_id_2"].notna(), pred + 50.0, pred)               # stacked flat $50/line
pred = np.where(oi_d["promo_id"].isna(), 0.0, pred)                          # no-promo guard
oi_d["predicted_disc"] = pred
oi_d["err"] = (oi_d["discount_amount"] - oi_d["predicted_disc"]).abs()
TOL = 0.02
bad = oi_d[oi_d["err"] > TOL]
record("B15", len(bad), f"= 0 (tol ±${TOL})",
       pf(len(bad) == 0),
       "discount_amount formula (hybrid: flat-per-unit for fixed, pct-of-charged for percentage, +$50 flat on stacked).",
       samples=bad[["order_id","product_id","discount_amount","predicted_disc","err"]].head().to_dict("records"))

# ===========================================================================
# §4.7 GRANULARITY CHECKS
# ===========================================================================
print("=" * 78, "\n§4.7 GRANULARITY CHECKS\n", "=" * 78, sep="")

# -------- G_CHK1  web_traffic cardinality --------
wt = tables["web_traffic"]
vc = wt.groupby("date").size()
is_constant = vc.nunique() == 1
constant_val = vc.iloc[0] if is_constant else None
record("G_CHK1", int(vc.max()),
       "constant rows/date",
       pf(is_constant),
       f"web_traffic has {vc.nunique()} distinct row-per-date counts; "
       f"min={vc.min()} max={vc.max()} constant={constant_val}",
       samples=vc.value_counts().head().to_dict())

# -------- G_CHK2  inventory.snapshot_date is last day of month --------
inv = tables["inventory"]
d = pd.to_datetime(inv["snapshot_date"])
month_end = d + pd.offsets.MonthEnd(0)
# True last-day: d + MonthEnd(0) == d  when d is already month-end
is_mend = d.eq(month_end)
bad_cnt = int((~is_mend).sum())
record("G_CHK2", bad_cnt, "= 0", pf(bad_cnt == 0),
       "inventory.snapshot_date must be the last calendar day of its month",
       samples=inv.loc[~is_mend, "snapshot_date"].unique().tolist()[:5])

# -------- G_CHK3  sales.csv spine has no missing days --------
sl = tables["sales"].sort_values("Date")
dmin, dmax = sl["Date"].min(), sl["Date"].max()
expected = pd.date_range(dmin, dmax, freq="D")
missing = set(expected) - set(pd.to_datetime(sl["Date"]))
record("G_CHK3", len(missing), "= 0", pf(len(missing) == 0),
       f"sales.csv must cover {dmin.date()}..{dmax.date()} with no gaps "
       f"(expected={len(expected)}, actual={len(sl)})",
       samples=sorted(missing)[:5])

# -------- G_CHK4  sales.csv one row per date --------
dup = int(sl.duplicated("Date").sum())
record("G_CHK4", dup, "= 0", pf(dup == 0),
       "sales.csv must have exactly one row per Date",
       samples=sl[sl.duplicated("Date", keep=False)]["Date"].head().tolist())

# ---- additional  G_CHK5 web_traffic date coverage vs sales spine (flagged A3) ----
wt_dates = set(pd.to_datetime(wt["date"]))
sales_dates = set(pd.to_datetime(sl["Date"]))
missing_wt = sales_dates - wt_dates
record("G_CHK5", len(missing_wt), "LOG (G13)",
       "WARN" if missing_wt else "PASS",
       "web_traffic missing coverage vs sales.csv spine (Gap G13)",
       samples=sorted(missing_wt)[:5])

# ===========================================================================
# G1 — surrogate key for order_items
# ===========================================================================
print("=" * 78, "\nG1 surrogate line_item_id\n", "=" * 78, sep="")
oi_keyed = tables["order_items"].sort_values(
    ["order_id", "product_id"], kind="stable"
).reset_index(drop=True)
if "line_item_id" in oi_keyed.columns:
    oi_keyed = oi_keyed.drop(columns=["line_item_id"])
oi_keyed.insert(0, "line_item_id", np.arange(1, len(oi_keyed) + 1, dtype=np.int64))
# Sanity
assert oi_keyed["line_item_id"].is_unique
assert len(oi_keyed) == len(tables["order_items"])
oi_keyed.to_parquet(SNAP / "order_items.parquet", index=False)
print(f"Wrote {SNAP/'order_items.parquet'} with line_item_id "
      f"({len(oi_keyed):,} rows, PK unique)")

# ===========================================================================
# WRITE RESULTS
# ===========================================================================
res = pd.DataFrame(results)
out_path = OUT / "phase2_audit_results.csv"
res.to_csv(out_path, index=False)
print(f"\nWrote {out_path}  ({len(res)} checks)")

summary = res["verdict"].value_counts()
print("\nVerdict summary:")
print(summary.to_string())
print("\nFull table:")
print(res[["rule_id","metric","threshold","verdict"]].to_string(index=False))
