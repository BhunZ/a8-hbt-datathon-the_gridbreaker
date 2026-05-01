"""
Validate the unit-price model:
    unit_price = list_price * (1 - discount_value/100) * (1 + epsilon)
    epsilon ~ Normal(0, sigma)   per-line, independent
    discount_value is a PERCENT (both 'fixed' and 'percentage' promo_types)
    "fixed" promos = 50% off (single fixed rate); "percentage" promos ∈ {10%, 20%}
    discount_amount column = qty * list_price * discount_value/100  (noise-free)
Residuals should be ~Normal(0, sigma) for all buckets with similar sigma.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")
import helpers as H

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

T = H.load_all()
oi = T["order_items"].merge(
    T["products"][["product_id","price"]].rename(columns={"price":"list_price"}),
    on="product_id", how="left")
oi = oi.merge(
    T["promotions"][["promo_id","discount_value","promo_type"]],
    on="promo_id", how="left")

# Effective discount fraction
oi["disc_frac"] = np.where(oi["promo_id"].notna(),
                           oi["discount_value"].astype(float)/100.0, 0.0)
oi["theoretical"] = oi["list_price"] * (1 - oi["disc_frac"])
oi["residual"]   = oi["unit_price"]/oi["theoretical"] - 1.0   # = epsilon

# Stratify
def bucket(r):
    if pd.isna(r["promo_id"]): return "NO_PROMO"
    return f"{r['promo_type']} / dv={int(r['discount_value'])}"
oi["bucket"] = oi.apply(bucket, axis=1)

print("=" * 78)
print("Residual distribution by bucket  (ε = unit_price / theoretical - 1)")
print("=" * 78)
g = oi.groupby("bucket")["residual"].agg(['count','mean','std','min','max'])
print(g.round(5))

# Check: is sigma consistent across buckets?
print("\nSigma ratio (bucket std / no-promo std):")
sigma0 = g.loc["NO_PROMO", "std"]
print((g["std"]/sigma0).round(4))

# Test noise Gaussianity: % of residuals within ±3σ
print("\nFraction of residuals within ±3σ (expect ~99.7% if Gaussian):")
for b, grp in oi.groupby("bucket"):
    s = grp["residual"].std()
    within = grp["residual"].abs().lt(3*s).mean()
    print(f"  {b:25s}  n={len(grp):>6,}  σ={s:.4f}  frac<3σ={within:.4f}")

# Verify discount_amount formula
oi["da_pred"] = oi["quantity"] * oi["list_price"] * oi["disc_frac"]
oi["da_err"]  = (oi["discount_amount"] - oi["da_pred"]).round(2)
print("\n" + "=" * 78)
print("discount_amount  vs  qty*list_price*disc_frac")
print("=" * 78)
print(oi["da_err"].describe())
print(f"exact zero (abs<0.01): {int(oi['da_err'].abs().lt(0.01).sum()):,} / {len(oi):,}")
print(f"abs err < 1          : {int(oi['da_err'].abs().lt(1.00).sum()):,} / {len(oi):,}")

# The 206 rows with two promos — do they stack multiplicatively?
two = oi[oi["promo_id_2"].notna()].copy()
# second promo's discount
two = two.merge(
    T["promotions"][["promo_id","discount_value"]].rename(
        columns={"promo_id":"promo_id_2","discount_value":"dv2"}),
    on="promo_id_2", how="left")
two["disc2_frac"] = two["dv2"].astype(float)/100.0
# model A: stacked multiplicatively
two["theo_stack"] = two["list_price"] * (1 - two["disc_frac"]) * (1 - two["disc2_frac"])
# model B: additively
two["theo_add"]   = two["list_price"] * (1 - two["disc_frac"] - two["disc2_frac"])
two["res_stack"]  = two["unit_price"]/two["theo_stack"] - 1
two["res_add"]    = two["unit_price"]/two["theo_add"]   - 1
print("\n" + "=" * 78)
print(f"206 two-promo rows — test stacking model")
print("=" * 78)
print(f"multiplicative: mean={two['res_stack'].mean():.4f}  std={two['res_stack'].std():.4f}")
print(f"additive      : mean={two['res_add'].mean():.4f}  std={two['res_add'].std():.4f}")

# Final summary ROW-EXACT reconciliation
# Line cash = qty * unit_price  — used by sales.Revenue (MAE=0)
# List cash = qty * list_price
# disc_amt  = qty * list_price * disc_frac  (noise-free)
# Customer paid: sum lines + shipping - sum disc_amt?  (on payments)
print("\n" + "=" * 78)
print("Written model")
print("=" * 78)
print("unit_price = products.price * (1 - disc_frac) * (1 + ε)")
print("   disc_frac  = promo.discount_value / 100   if a promo is applied,  else 0")
print("   ε          ≈ Normal(0, ~0.02)  i.i.d. per line")
print("discount_amount = quantity * products.price * disc_frac      (noise-free)")
print("sales.Revenue   = Σ_all_lines  quantity * unit_price")
