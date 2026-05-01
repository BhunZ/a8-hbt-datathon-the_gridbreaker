"""
Reverse-engineer how order_items.unit_price is computed.

Strategy (each step narrows the hypothesis space):
 1. Universe split — rows WITH vs WITHOUT any promo.
 2. On no-promo rows: is unit_price ≡ products.price?
 3. On promo rows, by promo_type: what transformation is applied?
 4. What does discount_amount actually represent?
 5. Does unit_price vary over time for the same product (re-pricing)?
 6. Payment reconciliation: does Σ(qty*unit_price) per order + shipping_fee == payment_value?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")
import helpers as H

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

DATA = Path("/sessions/lucid-relaxed-edison/mnt/Datathon")
T = H.load_all(DATA)
oi   = T["order_items"].copy()
prod = T["products"][["product_id","price","cogs","category"]].rename(columns={"price":"list_price"})
prom = T["promotions"].copy()
ords = T["orders"][["order_id","order_date","order_status"]]
pay  = T["payments"][["order_id","payment_value"]]
shp  = T["shipments"][["order_id","shipping_fee"]]

# Join list price + order date
oi = oi.merge(prod, on="product_id", how="left").merge(ords, on="order_id", how="left")
oi["has_promo1"] = oi["promo_id"].notna()
oi["has_promo2"] = oi["promo_id_2"].notna()
oi["has_any_promo"] = oi["has_promo1"] | oi["has_promo2"]

print("=" * 78)
print("STEP 1 — Promo usage split")
print("=" * 78)
print(oi.groupby(["has_promo1","has_promo2"]).size().rename("rows"))

# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 2 — On NO-PROMO rows: does unit_price == list_price?")
print("=" * 78)
no = oi[~oi["has_any_promo"]].copy()
no["diff"] = (no["unit_price"] - no["list_price"]).round(6)
print(f"no-promo rows: {len(no):,}")
print(no["diff"].describe())
print("\nTop 10 non-zero diff rows:")
print(no.loc[no["diff"].ne(0), ["order_id","product_id","list_price","unit_price","diff","order_date"]].head(10))

# Fraction at exact equality
n_eq = int(no["diff"].eq(0).sum())
print(f"\nexact equality  : {n_eq:,} / {len(no):,}  ({n_eq/len(no)*100:.3f}%)")

# ratio unit_price / list_price
no["ratio"] = no["unit_price"] / no["list_price"]
print("\nratio unit_price/list_price — quantiles")
print(no["ratio"].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).round(6))

# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 3 — On PROMO rows: relationship with promo_type")
print("=" * 78)
yes = oi[oi["has_promo1"]].merge(
    prom.rename(columns={"promo_id":"promo_id","discount_value":"d1","promo_type":"type1"}),
    on="promo_id", how="left",
)
yes["ratio"] = yes["unit_price"] / yes["list_price"]
print("\nBy promo_type — ratio & diff summaries")
for t, g in yes.groupby("type1"):
    print(f"\n  type = {t}   (n={len(g):,})")
    print(f"    discount_value range: [{g['d1'].min():g}, {g['d1'].max():g}]")
    print(f"    ratio  mean/median/std : {g['ratio'].mean():.4f} / {g['ratio'].median():.4f} / {g['ratio'].std():.4f}")
    diff = (g['list_price'] - g['unit_price']).round(4)
    print(f"    (list - unit) mean/median/std : {diff.mean():.4f} / {diff.median():.4f} / {diff.std():.4f}")

# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 4 — What is discount_amount?")
print("=" * 78)
print("On NO-PROMO rows, discount_amount summary:")
print(no["discount_amount"].describe())

print("\nOn PROMO rows, discount_amount summary:")
print(yes["discount_amount"].describe())

# hypothesis: discount_amount == qty*(list_price - unit_price)
yes["implied_disc"] = (yes["list_price"] - yes["unit_price"]) * yes["quantity"]
yes["da_gap"] = (yes["discount_amount"] - yes["implied_disc"]).round(4)
print("\ndiscount_amount  vs  qty*(list-unit)  — gap distribution:")
print(yes["da_gap"].describe())
print(f"exact-zero gaps: {int(yes['da_gap'].eq(0).sum()):,} / {len(yes):,}")

# hypothesis B: discount_amount == list_price - unit_price (per unit, not whole line)
yes["implied_disc_per_unit"] = (yes["list_price"] - yes["unit_price"])
yes["da_gap_pu"] = (yes["discount_amount"] - yes["implied_disc_per_unit"]).round(4)
print("\ndiscount_amount  vs  (list-unit) per-unit  — gap distribution:")
print(yes["da_gap_pu"].describe())
print(f"exact-zero gaps: {int(yes['da_gap_pu'].eq(0).sum()):,} / {len(yes):,}")

# hypothesis C: discount_amount == promo.discount_value itself (flat)
yes["da_gap_flat"] = (yes["discount_amount"] - yes["d1"]).round(4)
print("\ndiscount_amount  vs  promo.discount_value  — gap distribution:")
print(yes["da_gap_flat"].describe())
print(f"exact-zero gaps: {int(yes['da_gap_flat'].eq(0).sum()):,} / {len(yes):,}")

# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 5 — Does unit_price drift over time for the same product?")
print("=" * 78)
# Check per-product unit_price dispersion on no-promo rows only
disp = (no.groupby("product_id")["unit_price"]
          .agg(['nunique','min','max','std','count'])
          .sort_values('nunique', ascending=False))
print(f"products with >1 distinct no-promo unit_price: "
      f"{int((disp['nunique']>1).sum()):,} / {len(disp):,}")
print("\nTop 5 most-varied products (no-promo unit_price):")
print(disp.head(5))

# Sample one heavily-varied product & plot timeline summary
if (disp['nunique'] > 1).any():
    pid = disp.query("nunique>1").index[0]
    sample = no[no["product_id"].eq(pid)].sort_values("order_date")
    yearly = (sample.assign(year=sample["order_date"].dt.year)
                    .groupby("year")["unit_price"].agg(['min','max','mean','count']))
    print(f"\nProduct {pid} list_price = {prod.loc[prod.product_id.eq(pid),'list_price'].iloc[0]}  — yearly unit_price stats:")
    print(yearly)

# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 6 — Payment reconciliation per order")
print("=" * 78)
line_total = oi.assign(line=oi["quantity"]*oi["unit_price"]).groupby("order_id")["line"].sum()
line_total.name = "lines_sum"

recon = (ords[["order_id"]]
         .merge(line_total, on="order_id", how="left")
         .merge(pay, on="order_id", how="left")
         .merge(shp, on="order_id", how="left"))
recon["shipping_fee"] = recon["shipping_fee"].fillna(0.0)
recon["implied_pay"] = recon["lines_sum"] + recon["shipping_fee"]
recon["gap"] = (recon["payment_value"] - recon["implied_pay"]).round(4)

print(f"orders evaluated: {len(recon):,}")
print("gap = payment_value - (Σ line_total + shipping_fee):")
print(recon["gap"].describe())
print(f"exact-zero gaps : {int(recon['gap'].eq(0).sum()):,}")

# Does subtracting discount_amount close the gap?
disc_total = oi.groupby("order_id")["discount_amount"].sum()
disc_total.name = "disc_sum"
recon = recon.merge(disc_total, on="order_id", how="left")
recon["implied_pay_minus_disc"] = recon["lines_sum"] - recon["disc_sum"].fillna(0) + recon["shipping_fee"]
recon["gap_minus_disc"] = (recon["payment_value"] - recon["implied_pay_minus_disc"]).round(4)
print("\nIf we SUBTRACT Σdiscount_amount from line total:")
print(recon["gap_minus_disc"].describe())
print(f"exact-zero : {int(recon['gap_minus_disc'].eq(0).sum()):,}")

# Does ADDING discount_amount close the gap?
recon["implied_pay_plus_disc"] = recon["lines_sum"] + recon["disc_sum"].fillna(0) + recon["shipping_fee"]
recon["gap_plus_disc"] = (recon["payment_value"] - recon["implied_pay_plus_disc"]).round(4)
print("\nIf we ADD Σdiscount_amount to line total:")
print(recon["gap_plus_disc"].describe())
print(f"exact-zero : {int(recon['gap_plus_disc'].eq(0).sum()):,}")

print("\nDone.")
