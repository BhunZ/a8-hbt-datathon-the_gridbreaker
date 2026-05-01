"""Drill into B10, B11, B13 failures to inform triage."""
import sys, pandas as pd
sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")
import helpers as H
T = H.load_all()

orders = T["orders"][["order_id","order_status","order_date"]]

# ---- B10_has_ship ----
shp_cnt = T["shipments"].groupby("order_id").size().rename("n_ship")
j = orders.merge(shp_cnt, on="order_id", how="left").fillna({"n_ship": 0})
ship_statuses = {"shipped","delivered","returned"}
bad10 = j[j["order_status"].isin(ship_statuses) & (j["n_ship"].ne(1))]
print("=== B10_has_ship (564) ===")
print(bad10.groupby(["order_status","n_ship"]).size())
print("date range:", bad10["order_date"].min(), "→", bad10["order_date"].max())

# ---- B11 ----
ret_cnt = T["returns"].groupby("order_id").size().rename("n_ret")
jr = orders.merge(ret_cnt, on="order_id", how="left").fillna({"n_ret": 0})
bad11 = jr[(jr["order_status"].eq("returned")) & (jr["n_ret"].lt(1))]
print("\n=== B11 (80) ===")
print(bad11["order_date"].agg(["min","max","count"]))

# ---- B13 ----
cust = T["customers"][["customer_id","signup_date"]]
first_order = T["orders"].groupby("customer_id")["order_date"].min().rename("first_order_date").reset_index()
c = cust.merge(first_order, on="customer_id", how="left")
bad13 = c[c["first_order_date"].notna() & (c["signup_date"].gt(c["first_order_date"]))].copy()
bad13["delta_days"] = (bad13["signup_date"] - bad13["first_order_date"]).dt.days
print("\n=== B13 (80,623) signup - first_order days ===")
print(bad13["delta_days"].describe().round(1))
n_cust = len(c.dropna(subset=["first_order_date"]))
print(f"Fraction affected: {len(bad13):,}/{n_cust:,} = {len(bad13)/n_cust*100:.1f}%")
print("Quartiles (days):", bad13["delta_days"].quantile([.1,.25,.5,.75,.9,.99]).round(0).to_dict())
