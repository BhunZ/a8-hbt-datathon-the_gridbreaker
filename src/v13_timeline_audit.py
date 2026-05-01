"""V13 Step 2 — Full-history forensics across all 13 files.

Goal: explain WHY revenue dropped ~50% between 2018 and 2019, and locate any
other regime changes across the entire 2012-2022 history. Output drives the
training-cut decision for V13.

Per-dimension metrics computed by month:
  Orders / customers
    - n_orders, n_unique_customers, orders_per_active_customer
    - n_signups (new customer cohort size)
  Basket
    - aov, items_per_order, avg_unit_price, avg_discount_pct, gross, cogs_recon
  Mix
    - device share (mobile/desktop/tablet)
    - payment method share
    - order source share (paid_search/organic/email/social/referral/direct)
    - acquisition channel share for new signups
  Catalog & inventory
    - n_products_active (had >0 items ordered)
    - mean_price, mean_cogs_ratio (catalog level)
    - stockout rate, fill rate
  Web traffic
    - sessions, unique_visitors, page_views, bounce_rate, avg session duration
    - traffic-source mix
  Customer experience
    - n_reviews, rating_mean, rating_low_pct
    - n_returns, return_rate, refund_rate
  Promotions
    - n_active_promos, mean_discount_value
  Shipments
    - mean_shipping_fee, mean_delivery_days

Change-point detection: for each metric, compute month-over-month % change
and flag months whose change > 2 stddev OR cumulative drop > 30% from prior peak.

Outputs:
  figures/v13_timeline_<group>.png        (per-group multi-panel charts)
  figures/v13_timeline_summary.png        (one-page overview)
  data/processed/v13_timeline_monthly.parquet
  docs/v13_timeline_audit.md              (narrative)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from paths import RAW, PROCESSED, FIGURES, DOCS

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.1f}s] {msg}")

TRAIN_END = pd.Timestamp("2022-12-31")

# ============================================================================
# 1. LOAD all files
# ============================================================================
log("Loading orders.csv...")
orders = pd.read_csv(RAW / "orders.csv", parse_dates=["order_date"], low_memory=False)
orders = orders[orders.order_date <= TRAIN_END]
orders["ym"] = orders.order_date.dt.to_period("M").dt.to_timestamp()

log("Loading order_items.csv...")
items = pd.read_csv(RAW / "order_items.csv", low_memory=False)
items = items.merge(orders[["order_id", "order_date", "ym"]], on="order_id", how="inner")

log("Loading payments.csv...")
pay = pd.read_csv(RAW / "payments.csv", low_memory=False)
pay = pay.merge(orders[["order_id", "order_date", "ym"]], on="order_id", how="inner")

log("Loading customers.csv...")
customers = pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"], low_memory=False)
customers["ym"] = customers.signup_date.dt.to_period("M").dt.to_timestamp()
customers = customers[customers.signup_date <= TRAIN_END]

log("Loading products.csv...")
products = pd.read_csv(RAW / "products.csv", low_memory=False)

log("Loading reviews.csv...")
reviews = pd.read_csv(RAW / "reviews.csv", parse_dates=["review_date"], low_memory=False)
reviews = reviews[reviews.review_date <= TRAIN_END]
reviews["ym"] = reviews.review_date.dt.to_period("M").dt.to_timestamp()

log("Loading returns.csv...")
returns = pd.read_csv(RAW / "returns.csv", parse_dates=["return_date"], low_memory=False)
returns = returns[returns.return_date <= TRAIN_END]
returns["ym"] = returns.return_date.dt.to_period("M").dt.to_timestamp()

log("Loading shipments.csv...")
ship = pd.read_csv(RAW / "shipments.csv", parse_dates=["ship_date","delivery_date"], low_memory=False)
ship = ship.merge(orders[["order_id","ym"]], on="order_id", how="inner")
ship["delivery_days"] = (ship.delivery_date - ship.ship_date).dt.days

log("Loading web_traffic.csv...")
traffic = pd.read_csv(RAW / "web_traffic.csv", parse_dates=["date"], low_memory=False)
traffic = traffic[traffic.date <= TRAIN_END]
traffic["ym"] = traffic.date.dt.to_period("M").dt.to_timestamp()

log("Loading inventory.csv...")
inv = pd.read_csv(RAW / "inventory.csv", parse_dates=["snapshot_date"], low_memory=False)
inv = inv[inv.snapshot_date <= TRAIN_END]
inv["ym"] = inv.snapshot_date.dt.to_period("M").dt.to_timestamp()

log("Loading sales.csv...")
sales = pd.read_csv(RAW / "sales.csv", parse_dates=["Date"], low_memory=False).rename(columns={"Date":"date"})
sales = sales[sales.date <= TRAIN_END]
sales["ym"] = sales.date.dt.to_period("M").dt.to_timestamp()

log("Loading promotions.csv...")
promos = pd.read_csv(RAW / "promotions.csv", parse_dates=["start_date","end_date"], low_memory=False)

log("All files loaded.")

# ============================================================================
# 2. MONTHLY AGGREGATES
# ============================================================================
log("Computing monthly aggregates...")

# --- orders / customer activity
o = orders.copy()
o_monthly = o.groupby("ym").agg(
    n_orders          = ("order_id", "count"),
    n_unique_cust     = ("customer_id", "nunique"),
    pct_mobile        = ("device_type", lambda s: (s=="mobile").mean()),
    pct_desktop       = ("device_type", lambda s: (s=="desktop").mean()),
    pct_tablet        = ("device_type", lambda s: (s=="tablet").mean()),
    pct_cc            = ("payment_method", lambda s: (s=="credit_card").mean()),
    pct_cod           = ("payment_method", lambda s: (s=="cod").mean()),
    pct_paypal        = ("payment_method", lambda s: (s=="paypal").mean()),
    pct_paid_search   = ("order_source", lambda s: (s=="paid_search").mean()),
    pct_organic       = ("order_source", lambda s: (s=="organic").mean()),
    pct_email         = ("order_source", lambda s: (s=="email").mean()),
    pct_social        = ("order_source", lambda s: (s=="social").mean()),
    pct_direct        = ("order_source", lambda s: (s=="direct").mean()),
    pct_referral      = ("order_source", lambda s: (s=="referral").mean()),
    pct_returned      = ("order_status", lambda s: (s=="returned").mean()),
    pct_cancelled     = ("order_status", lambda s: (s=="cancelled").mean()),
).reset_index()
o_monthly["orders_per_cust"] = o_monthly.n_orders / o_monthly.n_unique_cust

# --- new signups
s_monthly = customers.groupby("ym").agg(
    n_signups          = ("customer_id","count"),
    pct_acq_social     = ("acquisition_channel", lambda s: (s=="social_media").mean()),
    pct_acq_email      = ("acquisition_channel", lambda s: (s=="email_campaign").mean()),
    pct_acq_search     = ("acquisition_channel", lambda s: (s=="search").mean()),
    pct_acq_referral   = ("acquisition_channel", lambda s: (s=="referral").mean()),
    pct_acq_paid       = ("acquisition_channel", lambda s: (s=="paid_ads").mean()),
).reset_index()

# --- basket / items
prod_lookup = products.set_index("product_id")[["price","cogs","category","segment"]]
i2 = items.merge(prod_lookup[["price","cogs"]], on="product_id", how="left")
i2["line_gross"]   = i2.quantity * i2.unit_price - i2.discount_amount.fillna(0)
i2["line_list"]    = i2.quantity * i2.unit_price
i2["line_disc"]    = i2.discount_amount.fillna(0)
i2["line_cogs"]    = i2.quantity * i2.cogs
i_monthly = i2.groupby("ym").agg(
    n_items_sold        = ("quantity","sum"),
    n_line_items        = ("quantity","count"),
    items_gross         = ("line_gross","sum"),
    items_cogs_recon    = ("line_cogs","sum"),
    items_list          = ("line_list","sum"),
    items_discount_sum  = ("line_disc","sum"),
    avg_unit_price      = ("unit_price","mean"),
    avg_discount        = ("discount_amount","mean"),
).reset_index()
items_per_order = i2.groupby(["ym","order_id"])["quantity"].sum().groupby("ym").mean().rename("items_per_order").reset_index()
i_monthly = i_monthly.merge(items_per_order, on="ym", how="left")
i_monthly["discount_pct"] = i_monthly.items_discount_sum / i_monthly.items_list
i_monthly["cogs_ratio"]   = i_monthly.items_cogs_recon / i_monthly.items_gross
i_monthly["aov"]          = i_monthly.items_gross / o_monthly.set_index("ym").reindex(i_monthly.ym).n_orders.values

# --- payments
p_monthly = pay.groupby("ym").agg(
    payment_value_sum = ("payment_value","sum"),
    mean_installments = ("installments","mean"),
).reset_index()

# --- shipments
sh_monthly = ship.groupby("ym").agg(
    mean_ship_fee     = ("shipping_fee","mean"),
    mean_delivery_days = ("delivery_days","mean"),
).reset_index()

# --- traffic
tot = traffic.groupby("ym").agg(
    sessions          = ("sessions","sum"),
    unique_visitors   = ("unique_visitors","sum"),
    page_views        = ("page_views","sum"),
    bounce_rate       = ("bounce_rate","mean"),
    avg_sess_dur      = ("avg_session_duration_sec","mean"),
).reset_index()
src = traffic.groupby(["ym","traffic_source"])["sessions"].sum().unstack(fill_value=0)
src_share = src.div(src.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
src_share.columns = [f"sess_share_{c}" for c in src_share.columns]
t_monthly = tot.merge(src_share.reset_index(), on="ym", how="left")

# --- reviews
rv_monthly = reviews.groupby("ym").agg(
    n_reviews         = ("review_id","count"),
    rating_mean       = ("rating","mean"),
    pct_low_rating    = ("rating", lambda s: (s<=2).mean()),
).reset_index()

# --- returns
rt_monthly = returns.groupby("ym").agg(
    n_returns         = ("return_id","count"),
    return_qty_sum    = ("return_quantity","sum"),
    refund_sum        = ("refund_amount","sum"),
).reset_index()
# return rate = n_returns / n_orders
rt_monthly = rt_monthly.merge(o_monthly[["ym","n_orders"]], on="ym", how="left")
rt_monthly["return_rate"] = rt_monthly.n_returns / rt_monthly.n_orders

# --- inventory
inv_monthly = inv.groupby("ym").agg(
    stockout_rate     = ("stockout_flag","mean"),
    overstock_rate    = ("overstock_flag","mean"),
    mean_fill_rate    = ("fill_rate","mean"),
    n_skus_tracked    = ("product_id","nunique"),
).reset_index()

# --- catalog activity
catalog_monthly = i2.groupby("ym").agg(
    n_active_skus     = ("product_id","nunique"),
).reset_index()

# --- promotions activity (count of promos active in month)
prom_rows = []
for _, r in promos.iterrows():
    mrange = pd.period_range(r.start_date, r.end_date, freq="M")
    for p in mrange:
        prom_rows.append({"ym": p.to_timestamp(), "discount": r.discount_value})
if prom_rows:
    prom_m = pd.DataFrame(prom_rows).groupby("ym").agg(
        n_active_promos = ("discount","count"),
        max_discount    = ("discount","max"),
        mean_discount   = ("discount","mean"),
    ).reset_index()
else:
    prom_m = pd.DataFrame(columns=["ym","n_active_promos","max_discount","mean_discount"])

# --- sales (already monthly-friendly)
sa_monthly = sales.groupby("ym").agg(
    revenue_actual = ("Revenue","mean"),
    cogs_actual    = ("COGS","mean"),
).reset_index()
sa_monthly["actual_cogs_ratio"] = sa_monthly.cogs_actual / sa_monthly.revenue_actual

# ============================================================================
# 3. JOIN ALL into single monthly table
# ============================================================================
M = o_monthly.merge(s_monthly, on="ym", how="outer") \
             .merge(i_monthly, on="ym", how="outer") \
             .merge(p_monthly, on="ym", how="outer") \
             .merge(sh_monthly, on="ym", how="outer") \
             .merge(t_monthly, on="ym", how="outer") \
             .merge(rv_monthly, on="ym", how="outer") \
             .merge(rt_monthly[["ym","n_returns","return_rate","refund_sum"]], on="ym", how="outer") \
             .merge(inv_monthly, on="ym", how="outer") \
             .merge(catalog_monthly, on="ym", how="outer") \
             .merge(prom_m, on="ym", how="outer") \
             .merge(sa_monthly, on="ym", how="outer")
M = M.sort_values("ym").reset_index(drop=True)
M["year"] = M.ym.dt.year

PROCESSED.mkdir(parents=True, exist_ok=True)
M_path = PROCESSED / "v13_timeline_monthly.parquet"
M.to_parquet(M_path, index=False)
log(f"monthly table -> {M_path}  ({len(M)} months × {len(M.columns)} cols)")

# ============================================================================
# 4. CHANGE-POINT DETECTION
# ============================================================================
log("Detecting change points...")
def detect_changes(series, window=12):
    """Return list of (ym, prev_avg, new_avg, pct_change) where rolling mean shifts."""
    s = series.dropna()
    if len(s) < 2*window: return []
    prev = s.rolling(window, min_periods=window//2).mean().shift(window)
    curr = s.rolling(window, min_periods=window//2).mean()
    delta = (curr - prev) / prev.abs().replace(0, np.nan)
    out = []
    for ym, d in delta.items():
        if pd.isna(d): continue
        if abs(d) > 0.20:  # >20% shift in 12-month window
            out.append((ym, prev.loc[ym], curr.loc[ym], d))
    return out

key_metrics = [
    "n_orders","n_unique_cust","n_signups","aov","items_per_order","avg_unit_price",
    "discount_pct","sessions","unique_visitors","n_active_skus","return_rate",
    "rating_mean","mean_ship_fee","revenue_actual","actual_cogs_ratio",
    "pct_mobile","pct_cod","pct_paid_search","mean_installments",
]
change_points = {}
for k in key_metrics:
    if k in M.columns:
        ts = M.set_index("ym")[k]
        change_points[k] = detect_changes(ts)

# print top change points per metric
for k, lst in change_points.items():
    if not lst: continue
    print(f"\n--- CHANGE POINTS: {k} ---")
    # take 3 biggest by abs %
    big = sorted(lst, key=lambda x: -abs(x[3]))[:3]
    for ym, prev, curr, pct in big:
        print(f"  {ym.date()}  prev_avg={prev:>12.3f}  new_avg={curr:>12.3f}  Δ={pct:+.1%}")

# ============================================================================
# 5. PLOTS
# ============================================================================
log("Generating figures...")
FIGURES.mkdir(parents=True, exist_ok=True)

def setup_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # mark the 2018→2019 cliff
    ax.axvspan(pd.Timestamp("2018-09-01"), pd.Timestamp("2019-12-31"),
               color="orange", alpha=0.10)
    # mark COVID
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
               color="red", alpha=0.08)

# ---- Group A: business volume
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
axes = axes.flatten()
axes[0].plot(M.ym, M.n_orders/1000, color="tab:blue");           setup_ax(axes[0], "Orders per month (k)",         "thousands")
axes[1].plot(M.ym, M.n_unique_cust/1000, color="tab:orange");    setup_ax(axes[1], "Unique customers per month (k)", "thousands")
axes[2].plot(M.ym, M.n_signups/1000, color="tab:green");         setup_ax(axes[2], "New signups per month (k)",     "thousands")
axes[3].plot(M.ym, M.orders_per_cust, color="tab:red");          setup_ax(axes[3], "Orders per active customer",     "ratio")
axes[4].plot(M.ym, M.revenue_actual/1e6, color="tab:purple");    setup_ax(axes[4], "Mean daily Revenue (M)",         "$M")
axes[5].plot(M.ym, M.payment_value_sum/1e6, color="tab:brown");  setup_ax(axes[5], "Total payment value per month (M)","$M")
plt.suptitle("Group A — Business volume timeline (orange = 2018→2019 cliff,  red = COVID window)", y=1.00)
plt.tight_layout()
plt.savefig(FIGURES / "v13_timeline_volume.png", dpi=130); plt.close()

# ---- Group B: basket economics
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
axes = axes.flatten()
axes[0].plot(M.ym, M.aov, color="tab:blue");           setup_ax(axes[0], "AOV (gross / order)",            "$")
axes[1].plot(M.ym, M.items_per_order, color="tab:orange");  setup_ax(axes[1], "Items per order",            "qty")
axes[2].plot(M.ym, M.avg_unit_price, color="tab:green");    setup_ax(axes[2], "Average unit price",         "$")
axes[3].plot(M.ym, M.discount_pct*100, color="tab:red");    setup_ax(axes[3], "Discount as % of list",      "%")
axes[4].plot(M.ym, M.cogs_ratio, color="tab:purple");       setup_ax(axes[4], "COGS / Revenue (recon)",      "ratio")
axes[5].plot(M.ym, M.actual_cogs_ratio, color="tab:brown"); setup_ax(axes[5], "COGS / Revenue (actual sales)", "ratio")
plt.suptitle("Group B — Basket economics timeline", y=1.00)
plt.tight_layout()
plt.savefig(FIGURES / "v13_timeline_basket.png", dpi=130); plt.close()

# ---- Group C: channel mix
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
axes = axes.flatten()
for src in ["pct_mobile","pct_desktop","pct_tablet"]:
    axes[0].plot(M.ym, M[src], label=src.replace("pct_",""))
setup_ax(axes[0], "Device-type mix", "share"); axes[0].legend(fontsize=8)
for src in ["pct_cc","pct_cod","pct_paypal"]:
    axes[1].plot(M.ym, M[src], label=src.replace("pct_",""))
setup_ax(axes[1], "Payment-method mix", "share"); axes[1].legend(fontsize=8)
for src in ["pct_paid_search","pct_organic","pct_email","pct_social","pct_direct","pct_referral"]:
    if src in M.columns: axes[2].plot(M.ym, M[src], label=src.replace("pct_",""))
setup_ax(axes[2], "Order source mix", "share"); axes[2].legend(fontsize=7, ncol=2)
for src in ["pct_acq_social","pct_acq_email","pct_acq_search","pct_acq_referral","pct_acq_paid"]:
    if src in M.columns: axes[3].plot(M.ym, M[src], label=src.replace("pct_acq_",""))
setup_ax(axes[3], "Customer acquisition channel (new signups)", "share"); axes[3].legend(fontsize=7, ncol=2)
sess_share_cols = [c for c in M.columns if c.startswith("sess_share_")]
for c in sess_share_cols:
    axes[4].plot(M.ym, M[c], label=c.replace("sess_share_",""))
setup_ax(axes[4], "Web-traffic source share", "share"); axes[4].legend(fontsize=7, ncol=2)
axes[5].plot(M.ym, M.mean_installments, color="tab:purple")
setup_ax(axes[5], "Mean installments per payment", "count")
plt.suptitle("Group C — Mix shifts timeline", y=1.00)
plt.tight_layout()
plt.savefig(FIGURES / "v13_timeline_mix.png", dpi=130); plt.close()

# ---- Group D: customer experience + ops
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
axes = axes.flatten()
axes[0].plot(M.ym, M.return_rate*100, color="tab:red");          setup_ax(axes[0], "Return rate (%)", "%")
axes[1].plot(M.ym, M.rating_mean, color="tab:green");            setup_ax(axes[1], "Mean rating", "stars")
axes[2].plot(M.ym, M.pct_low_rating*100, color="tab:orange");    setup_ax(axes[2], "% low ratings (≤2)", "%")
axes[3].plot(M.ym, M.mean_ship_fee, color="tab:blue");           setup_ax(axes[3], "Mean shipping fee ($)", "$")
axes[4].plot(M.ym, M.mean_delivery_days, color="tab:purple");    setup_ax(axes[4], "Mean delivery days", "days")
axes[5].plot(M.ym, M.n_active_skus, color="tab:brown");          setup_ax(axes[5], "Active SKUs per month", "count")
plt.suptitle("Group D — Customer experience & operations", y=1.00)
plt.tight_layout()
plt.savefig(FIGURES / "v13_timeline_ops.png", dpi=130); plt.close()

# ---- Group E: web traffic & promotions
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
axes = axes.flatten()
axes[0].plot(M.ym, M.sessions/1e3, color="tab:blue");            setup_ax(axes[0], "Total sessions per month (k)", "k")
axes[1].plot(M.ym, M.unique_visitors/1e3, color="tab:orange");   setup_ax(axes[1], "Unique visitors per month (k)", "k")
axes[2].plot(M.ym, M.bounce_rate*100, color="tab:red");          setup_ax(axes[2], "Bounce rate (%)", "%")
axes[3].plot(M.ym, M.avg_sess_dur, color="tab:green");           setup_ax(axes[3], "Avg session duration (s)", "sec")
axes[4].plot(M.ym, M.n_active_promos, color="tab:purple");       setup_ax(axes[4], "Active promotions in month", "count")
axes[5].plot(M.ym, M.mean_discount, color="tab:brown");          setup_ax(axes[5], "Mean discount value (active promos)", "%")
plt.suptitle("Group E — Web traffic & promotions", y=1.00)
plt.tight_layout()
plt.savefig(FIGURES / "v13_timeline_web_promo.png", dpi=130); plt.close()

# ---- ONE-PAGE SUMMARY
fig, axes = plt.subplots(4, 2, figsize=(15, 13))
axes = axes.flatten()
axes[0].plot(M.ym, M.revenue_actual/1e6, color="tab:purple", lw=1.5); setup_ax(axes[0], "Revenue (mean daily, $M)", "$M")
axes[1].plot(M.ym, M.n_orders/1000, color="tab:blue");                setup_ax(axes[1], "Orders / month (k)", "k")
axes[2].plot(M.ym, M.aov, color="tab:orange");                        setup_ax(axes[2], "AOV ($)", "$")
axes[3].plot(M.ym, M.items_per_order, color="tab:green");             setup_ax(axes[3], "Items per order", "qty")
axes[4].plot(M.ym, M.avg_unit_price, color="tab:red");                setup_ax(axes[4], "Avg unit price ($)", "$")
axes[5].plot(M.ym, M.n_active_skus, color="tab:brown");               setup_ax(axes[5], "Active SKUs/month", "count")
axes[6].plot(M.ym, M.sessions/1e3, color="tab:cyan");                 setup_ax(axes[6], "Sessions / month (k)", "k")
axes[7].plot(M.ym, M.return_rate*100, color="tab:pink");              setup_ax(axes[7], "Return rate (%)", "%")
plt.suptitle("V13 Timeline — One-page summary  (orange = 2018→2019 cliff,  red = COVID window)",
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(FIGURES / "v13_timeline_summary.png", dpi=140); plt.close()

log("All figures written.")

# ============================================================================
# 6. NARRATIVE REPORT
# ============================================================================
log("Writing narrative...")

# helper to print yearly mean of a column
def yearly(col):
    s = M.groupby("year")[col].mean()
    return s.round(3).to_dict()

# build comparison: 2018 vs 2019, 2019 vs 2022
def cmp(col, y1, y2):
    a = M[M.year==y1][col].mean()
    b = M[M.year==y2][col].mean()
    if a == 0 or pd.isna(a) or pd.isna(b): return "—"
    return f"{a:.3f} → {b:.3f}  ({(b-a)/abs(a)*100:+.1f}%)"

key_compare = [
    "n_orders","n_unique_cust","n_signups","orders_per_cust","aov","items_per_order",
    "avg_unit_price","discount_pct","cogs_ratio","sessions","unique_visitors","bounce_rate",
    "n_active_skus","return_rate","rating_mean","mean_ship_fee","mean_delivery_days",
    "mean_installments","pct_cod","pct_mobile","pct_paid_search","pct_acq_paid",
    "n_active_promos","mean_discount",
]

md = [
    "# V13 Step 2 — Full-history Timeline Forensics",
    "",
    "Produced by `src/v13_timeline_audit.py`.  Data: 13 raw files, 2012-07 → 2022-12.",
    "",
    "## 1. Headline finding",
    "",
    "Revenue dropped ~50% between 2018 and 2019. The audit attributes this to **specific dimensions** "
    "below; multiply through to confirm whether the cause was volume (orders/customers) or unit economics "
    "(AOV/items/price).",
    "",
    "## 2. 2018 vs 2019 — what changed (yearly mean)",
    "",
    "| Metric | 2018 → 2019 | Direction |",
    "|---|---|---|",
]
for k in key_compare:
    if k in M.columns:
        md.append(f"| `{k}` | {cmp(k, 2018, 2019)} | |")

md += [
    "",
    "## 3. 2019 vs 2022 — same-regime comparison",
    "",
    "| Metric | 2019 → 2022 | |",
    "|---|---|---|",
]
for k in key_compare:
    if k in M.columns:
        md.append(f"| `{k}` | {cmp(k, 2019, 2022)} | |")

md += [
    "",
    "## 4. Yearly mean — every key metric",
    "",
    "| Year | "
    + " | ".join(key_compare) + " |",
    "|---|" + "|".join(["---"] * len(key_compare)) + "|",
]
for y in sorted(M.year.dropna().unique()):
    row = f"| {int(y)} |"
    for k in key_compare:
        if k in M.columns:
            v = M[M.year==y][k].mean()
            row += f" {v:.3f} |" if not pd.isna(v) else " — |"
        else:
            row += " — |"
    md.append(row)

md += [
    "",
    "## 5. Detected change points (>20% shift in 12-month rolling mean)",
    "",
    "| Metric | Date | Prev avg | New avg | Δ |",
    "|---|---|---:|---:|---:|",
]
for k, lst in change_points.items():
    if not lst: continue
    big = sorted(lst, key=lambda x: -abs(x[3]))[:3]
    for ym, prev, curr, pct in big:
        md.append(f"| `{k}` | {ym.date()} | {prev:.3f} | {curr:.3f} | {pct:+.1%} |")

md += [
    "",
    "## 6. Figures",
    "",
    "- `figures/v13_timeline_summary.png` — one-page overview of 8 key metrics.",
    "- `figures/v13_timeline_volume.png` — orders, customers, signups, revenue.",
    "- `figures/v13_timeline_basket.png` — AOV, items per order, unit price, discount, COGS ratio.",
    "- `figures/v13_timeline_mix.png` — device, payment, channel, acquisition.",
    "- `figures/v13_timeline_ops.png` — returns, ratings, shipping, delivery, SKUs.",
    "- `figures/v13_timeline_web_promo.png` — sessions, visitors, bounce, promotions.",
    "",
    "Orange shading = 2018-09 → 2019-12 (the cliff).  Red shading = 2020-03 → 2021-12 (COVID window).",
    "",
    "## 7. How this drives V13",
    "",
    "Inspect the change-point table and per-group figures, then decide:",
    "1. **Training cut.** If most key metrics shifted in 2019 and stayed shifted, train on 2019+ only.",
    "2. **COVID flags.** If COVID didn't materially move metrics relative to 2019 baseline, "
    "demote the COVID feature block to a single `covid_severity` scalar (or drop it).",
    "3. **Per-feature regime adjustment.** For metrics that shifted (e.g. mean_installments doubling), "
    "consider regime indicators rather than treating them as continuous.",
    "",
]
DOCS.mkdir(parents=True, exist_ok=True)
md_path = DOCS / "v13_timeline_audit.md"
md_path.write_text("\n".join(md), encoding="utf-8")
log(f"narrative -> {md_path}")
log(f"DONE in {time.time()-t0:.1f}s")
