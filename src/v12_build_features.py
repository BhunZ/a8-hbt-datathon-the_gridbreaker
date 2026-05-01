"""V12 Step 2 — Build daily_features_v12.parquet.

Produces a single daily-grain table covering 2017-01-01 → 2024-07-01.
  - rows ≤ 2022-12-31: actual daily rollups from the 13 source files
  - rows 2023-01-01 → 2024-07-01: 5D profile-projected values
    (profile key = month, dow, is_dd, is_pd, is_tet; median aggregation)

Design principles:
  1. Every aux-derived numeric feature gets the SAME profile treatment as friend's `sessions`.
  2. Calendar / event flags are computed directly for the full horizon — no projection needed.
  3. AOV has its own linear-trend projection (monotonically rising signal).
  4. We DO NOT include Revenue / COGS in the feature set. Those are targets.

Outputs:
  data/processed/daily_features_v12.parquet
  data/processed/daily_features_v12_summary.txt   (stats, null audit)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from paths import RAW, PROCESSED

t0 = time.time()

# ---------- horizon
TRAIN_START = pd.Timestamp("2017-01-01")
TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")

ALL_DATES = pd.date_range(TRAIN_START, TEST_END, freq="D")

# ---------- Tet / event calendars
TET = [pd.Timestamp(d) for d in [
    "2017-01-28","2018-02-16","2019-02-05","2020-01-25",
    "2021-02-12","2022-02-01","2023-01-22","2024-02-10"
]]

def nth_weekday_of_month(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    shift = (wd - d.weekday()) % 7
    return d + pd.Timedelta(days=shift + 7 * (n - 1))

# ---------- calendar features (FULL horizon, no projection)
def build_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates})
    df["year"]        = df.date.dt.year
    df["month"]       = df.date.dt.month
    df["day"]         = df.date.dt.day
    df["dow"]         = df.date.dt.dayofweek
    df["woy"]         = df.date.dt.isocalendar().week.astype(int)
    df["day_of_year"] = df.date.dt.dayofyear
    df["is_weekend"]  = (df.dow >= 5).astype(int)
    df["is_dd"]       = (df.month == df.day).astype(int)
    df["is_pd"]       = ((df.day >= 25) | (df.day <= 5)).astype(int)
    # Tet season = 21 days before Lunar New Year
    df["is_tet"] = 0
    for tet in TET:
        mask = (df.date >= (tet - pd.Timedelta(days=21))) & (df.date < tet)
        df.loc[mask, "is_tet"] = 1
    # Named events
    df["is_singles_day"]    = ((df.month == 11) & (df.day == 11)).astype(int)
    df["is_twelve_twelve"]  = ((df.month == 12) & (df.day == 12)).astype(int)
    df["is_nine_nine"]      = ((df.month == 9)  & (df.day == 9)).astype(int)
    df["is_cyber_monday"]   = 0
    df["is_black_friday"]   = 0
    for y in range(TRAIN_START.year, TEST_END.year + 1):
        bf = nth_weekday_of_month(y, 11, 4, 4)      # 4th Friday of Nov
        cm = bf + pd.Timedelta(days=3)              # Monday after
        df.loc[df.date == bf, "is_black_friday"] = 1
        df.loc[df.date == cm, "is_cyber_monday"] = 1
    df["is_womens_day_vn"]  = ((df.month == 10) & (df.day == 20)).astype(int)   # 20/10
    df["is_intl_women_day"] = ((df.month == 3)  & (df.day == 8)).astype(int)    # 8/3
    df["is_mothers_day_vn"] = 0  # Vu Lan — movable; skip for now (rare signal)
    return df


# ---------- HISTORICAL DAILY ROLLUPS -------------------------------------------------
print(f"[{time.time()-t0:5.1f}s] Loading source files...")

orders = pd.read_csv(RAW / "orders.csv",
                     parse_dates=["order_date"],
                     low_memory=False)
orders = orders[(orders.order_date >= TRAIN_START) & (orders.order_date <= TRAIN_END)]

items  = pd.read_csv(RAW / "order_items.csv", low_memory=False)
items  = items.merge(orders[["order_id", "order_date"]], on="order_id", how="inner")

pay    = pd.read_csv(RAW / "payments.csv", low_memory=False)
pay    = pay.merge(orders[["order_id", "order_date"]], on="order_id", how="inner")

traffic = pd.read_csv(RAW / "web_traffic.csv", parse_dates=["date"], low_memory=False)
traffic = traffic[traffic.date <= TRAIN_END]

reviews = pd.read_csv(RAW / "reviews.csv", parse_dates=["review_date"], low_memory=False)
reviews = reviews[reviews.review_date <= TRAIN_END]

returns = pd.read_csv(RAW / "returns.csv", parse_dates=["return_date"], low_memory=False)
returns = returns[returns.return_date <= TRAIN_END]

customers = pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"], low_memory=False)
customers = customers[customers.signup_date <= TRAIN_END]

promos  = pd.read_csv(RAW / "promotions.csv",
                      parse_dates=["start_date", "end_date"], low_memory=False)

inventory = pd.read_csv(RAW / "inventory.csv",
                        parse_dates=["snapshot_date"], low_memory=False)
inventory = inventory[inventory.snapshot_date <= TRAIN_END]

sales = pd.read_csv(RAW / "sales.csv", parse_dates=["Date"], low_memory=False)
sales = sales[(sales.Date >= TRAIN_START) & (sales.Date <= TRAIN_END)].rename(columns={"Date":"date"})

print(f"[{time.time()-t0:5.1f}s] Source rows: orders={len(orders):,} items={len(items):,} "
      f"pay={len(pay):,} traffic={len(traffic):,} reviews={len(reviews):,} "
      f"returns={len(returns):,} cust={len(customers):,} inv={len(inventory):,}")


# ---------- orders rollup
print(f"[{time.time()-t0:5.1f}s] Rolling up orders / payments / items...")

o = orders.copy()
o["is_mobile"]   = (o.device_type == "mobile").astype(int)
o["is_desktop"]  = (o.device_type == "desktop").astype(int)
o["is_tablet"]   = (o.device_type == "tablet").astype(int)
o["is_cc"]       = (o.payment_method == "credit_card").astype(int)
o["is_cod"]      = (o.payment_method == "cod").astype(int)
o["is_paypal"]   = (o.payment_method == "paypal").astype(int)
orders_daily = o.groupby("order_date").agg(
    n_orders        = ("order_id", "count"),
    n_unique_cust   = ("customer_id", "nunique"),
    pct_mobile      = ("is_mobile", "mean"),
    pct_desktop     = ("is_desktop", "mean"),
    pct_tablet      = ("is_tablet", "mean"),
    pct_cc          = ("is_cc", "mean"),
    pct_cod         = ("is_cod", "mean"),
    pct_paypal      = ("is_paypal", "mean"),
).reset_index().rename(columns={"order_date":"date"})


# ---------- items rollup  (quantity, gross, cogs-reconstruct)
prods = pd.read_csv(RAW / "products.csv", usecols=["product_id","price","cogs"])
i2 = items.merge(prods, on="product_id", how="left")
i2["line_gross"] = i2.quantity * i2.unit_price - i2.discount_amount.fillna(0)
i2["line_cogs"]  = i2.quantity * i2.cogs
items_daily_order_lvl = i2.groupby(["order_date","order_id"]).agg(
    qty_in_order = ("quantity","sum")
).reset_index()
items_daily = i2.groupby("order_date").agg(
    n_items             = ("quantity","sum"),
    n_line_items        = ("quantity","count"),
    items_gross         = ("line_gross","sum"),
    items_cogs_recon    = ("line_cogs","sum"),
    avg_unit_price      = ("unit_price","mean"),
).reset_index().rename(columns={"order_date":"date"})
items_per_ord = items_daily_order_lvl.groupby("order_date")["qty_in_order"].mean()\
    .reset_index().rename(columns={"order_date":"date","qty_in_order":"items_per_order"})
items_daily = items_daily.merge(items_per_ord, on="date", how="left")


# ---------- payments rollup
pay_daily = pay.groupby("order_date").agg(
    payment_value_sum = ("payment_value","sum"),
    mean_installments = ("installments","mean"),
).reset_index().rename(columns={"order_date":"date"})


# ---------- traffic rollup (aggregate 7 sources into totals + shares)
tot = traffic.groupby("date").agg(
    sessions           = ("sessions","sum"),
    unique_visitors    = ("unique_visitors","sum"),
    page_views         = ("page_views","sum"),
    bounce_rate        = ("bounce_rate","mean"),
    avg_sess_dur       = ("avg_session_duration_sec","mean"),
).reset_index()
src_sum = traffic.groupby(["date","traffic_source"])["sessions"].sum().unstack(fill_value=0)
src_sum.columns = [f"sess_{c}" for c in src_sum.columns]
share = src_sum.div(src_sum.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
share.columns = [c.replace("sess_","pct_") for c in share.columns]
traffic_daily = tot.merge(share.reset_index(), on="date", how="left")


# ---------- reviews rollup
rv_daily = reviews.groupby("review_date").agg(
    n_reviews       = ("review_id","count"),
    rating_mean     = ("rating","mean"),
    n_low_ratings   = ("rating", lambda s: (s<=2).sum()),
).reset_index().rename(columns={"review_date":"date"})


# ---------- returns rollup
ret_daily = returns.groupby("return_date").agg(
    n_returns       = ("return_id","count"),
    return_qty_sum  = ("return_quantity","sum"),
    refund_sum      = ("refund_amount","sum"),
).reset_index().rename(columns={"return_date":"date"})


# ---------- customers rollup
cust_daily = customers.groupby("signup_date").agg(
    n_signups = ("customer_id","count")
).reset_index().rename(columns={"signup_date":"date"})


# ---------- inventory rollup
inv_daily = inventory.groupby("snapshot_date").agg(
    n_stockouts       = ("stockout_flag","sum"),
    mean_fill_rate    = ("fill_rate","mean"),
).reset_index().rename(columns={"snapshot_date":"date"})


# ---------- promotions — expand each promo into its active date range
# (this is small — only 50 promos; expanded ~ a few hundred days)
prom_rows = []
for _, r in promos.iterrows():
    rng = pd.date_range(r.start_date, r.end_date, freq="D")
    for d in rng:
        prom_rows.append({
            "date": d,
            "discount_value": r.discount_value if r.promo_type == "percentage"
                              else r.discount_value,  # treat fixed as-is (rare, only 5)
            "is_stackable": int(r.stackable_flag),
        })
promo_exp = pd.DataFrame(prom_rows)
prom_daily = promo_exp.groupby("date").agg(
    n_active_promos     = ("discount_value","count"),
    max_discount_active = ("discount_value","max"),
    n_stackable_active  = ("is_stackable","sum"),
).reset_index()


# ---------- sales (AOV history for trend projection)
sales["aov"] = sales.Revenue / orders_daily.set_index("date").reindex(sales.date).n_orders.values
sales_aov = sales[["date","aov"]]


# ---------- ASSEMBLE HISTORICAL
print(f"[{time.time()-t0:5.1f}s] Merging historical rollups...")
hist = build_calendar(pd.date_range(TRAIN_START, TRAIN_END, freq="D"))
for tbl in [orders_daily, items_daily, pay_daily, traffic_daily,
            rv_daily, ret_daily, cust_daily, inv_daily, prom_daily, sales_aov]:
    hist = hist.merge(tbl, on="date", how="left")

# fill sensible zeros for count-type features
zero_fill = ["n_orders","n_unique_cust","n_items","n_line_items","items_gross",
             "items_cogs_recon","payment_value_sum","n_reviews","n_low_ratings",
             "n_returns","return_qty_sum","refund_sum","n_signups","n_stockouts",
             "n_active_promos","n_stackable_active"]
for c in zero_fill:
    if c in hist.columns: hist[c] = hist[c].fillna(0)
# means / shares: forward fill then zero for early missing
share_cols = [c for c in hist.columns if c.startswith("pct_") and c.startswith("pct_")]
for c in share_cols:
    hist[c] = hist[c].fillna(0)
# float fill remaining numerics with column median
for c in hist.select_dtypes(include=[float]).columns:
    if hist[c].isna().any():
        hist[c] = hist[c].fillna(hist[c].median())

PROFILE_KEYS = ["month","dow","is_dd","is_pd","is_tet"]
FEATURE_COLS = [c for c in hist.columns
                if c not in ["date","year","day","woy","day_of_year"] + PROFILE_KEYS
                and not c.startswith("is_")]  # event flags are calendar, not projected

print(f"[{time.time()-t0:5.1f}s] Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")


# ---------- BUILD 5D PROFILE on 2017-2022 feature values
print(f"[{time.time()-t0:5.1f}s] Building 5D median profile...")
profile_5d = hist.groupby(PROFILE_KEYS)[FEATURE_COLS].median().reset_index()
profile_md = hist.groupby(["month","dow"])[FEATURE_COLS].median().reset_index()   # fallback
profile_m  = hist.groupby(["month"])[FEATURE_COLS].median().reset_index()          # 2nd fallback


# ---------- AOV linear trend (special case — monotonic)
aov_trend = hist[["date","aov","year","day_of_year"]].dropna().copy()
aov_trend["t"] = (aov_trend.date - TRAIN_START).dt.days
from numpy.polynomial.polynomial import Polynomial
coef = np.polyfit(aov_trend.t.values, aov_trend.aov.values, deg=1)
def aov_proj(d): return coef[1] + coef[0] * (d - TRAIN_START).days


# ---------- TEST HORIZON — build calendar + apply profile
print(f"[{time.time()-t0:5.1f}s] Projecting test horizon via 5D profile...")
test_cal = build_calendar(pd.date_range(TEST_START, TEST_END, freq="D"))
test_proj = test_cal.merge(profile_5d, on=PROFILE_KEYS, how="left", suffixes=("","_5d"))

# Per-column fallback: first to (month, dow) median, then to month-only median
def fillna_from(src_df: pd.DataFrame, keys: list[str]):
    """For any FEATURE_COL still NaN in test_proj, fill from src_df joined on `keys`."""
    lookup = src_df.set_index(keys)[FEATURE_COLS]
    for c in FEATURE_COLS:
        na_mask = test_proj[c].isna()
        if not na_mask.any():
            continue
        key_idx = pd.MultiIndex.from_frame(test_proj.loc[na_mask, keys]) if len(keys) > 1 \
                  else test_proj.loc[na_mask, keys[0]]
        test_proj.loc[na_mask, c] = lookup[c].reindex(key_idx).values

fillna_from(profile_md, ["month","dow"])
fillna_from(profile_m,  ["month"])
# any still-NaN → historical overall median
for c in FEATURE_COLS:
    if test_proj[c].isna().any():
        test_proj[c] = test_proj[c].fillna(hist[c].median())

# AOV override with linear projection
test_proj["aov"] = test_proj["date"].apply(aov_proj)


# ---------- UNION historical + projected
hist_tagged = hist.copy();       hist_tagged["set"] = "train"
test_tagged = test_proj.copy();  test_tagged["set"] = "test"
full = pd.concat([hist_tagged, test_tagged], ignore_index=True, sort=False)
full = full.sort_values("date").reset_index(drop=True)

# target-join for training rows
full = full.merge(sales[["date","Revenue","COGS"]].rename(columns={"date":"date"}), on="date", how="left")


# ---------- WRITE
PROCESSED.mkdir(parents=True, exist_ok=True)
out = PROCESSED / "daily_features_v12.parquet"
full.to_parquet(out, index=False)

# summary
summary_path = PROCESSED / "daily_features_v12_summary.txt"
with summary_path.open("w") as fh:
    fh.write(f"V12 features — {len(full)} rows, {len(full.columns)} cols\n")
    fh.write(f"Train rows: {(full.set=='train').sum()}    Test rows: {(full.set=='test').sum()}\n\n")
    fh.write("COLUMN SUMMARY\n")
    fh.write("-"*80 + "\n")
    for c in full.columns:
        nn = full[c].isna().sum()
        ty = str(full[c].dtype)
        mn = full[c].min() if ty not in ("object",) else ""
        mx = full[c].max() if ty not in ("object",) else ""
        fh.write(f"  {c:28s}  {ty:12s}  null={nn:>5}  min={mn}  max={mx}\n")
    fh.write("\nTRAIN vs TEST — sanity\n")
    fh.write("-"*80 + "\n")
    for c in FEATURE_COLS:
        a = full.loc[full.set=="train", c]; b = full.loc[full.set=="test", c]
        fh.write(f"  {c:28s}  train μ={a.mean():12.3f}   test μ={b.mean():12.3f}   "
                 f"ratio={b.mean()/a.mean() if a.mean() else float('nan'):6.3f}\n")

print(f"\n[{time.time()-t0:5.1f}s] WROTE {out}  ({len(full)} rows × {len(full.columns)} cols)")
print(f"                                     {summary_path}")
print(f"\nTrain set: {(full.set=='train').sum()} rows  Test set: {(full.set=='test').sum()} rows")
print(f"Total feature columns (non-flag, projectable): {len(FEATURE_COLS)}")
print(f"Calendar/event flags:                          {sum(1 for c in full.columns if c.startswith('is_'))}")
print(f"=== DONE in {time.time()-t0:.1f}s ===")
 {(full.set=='test').sum()} rows")
print(f"Total feature columns (non-flag, projectable): {len(FEATURE_COLS)}")
print(f"Calendar/event flags:                          {sum(1 for c in full.columns if c.startswith('is_'))}")
print(f"=== DONE in {time.time()-t0:.1f}s ===")
