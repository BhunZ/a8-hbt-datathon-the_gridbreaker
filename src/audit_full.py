"""Comprehensive column-level audit. Prints everything we can discover."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW

BASE = str(RAW)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)

def head(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def inspect(name, df, date_cols=None):
    print(f"\n### {name}  shape={df.shape}")
    print("Columns + dtypes:")
    for c in df.columns:
        nn = df[c].isna().sum()
        pct_na = 100*nn/len(df)
        print(f"  {c:30s}  {str(df[c].dtype):10s}  null={nn:>7}  ({pct_na:5.2f}%)  uniq={df[c].nunique()}")
    if date_cols:
        for dc in date_cols:
            if dc in df.columns:
                d = pd.to_datetime(df[dc], errors='coerce')
                print(f"  DATE[{dc}]: min={d.min()}  max={d.max()}")
    # For each categorical with <20 unique, show value counts
    for c in df.columns:
        try:
            if df[c].dtype == 'object' and df[c].nunique() < 25:
                vc = df[c].value_counts(dropna=False).head(20)
                print(f"  VALS[{c}]: {dict(vc)}")
        except:
            pass
    # Numeric summary
    num = df.select_dtypes(include=[np.number])
    if len(num.columns):
        print("Numeric summary:")
        print(num.describe().T[['min','max','mean','std']].round(2).to_string())

# ------------------------------------------------------------------
head("SALES")
sales = pd.read_csv(f"{BASE}/sales.csv", parse_dates=['Date'])
inspect("sales.csv", sales, ['Date'])

head("WEB TRAFFIC")
tr = pd.read_csv(f"{BASE}/web_traffic.csv", parse_dates=['date'])
inspect("web_traffic.csv", tr, ['date'])
# Sources count
print(f"\ntraffic_source x date uniqueness:")
print(f"  rows={len(tr)}  unique dates={tr.date.nunique()}  unique sources={tr.traffic_source.nunique()}")
print(f"  rows per (date, source): {tr.groupby(['date','traffic_source']).size().value_counts().to_dict()}")

head("ORDERS")
orders = pd.read_csv(f"{BASE}/orders.csv", parse_dates=['order_date'], low_memory=False)
inspect("orders.csv", orders, ['order_date'])
# Status, source, device breakdowns
for c in ['order_status','payment_method','device_type','order_source']:
    print(f"\n[{c}] value counts (with %):")
    print((orders[c].value_counts(normalize=True)*100).round(2))

head("ORDER ITEMS")
items = pd.read_csv(f"{BASE}/order_items.csv")
inspect("order_items.csv", items)
print(f"\nItems per order distribution: {items.groupby('order_id').size().describe().round(2).to_dict()}")
print(f"Unique products referenced: {items.product_id.nunique()}")
print(f"Rows with promo_id not null: {items.promo_id.notna().sum()} ({100*items.promo_id.notna().mean():.2f}%)")
print(f"Rows with promo_id_2 not null: {items.promo_id_2.notna().sum()}")

head("PAYMENTS")
pay = pd.read_csv(f"{BASE}/payments.csv")
inspect("payments.csv", pay)
print(f"\nInstallments distribution:")
print(pay.installments.value_counts().sort_index())

head("SHIPMENTS")
ship = pd.read_csv(f"{BASE}/shipments.csv", parse_dates=['ship_date','delivery_date'])
inspect("shipments.csv", ship, ['ship_date','delivery_date'])
# Compute ship-to-deliver days
ship['days_to_deliver'] = (ship.delivery_date - ship.ship_date).dt.days
# Compute order-to-ship gap (join with orders)
ship_m = ship.merge(orders[['order_id','order_date']], on='order_id', how='left')
ship_m['days_to_ship'] = (ship_m.ship_date - ship_m.order_date).dt.days
print(f"\ndays_to_deliver: min={ship.days_to_deliver.min()} med={ship.days_to_deliver.median()} max={ship.days_to_deliver.max()}  mean={ship.days_to_deliver.mean():.2f}")
print(f"days_to_ship   : min={ship_m.days_to_ship.min()} med={ship_m.days_to_ship.median()} max={ship_m.days_to_ship.max()}  mean={ship_m.days_to_ship.mean():.2f}")
print(f"Shipping_fee summary: min={ship.shipping_fee.min():.2f} median={ship.shipping_fee.median():.2f} max={ship.shipping_fee.max():.2f}")
# What fraction of orders have shipments? (may indicate order_status correlation)
print(f"orders with shipment: {ship.order_id.nunique()} / {orders.order_id.nunique()}  ({100*ship.order_id.nunique()/orders.order_id.nunique():.1f}%)")

head("REVIEWS")
rv = pd.read_csv(f"{BASE}/reviews.csv", parse_dates=['review_date'])
inspect("reviews.csv", rv, ['review_date'])
print(f"\nRating distribution:")
print(rv.rating.value_counts().sort_index())
# Average rating by year
rv['year'] = rv.review_date.dt.year
print("\nMean rating per year:")
print(rv.groupby('year').rating.agg(['mean','count']).round(3))
# review_title uniqueness (any pattern?)
print(f"\ntop 10 review_title values:")
print(rv.review_title.value_counts().head(10))

head("RETURNS")
ret = pd.read_csv(f"{BASE}/returns.csv", parse_dates=['return_date'])
inspect("returns.csv", ret, ['return_date'])
print(f"\nReturn reasons:")
print(ret.return_reason.value_counts(normalize=True).round(3)*100)
# Return delay (return_date - order_date)
ret_m = ret.merge(orders[['order_id','order_date']], on='order_id', how='left')
ret_m['delay_days'] = (ret_m.return_date - ret_m.order_date).dt.days
print(f"\nReturn delay days: min={ret_m.delay_days.min()} med={ret_m.delay_days.median()} max={ret_m.delay_days.max()} mean={ret_m.delay_days.mean():.1f}")

head("PRODUCTS")
prod = pd.read_csv(f"{BASE}/products.csv")
inspect("products.csv", prod)
print(f"\nCategory distribution:")
print(prod.category.value_counts())
print(f"\nSegment distribution:")
print(prod.segment.value_counts())
print(f"Sizes: {prod['size'].value_counts().to_dict()}")
print(f"Colors: {prod.color.value_counts().head(10).to_dict()}")
print(f"Price range: min={prod.price.min():.2f} median={prod.price.median():.2f} max={prod.price.max():.2f}")
print(f"Margin (price-cogs): min={(prod.price-prod.cogs).min():.2f} median={(prod.price-prod.cogs).median():.2f} max={(prod.price-prod.cogs).max():.2f}")
print(f"Margin %: min={((prod.price-prod.cogs)/prod.price*100).min():.2f} median={((prod.price-prod.cogs)/prod.price*100).median():.2f} max={((prod.price-prod.cogs)/prod.price*100).max():.2f}")

head("GEOGRAPHY")
geo = pd.read_csv(f"{BASE}/geography.csv")
inspect("geography.csv", geo)
print(f"\nCities (top 20):")
print(geo.city.value_counts().head(20))
print(f"\nRegions:")
print(geo.region.value_counts())

head("CUSTOMERS")
cu = pd.read_csv(f"{BASE}/customers.csv", parse_dates=['signup_date'])
inspect("customers.csv", cu, ['signup_date'])
print(f"\nGender: {cu.gender.value_counts(normalize=True).round(3).to_dict()}")
print(f"Age group: {cu.age_group.value_counts().to_dict()}")
print(f"Acquisition channel: {cu.acquisition_channel.value_counts().to_dict()}")

head("PROMOTIONS")
pr = pd.read_csv(f"{BASE}/promotions.csv", parse_dates=['start_date','end_date'])
inspect("promotions.csv", pr, ['start_date','end_date'])
pr['duration_days'] = (pr.end_date - pr.start_date).dt.days
print(f"\nPromo duration: min={pr.duration_days.min()} med={pr.duration_days.median()} max={pr.duration_days.max()}")
print(f"Discount values:")
print(f"  type=percentage: min={pr[pr.promo_type=='percentage'].discount_value.min()} max={pr[pr.promo_type=='percentage'].discount_value.max()}")
print(f"  type=fixed     : min={pr[pr.promo_type=='fixed'].discount_value.min()} max={pr[pr.promo_type=='fixed'].discount_value.max()}")
print(f"\nChannel: {pr.promo_channel.value_counts().to_dict()}")
print(f"Stackable: {pr.stackable_flag.value_counts().to_dict()}")
print(f"Applicable category: {pr.applicable_category.value_counts(dropna=False).to_dict()}")

head("INVENTORY")
inv = pd.read_csv(f"{BASE}/inventory.csv", parse_dates=['snapshot_date'])
inspect("inventory.csv", inv, ['snapshot_date'])
print(f"\nSnapshots per product id: {inv.groupby('product_id').size().describe().round(2).to_dict()}")
print(f"Products with inventory data: {inv.product_id.nunique()} / {prod.product_id.nunique() if 'prod' in dir() else '?'}")

print("\n\n=== FINAL SUMMARY ===")
print(f"sales        : {len(sales):,} rows   {sales.Date.min()} → {sales.Date.max()}")
print(f"web_traffic  : {len(tr):,} rows   {tr.date.min()} → {tr.date.max()}")
print(f"orders       : {len(orders):,} rows   {orders.order_date.min()} → {orders.order_date.max()}")
print(f"order_items  : {len(items):,} rows")
print(f"payments     : {len(pay):,} rows")
print(f"shipments    : {len(ship):,} rows   {ship.ship_date.min()} → {ship.ship_date.max()}")
print(f"reviews      : {len(rv):,} rows   {rv.review_date.min()} → {rv.review_date.max()}")
print(f"returns      : {len(ret):,} rows   {ret.return_date.min()} → {ret.return_date.max()}")
print(f"products     : {len(prod):,} rows")
print(f"geography    : {len(geo):,} rows")
print(f"customers    : {len(cu):,} rows   {cu.signup_date.min()} → {cu.signup_date.max()}")
print(f"promotions   : {len(pr):,} rows   {pr.start_date.min()} → {pr.end_date.max()}")
print(f"inventory    : {len(inv):,} rows   {inv.snapshot_date.min()} → {inv.snapshot_date.max()}")
