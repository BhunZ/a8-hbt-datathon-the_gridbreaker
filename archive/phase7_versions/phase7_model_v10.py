"""Phase 7 v10 — enrich friend's 740k / our 781k submission with promo + stockout + named-event features.

Current: friend's feature set = calendar + traffic + payday + double_day + tet + holidays. Scored 781k.
Top of leaderboard: 658k. Gap to close: 123k (~16%).

v10 adds untapped master-dataset signals:
  • PROMO features: daily count of active promos + avg discount value (from promotions.csv).
  • STOCKOUT: daily sum of stockout_days across SKUs (from inventory.csv).
  • NEW-CUSTOMER signups: 30-day rolling mean (from customers.csv, leading indicator).
  • NAMED EVENTS: Singles Day 11/11, 12/12 Online, Black Friday (4th Fri of November),
    Cyber Monday, Mother's Day (2nd Sun of May) — all major VN e-commerce spikes.
  • Traffic YoY ratio vs (month,dow) average.

Models: keeps friend's RF+ET+HGB core (this was proven to produce the right level)
plus multi-seed RF+ET averaging to reduce variance.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"; OUT = os.path.join(BASE, "outputs")
SEED = 42
TRAIN_END=pd.Timestamp("2022-12-31"); TEST_START=pd.Timestamp("2023-01-01"); TEST_END=pd.Timestamp("2024-07-01")
VAL_START=pd.Timestamp("2021-07-01"); VAL_END=pd.Timestamp("2022-12-31"); TRAIN_START=pd.Timestamp("2017-01-01")
TET = [pd.Timestamp(d) for d in ["2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10"]]

# --- Named VN e-commerce event dates (return True if date matches) ---
def nth_weekday_of_month(year, month, weekday, n):
    """Return nth occurrence (1-based) of weekday (0=Mon) in given month."""
    d = pd.Timestamp(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + pd.Timedelta(days=shift + 7*(n-1))

def is_black_friday(date):
    """4th Friday of November — VN retailers adopt US Black Friday."""
    return date.month == 11 and date == nth_weekday_of_month(date.year, 11, 4, 4)

def is_mothers_day(date):
    """2nd Sunday of May."""
    return date.month == 5 and date == nth_weekday_of_month(date.year, 5, 6, 2)

def is_cyber_monday(date):
    """Monday after US Thanksgiving (4th Thursday of Nov + 4 days)."""
    thx = nth_weekday_of_month(date.year, 11, 3, 4)
    return date == thx + pd.Timedelta(days=4)

def is_singles_day(date):
    """11/11 — largest e-commerce day in Vietnam."""
    return date.month == 11 and date.day == 11

def is_twelve_twelve(date):
    """12/12 — major year-end VN online shopping day."""
    return date.month == 12 and date.day == 12

def is_nine_nine(date):
    """9/9 — rising VN Shopee sale day."""
    return date.month == 9 and date.day == 9

# --- Load all relevant data ---
sales   = pd.read_csv(f"{BASE}/sales.csv",         parse_dates=["Date"])
traffic = pd.read_csv(f"{BASE}/web_traffic.csv",   parse_dates=["date"])
sample  = pd.read_csv(f"{BASE}/sample_submission.csv", parse_dates=["Date"])
promos  = pd.read_csv(f"{BASE}/promotions.csv",    parse_dates=["start_date","end_date"])
inv     = pd.read_csv(f"{BASE}/inventory.csv",     parse_dates=["snapshot_date"])
cust    = pd.read_csv(f"{BASE}/customers.csv",     parse_dates=["signup_date"])

# --- Aggregate traffic ---
daily_t = traffic.groupby("date").agg(
    sessions=("sessions","sum"), page_views=("page_views","sum"),
    unique_visitors=("unique_visitors","sum"),
    bounce_rate=("bounce_rate","mean"),
    avg_sess=("avg_session_duration_sec","mean")).reset_index().rename(columns={"date":"Date"})
daily_t["month"]=daily_t.Date.dt.month; daily_t["dow"]=daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]].mean().reset_index()

# --- Aggregate promotions to daily ---
full_date_range = pd.date_range(pd.Timestamp("2012-01-01"), TEST_END, freq="D")
promo_daily = pd.DataFrame({"Date": full_date_range, "n_promo": 0.0, "avg_promo_discount": 0.0, "has_percentage_promo": 0, "has_fixed_promo": 0})
promo_daily = promo_daily.set_index("Date")
for _, r in promos.iterrows():
    mask = (promo_daily.index >= r.start_date) & (promo_daily.index <= r.end_date)
    promo_daily.loc[mask, "n_promo"] += 1
    promo_daily.loc[mask, "avg_promo_discount"] += r.discount_value
    if r.promo_type == "percentage":
        promo_daily.loc[mask, "has_percentage_promo"] = 1
    else:
        promo_daily.loc[mask, "has_fixed_promo"] = 1
promo_daily["avg_promo_discount"] = np.where(promo_daily["n_promo"]>0,
                                              promo_daily["avg_promo_discount"]/promo_daily["n_promo"].clip(1),
                                              0)
promo_daily = promo_daily.reset_index()
print(f"[promo] aggregated {len(promos)} promos, {promo_daily.n_promo.gt(0).sum()} days with active promo")

# --- Aggregate stockouts to daily ---
stockout_daily = inv.groupby("snapshot_date").agg(
    total_stockout_days=("stockout_days","sum"),
    n_skus_stockout=("stockout_days", lambda s: (s>0).sum()),
    mean_stock=("stock_on_hand","mean")
).reset_index().rename(columns={"snapshot_date":"Date"})
print(f"[stockout] {len(stockout_daily)} inventory snapshots")

# --- Aggregate signups (30d rolling mean as leading indicator) ---
signup_daily = cust.groupby("signup_date").size().reset_index(name="n_signups").rename(columns={"signup_date":"Date"})
# Reindex to full range, fill with 0, then 30-day rolling mean
signup_full = pd.DataFrame({"Date": full_date_range}).merge(signup_daily, on="Date", how="left").fillna(0)
signup_full["signups_30d"] = signup_full["n_signups"].rolling(30, min_periods=1).mean()
signup_full["signups_90d"] = signup_full["n_signups"].rolling(90, min_periods=1).mean()
print(f"[signup] {signup_full.n_signups.sum():.0f} total signups")

def build_features(df, is_train_traffic):
    df = df.copy()
    df["year"]=df.Date.dt.year; df["month"]=df.Date.dt.month; df["day"]=df.Date.dt.day
    df["dow"]=df.Date.dt.dayofweek; df["doy"]=df.Date.dt.dayofyear
    if is_train_traffic:
        df = df.merge(daily_t[["Date","sessions","page_views","unique_visitors","bounce_rate","avg_sess"]], on="Date", how="left")
    else:
        df = df.merge(traffic_avg, on=["month","dow"], how="left")
    for c in ["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]:
        df[c] = df[c].fillna(0)
    # Merge promos, stockouts, signups
    df = df.merge(promo_daily,       on="Date", how="left").fillna({"n_promo":0,"avg_promo_discount":0,"has_percentage_promo":0,"has_fixed_promo":0})
    df = df.merge(stockout_daily,    on="Date", how="left").fillna({"total_stockout_days":0,"n_skus_stockout":0,"mean_stock":0})
    df = df.merge(signup_full[["Date","signups_30d","signups_90d"]], on="Date", how="left").fillna({"signups_30d":0,"signups_90d":0})

    df["month_sin"]=np.sin(2*np.pi*df.month/12); df["month_cos"]=np.cos(2*np.pi*df.month/12)
    df["dow_sin"]=np.sin(2*np.pi*df.dow/7);      df["dow_cos"]=np.cos(2*np.pi*df.dow/7)
    df["is_payday"]=((df.day>=25)|(df.day<=5)).astype(int)
    df["is_double_day"]=(df.month==df.day).astype(int)
    # Tet window 21-day pre
    df["is_tet_season"] = 0
    for td in TET:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m, "is_tet_season"] = 1
    # Generic VN holidays
    df["is_holiday_period"] = 0
    for mh,dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            df.loc[(df.month==mh)&(df.day==(dh-off)), "is_holiday_period"] = 1
    # Named events (major VN e-commerce spikes)
    df["is_singles_day"]   = df.Date.apply(is_singles_day).astype(int)
    df["is_nine_nine"]     = df.Date.apply(is_nine_nine).astype(int)
    df["is_twelve_twelve"] = df.Date.apply(is_twelve_twelve).astype(int)
    df["is_black_friday"]  = df.Date.apply(is_black_friday).astype(int)
    df["is_cyber_monday"]  = df.Date.apply(is_cyber_monday).astype(int)
    df["is_mothers_day"]   = df.Date.apply(is_mothers_day).astype(int)
    # Event-adjacent window (±3 days around big events)
    df["near_flash_event"] = 0
    for flag in ["is_singles_day","is_twelve_twelve","is_black_friday","is_nine_nine","is_cyber_monday"]:
        ev_dates = df.loc[df[flag]==1, "Date"].tolist()
        for ed in ev_dates:
            m = (df.Date >= ed - pd.Timedelta(days=3)) & (df.Date <= ed + pd.Timedelta(days=3))
            df.loc[m, "near_flash_event"] = 1
    return df

FEATS = ['month_sin','month_cos','dow_sin','dow_cos','day','month','dow',
         'is_payday','is_double_day','is_tet_season','is_holiday_period',
         'is_singles_day','is_nine_nine','is_twelve_twelve',
         'is_black_friday','is_cyber_monday','is_mothers_day','near_flash_event',
         'sessions','page_views','unique_visitors','bounce_rate','avg_sess',
         'n_promo','avg_promo_discount','has_percentage_promo','has_fixed_promo',
         'total_stockout_days','n_skus_stockout','mean_stock',
         'signups_30d','signups_90d']
print(f"[features] {len(FEATS)} total")

tr_all = sales[sales.Date >= TRAIN_START].reset_index(drop=True)
tr_f   = build_features(tr_all, is_train_traffic=True)
trn_mask = (tr_f.Date < VAL_START)
va_f = build_features(tr_all[(tr_all.Date >= VAL_START) & (tr_all.Date <= VAL_END)], is_train_traffic=False)
print(f"[val] train={trn_mask.sum()} val={len(va_f)}")

def swt(years):
    return np.clip((years - 2016).astype(float), 1.0, None) ** 1.2

def run_val(tgt):
    Xt = tr_f.loc[trn_mask, FEATS]; yt = np.log1p(tr_f.loc[trn_mask, tgt].values)
    Xv = va_f[FEATS];               yvr = va_f[tgt].values
    w = swt(tr_f.loc[trn_mask, "year"].values)
    print(f"  [{tgt}] RF...", flush=True)
    rf = RandomForestRegressor(n_estimators=400, max_depth=15, random_state=SEED, n_jobs=-1); rf.fit(Xt, yt, sample_weight=w)
    print(f"  [{tgt}] ET...", flush=True)
    et = ExtraTreesRegressor(n_estimators=500, max_depth=16, random_state=SEED, n_jobs=-1); et.fit(Xt, yt, sample_weight=w)
    print(f"  [{tgt}] HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(Xt, yt, sample_weight=w)
    p_rf = np.expm1(rf.predict(Xv)); p_et = np.expm1(et.predict(Xv)); p_h = np.expm1(hgb.predict(Xv))
    p_blend = 0.3*p_rf + 0.5*p_et + 0.2*p_h
    for n,p in [("rf",p_rf),("et",p_et),("hgb",p_h),("blend",p_blend)]:
        rmse = np.sqrt(mean_squared_error(yvr, p))
        print(f"   {n:6s} RMSE={rmse:>10,.0f} mean={p.mean()/1e6:.2f}M actual={yvr.mean()/1e6:.2f}M")

print("\n=== VAL Revenue ==="); run_val("Revenue")
print("\n=== VAL COGS ==="   ); run_val("COGS")

# FINAL fit: MULTI-SEED RF/ET for variance reduction
def fit_final_multi_seed(tgt, n_seeds=3):
    X = tr_f[FEATS]; y = np.log1p(tr_f[tgt].values); w = swt(tr_f["year"].values)
    rfs, ets = [], []
    for s in [42, 17, 91, 2024, 777][:n_seeds]:
        print(f"  [{tgt}] RF seed={s}...", flush=True)
        rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=s, n_jobs=-1); rf.fit(X, y, sample_weight=w)
        rfs.append(rf)
        print(f"  [{tgt}] ET seed={s}...", flush=True)
        et = ExtraTreesRegressor(n_estimators=600, max_depth=16, random_state=s, n_jobs=-1); et.fit(X, y, sample_weight=w)
        ets.append(et)
    print(f"  [{tgt}] HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(X, y, sample_weight=w)
    return rfs, ets, hgb

print("\n[final Revenue]"); rev_rfs, rev_ets, rev_hgb = fit_final_multi_seed("Revenue", n_seeds=3)
print("[final COGS]");      cogs_rfs, cogs_ets, cogs_hgb = fit_final_multi_seed("COGS", n_seeds=3)

test_f = build_features(sample[["Date"]], is_train_traffic=False)
Xt_test = test_f[FEATS]

def avg_predict(models, X):
    return np.mean([np.expm1(m.predict(X)) for m in models], axis=0)

p_rf_r  = avg_predict(rev_rfs, Xt_test)
p_et_r  = avg_predict(rev_ets, Xt_test)
p_hgb_r = np.expm1(rev_hgb.predict(Xt_test))

p_rf_c  = avg_predict(cogs_rfs, Xt_test)
p_et_c  = avg_predict(cogs_ets, Xt_test)
p_hgb_c = np.expm1(cogs_hgb.predict(Xt_test))

# Friend's blend weights [0.3, 0.5, 0.2] (preserved — what scored 740-781k)
p_rev  = 0.3*p_rf_r + 0.5*p_et_r + 0.2*p_hgb_r
p_cogs = 0.3*p_rf_c + 0.5*p_et_c + 0.2*p_hgb_c

sub = pd.DataFrame({"Date": sample.Date.dt.strftime("%Y-%m-%d"),
                    "Revenue": np.round(p_rev, 2),
                    "COGS":    np.round(p_cogs, 2)})
V10_OUT = os.path.join(OUT, "submission_v10.csv")
sub.to_csv(V10_OUT, index=False)
print(f"\n[v10] -> {V10_OUT}")
print(f"  mean Rev={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum  Rev={sub.Revenue.sum()/1e9:.3f}B  sum  COGS={sub.COGS.sum()/1e9:.3f}B")

# Inspect flash-event days specifically (they should stand out)
test_dates = pd.to_datetime(sub.Date)
for evt_fn, name in [(is_singles_day,"11/11"), (is_twelve_twelve,"12/12"), (is_black_friday,"Black Fri")]:
    mask = test_dates.apply(evt_fn)
    if mask.any():
        idx = mask[mask].index.tolist()
        print(f"  {name}: test days = {list(test_dates[idx])}  preds = {[round(p_rev[i]/1e6,2) for i in idx]}M")
print("=== DONE ===")
