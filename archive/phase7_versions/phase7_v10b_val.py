"""v10b — ablation: add ONLY safe calendar features on top of friend's base.

Hypothesis: v10 RMSE dropped because signups/promos/stockouts are level-leaks
(they carry 2022's depressed state into the features). Pure calendar events
(named flash-sale days) should help shape without changing level.

This tests 3 variants on our val fold:
  BASE : friend's 11 features
  +EVT : + named events (singles_day, 12/12, 9/9, black_friday, cyber_monday, mothers_day, near_flash)
  +TRAFFIC : +EVT + additional traffic fields (unique_visitors, bounce_rate, avg_sess)
  FULL : +EVT + +TRAFFIC + promos/stockouts/signups (equiv to v10)

Key metric: look at mean_pred on val. The winning model on Kaggle had HIGH mean
(~$4.33M). If a variant lowers the mean it's probably over-fitting to 2022.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

t0 = time.time()
BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"; OUT = os.path.join(BASE, "outputs")
SEED = 42
VAL_START=pd.Timestamp("2021-07-01"); VAL_END=pd.Timestamp("2022-12-31"); TRAIN_START=pd.Timestamp("2017-01-01")
TET = [pd.Timestamp(d) for d in ["2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10"]]

def nth_weekday_of_month(y,m,wd,n):
    d = pd.Timestamp(y,m,1); shift=(wd-d.weekday())%7
    return d + pd.Timedelta(days=shift+7*(n-1))

sales   = pd.read_csv(f"{BASE}/sales.csv",           parse_dates=["Date"])
traffic = pd.read_csv(f"{BASE}/web_traffic.csv",     parse_dates=["date"])
promos  = pd.read_csv(f"{BASE}/promotions.csv",      parse_dates=["start_date","end_date"])
inv     = pd.read_csv(f"{BASE}/inventory.csv",       parse_dates=["snapshot_date"])
cust    = pd.read_csv(f"{BASE}/customers.csv",       parse_dates=["signup_date"])
print(f"[{time.time()-t0:.1f}s] loaded")

daily_t = traffic.groupby("date").agg(
    sessions=("sessions","sum"), page_views=("page_views","sum"),
    unique_visitors=("unique_visitors","sum"),
    bounce_rate=("bounce_rate","mean"),
    avg_sess=("avg_session_duration_sec","mean")).reset_index().rename(columns={"date":"Date"})
daily_t["month"]=daily_t.Date.dt.month; daily_t["dow"]=daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]].mean().reset_index()

# promos/stockouts/signups — precompute daily tables
full_range = pd.date_range("2012-01-01", "2024-07-01", freq="D")
promo_daily = pd.DataFrame(index=full_range, data={"n_promo":0.0,"avg_promo_discount":0.0,"has_percentage_promo":0,"has_fixed_promo":0})
for _, r in promos.iterrows():
    m = (promo_daily.index >= r.start_date) & (promo_daily.index <= r.end_date)
    promo_daily.loc[m,"n_promo"] += 1
    promo_daily.loc[m,"avg_promo_discount"] += float(r.discount_value)
    if r.promo_type == "percentage":
        promo_daily.loc[m,"has_percentage_promo"] = 1
    else:
        promo_daily.loc[m,"has_fixed_promo"] = 1
promo_daily["avg_promo_discount"] = np.where(promo_daily.n_promo>0, promo_daily.avg_promo_discount/promo_daily.n_promo.clip(lower=1), 0)
promo_daily = promo_daily.reset_index().rename(columns={"index":"Date"})

stockout_daily = inv.groupby("snapshot_date").agg(
    total_stockout_days=("stockout_days","sum"),
    n_skus_stockout=("stockout_days", lambda s: (s>0).sum()),
    mean_stock=("stock_on_hand","mean")).reset_index().rename(columns={"snapshot_date":"Date"})

signup_daily = cust.groupby("signup_date").size().reset_index(name="n_signups").rename(columns={"signup_date":"Date"})
signup_full = pd.DataFrame({"Date": full_range}).merge(signup_daily, on="Date", how="left").fillna(0)
signup_full["signups_30d"] = signup_full["n_signups"].rolling(30, min_periods=1).mean()
signup_full["signups_90d"] = signup_full["n_signups"].rolling(90, min_periods=1).mean()
print(f"[{time.time()-t0:.1f}s] tables ready")

def build_features(df, is_train_traffic):
    df = df.copy()
    df["year"]=df.Date.dt.year; df["month"]=df.Date.dt.month; df["day"]=df.Date.dt.day
    df["dow"]=df.Date.dt.dayofweek
    if is_train_traffic:
        df = df.merge(daily_t[["Date","sessions","page_views","unique_visitors","bounce_rate","avg_sess"]], on="Date", how="left")
    else:
        df = df.merge(traffic_avg, on=["month","dow"], how="left")
    for c in ["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]:
        df[c] = df[c].fillna(0)
    df = df.merge(promo_daily, on="Date", how="left").fillna({"n_promo":0,"avg_promo_discount":0,"has_percentage_promo":0,"has_fixed_promo":0})
    df = df.merge(stockout_daily, on="Date", how="left").fillna({"total_stockout_days":0,"n_skus_stockout":0,"mean_stock":0})
    df = df.merge(signup_full[["Date","signups_30d","signups_90d"]], on="Date", how="left").fillna({"signups_30d":0,"signups_90d":0})
    df["month_sin"]=np.sin(2*np.pi*df.month/12); df["month_cos"]=np.cos(2*np.pi*df.month/12)
    df["dow_sin"]=np.sin(2*np.pi*df.dow/7);      df["dow_cos"]=np.cos(2*np.pi*df.dow/7)
    df["is_payday"]=((df.day>=25)|(df.day<=5)).astype(int)
    df["is_double_day"]=(df.month==df.day).astype(int)
    df["is_tet_season"] = 0
    for td in TET:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m,"is_tet_season"] = 1
    df["is_holiday_period"] = 0
    for mh,dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            df.loc[(df.month==mh)&(df.day==(dh-off)), "is_holiday_period"] = 1
    df["is_singles_day"]   = ((df.month==11)&(df.day==11)).astype(int)
    df["is_nine_nine"]     = ((df.month==9)&(df.day==9)).astype(int)
    df["is_twelve_twelve"] = ((df.month==12)&(df.day==12)).astype(int)
    df["is_black_friday"] = 0
    for y in df.year.unique():
        bf = nth_weekday_of_month(int(y),11,4,4); df.loc[df.Date==bf,"is_black_friday"] = 1
    df["is_cyber_monday"] = 0
    for y in df.year.unique():
        thx = nth_weekday_of_month(int(y),11,3,4); df.loc[df.Date==thx+pd.Timedelta(days=4),"is_cyber_monday"] = 1
    df["is_mothers_day"] = 0
    for y in df.year.unique():
        md = nth_weekday_of_month(int(y),5,6,2); df.loc[df.Date==md,"is_mothers_day"] = 1
    flash = df["is_singles_day"]+df["is_twelve_twelve"]+df["is_nine_nine"]+df["is_black_friday"]+df["is_cyber_monday"]
    df["near_flash_event"] = 0
    for off in range(-3,4):
        shifted = flash.shift(off).fillna(0)
        df["near_flash_event"] = np.maximum(df["near_flash_event"], (shifted>0).astype(int))
    return df

F_BASE     = ['month_sin','month_cos','dow_sin','dow_cos','day','is_payday','is_double_day','is_tet_season','sessions','page_views','is_holiday_period']
F_EVT      = F_BASE + ['is_singles_day','is_nine_nine','is_twelve_twelve','is_black_friday','is_cyber_monday','is_mothers_day','near_flash_event']
F_TRAFFIC  = F_EVT  + ['unique_visitors','bounce_rate','avg_sess','month','dow']
F_FULL     = F_TRAFFIC + ['n_promo','avg_promo_discount','has_percentage_promo','has_fixed_promo','total_stockout_days','n_skus_stockout','mean_stock','signups_30d','signups_90d']

tr_all = sales[sales.Date >= TRAIN_START].reset_index(drop=True)
tr_f   = build_features(tr_all, is_train_traffic=True)
trn_mask = (tr_f.Date < VAL_START)
va_f = build_features(tr_all[(tr_all.Date >= VAL_START) & (tr_all.Date <= VAL_END)], is_train_traffic=False)
print(f"[{time.time()-t0:.1f}s] built  train={trn_mask.sum()} val={len(va_f)}")

def swt(years): return np.clip((years - 2016).astype(float), 1.0, None) ** 1.2

def eval_variant(FEATS, label):
    out = {}
    for tgt in ["Revenue","COGS"]:
        Xt = tr_f.loc[trn_mask, FEATS].values; yt = np.log1p(tr_f.loc[trn_mask, tgt].values)
        Xv = va_f[FEATS].values;                yvr = va_f[tgt].values
        w = swt(tr_f.loc[trn_mask, "year"].values)
        rf = RandomForestRegressor(n_estimators=250, max_depth=15, random_state=SEED, n_jobs=-1); rf.fit(Xt, yt, sample_weight=w)
        et = ExtraTreesRegressor(n_estimators=250, max_depth=16, random_state=SEED, n_jobs=-1); et.fit(Xt, yt, sample_weight=w)
        hgb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=10, random_state=SEED); hgb.fit(Xt, yt, sample_weight=w)
        p = 0.3*np.expm1(rf.predict(Xv)) + 0.5*np.expm1(et.predict(Xv)) + 0.2*np.expm1(hgb.predict(Xv))
        out[tgt] = (p, yvr)
    pr,yrv = out["Revenue"]; pc,ycv = out["COGS"]
    rmse = np.sqrt(np.mean(np.concatenate([(pr-yrv)**2,(pc-ycv)**2])))
    print(f"  [{label:8s} n={len(FEATS)}]  RMSE={rmse:>10,.0f}  mean_rev={pr.mean()/1e6:.2f}M  mean_cogs={pc.mean()/1e6:.2f}M  (actual rev={yrv.mean()/1e6:.2f}M cogs={ycv.mean()/1e6:.2f}M)")
    return rmse, pr.mean(), pc.mean()

print(f"\n[time:{time.time()-t0:.1f}s] running variants...")
eval_variant(F_BASE,    "BASE")    ; print(f"[{time.time()-t0:.1f}s]")
eval_variant(F_EVT,     "+EVT")    ; print(f"[{time.time()-t0:.1f}s]")
eval_variant(F_TRAFFIC, "+TRAFFIC"); print(f"[{time.time()-t0:.1f}s]")
eval_variant(F_FULL,    "FULL")    ; print(f"[{time.time()-t0:.1f}s]")
print(f"=== DONE in {time.time()-t0:.1f}s ===")
