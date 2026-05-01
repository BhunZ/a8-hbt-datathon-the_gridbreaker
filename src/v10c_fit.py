"""v10c — final submission: BASE + named events + extended traffic.
Keeps friend's 781k level (~$4.3M/day mean), adds event calendar signal and
extra traffic channels, with multi-seed RF/ET for variance reduction.

Written to fit/predict inside 45s. Two seeds, 400 trees each.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW, REFERENCE, SUBMISSIONS
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

t0 = time.time()
TRAIN_START = pd.Timestamp("2017-01-01")
TET = [pd.Timestamp(d) for d in ["2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10"]]

def nth_weekday_of_month(y,m,wd,n):
    d = pd.Timestamp(y,m,1); shift=(wd-d.weekday())%7
    return d + pd.Timedelta(days=shift+7*(n-1))

sales   = pd.read_csv(RAW / "sales.csv",                  parse_dates=["Date"])
traffic = pd.read_csv(RAW / "web_traffic.csv",            parse_dates=["date"])
sample  = pd.read_csv(REFERENCE / "sample_submission.csv", parse_dates=["Date"])

daily_t = traffic.groupby("date").agg(
    sessions=("sessions","sum"), page_views=("page_views","sum"),
    unique_visitors=("unique_visitors","sum"),
    bounce_rate=("bounce_rate","mean"),
    avg_sess=("avg_session_duration_sec","mean")).reset_index().rename(columns={"date":"Date"})
daily_t["month"]=daily_t.Date.dt.month; daily_t["dow"]=daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]].mean().reset_index()

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

FEATS = ['month_sin','month_cos','dow_sin','dow_cos','day','month','dow',
         'is_payday','is_double_day','is_tet_season','is_holiday_period',
         'is_singles_day','is_nine_nine','is_twelve_twelve','is_black_friday','is_cyber_monday','is_mothers_day','near_flash_event',
         'sessions','page_views','unique_visitors','bounce_rate','avg_sess']
print(f"[{time.time()-t0:.1f}s] {len(FEATS)} features")

tr_all = sales[sales.Date >= TRAIN_START].reset_index(drop=True)
tr_f = build_features(tr_all, is_train_traffic=True)
test_f = build_features(sample[["Date"]], is_train_traffic=False)
print(f"[{time.time()-t0:.1f}s] train={len(tr_f)} test={len(test_f)}")

def swt(years): return np.clip((years - 2016).astype(float), 1.0, None) ** 1.2

X = tr_f[FEATS].values; w = swt(tr_f["year"].values)
Xt = test_f[FEATS].values

# Two seeds, moderate tree count; HGB single-seed
SEEDS = [42, 17]
preds = {"Revenue": [], "COGS": []}

for tgt in ["Revenue","COGS"]:
    y = np.log1p(tr_f[tgt].values)
    for s in SEEDS:
        print(f"  [{tgt}] RF s={s}...", flush=True)
        rf = RandomForestRegressor(n_estimators=350, max_depth=15, random_state=s, n_jobs=-1); rf.fit(X, y, sample_weight=w)
        preds[tgt].append(("rf", 0.3/len(SEEDS), np.expm1(rf.predict(Xt))))
        print(f"  [{tgt}] ET s={s}...", flush=True)
        et = ExtraTreesRegressor(n_estimators=450, max_depth=16, random_state=s, n_jobs=-1); et.fit(X, y, sample_weight=w)
        preds[tgt].append(("et", 0.5/len(SEEDS), np.expm1(et.predict(Xt))))
    print(f"  [{tgt}] HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42); hgb.fit(X, y, sample_weight=w)
    preds[tgt].append(("hgb", 0.2, np.expm1(hgb.predict(Xt))))
    print(f"  [{tgt}] done at {time.time()-t0:.1f}s")

p_rev  = sum(wt*p for _,wt,p in preds["Revenue"])
p_cogs = sum(wt*p for _,wt,p in preds["COGS"])

sub = pd.DataFrame({"Date": sample.Date.dt.strftime("%Y-%m-%d"),
                    "Revenue": np.round(p_rev, 2),
                    "COGS":    np.round(p_cogs, 2)})
V10C_OUT = SUBMISSIONS / "submission_v10c.csv"
sub.to_csv(V10C_OUT, index=False)
print(f"\n[v10c] -> {V10C_OUT}")
print(f"  mean  Rev={sub.Revenue.mean()/1e6:.2f}M  COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum   Rev={sub.Revenue.sum()/1e9:.3f}B  COGS={sub.COGS.sum()/1e9:.3f}B")

# Print named event days
test_dates = pd.to_datetime(sub.Date)
for evt_col in ["is_singles_day","is_twelve_twelve","is_black_friday","is_nine_nine","is_cyber_monday","is_mothers_day"]:
    mask = test_f[evt_col]==1
    if mask.any():
        for i in np.where(mask.values)[0]:
            print(f"  {evt_col}: {test_dates.iloc[i].date()}  rev={p_rev[i]/1e6:.2f}M cogs={p_cogs[i]/1e6:.2f}M")
print(f"=== DONE in {time.time()-t0:.1f}s ===")
