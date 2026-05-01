"""Validate friend's weighted_god_mode.py on our 2021-07-01 -> 2022-12-31 fold."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"
VAL_START = pd.Timestamp("2021-07-01"); VAL_END = pd.Timestamp("2022-12-31")

sales   = pd.read_csv(f"{BASE}/sales.csv",         parse_dates=["Date"])
traffic = pd.read_csv(f"{BASE}/web_traffic.csv",   parse_dates=["date"])

daily_t = traffic.groupby("date")[["sessions","page_views"]].sum().reset_index().rename(columns={"date":"Date"})
daily_t["month"] = daily_t.Date.dt.month; daily_t["dow"] = daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","page_views"]].mean().reset_index()

tet_dates = pd.to_datetime(['2017-01-28','2018-02-16','2019-02-05','2020-01-25','2021-02-12','2022-02-01','2023-01-22','2024-02-10'])

def feats(df, is_train):
    df = df.copy()
    df["year"]=df.Date.dt.year; df["month"]=df.Date.dt.month; df["day"]=df.Date.dt.day; df["dow"]=df.Date.dt.dayofweek
    if is_train:
        df = df.merge(daily_t[["Date","sessions","page_views"]], on="Date", how="left").fillna(0)
    else:
        df = df.merge(traffic_avg, on=["month","dow"], how="left").fillna(0)
    df["is_payday"] = ((df.day>=25)|(df.day<=5)).astype(int)
    df["is_double_day"] = (df.month==df.day).astype(int)
    df["is_tet_season"] = 0
    for td in tet_dates:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m, "is_tet_season"] = 1
    df["is_holiday_period"] = 0
    for mh,dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            df.loc[(df.month==mh)&(df.day==(dh-off)), "is_holiday_period"] = 1
    df["month_sin"]=np.sin(2*np.pi*df.month/12); df["month_cos"]=np.cos(2*np.pi*df.month/12)
    df["dow_sin"]=np.sin(2*np.pi*df.dow/7); df["dow_cos"]=np.cos(2*np.pi*df.dow/7)
    return df

# Train 2017 .. 2021-06-30, val 2021-07-01 .. 2022-12-31
tr = sales[(sales.Date.dt.year>=2017) & (sales.Date < VAL_START)].reset_index(drop=True)
va = sales[(sales.Date>=VAL_START) & (sales.Date<=VAL_END)].reset_index(drop=True)
tr_f = feats(tr, is_train=True)
va_f = feats(va, is_train=False)  # use (month,dow) avg traffic as in friend's test path

features = ['month_sin','month_cos','dow_sin','dow_cos','day','is_payday','is_double_day','is_tet_season','sessions','page_views','is_holiday_period']
y_rev = np.log1p(tr_f['Revenue']); y_cogs = np.log1p(tr_f['COGS'])
w = (tr_f['year'] - 2016).clip(lower=1) ** 1.2
print(f"[val] tr={len(tr_f)} va={len(va_f)}  weight range: {w.min():.2f}-{w.max():.2f}")

def fit_blend(y, label):
    rf  = RandomForestRegressor(n_estimators=400, max_depth=15, random_state=42, n_jobs=-1)
    et  = ExtraTreesRegressor(n_estimators=500, max_depth=16, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42)
    print(f"  [{label}] fit RF...", flush=True); rf.fit(tr_f[features], y, sample_weight=w)
    print(f"  [{label}] fit ET...", flush=True); et.fit(tr_f[features], y, sample_weight=w)
    print(f"  [{label}] fit HGB...", flush=True); hgb.fit(tr_f[features], y, sample_weight=w)
    p_rf  = np.expm1(rf.predict(va_f[features]))
    p_et  = np.expm1(et.predict(va_f[features]))
    p_hgb = np.expm1(hgb.predict(va_f[features]))
    p_blend = 0.3*p_rf + 0.5*p_et + 0.2*p_hgb
    actual = va_f[label].values
    print(f"\n  --- {label} val RMSE ---")
    for n, p in [("rf",p_rf),("et",p_et),("hgb",p_hgb),("blend_3_5_2",p_blend)]:
        rmse = np.sqrt(mean_squared_error(actual, p))
        mae  = mean_absolute_error(actual, p)
        print(f"   {n:12s} RMSE={rmse:>10,.0f} MAE={mae:>9,.0f} p.mean={p.mean()/1e6:.2f}M actual.mean={actual.mean()/1e6:.2f}M")
    return p_blend

p_rev_pred  = fit_blend(y_rev,  "Revenue")
p_cogs_pred = fit_blend(y_cogs, "COGS")
print("\nConcatenated RMSE (Revenue+COGS):", np.sqrt(np.mean(np.concatenate([
    (p_rev_pred - va_f['Revenue'].values)**2,
    (p_cogs_pred - va_f['COGS'].values)**2]))) )
print("DONE")
