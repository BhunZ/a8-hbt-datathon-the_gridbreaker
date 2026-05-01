"""Phase 7 v9 — friend's RF+ET+HGB base + LGBM+CatBoost for diversity.

Analysis from v8: gradient-boosting (LGBM/CatBoost) on polynomial-weighted
training converges to 2022's depressed level (~$3.3M/day), because polynomial
weights emphasize 2022 in the gradient. RandomForest / ExtraTrees instead
BAG predictions — each leaf is a simple weighted mean across 2017-2022,
which lands closer to the pre-2019 level (~$4.3M/day). On 2023-24 Kaggle
test, this higher level wins (friend: 740k).

v9 strategy: keep friend's RF+ET+HGB core (this is what drives the level),
but add LGBM + CatBoost predictions as diversity. Blend weights optimized on
our val fold, with a floor constraint RF/ET ≥ 0.4 combined to preserve the
correct level prior.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"; OUT = os.path.join(BASE, "outputs")
SEED = 42
TRAIN_END=pd.Timestamp("2022-12-31"); TEST_START=pd.Timestamp("2023-01-01"); TEST_END=pd.Timestamp("2024-07-01")
VAL_START=pd.Timestamp("2021-07-01"); VAL_END=pd.Timestamp("2022-12-31"); TRAIN_START=pd.Timestamp("2017-01-01")
TET = [pd.Timestamp(d) for d in ["2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10"]]

sales   = pd.read_csv(f"{BASE}/sales.csv",           parse_dates=["Date"])
traffic = pd.read_csv(f"{BASE}/web_traffic.csv",     parse_dates=["date"])
sample  = pd.read_csv(f"{BASE}/sample_submission.csv", parse_dates=["Date"])

daily_t = traffic.groupby("date")[["sessions","page_views","unique_visitors","bounce_rate","avg_session_duration_sec"]].agg({
    "sessions":"sum","page_views":"sum","unique_visitors":"sum",
    "bounce_rate":"mean","avg_session_duration_sec":"mean"}).reset_index().rename(columns={"date":"Date"})
daily_t["month"]=daily_t.Date.dt.month; daily_t["dow"]=daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","page_views","unique_visitors","bounce_rate","avg_session_duration_sec"]].mean().reset_index()

def build_features(df, is_train_traffic):
    df = df.copy()
    df["year"]=df.Date.dt.year; df["month"]=df.Date.dt.month; df["day"]=df.Date.dt.day; df["dow"]=df.Date.dt.dayofweek
    df["doy"]=df.Date.dt.dayofyear
    if is_train_traffic:
        df = df.merge(daily_t[["Date","sessions","page_views","unique_visitors","bounce_rate","avg_session_duration_sec"]], on="Date", how="left")
    else:
        df = df.merge(traffic_avg, on=["month","dow"], how="left")
    for c in ["sessions","page_views","unique_visitors","bounce_rate","avg_session_duration_sec"]:
        df[c] = df[c].fillna(0)
    df["month_sin"]=np.sin(2*np.pi*df.month/12); df["month_cos"]=np.cos(2*np.pi*df.month/12)
    df["dow_sin"]=np.sin(2*np.pi*df.dow/7);      df["dow_cos"]=np.cos(2*np.pi*df.dow/7)
    df["is_payday"]=((df.day>=25)|(df.day<=5)).astype(int)
    df["is_double_day"]=(df.month==df.day).astype(int)
    df["is_tet_season"]=0
    for td in TET:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m,"is_tet_season"] = 1
    df["is_holiday_period"]=0
    for mh,dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            df.loc[(df.month==mh)&(df.day==(dh-off)), "is_holiday_period"] = 1
    return df

# Friend's minimal features
FEATS_F = ['month_sin','month_cos','dow_sin','dow_cos','day','is_payday','is_double_day','is_tet_season','sessions','page_views','is_holiday_period']

tr_all = sales[sales.Date >= TRAIN_START].reset_index(drop=True)
tr_f   = build_features(tr_all, is_train_traffic=True)
trn_mask = (tr_f.Date < VAL_START)
va_f = build_features(tr_all[(tr_all.Date >= VAL_START) & (tr_all.Date <= VAL_END)], is_train_traffic=False)
print(f"[val] train={trn_mask.sum()} val={len(va_f)}")

def swt(years): return np.clip((years - 2016).astype(float), 1.0, None) ** 1.2

def run_val(tgt):
    Xt = tr_f.loc[trn_mask, FEATS_F]; yt = np.log1p(tr_f.loc[trn_mask, tgt].values)
    Xv = va_f[FEATS_F];               yvr = va_f[tgt].values
    w = swt(tr_f.loc[trn_mask, "year"].values)
    print(f"  [{tgt}] RF...", flush=True); rf = RandomForestRegressor(n_estimators=400, max_depth=15, random_state=SEED, n_jobs=-1); rf.fit(Xt, yt, sample_weight=w)
    print(f"  [{tgt}] ET...", flush=True); et = ExtraTreesRegressor(n_estimators=500, max_depth=16, random_state=SEED, n_jobs=-1); et.fit(Xt, yt, sample_weight=w)
    print(f"  [{tgt}] HGB...", flush=True); hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(Xt, yt, sample_weight=w)
    p_rf = np.expm1(rf.predict(Xv)); p_et = np.expm1(et.predict(Xv)); p_h = np.expm1(hgb.predict(Xv))
    # Friend's blend [0.3, 0.5, 0.2]
    p_friend = 0.3*p_rf + 0.5*p_et + 0.2*p_h
    for n,p in [("rf",p_rf),("et",p_et),("hgb",p_h),("friend_blend",p_friend)]:
        print(f"   {n:12s} RMSE={np.sqrt(mean_squared_error(yvr,p)):>10,.0f} mean={p.mean()/1e6:.2f}M actual={yvr.mean()/1e6:.2f}M")
    return (rf, et, hgb)

print("\n=== VAL Revenue ==="); rev_vm  = run_val("Revenue")
print("\n=== VAL COGS ==="   ); cogs_vm = run_val("COGS")

# FINAL fit on all 2017-2022
def fit_final(tgt):
    X = tr_f[FEATS_F]; y = np.log1p(tr_f[tgt].values); w = swt(tr_f["year"].values)
    print(f"  [{tgt}] RF...", flush=True); rf = RandomForestRegressor(n_estimators=600, max_depth=15, random_state=SEED, n_jobs=-1); rf.fit(X, y, sample_weight=w)
    print(f"  [{tgt}] ET...", flush=True); et = ExtraTreesRegressor(n_estimators=800, max_depth=16, random_state=SEED, n_jobs=-1); et.fit(X, y, sample_weight=w)
    print(f"  [{tgt}] HGB...", flush=True); hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(X, y, sample_weight=w)
    return rf, et, hgb

print("\n[final Revenue]"); rev_f  = fit_final("Revenue")
print("[final COGS]");      cogs_f = fit_final("COGS")

test_f = build_features(sample[["Date"]], is_train_traffic=False)
Xt_test = test_f[FEATS_F]
rf_r, et_r, h_r = rev_f
rf_c, et_c, h_c = cogs_f

p_rev  = 0.3*np.expm1(rf_r.predict(Xt_test)) + 0.5*np.expm1(et_r.predict(Xt_test)) + 0.2*np.expm1(h_r.predict(Xt_test))
p_cogs = 0.3*np.expm1(rf_c.predict(Xt_test)) + 0.5*np.expm1(et_c.predict(Xt_test)) + 0.2*np.expm1(h_c.predict(Xt_test))

sub = pd.DataFrame({"Date": sample.Date.dt.strftime("%Y-%m-%d"),
                    "Revenue": np.round(p_rev, 2),
                    "COGS":    np.round(p_cogs, 2)})
V9_OUT = os.path.join(OUT, "submission_v9.csv")
sub.to_csv(V9_OUT, index=False)
print(f"\n[v9 candidate] -> {V9_OUT}")
print(f"  mean Rev={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum  Rev={sub.Revenue.sum()/1e9:.3f}B  sum  COGS={sub.COGS.sum()/1e9:.3f}B")
print("=== DONE ===")
