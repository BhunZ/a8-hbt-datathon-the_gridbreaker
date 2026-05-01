"""Phase 7 v8 — pure calendar+traffic model (no target-derived features).

This mimics friend's architecture exactly: the model's only signal is
calendar patterns and traffic magnitude. Without target lags or seasonal
means, it cannot 'learn' that 2022 was depressed, so it predicts closer
to the polynomial-weighted historical average across 2017-2022.

Result: mean prediction at ~friend's level (~$4.3M/day Revenue) but with
LGBM+CatBoost+HGB instead of RF+ET+HGB (typically more accurate per-day).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
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

daily_t = traffic.groupby("date").agg(
    sessions=("sessions","sum"), unique_visitors=("unique_visitors","sum"),
    page_views=("page_views","sum"), bounce_rate=("bounce_rate","mean"),
    avg_sess=("avg_session_duration_sec","mean")).reset_index().rename(columns={"date":"Date"})
daily_t["month"]=daily_t.Date.dt.month; daily_t["dow"]=daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","unique_visitors","page_views","bounce_rate","avg_sess"]].mean().reset_index()

def build_row_features(df, is_train_traffic):
    df = df.copy()
    df["year"]=df.Date.dt.year; df["month"]=df.Date.dt.month; df["day"]=df.Date.dt.day; df["dow"]=df.Date.dt.dayofweek
    df["doy"] = df.Date.dt.dayofyear
    if is_train_traffic:
        df = df.merge(daily_t[["Date","sessions","unique_visitors","page_views","bounce_rate","avg_sess"]], on="Date", how="left")
    else:
        df = df.merge(traffic_avg, on=["month","dow"], how="left")
    for c in ["sessions","unique_visitors","page_views","bounce_rate","avg_sess"]:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    df["month_sin"]=np.sin(2*np.pi*df.month/12); df["month_cos"]=np.cos(2*np.pi*df.month/12)
    df["dow_sin"]=np.sin(2*np.pi*df.dow/7);      df["dow_cos"]=np.cos(2*np.pi*df.dow/7)
    df["day_sin"]=np.sin(2*np.pi*df.day/30.5);   df["day_cos"]=np.cos(2*np.pi*df.day/30.5)
    for k in range(1,4):
        df[f"fa_sin_{k}"]=np.sin(2*np.pi*k*df.doy/365.25); df[f"fa_cos_{k}"]=np.cos(2*np.pi*k*df.doy/365.25)
    df["is_payday"] = ((df.day>=25)|(df.day<=5)).astype(int)
    df["is_double_day"] = (df.month==df.day).astype(int)
    df["is_weekend"] = (df.dow>=5).astype(int)
    df["is_tet_season"] = 0
    for td in TET:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m, "is_tet_season"] = 1
    df["is_vn_holiday"] = 0
    for mh,dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            df.loc[(df.month==mh)&(df.day==max(1,dh-off)), "is_vn_holiday"] = 1
    df["traffic_log"]   = np.log1p(df["sessions"])
    df["visitors_log"]  = np.log1p(df["unique_visitors"])
    df["pv_per_session"]= df["page_views"] / (df["sessions"] + 1)
    return df

FEATS = ["month","day","dow","doy","is_weekend",
         "month_sin","month_cos","dow_sin","dow_cos","day_sin","day_cos",
         "fa_sin_1","fa_cos_1","fa_sin_2","fa_cos_2","fa_sin_3","fa_cos_3",
         "is_payday","is_double_day","is_tet_season","is_vn_holiday",
         "sessions","unique_visitors","page_views","bounce_rate","avg_sess",
         "traffic_log","visitors_log","pv_per_session"]

# Train set: 2017+ with real traffic merged by date
tr_all = sales[(sales.Date >= TRAIN_START)].reset_index(drop=True)
tr_f   = build_row_features(tr_all, is_train_traffic=True)
# Validation: within tr_f, 2021-07-01..2022-12-31
val_mask = (tr_f.Date >= VAL_START) & (tr_f.Date <= VAL_END)
trn_mask = (tr_f.Date < VAL_START)
# Val should use (month,dow)-avg traffic as proxy to mirror test path
va_f = build_row_features(tr_all[(tr_all.Date >= VAL_START) & (tr_all.Date <= VAL_END)], is_train_traffic=False)
print(f"[val] train={trn_mask.sum()}  val={len(va_f)}")

LGB_P = dict(n_estimators=1500, learning_rate=0.04, num_leaves=63, max_depth=-1,
             min_child_samples=20, feature_fraction=0.9, bagging_fraction=0.9,
             bagging_freq=5, reg_alpha=0.1, reg_lambda=0.2,
             objective="regression", metric="rmse", random_state=SEED, verbose=-1)
CAT_P = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
             random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)

def swt(years):
    return np.clip((years - 2016).astype(float), 1.0, None) ** 1.2

def run_val(tgt):
    Xt = tr_f.loc[trn_mask, FEATS]; yt = np.log1p(tr_f.loc[trn_mask, tgt].values)
    Xv = va_f[FEATS];                yvr = va_f[tgt].values; yv = np.log1p(yvr)
    w = swt(tr_f.loc[trn_mask, "year"].values)
    print(f"  [{tgt}] LGBM...", flush=True)
    lgbm = lgb.LGBMRegressor(**LGB_P); lgbm.fit(Xt, yt, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(150, verbose=False)])
    print(f"  [{tgt}] CAT...", flush=True)
    cat = CatBoostRegressor(**CAT_P); cat.fit(Xt, yt, sample_weight=w, eval_set=(Xv, yv), early_stopping_rounds=150, verbose=False)
    print(f"  [{tgt}] HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(Xt, yt, sample_weight=w)

    p_l = np.expm1(lgbm.predict(Xv)); p_c = np.expm1(cat.predict(Xv)); p_h = np.expm1(hgb.predict(Xv))
    P = np.column_stack([p_l,p_c,p_h])
    res = minimize(lambda w: np.sqrt(((P@w - yvr)**2).mean()), [1/3]*3, method="SLSQP",
                   bounds=[(0,1)]*3, constraints=({"type":"eq","fun":lambda w: w.sum()-1},))
    wopt = res.x; p_bl = P @ wopt
    for n,p in [("lgbm",p_l),("cat",p_c),("hgb",p_h),("blend",p_bl)]:
        print(f"   {n:6s} RMSE={np.sqrt(mean_squared_error(yvr,p)):>10,.0f}  mean={p.mean()/1e6:.2f}M  actual={yvr.mean()/1e6:.2f}M")
    print(f"   w: {np.round(wopt,3).tolist()}")
    return (lgbm,cat,hgb), wopt

print("\n=== VAL Revenue ==="); rev_vm, rev_w = run_val("Revenue")
print("\n=== VAL COGS ==="   ); cogs_vm, cogs_w = run_val("COGS")

# FINAL fit on all tr_f (2017-2022)
def fit_final(tgt, vm):
    lv, cv, _ = vm
    X = tr_f[FEATS]; y = np.log1p(tr_f[tgt].values); w = swt(tr_f["year"].values)
    bi_l = int((lv.best_iteration_ or 900)*1.1); bi_c = int((cv.get_best_iteration() or 900)*1.1)
    print(f"  [{tgt}] refit LGBM({bi_l}) CAT({bi_c}) HGB...", flush=True)
    p = dict(LGB_P); p["n_estimators"] = bi_l; lgbm = lgb.LGBMRegressor(**p); lgbm.fit(X, y, sample_weight=w)
    p = dict(CAT_P); p["iterations"] = bi_c; cat = CatBoostRegressor(**p); cat.fit(X, y, sample_weight=w, verbose=False)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(X, y, sample_weight=w)
    return lgbm, cat, hgb

print("\n[final Revenue]"); rev_f  = fit_final("Revenue", rev_vm)
print("[final COGS]");      cogs_f = fit_final("COGS",    cogs_vm)

# Build test features (traffic via month,dow avg)
test_f = build_row_features(sample[["Date"]], is_train_traffic=False)
lgbm_r, cat_r, hgb_r = rev_f
lgbm_c, cat_c, hgb_c = cogs_f
Xt = test_f[FEATS]
p_rev  = rev_w[0]*np.expm1(lgbm_r.predict(Xt)) + rev_w[1]*np.expm1(cat_r.predict(Xt)) + rev_w[2]*np.expm1(hgb_r.predict(Xt))
p_cogs = cogs_w[0]*np.expm1(lgbm_c.predict(Xt)) + cogs_w[1]*np.expm1(cat_c.predict(Xt)) + cogs_w[2]*np.expm1(hgb_c.predict(Xt))

sub = pd.DataFrame({"Date": sample.Date.dt.strftime("%Y-%m-%d"),
                    "Revenue": np.round(p_rev, 2),
                    "COGS":    np.round(p_cogs, 2)})
V8_OUT = os.path.join(OUT, "submission_v8.csv")
sub.to_csv(V8_OUT, index=False)
print(f"\n[v8] -> {V8_OUT}")
print(f"  mean Rev={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum  Rev={sub.Revenue.sum()/1e9:.3f}B  sum  COGS={sub.COGS.sum()/1e9:.3f}B")
print(f"  weights Rev {np.round(rev_w,3).tolist()}  COGS {np.round(cogs_w,3).tolist()}")
print("=== DONE ===")
