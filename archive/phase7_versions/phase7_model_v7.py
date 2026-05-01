"""Phase 7 v7 — stay true to friend's core idea: NO LAG FEATURES.

Why: Our v2/v6 used Revenue_lag_365 etc., which anchor predictions to the
depressed 2020-2022 level (~$3.2M/day). Friend's model, with no lags, lets
the trend + traffic + calendar signals extrapolate UP toward the pre-2019
level. His submission predicts $4.33M/day and scored 740k on Kaggle.

v7: mimic friend's architecture but swap RF+ET for LGBM+CatBoost+HGB and
optimize blend weights on validation. Keep his feature set, his polynomial
weighting, his 2017+ training window. Add ONE extra: seasonal same-(month,day)
expanding mean (computed leak-free on the spine BEFORE 2023) so the model
knows the typical March-vs-November shape.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"
OUT  = os.path.join(BASE, "outputs")
SEED = 42
TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")
VAL_START   = pd.Timestamp("2021-07-01")
VAL_END     = pd.Timestamp("2022-12-31")
TRAIN_START = pd.Timestamp("2017-01-01")
TET_DATES = [pd.Timestamp(d) for d in [
    "2013-02-10","2014-01-31","2015-02-19","2016-02-08","2017-01-28","2018-02-16",
    "2019-02-05","2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10",
]]

sales   = pd.read_csv(os.path.join(BASE, "sales.csv"),           parse_dates=["Date"])
traffic = pd.read_csv(os.path.join(BASE, "web_traffic.csv"),     parse_dates=["date"])
sample  = pd.read_csv(os.path.join(BASE, "sample_submission.csv"), parse_dates=["Date"])

daily_t = traffic.groupby("date").agg(
    sessions=("sessions","sum"), unique_visitors=("unique_visitors","sum"),
    page_views=("page_views","sum"), bounce_rate=("bounce_rate","mean"),
    avg_sess=("avg_session_duration_sec","mean"),
).reset_index().rename(columns={"date":"Date"})
daily_t["month"] = daily_t.Date.dt.month; daily_t["dow"] = daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","unique_visitors","page_views","bounce_rate","avg_sess"]].mean().reset_index()

spine = pd.DataFrame({"Date": pd.date_range(sales.Date.min(), TEST_END, freq="D")})
spine = spine.merge(sales, on="Date", how="left")
spine = spine.merge(daily_t[["Date","sessions","unique_visitors","page_views","bounce_rate","avg_sess"]],
                    on="Date", how="left")
spine["month"] = spine.Date.dt.month; spine["dow"] = spine.Date.dt.dayofweek
need = spine.sessions.isna()
fill = spine.loc[need, ["month","dow"]].merge(traffic_avg, on=["month","dow"], how="left")
for c in ["sessions","unique_visitors","page_views","bounce_rate","avg_sess"]:
    spine.loc[need, c] = fill[c].values
print(f"[traffic] filled {need.sum()} rows with (month,dow) avg")

def build_features(d):
    d = d.copy().sort_values("Date").reset_index(drop=True)
    dt = d.Date.dt
    d["year"]=dt.year; d["month"]=dt.month; d["day"]=dt.day
    d["dow"]=dt.dayofweek; d["doy"]=dt.dayofyear
    d["month_sin"]=np.sin(2*np.pi*d.month/12); d["month_cos"]=np.cos(2*np.pi*d.month/12)
    d["dow_sin"]=np.sin(2*np.pi*d.dow/7);      d["dow_cos"]=np.cos(2*np.pi*d.dow/7)
    d["day_sin"]=np.sin(2*np.pi*d.day/30.5);   d["day_cos"]=np.cos(2*np.pi*d.day/30.5)
    for k in range(1, 4):
        d[f"fa_sin_{k}"] = np.sin(2*np.pi*k*d.doy/365.25)
        d[f"fa_cos_{k}"] = np.cos(2*np.pi*k*d.doy/365.25)
    d["is_payday"]     = ((d.day >= 25) | (d.day <= 5)).astype(int)
    d["is_double_day"] = (d.month == d.day).astype(int)
    d["is_tet_season"] = 0
    for tet in TET_DATES:
        m = (d.Date >= tet - pd.Timedelta(days=21)) & (d.Date < tet)
        d.loc[m, "is_tet_season"] = 1
    d["is_vn_holiday"] = 0
    for mh, dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            d.loc[(d.month==mh) & (d.day == max(1, dh-off)), "is_vn_holiday"] = 1
    # Seasonal same-(month,day) expanding mean — leak-free (shift 1)
    for col in ["Revenue","COGS"]:
        grp = d.groupby(["month","day"])[col]
        d[f"{col}_seas_mean"] = grp.expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
    # Traffic-derived
    d["traffic_log"]    = np.log1p(d["sessions"])
    d["pv_per_session"] = d["page_views"] / (d["sessions"] + 1)
    d["visitors_log"]   = np.log1p(d["unique_visitors"])
    return d

feat = build_features(spine)
print(f"[features] shape={feat.shape}")

# NO LAG FEATURES — let trend + traffic + calendar extrapolate the level
FEATS = ["month","day","dow","doy",
         "month_sin","month_cos","dow_sin","dow_cos","day_sin","day_cos",
         "fa_sin_1","fa_cos_1","fa_sin_2","fa_cos_2","fa_sin_3","fa_cos_3",
         "is_payday","is_double_day","is_tet_season","is_vn_holiday",
         "sessions","unique_visitors","page_views","bounce_rate","avg_sess",
         "traffic_log","pv_per_session","visitors_log",
         "Revenue_seas_mean","COGS_seas_mean"]

LGB_P = dict(n_estimators=1200, learning_rate=0.04, num_leaves=63, max_depth=-1,
             min_child_samples=20, feature_fraction=0.9, bagging_fraction=0.9,
             bagging_freq=5, reg_alpha=0.1, reg_lambda=0.2,
             objective="regression", metric="rmse", random_state=SEED, verbose=-1)
CAT_P = dict(iterations=1200, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
             random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)

def swt(dates):
    yrs = dates.dt.year.values
    return np.clip((yrs - 2016).astype(float), 1.0, None) ** 1.2

train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START)  & (feat.Date <= VAL_END)

def run_val(tgt):
    Xt = feat.loc[train_mask, FEATS]; yt = np.log1p(feat.loc[train_mask, tgt].values)
    Xv = feat.loc[val_mask,   FEATS]; yvr = feat.loc[val_mask, tgt].values; yv = np.log1p(yvr)
    w = swt(feat.loc[train_mask, "Date"])
    print(f"  [{tgt}] LGBM...", flush=True)
    lgbm = lgb.LGBMRegressor(**LGB_P); lgbm.fit(Xt, yt, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(150, verbose=False)])
    print(f"  [{tgt}] CatBoost...", flush=True)
    cat = CatBoostRegressor(**CAT_P); cat.fit(Xt, yt, sample_weight=w, eval_set=(Xv, yv), early_stopping_rounds=150, verbose=False)
    print(f"  [{tgt}] HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED); hgb.fit(Xt, yt, sample_weight=w)
    p_l = np.expm1(lgbm.predict(Xv)); p_c = np.expm1(cat.predict(Xv)); p_h = np.expm1(hgb.predict(Xv))
    P = np.column_stack([p_l, p_c, p_h])
    res = minimize(lambda w: np.sqrt(((P@w - yvr)**2).mean()), [1/3]*3,
                   method="SLSQP", bounds=[(0,1)]*3, constraints=({"type":"eq","fun":lambda w: w.sum()-1},))
    w_opt = res.x; p_bl = P @ w_opt
    for name, p in [("lgbm",p_l),("cat",p_c),("hgb",p_h),("blend",p_bl)]:
        rmse = np.sqrt(mean_squared_error(yvr, p))
        print(f"   {name:6s} RMSE={rmse:>10,.0f}  mean={p.mean()/1e6:.2f}M  actual={yvr.mean()/1e6:.2f}M")
    print(f"   blend w: {np.round(w_opt,3).tolist()}")
    return (lgbm, cat, hgb), w_opt

print("\n=== VAL Revenue ==="); rev_vm, rev_w = run_val("Revenue")
print("\n=== VAL COGS ===");    cogs_vm, cogs_w = run_val("COGS")

# FINAL train 2017-01-01 .. 2022-12-31
full_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)

def fit_final(tgt, vm):
    lv, cv, hv = vm
    X = feat.loc[full_mask, FEATS]; y = np.log1p(feat.loc[full_mask, tgt].values)
    w = swt(feat.loc[full_mask, "Date"])
    bi_l = int((lv.best_iteration_ or 800)*1.1); bi_c = int((cv.get_best_iteration() or 800)*1.1)
    print(f"  [{tgt}] refit LGBM({bi_l})...", flush=True)
    p = dict(LGB_P); p["n_estimators"] = bi_l; lgbm = lgb.LGBMRegressor(**p); lgbm.fit(X, y, sample_weight=w)
    print(f"  [{tgt}] refit CAT({bi_c})...", flush=True)
    p = dict(CAT_P); p["iterations"] = bi_c; cat = CatBoostRegressor(**p); cat.fit(X, y, sample_weight=w, verbose=False)
    print(f"  [{tgt}] refit HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED)
    hgb.fit(X, y, sample_weight=w)
    return lgbm, cat, hgb

print("\n[final Revenue]"); rev_f  = fit_final("Revenue", rev_vm)
print("[final COGS]");      cogs_f = fit_final("COGS",    cogs_vm)

# Predict — no recursion needed (no lags)
test_mask = (feat.Date >= TEST_START) & (feat.Date <= TEST_END)
Xt_test = feat.loc[test_mask, FEATS].reset_index(drop=True)
dates_test = feat.loc[test_mask, "Date"].reset_index(drop=True)

lgbm_r, cat_r, hgb_r = rev_f
lgbm_c, cat_c, hgb_c = cogs_f
p_lr = np.expm1(lgbm_r.predict(Xt_test)); p_cr = np.expm1(cat_r.predict(Xt_test)); p_hr = np.expm1(hgb_r.predict(Xt_test))
p_lc = np.expm1(lgbm_c.predict(Xt_test)); p_cc = np.expm1(cat_c.predict(Xt_test)); p_hc = np.expm1(hgb_c.predict(Xt_test))
p_rev  = rev_w[0]*p_lr  + rev_w[1]*p_cr  + rev_w[2]*p_hr
p_cogs = cogs_w[0]*p_lc + cogs_w[1]*p_cc + cogs_w[2]*p_hc

pred_df = pd.DataFrame({"Date": dates_test, "Revenue": p_rev, "COGS": p_cogs})
sub = sample[["Date"]].merge(pred_df, on="Date", how="left")
assert sub.Revenue.notna().all() and sub.COGS.notna().all()
sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2); sub["COGS"] = sub["COGS"].round(2)

V7_OUT = os.path.join(OUT, "submission_v7.csv")
sub.to_csv(V7_OUT, index=False)
print(f"\n[v7 candidate] -> {V7_OUT}")
print(f"  mean Rev={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum Rev={sub.Revenue.sum()/1e9:.3f}B  sum COGS={sub.COGS.sum()/1e9:.3f}B")
print(f"  weights — Rev: {np.round(rev_w,3).tolist()}  COGS: {np.round(cogs_w,3).tolist()}")
print("=== DONE ===")
