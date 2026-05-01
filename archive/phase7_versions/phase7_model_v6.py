"""Phase 7 v6 — FAST iteration on friend's 740k approach.

Keeps friend's core insights:
  - web_traffic.csv: sessions + page_views + unique_visitors features
  - payday (day <=5 or >=25)
  - double-day (day == month, Vietnamese flash-sale days)
  - is_tet_season (21-day pre-Tet window)
  - Vietnamese public holidays
  - polynomial sample-weighting (year-2016)^1.2
  - train from 2017+

Improvements:
  - LGBM + CatBoost + HistGradientBoosting ensemble (faster and usually
    beats RF+ET; leaves friend's RF/ET approach for reference)
  - seasonal same-(month,day) expanding mean feature
  - year-over-year traffic ratio
  - scipy-optimized blend weights on validation fold
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"
OUT  = os.path.join(BASE, "outputs")
SUB_OUT = os.path.join(OUT, "submission.csv")
SEED = 42

TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")
VAL_START   = pd.Timestamp("2021-07-01")
VAL_END     = pd.Timestamp("2022-12-31")
TRAIN_START = pd.Timestamp("2017-01-01")

TET_DATES = [pd.Timestamp(d) for d in [
    "2013-02-10","2014-01-31","2015-02-19","2016-02-08","2017-01-28","2018-02-16",
    "2019-02-05","2020-01-25","2021-02-12","2022-02-01","2023-01-22","2024-02-10","2025-01-29",
]]

sales   = pd.read_csv(os.path.join(BASE, "sales.csv"),         parse_dates=["Date"])
traffic = pd.read_csv(os.path.join(BASE, "web_traffic.csv"),   parse_dates=["date"])
sample  = pd.read_csv(os.path.join(BASE, "sample_submission.csv"), parse_dates=["Date"])

daily_t = traffic.groupby("date").agg(
    sessions=("sessions","sum"),
    unique_visitors=("unique_visitors","sum"),
    page_views=("page_views","sum"),
    bounce_rate=("bounce_rate","mean"),
    avg_session_duration_sec=("avg_session_duration_sec","mean"),
).reset_index().rename(columns={"date":"Date"})
daily_t["month"] = daily_t.Date.dt.month
daily_t["dow"]   = daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]].mean().reset_index()

spine = pd.DataFrame({"Date": pd.date_range(sales.Date.min(), TEST_END, freq="D")})
spine = spine.merge(sales, on="Date", how="left")
spine = spine.merge(daily_t[["Date","sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]],
                    on="Date", how="left")
spine["month"] = spine.Date.dt.month; spine["dow"] = spine.Date.dt.dayofweek
# Fill post-2022 with (month,dow) avg
need = spine.sessions.isna()
fill = spine.loc[need, ["month","dow"]].merge(traffic_avg, on=["month","dow"], how="left")
for c in ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]:
    spine.loc[need, c] = fill[c].values
print(f"[traffic] filled {need.sum()} rows with (month,dow) avg")

def build_features(d):
    d = d.copy().sort_values("Date").reset_index(drop=True)
    dt = d.Date.dt
    d["year"]=dt.year; d["month"]=dt.month; d["day"]=dt.day
    d["dow"]=dt.dayofweek; d["doy"]=dt.dayofyear
    d["is_weekend"]=(d.dow>=5).astype(int)
    d["month_sin"]=np.sin(2*np.pi*d.month/12); d["month_cos"]=np.cos(2*np.pi*d.month/12)
    d["dow_sin"]=np.sin(2*np.pi*d.dow/7);      d["dow_cos"]=np.cos(2*np.pi*d.dow/7)
    d["day_sin"]=np.sin(2*np.pi*d.day/30.5);   d["day_cos"]=np.cos(2*np.pi*d.day/30.5)
    d["doy_sin"]=np.sin(2*np.pi*d.doy/365.25); d["doy_cos"]=np.cos(2*np.pi*d.doy/365.25)
    d["is_payday"] = ((d.day >= 25) | (d.day <= 5)).astype(int)
    d["is_double_day"] = (d.month == d.day).astype(int)
    d["is_tet_season"] = 0
    for tet in TET_DATES:
        m = (d.Date >= tet - pd.Timedelta(days=21)) & (d.Date < tet)
        d.loc[m, "is_tet_season"] = 1
    d["is_vn_holiday"] = 0
    for mh, dh in [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]:
        for off in range(0,4):
            d.loc[(d.month==mh) & (d.day == max(1, dh-off)), "is_vn_holiday"] = 1
    # seasonal same-(month,day) expanding mean of target
    for col in ["Revenue","COGS"]:
        grp = d.groupby(["month","day"])[col]
        d[f"{col}_seas_mean"] = grp.expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
    # lag365 for recursive fill
    for col in ["Revenue","COGS"]:
        d[f"{col}_lag_365"]  = d[col].shift(365)
        d[f"{col}_lag_730"]  = d[col].shift(730)
        d[f"{col}_lag_1095"] = d[col].shift(1095)
    # Traffic YoY
    d["sessions_lag_365"]   = d["sessions"].shift(365)
    d["sessions_ratio_365"] = d["sessions"] / (d["sessions_lag_365"] + 1)
    d["traffic_log"]        = np.log1p(d["sessions"])
    d["pv_per_session"]     = d["page_views"] / (d["sessions"] + 1)
    d["visitors_log"]       = np.log1p(d["unique_visitors"])
    return d

feat = build_features(spine)
print(f"[features] shape={feat.shape}")

BASE_FEATS = ["month","day","dow","doy","is_weekend",
              "month_sin","month_cos","dow_sin","dow_cos","day_sin","day_cos","doy_sin","doy_cos",
              "is_payday","is_double_day","is_tet_season","is_vn_holiday",
              "sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec",
              "traffic_log","pv_per_session","visitors_log","sessions_lag_365","sessions_ratio_365"]
REV_FEATS  = BASE_FEATS + ["Revenue_lag_365","Revenue_lag_730","Revenue_lag_1095",
                           "Revenue_seas_mean","COGS_seas_mean"]
COGS_FEATS = BASE_FEATS + ["COGS_lag_365","COGS_lag_730","COGS_lag_1095",
                           "COGS_seas_mean","Revenue_seas_mean"]

LGB_P = dict(n_estimators=1500, learning_rate=0.04, num_leaves=47, max_depth=-1,
             min_child_samples=25, feature_fraction=0.9, bagging_fraction=0.9,
             bagging_freq=5, reg_alpha=0.1, reg_lambda=0.2,
             objective="regression", metric="rmse", random_state=SEED, verbose=-1)
CAT_P = dict(iterations=1500, learning_rate=0.045, depth=7, l2_leaf_reg=3.0,
             random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)

def swt(dates):
    yrs = dates.dt.year.values
    return np.clip((yrs - 2016).astype(float), 1.0, None) ** 1.2

train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START)  & (feat.Date <= VAL_END)
print(f"[val] train={train_mask.sum()}  val={val_mask.sum()}")

def run_val(tgt, feats):
    Xt = feat.loc[train_mask, feats]; yt = np.log1p(feat.loc[train_mask, tgt].values)
    Xv = feat.loc[val_mask,   feats]; yvr = feat.loc[val_mask, tgt].values; yv = np.log1p(yvr)
    w = swt(feat.loc[train_mask, "Date"])

    print(f"  [{tgt}] fit LGBM...", flush=True)
    lgbm = lgb.LGBMRegressor(**LGB_P)
    lgbm.fit(Xt, yt, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(120, verbose=False)])

    print(f"  [{tgt}] fit CatBoost...", flush=True)
    cat = CatBoostRegressor(**CAT_P)
    cat.fit(Xt, yt, sample_weight=w, eval_set=(Xv, yv), early_stopping_rounds=120, verbose=False)

    print(f"  [{tgt}] fit HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED)
    hgb.fit(Xt, yt, sample_weight=w)

    p_l = np.expm1(lgbm.predict(Xv)); p_c = np.expm1(cat.predict(Xv)); p_h = np.expm1(hgb.predict(Xv))
    P = np.column_stack([p_l, p_c, p_h])
    res = minimize(lambda w: np.sqrt(((P@w - yvr)**2).mean()), [1/3]*3,
                   method="SLSQP", bounds=[(0,1)]*3, constraints=({"type":"eq","fun":lambda w: w.sum()-1},))
    wopt = res.x
    p_bl = P @ wopt
    for name, p in [("lgbm",p_l),("cat",p_c),("hgb",p_h),("blend",p_bl)]:
        rmse = np.sqrt(mean_squared_error(yvr, p)); mae = mean_absolute_error(yvr, p); mean = p.mean()/1e6
        print(f"   {name:6s} RMSE={rmse:>10,.0f}  MAE={mae:>10,.0f}  mean={mean:.2f}M  actual={yvr.mean()/1e6:.2f}M")
    print(f"   blend weights: {np.round(wopt,3).tolist()}")
    return (lgbm, cat, hgb), wopt

print("\n=== VAL Revenue ==="); rev_val_models, rev_w = run_val("Revenue", REV_FEATS)
print("\n=== VAL COGS ==="   ); cogs_val_models, cogs_w = run_val("COGS",    COGS_FEATS)

# FINAL TRAIN 2017-01-01 .. 2022-12-31
full_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)
test_mask = (feat.Date >= TEST_START)  & (feat.Date <= TEST_END)

def fit_final(tgt, feats, val_models):
    lv, cv, hv = val_models
    X = feat.loc[full_mask, feats]; y = np.log1p(feat.loc[full_mask, tgt].values)
    w = swt(feat.loc[full_mask, "Date"])
    bi_l = int((lv.best_iteration_ or 1200) * 1.1); bi_c = int((cv.get_best_iteration() or 1200) * 1.1)
    print(f"  [{tgt}] refit LGBM({bi_l})...", flush=True)
    p = dict(LGB_P); p["n_estimators"] = bi_l; lgbm = lgb.LGBMRegressor(**p); lgbm.fit(X, y, sample_weight=w)
    print(f"  [{tgt}] refit CatBoost({bi_c})...", flush=True)
    p = dict(CAT_P); p["iterations"] = bi_c; cat = CatBoostRegressor(**p); cat.fit(X, y, sample_weight=w, verbose=False)
    print(f"  [{tgt}] refit HGB...", flush=True)
    hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=SEED)
    hgb.fit(X, y, sample_weight=w)
    return lgbm, cat, hgb

print("\n[final Revenue]"); rev_final  = fit_final("Revenue", REV_FEATS, rev_val_models)
print("[final COGS]"   );  cogs_final = fit_final("COGS",    COGS_FEATS, cogs_val_models)

# Recursive lag_365 predict
lgbm_r, cat_r, hgb_r = rev_final
lgbm_c, cat_c, hgb_c = cogs_final
pf = feat.copy()
test_idx = pf.index[pf.Date.between(TEST_START, TEST_END)]
prev_r, prev_c = {}, {}
dates_out, rev_out, cogs_out = [], [], []
for i in test_idx:
    dt = pf.at[i, "Date"]; dates_out.append(dt)
    lagd = dt - pd.Timedelta(days=365)
    if lagd >= TEST_START:
        pf.at[i, "Revenue_lag_365"] = prev_r.get(lagd, np.nan)
        pf.at[i, "COGS_lag_365"]    = prev_c.get(lagd, np.nan)
    Xr = pf.loc[[i], REV_FEATS]; Xc = pf.loc[[i], COGS_FEATS]
    p_lr = float(np.expm1(lgbm_r.predict(Xr)[0])); p_cr = float(np.expm1(cat_r.predict(Xr)[0])); p_hr = float(np.expm1(hgb_r.predict(Xr)[0]))
    p_lc = float(np.expm1(lgbm_c.predict(Xc)[0])); p_cc = float(np.expm1(cat_c.predict(Xc)[0])); p_hc = float(np.expm1(hgb_c.predict(Xc)[0]))
    pr = rev_w[0]*p_lr + rev_w[1]*p_cr + rev_w[2]*p_hr
    pc = cogs_w[0]*p_lc + cogs_w[1]*p_cc + cogs_w[2]*p_hc
    prev_r[dt] = pr; prev_c[dt] = pc
    rev_out.append(pr); cogs_out.append(pc)

pred_df = pd.DataFrame({"Date": dates_out, "Revenue": rev_out, "COGS": cogs_out})
sub = sample[["Date"]].merge(pred_df, on="Date", how="left")
assert sub.Revenue.notna().all() and sub.COGS.notna().all()
sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2); sub["COGS"] = sub["COGS"].round(2)

# Save as v6 candidate (don't overwrite the safe friend baseline yet)
V6_OUT = os.path.join(OUT, "submission_v6.csv")
sub.to_csv(V6_OUT, index=False)
print(f"\n[submission v6 candidate] -> {V6_OUT}")
print(f"  mean Rev={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum Rev={sub.Revenue.sum()/1e9:.3f}B  sum COGS={sub.COGS.sum()/1e9:.3f}B")
print(f"  val-weights — Revenue: {np.round(rev_w,3).tolist()}  COGS: {np.round(cogs_w,3).tolist()}")
print("=== DONE ===")
