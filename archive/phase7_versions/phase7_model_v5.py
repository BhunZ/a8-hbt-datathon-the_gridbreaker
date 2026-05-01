"""Phase 7 v5 — incorporates the insights from friend's 740k submission.

Key additions over v2:
  (A) web_traffic.csv: daily sessions + page_views + unique_visitors.
      For test (2023-2024) where traffic is not observed, use the per
      (month, dow) historical average.
  (B) is_payday: day <= 5 OR day >= 25 (Vietnamese salary calendar).
  (C) is_double_day: day == month (3/3, 9/9, 10/10, 11/11, 12/12 are the
      biggest Vietnamese e-commerce flash sale days).
  (D) is_tet_season: 21-day pre-Tet window (not ±10 like v2), reflecting
      actual Vietnamese shopping behaviour.
  (E) Vietnamese public holidays: Reunification (4/30), Labour (5/1),
      National Day (9/2), Christmas (12/24-25), New Year (1/1).
  (F) Keeps v2's tree stack (LGBM + CatBoost) but adds them as an ensemble
      with friend's RandomForest + ExtraTrees approach.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

SALES_CSV   = os.path.join(BASE, "sales.csv")
TRAFFIC_CSV = os.path.join(BASE, "web_traffic.csv")
SAMPLE_SUB  = os.path.join(BASE, "sample_submission.csv")
SUB_OUT     = os.path.join(OUT,  "submission.csv")

TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")
VAL_START   = pd.Timestamp("2021-07-01")
VAL_END     = pd.Timestamp("2022-12-31")
TRAIN_START = pd.Timestamp("2017-01-01")   # match friend: 2017+
SEED = 42
np.random.seed(SEED)

TET_DATES = [pd.Timestamp(d) for d in [
    "2013-02-10","2014-01-31","2015-02-19","2016-02-08","2017-01-28",
    "2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01",
    "2023-01-22","2024-02-10","2025-01-29",
]]

# 1. Load -------------------------------------------------------------------
sales   = pd.read_csv(SALES_CSV,   parse_dates=["Date"])
traffic = pd.read_csv(TRAFFIC_CSV, parse_dates=["date"])
sample  = pd.read_csv(SAMPLE_SUB,  parse_dates=["Date"])

# 2. Aggregate traffic by date ---------------------------------------------
daily_t = traffic.groupby("date")[["sessions","unique_visitors","page_views",
                                    "bounce_rate","avg_session_duration_sec"]].agg({
    "sessions":"sum", "unique_visitors":"sum", "page_views":"sum",
    "bounce_rate":"mean", "avg_session_duration_sec":"mean"
}).reset_index().rename(columns={"date":"Date"})
daily_t["month"] = daily_t.Date.dt.month
daily_t["dow"]   = daily_t.Date.dt.dayofweek
traffic_avg = daily_t.groupby(["month","dow"])[
    ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]
].mean().reset_index()
print(f"[traffic] {len(daily_t)} daily rows, cols={list(daily_t.columns)[:5]}...")

# 3. Build full spine, merge actual traffic where available, avg profile elsewhere
spine = pd.DataFrame({"Date": pd.date_range(sales.Date.min(), TEST_END, freq="D")})
spine = spine.merge(sales, on="Date", how="left")
# Merge actual traffic for dates <= 2022
spine = spine.merge(daily_t[["Date","sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]],
                    on="Date", how="left")
# Fill post-2022 with (month, dow) average
spine["month"] = spine.Date.dt.month
spine["dow"]   = spine.Date.dt.dayofweek
needs_avg = spine.sessions.isna()
print(f"[traffic-fill] filling {needs_avg.sum()} rows with (month,dow) averages")
spine_avg = spine.loc[needs_avg, ["month","dow"]].merge(traffic_avg, on=["month","dow"], how="left")
for col in ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec"]:
    spine.loc[needs_avg, col] = spine_avg[col].values

# 4. Feature engineering --------------------------------------------------
def build_features(d):
    d = d.copy().sort_values("Date").reset_index(drop=True)
    dt = d.Date.dt
    d["year"]   = dt.year; d["month"] = dt.month; d["day"] = dt.day
    d["dow"]    = dt.dayofweek; d["doy"] = dt.dayofyear
    d["woy"]    = dt.isocalendar().week.astype(int)
    d["is_weekend"]       = (d.dow >= 5).astype(int)
    d["is_month_start"]   = dt.is_month_start.astype(int)
    d["is_month_end"]     = dt.is_month_end.astype(int)
    d["is_quarter_start"] = dt.is_quarter_start.astype(int)
    d["is_quarter_end"]   = dt.is_quarter_end.astype(int)
    d["t"]  = (d.Date - d.Date.min()).dt.days.astype(np.int32)
    d["t2"] = (d["t"] ** 2) / 1e6

    # Cyclical (friend-style simpler encoding in addition to Fourier)
    d["month_sin"] = np.sin(2*np.pi*d.month/12); d["month_cos"] = np.cos(2*np.pi*d.month/12)
    d["dow_sin"]   = np.sin(2*np.pi*d.dow/7);    d["dow_cos"]   = np.cos(2*np.pi*d.dow/7)
    d["day_sin"]   = np.sin(2*np.pi*d.day/30.5); d["day_cos"]   = np.cos(2*np.pi*d.day/30.5)

    # Fourier annual
    for k in range(1, 4):
        d[f"fa_sin_{k}"] = np.sin(2*np.pi*k*d.doy/365.25)
        d[f"fa_cos_{k}"] = np.cos(2*np.pi*k*d.doy/365.25)

    # NEW from friend: payday, double-day, VN holidays
    d["is_payday"]     = ((d.day >= 25) | (d.day <= 5)).astype(int)
    d["is_double_day"] = (d.month == d.day).astype(int)

    # Tet-related (21-day PRE window, plus ±3 day actual Tet)
    d["is_tet_season"] = 0
    d["is_tet_core"]   = 0
    d["tet_offset"]    = 99
    for tet in TET_DATES:
        pre_mask = (d.Date >= tet - pd.Timedelta(days=21)) & (d.Date < tet)
        d.loc[pre_mask, "is_tet_season"] = 1
        core_mask = (d.Date >= tet - pd.Timedelta(days=3)) & (d.Date <= tet + pd.Timedelta(days=3))
        d.loc[core_mask, "is_tet_core"] = 1
        # offset (days before Tet, negative for after, clipped)
        same_yr = (d.Date.dt.year == tet.year)
        diff_days = (d.Date - tet).dt.days.where(same_yr, other=np.nan)
        d.loc[same_yr, "tet_offset"] = diff_days.loc[same_yr].clip(-30, 30).fillna(99).astype(int).values

    # Vietnamese public holidays (±3 days window)
    d["is_vn_holiday"] = 0
    vn_holidays = [(1,1),(4,30),(5,1),(9,2),(12,24),(12,25),(12,31)]
    for mh, dh in vn_holidays:
        for off in range(0, 4):
            d.loc[(d.month==mh) & (d.day == max(1, dh-off)), "is_vn_holiday"] = 1

    # LAG / level features on the target (leak-free: shift on the spine)
    for col in ["Revenue","COGS"]:
        d[f"{col}_lag_365"]  = d[col].shift(365)
        d[f"{col}_lag_548"]  = d[col].shift(548)
        d[f"{col}_lag_730"]  = d[col].shift(730)
        d[f"{col}_lag_1095"] = d[col].shift(1095)
    for col in ["Revenue","COGS"]:
        grp = d.groupby(["month","day"])[col]
        d[f"{col}_seas_expand_mean"] = grp.expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
    for col in ["Revenue","COGS"]:
        roll = d[col].rolling(365, min_periods=180).mean()
        d[f"{col}_level_548"] = roll.shift(548)
        d[f"{col}_level_730"] = roll.shift(730)

    # Traffic-derived features
    d["traffic_log"]        = np.log1p(d["sessions"])
    d["pv_per_session"]     = d["page_views"] / (d["sessions"] + 1)
    d["visitors_log"]       = np.log1p(d["unique_visitors"])
    # YoY-style traffic lag
    d["sessions_lag_365"]   = d["sessions"].shift(365)
    d["sessions_ratio_365"] = d["sessions"] / (d["sessions_lag_365"] + 1)

    return d

feat = build_features(spine)
print(f"[features] shape={feat.shape}  traffic cols present: {'sessions' in feat.columns}")

# 5. Feature lists --------------------------------------------------------
TRAFFIC_FEATS = ["sessions","unique_visitors","page_views","bounce_rate","avg_session_duration_sec",
                 "traffic_log","pv_per_session","visitors_log","sessions_lag_365","sessions_ratio_365"]
BASE_FEATS = [
    "year","month","day","dow","doy","woy","is_weekend","is_month_start","is_month_end",
    "is_quarter_start","is_quarter_end","t","t2",
    "month_sin","month_cos","dow_sin","dow_cos","day_sin","day_cos",
    "fa_sin_1","fa_cos_1","fa_sin_2","fa_cos_2","fa_sin_3","fa_cos_3",
    "is_payday","is_double_day","is_tet_season","is_tet_core","tet_offset","is_vn_holiday",
] + TRAFFIC_FEATS
REV_FEATS = BASE_FEATS + [
    "Revenue_lag_365","Revenue_lag_548","Revenue_lag_730","Revenue_lag_1095",
    "Revenue_seas_expand_mean","COGS_seas_expand_mean",
    "Revenue_level_548","Revenue_level_730",
]
COGS_FEATS = BASE_FEATS + [
    "COGS_lag_365","COGS_lag_548","COGS_lag_730","COGS_lag_1095",
    "COGS_seas_expand_mean","Revenue_seas_expand_mean",
    "COGS_level_548","COGS_level_730",
]

LGB_P = dict(n_estimators=2500, learning_rate=0.03, num_leaves=47, max_depth=-1,
             min_child_samples=25, feature_fraction=0.9, bagging_fraction=0.9,
             bagging_freq=5, reg_alpha=0.1, reg_lambda=0.2,
             objective="regression", metric="rmse", random_state=SEED, verbose=-1)
CAT_P = dict(iterations=2500, learning_rate=0.035, depth=7, l2_leaf_reg=3.0,
             random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)

def swt(dates):
    # Friend's polynomial weighting: (year - 2016)^1.2, clipped to min 1
    yrs = dates.dt.year.values
    return np.clip((yrs - 2016).astype(float), 1.0, None) ** 1.2

def fit_lgb(X, y, w, Xv=None, yv=None):
    m = lgb.LGBMRegressor(**LGB_P)
    if Xv is not None:
        m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m.fit(X, y, sample_weight=w)
    return m

def fit_cat(X, y, w, Xv=None, yv=None):
    p = dict(CAT_P)
    m = CatBoostRegressor(**p)
    if Xv is not None:
        m.fit(X, y, sample_weight=w, eval_set=(Xv, yv),
              early_stopping_rounds=150, verbose=False)
    else:
        m.fit(X, y, sample_weight=w, verbose=False)
    return m

def mtx(y, p):
    return dict(MAE=mean_absolute_error(y,p), RMSE=np.sqrt(mean_squared_error(y,p)), R2=r2_score(y,p))

# 6. VALIDATION FOLD ------------------------------------------------------
train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START)  & (feat.Date <= VAL_END)
print(f"[val] train={train_mask.sum()}  val={val_mask.sum()}")

def run_val(tgt, feats):
    Xt = feat.loc[train_mask, feats].copy()
    yt = np.log1p(feat.loc[train_mask, tgt].values)
    Xv = feat.loc[val_mask,   feats].copy()
    yvr = feat.loc[val_mask,  tgt].values
    yv  = np.log1p(yvr)
    w_tr = swt(feat.loc[train_mask, "Date"])

    print(f"  [{tgt}] fit LGBM...", flush=True)
    lgbm = fit_lgb(Xt, yt, w_tr, Xv, yv)
    print(f"  [{tgt}] fit CatBoost...", flush=True)
    cat  = fit_cat(Xt, yt, w_tr, Xv, yv)
    print(f"  [{tgt}] fit RandomForest...", flush=True)
    rf = RandomForestRegressor(n_estimators=400, max_depth=15, random_state=SEED, n_jobs=-1)
    rf.fit(Xt, yt, sample_weight=w_tr)
    print(f"  [{tgt}] fit ExtraTrees...", flush=True)
    et = ExtraTreesRegressor(n_estimators=500, max_depth=16, random_state=SEED, n_jobs=-1)
    et.fit(Xt, yt, sample_weight=w_tr)

    p_l = np.expm1(lgbm.predict(Xv))
    p_c = np.expm1(cat.predict(Xv))
    p_r = np.expm1(rf.predict(Xv))
    p_e = np.expm1(et.predict(Xv))

    # Optimize 4-way blend
    P = np.column_stack([p_l, p_c, p_r, p_e])
    def obj(w): return np.sqrt(((P @ w - yvr)**2).mean())
    cons = ({"type":"eq","fun":lambda w: w.sum()-1},)
    res = minimize(obj, [0.25]*4, method="SLSQP", bounds=[(0.0,1.0)]*4, constraints=cons)
    w_opt = res.x
    p_bl = P @ w_opt

    rows = [
        {"target":tgt,"model":"lgbm","RMSE":mtx(yvr,p_l)["RMSE"],"MAE":mtx(yvr,p_l)["MAE"],"mean":p_l.mean()/1e6},
        {"target":tgt,"model":"cat", "RMSE":mtx(yvr,p_c)["RMSE"],"MAE":mtx(yvr,p_c)["MAE"],"mean":p_c.mean()/1e6},
        {"target":tgt,"model":"rf",  "RMSE":mtx(yvr,p_r)["RMSE"],"MAE":mtx(yvr,p_r)["MAE"],"mean":p_r.mean()/1e6},
        {"target":tgt,"model":"et",  "RMSE":mtx(yvr,p_e)["RMSE"],"MAE":mtx(yvr,p_e)["MAE"],"mean":p_e.mean()/1e6},
        {"target":tgt,"model":"blend","RMSE":mtx(yvr,p_bl)["RMSE"],"MAE":mtx(yvr,p_bl)["MAE"],"mean":p_bl.mean()/1e6,"w":str(np.round(w_opt,3).tolist())},
    ]
    for r in rows:
        extra = "  w="+r.get("w","") if "w" in r else ""
        print(f"   {r['model']:6s} RMSE={r['RMSE']:>10,.0f}  MAE={r['MAE']:>10,.0f}  mean={r['mean']:.2f}M  actual={yvr.mean()/1e6:.2f}M{extra}")
    return rows, (lgbm, cat, rf, et), w_opt

print("\n=== VAL Revenue ===")
rev_rows, rev_val_models, rev_w = run_val("Revenue", REV_FEATS)
print("\n=== VAL COGS ===")
cogs_rows, cogs_val_models, cogs_w = run_val("COGS", COGS_FEATS)
pd.DataFrame(rev_rows + cogs_rows).to_csv(os.path.join(OUT, "phase7_v5_val_metrics.csv"), index=False)

# 7. FINAL TRAIN on 2017-01-01 -> 2022-12-31 -----------------------------
full_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)
test_mask = (feat.Date >= TEST_START)  & (feat.Date <= TEST_END)
dates_full = feat.loc[full_mask, "Date"]
print(f"\n[final] train={full_mask.sum()}  test={test_mask.sum()}")

def fit_final(tgt, feats, val_models):
    lv, cv, rv, ev = val_models
    X = feat.loc[full_mask, feats].copy()
    y = np.log1p(feat.loc[full_mask, tgt].values)
    w = swt(dates_full)

    bi_l = int((lv.best_iteration_ or 1800) * 1.1)
    bi_c = int((cv.get_best_iteration() or 1800) * 1.1)
    print(f"  [{tgt}] refit lgbm ({bi_l})...", flush=True)
    lp = dict(LGB_P); lp["n_estimators"] = bi_l
    lgbm = lgb.LGBMRegressor(**lp); lgbm.fit(X, y, sample_weight=w)
    print(f"  [{tgt}] refit cat ({bi_c})...", flush=True)
    cp = dict(CAT_P); cp["iterations"] = bi_c
    cat = CatBoostRegressor(**cp); cat.fit(X, y, sample_weight=w, verbose=False)
    print(f"  [{tgt}] refit rf...", flush=True)
    rf = RandomForestRegressor(n_estimators=400, max_depth=15, random_state=SEED, n_jobs=-1)
    rf.fit(X, y, sample_weight=w)
    print(f"  [{tgt}] refit et...", flush=True)
    et = ExtraTreesRegressor(n_estimators=500, max_depth=16, random_state=SEED, n_jobs=-1)
    et.fit(X, y, sample_weight=w)
    return lgbm, cat, rf, et

print("\n[final Revenue]")
rev_final = fit_final("Revenue", REV_FEATS, rev_val_models)
print("[final COGS]")
cogs_final = fit_final("COGS", COGS_FEATS, cogs_val_models)

# 8. Recursive predict with val-optimal blend -----------------------------
lgbm_r, cat_r, rf_r, et_r = rev_final
lgbm_c, cat_c, rf_c, et_c = cogs_final
pf = feat.copy()
test_idx = pf.index[pf.Date.between(TEST_START, TEST_END)]
prev_rev, prev_cogs = {}, {}
dates_out, p_rev_out, p_cogs_out = [], [], []

for i in test_idx:
    dt = pf.at[i, "Date"]
    dates_out.append(dt)
    lagd = dt - pd.Timedelta(days=365)
    if lagd >= TEST_START:
        pf.at[i, "Revenue_lag_365"] = prev_rev.get(lagd, np.nan)
        pf.at[i, "COGS_lag_365"]    = prev_cogs.get(lagd, np.nan)
    Xr = pf.loc[[i], REV_FEATS]
    Xc = pf.loc[[i], COGS_FEATS]

    p_lr = float(np.expm1(lgbm_r.predict(Xr)[0]))
    p_cr = float(np.expm1(cat_r.predict(Xr)[0]))
    p_rr = float(np.expm1(rf_r.predict(Xr)[0]))
    p_er = float(np.expm1(et_r.predict(Xr)[0]))

    p_lc = float(np.expm1(lgbm_c.predict(Xc)[0]))
    p_cc = float(np.expm1(cat_c.predict(Xc)[0]))
    p_rc = float(np.expm1(rf_c.predict(Xc)[0]))
    p_ec = float(np.expm1(et_c.predict(Xc)[0]))

    pr = rev_w[0]*p_lr + rev_w[1]*p_cr + rev_w[2]*p_rr + rev_w[3]*p_er
    pc = cogs_w[0]*p_lc + cogs_w[1]*p_cc + cogs_w[2]*p_rc + cogs_w[3]*p_ec

    prev_rev[dt]  = pr
    prev_cogs[dt] = pc
    p_rev_out.append(pr)
    p_cogs_out.append(pc)

pred_df = pd.DataFrame({"Date": dates_out, "Revenue": p_rev_out, "COGS": p_cogs_out})
sub = sample[["Date"]].merge(pred_df, on="Date", how="left")
assert sub.Revenue.notna().all() and sub.COGS.notna().all()
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub.to_csv(SUB_OUT, index=False)
print(f"\n[submission] -> {SUB_OUT}")
print(f"  mean Revenue={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum  Revenue={sub.Revenue.sum()/1e9:.3f}B  sum  COGS={sub.COGS.sum()/1e9:.3f}B")
print(f"  val-optimal weights — Revenue: {np.round(rev_w,3).tolist()}  COGS: {np.round(cogs_w,3).tolist()}")
print("\n=== DONE ===")
