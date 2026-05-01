"""Phase 7 v4 — level-diverse blend (LGBM + CatBoost × 3 weight schemes + Prophet).

Motivation: Kaggle score on v2 was 1.24M — about 55% worse than val 782k, indicating
large LEVEL bias. v4 uses scipy.optimize for blend weights, fits in 45s, and outputs
predictions from each base model so we can pick a level-diverse final submission.

Base models:
  M1: LightGBM, half-life 2y
  M2: CatBoost,  half-life 2y   (= v2 winner)
  M3: CatBoost,  half-life 6y   (pulls in 2015-2018 higher levels)
  M4: CatBoost,  no recency weight (pure historical average)
  M5: Prophet    (linear trend, flexible changepoints, yearly + weekly + Tet)
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
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

SALES_CSV  = os.path.join(BASE, "sales.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
SUB_OUT    = os.path.join(OUT,  "submission.csv")
DIAG_CSV   = os.path.join(OUT,  "phase7_v4_candidates.csv")

TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")
VAL_START   = pd.Timestamp("2021-07-01")
VAL_END     = pd.Timestamp("2022-12-31")
TRAIN_START = pd.Timestamp("2013-01-01")
SEED = 42
np.random.seed(SEED)

TET_DATES = {y: pd.Timestamp(d) for y, d in {
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
    2025: "2025-01-29",
}.items()}

sales  = pd.read_csv(SALES_CSV, parse_dates=["Date"])
sample = pd.read_csv(SAMPLE_SUB, parse_dates=["Date"])
spine  = pd.DataFrame({"Date": pd.date_range(sales.Date.min(), TEST_END, freq="D")})
df     = spine.merge(sales, on="Date", how="left")
print(f"[spine] {len(df):,} rows")

def build_features(d):
    d = d.copy().sort_values("Date").reset_index(drop=True)
    dt = d.Date.dt
    d["year"]=dt.year; d["month"]=dt.month; d["day"]=dt.day
    d["dow"]=dt.dayofweek; d["doy"]=dt.dayofyear
    d["woy"]=dt.isocalendar().week.astype(int)
    d["is_weekend"]=(d.dow>=5).astype(int)
    d["is_month_start"]=dt.is_month_start.astype(int)
    d["is_month_end"]=dt.is_month_end.astype(int)
    d["is_quarter_start"]=dt.is_quarter_start.astype(int)
    d["is_quarter_end"]=dt.is_quarter_end.astype(int)
    d["t"]=(d.Date-d.Date.min()).dt.days.astype(np.int32)
    d["t2"]=(d["t"]**2)/1e6
    for k in range(1,6):
        d[f"fa_sin_{k}"]=np.sin(2*np.pi*k*d.doy.values/365.25)
        d[f"fa_cos_{k}"]=np.cos(2*np.pi*k*d.doy.values/365.25)
    for k in range(1,4):
        d[f"fw_sin_{k}"]=np.sin(2*np.pi*k*d.dow.values/7)
        d[f"fw_cos_{k}"]=np.cos(2*np.pi*k*d.dow.values/7)
    for k in range(1,3):
        d[f"fm_sin_{k}"]=np.sin(2*np.pi*k*d.day/30.5)
        d[f"fm_cos_{k}"]=np.cos(2*np.pi*k*d.day/30.5)
    for lag in [365, 548, 730, 1095, 1460]:
        d[f"Revenue_lag_{lag}"]=d["Revenue"].shift(lag)
        d[f"COGS_lag_{lag}"]=d["COGS"].shift(lag)
    for col in ["Revenue","COGS"]:
        grp=d.groupby(["month","day"])[col]
        d[f"{col}_seas_expand_mean"]=grp.expanding().mean().shift(1).reset_index(level=[0,1],drop=True)
        d[f"{col}_seas_roll3_mean"]=grp.transform(lambda s: s.shift(1).rolling(3,min_periods=1).mean())
    for col in ["Revenue","COGS"]:
        roll=d[col].rolling(365,min_periods=180).mean()
        d[f"{col}_level_548"]=roll.shift(548)
        d[f"{col}_level_730"]=roll.shift(730)
        d[f"{col}_level_913"]=roll.shift(913)
    d["rev_yoy_ratio"]=d["Revenue_level_548"]/d["Revenue_level_913"]
    d["cogs_yoy_ratio"]=d["COGS_level_548"]/d["COGS_level_913"]
    tet=pd.Series(d.Date.map(lambda x: TET_DATES.get(x.year, pd.NaT)))
    diff=(d.Date - tet).dt.days
    d["tet_offset"]=diff.clip(-30,30).fillna(99).astype(int)
    d["tet_is_window"]=(diff.abs()<=10).fillna(False).astype(int)
    d["tet_pre"]=((diff>=-10)&(diff<0)).fillna(False).astype(int)
    d["tet_post"]=((diff>0)&(diff<=10)).fillna(False).astype(int)
    d["tet_day"]=(diff==0).fillna(False).astype(int)
    return d

feat = build_features(df)
print(f"[features] shape={feat.shape}")

BASE_FEATS=["year","month","day","dow","doy","woy","is_weekend","is_month_start","is_month_end",
            "is_quarter_start","is_quarter_end","t","t2",
            *[f"fa_sin_{k}" for k in range(1,6)], *[f"fa_cos_{k}" for k in range(1,6)],
            *[f"fw_sin_{k}" for k in range(1,4)], *[f"fw_cos_{k}" for k in range(1,4)],
            *[f"fm_sin_{k}" for k in range(1,3)], *[f"fm_cos_{k}" for k in range(1,3)],
            "tet_offset","tet_is_window","tet_pre","tet_post","tet_day"]
REV_FEATS=BASE_FEATS+["Revenue_lag_365","Revenue_lag_548","Revenue_lag_730","Revenue_lag_1095","Revenue_lag_1460",
                      "Revenue_seas_expand_mean","Revenue_seas_roll3_mean","COGS_seas_expand_mean","COGS_seas_roll3_mean",
                      "Revenue_level_548","Revenue_level_730","Revenue_level_913","rev_yoy_ratio"]
COGS_FEATS=BASE_FEATS+["COGS_lag_365","COGS_lag_548","COGS_lag_730","COGS_lag_1095","COGS_lag_1460",
                       "COGS_seas_expand_mean","COGS_seas_roll3_mean","Revenue_seas_expand_mean","Revenue_seas_roll3_mean",
                       "COGS_level_548","COGS_level_730","COGS_level_913","cogs_yoy_ratio"]

LGB_P=dict(n_estimators=3000,learning_rate=0.025,num_leaves=47,max_depth=-1,min_child_samples=25,
           feature_fraction=0.88,bagging_fraction=0.9,bagging_freq=5,reg_alpha=0.1,reg_lambda=0.2,
           objective="regression",metric="rmse",random_state=SEED,verbose=-1)
CAT_P=dict(iterations=3000,learning_rate=0.03,depth=7,l2_leaf_reg=3.0,random_seed=SEED,
           loss_function="RMSE",verbose=False,allow_writing_files=False)

def swt(dates, hl):
    if hl is None: return np.ones(len(dates))
    ya = (dates.max()-dates).dt.days/365.25
    return (0.5**(ya/hl)).values

def fit_cat(X,y,w,Xv=None,yv=None,p=None):
    p = dict(p or CAT_P)
    m = CatBoostRegressor(**p)
    if Xv is not None:
        m.fit(X,y,sample_weight=w,eval_set=(Xv,yv),early_stopping_rounds=150,verbose=False)
    else:
        m.fit(X,y,sample_weight=w,verbose=False)
    return m

def fit_lgb(X,y,w,Xv=None,yv=None):
    m = lgb.LGBMRegressor(**LGB_P)
    if Xv is not None:
        m.fit(X,y,sample_weight=w,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(150,verbose=False)])
    else:
        m.fit(X,y,sample_weight=w)
    return m

def mtx(y,p): return dict(MAE=mean_absolute_error(y,p), RMSE=np.sqrt(mean_squared_error(y,p)), R2=r2_score(y,p))

from prophet import Prophet
def prophet_fc(sdf, test_dates, tgt, cps=0.08):
    dfp=sdf[["Date",tgt]].rename(columns={"Date":"ds",tgt:"y"})
    m=Prophet(growth="linear",yearly_seasonality=20,weekly_seasonality=True,
              daily_seasonality=False,changepoint_prior_scale=cps,seasonality_prior_scale=10,
              holidays_prior_scale=5.0,n_changepoints=25)
    m.holidays=pd.DataFrame({"holiday":"tet","ds":pd.to_datetime(list(TET_DATES.values())),
                             "lower_window":-10,"upper_window":10})
    m.fit(dfp)
    fc=m.predict(pd.DataFrame({"ds":test_dates}))
    return np.clip(fc["yhat"].values, 1.0, None)

# ----- VALIDATION -----
train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START) & (feat.Date <= VAL_END)
print(f"[val] train={train_mask.sum()}  val={val_mask.sum()}")

def run_target_val(tgt, feats):
    Xt = feat.loc[train_mask, feats].copy()
    yt = np.log1p(feat.loc[train_mask, tgt].values)
    Xv = feat.loc[val_mask,   feats].copy()
    yvr = feat.loc[val_mask,  tgt].values
    yv  = np.log1p(yvr)
    dates_tr = feat.loc[train_mask, "Date"]

    w2 = swt(dates_tr, 2.0); w6 = swt(dates_tr, 6.0); weq = swt(dates_tr, None)

    print(f"  [{tgt}] fit LGBM...", flush=True)
    lgbm = fit_lgb(Xt, yt, w2, Xv, yv)
    print(f"  [{tgt}] fit CatBoost (2y)...", flush=True)
    cf   = fit_cat(Xt, yt, w2, Xv, yv)
    print(f"  [{tgt}] fit CatBoost (6y)...", flush=True)
    cs   = fit_cat(Xt, yt, w6, Xv, yv)
    print(f"  [{tgt}] fit CatBoost (eq)...", flush=True)
    ce   = fit_cat(Xt, yt, weq, Xv, yv)

    p_l  = np.expm1(lgbm.predict(Xv))
    p_cf = np.expm1(cf.predict(Xv))
    p_cs = np.expm1(cs.predict(Xv))
    p_ce = np.expm1(ce.predict(Xv))

    print(f"  [{tgt}] fit Prophet...", flush=True)
    sales_tr = sales[sales.Date < VAL_START].copy()
    val_dates = feat.loc[val_mask, "Date"].values
    p_pr = prophet_fc(sales_tr, val_dates, tgt)

    # Optimize 5-way non-negative weights summing to 1
    P = np.column_stack([p_l, p_cf, p_cs, p_ce, p_pr])
    def obj(w): return np.sqrt(((P @ w - yvr)**2).mean())
    x0 = np.array([0.2]*5)
    cons = ({"type":"eq","fun":lambda w: w.sum()-1},)
    bnds = [(0.0, 1.0)]*5
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter":500})
    w_opt = r.x
    p_bl  = P @ w_opt

    rows = [
        {"target":tgt,"model":"lgbm",     **mtx(yvr, p_l),  "mean":p_l.mean()},
        {"target":tgt,"model":"cat_2y",   **mtx(yvr, p_cf), "mean":p_cf.mean()},
        {"target":tgt,"model":"cat_6y",   **mtx(yvr, p_cs), "mean":p_cs.mean()},
        {"target":tgt,"model":"cat_eq",   **mtx(yvr, p_ce), "mean":p_ce.mean()},
        {"target":tgt,"model":"prophet",  **mtx(yvr, p_pr), "mean":p_pr.mean()},
        {"target":tgt,"model":"blend",    **mtx(yvr, p_bl), "mean":p_bl.mean(), "w":str(np.round(w_opt,3).tolist())},
    ]
    for r_ in rows: print(f"   {r_['model']:8s} RMSE={r_['RMSE']:>10,.0f}  mean={r_['mean']/1e6:.2f}M  actual={yvr.mean()/1e6:.2f}M")
    return rows, (lgbm, cf, cs, ce), w_opt

print("\n=== VAL: Revenue ===")
rev_rows, rev_val_models, rev_w = run_target_val("Revenue", REV_FEATS)
print("\n=== VAL: COGS ===")
cogs_rows, cogs_val_models, cogs_w = run_target_val("COGS", COGS_FEATS)

pd.DataFrame(rev_rows + cogs_rows).to_csv(os.path.join(OUT, "phase7_v4_val_metrics.csv"), index=False)

# ----- FINAL TRAIN (2013-2022) and PREDICT (2023-07-2024) -----
full_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)
test_mask = (feat.Date >= TEST_START)  & (feat.Date <= TEST_END)
dates_full = feat.loc[full_mask, "Date"]
print(f"\n[final] train={full_mask.sum()}  test={test_mask.sum()}")

def fit_final_models(tgt, feats, val_models):
    lv, cfv, csv, cev = val_models
    X = feat.loc[full_mask, feats].copy()
    y = np.log1p(feat.loc[full_mask, tgt].values)
    w2 = swt(dates_full, 2.0); w6 = swt(dates_full, 6.0); weq = swt(dates_full, None)

    bi_l  = int((lv.best_iteration_ or 1800)*1.1)
    bi_cf = int((cfv.get_best_iteration() or 1800)*1.1)
    bi_cs = int((csv.get_best_iteration() or 1800)*1.1)
    bi_ce = int((cev.get_best_iteration() or 1800)*1.1)

    print(f"  [{tgt}] refit lgbm ({bi_l})...", flush=True)
    lp = dict(LGB_P); lp["n_estimators"] = bi_l
    lgbm = lgb.LGBMRegressor(**lp); lgbm.fit(X, y, sample_weight=w2)
    print(f"  [{tgt}] refit cat_2y ({bi_cf})...", flush=True)
    cp = dict(CAT_P); cp["iterations"] = bi_cf
    cf = CatBoostRegressor(**cp); cf.fit(X, y, sample_weight=w2, verbose=False)
    print(f"  [{tgt}] refit cat_6y ({bi_cs})...", flush=True)
    cp = dict(CAT_P); cp["iterations"] = bi_cs
    cs = CatBoostRegressor(**cp); cs.fit(X, y, sample_weight=w6, verbose=False)
    print(f"  [{tgt}] refit cat_eq ({bi_ce})...", flush=True)
    cp = dict(CAT_P); cp["iterations"] = bi_ce
    ce = CatBoostRegressor(**cp); ce.fit(X, y, sample_weight=weq, verbose=False)
    return lgbm, cf, cs, ce

print("\n[final Revenue]")
rev_final = fit_final_models("Revenue", REV_FEATS, rev_val_models)
print("[final COGS]")
cogs_final = fit_final_models("COGS", COGS_FEATS, cogs_val_models)

print("[final Prophet Revenue]")
test_dates = sample.Date.values
rev_pr = prophet_fc(sales, test_dates, "Revenue")
print("[final Prophet COGS]")
cogs_pr = prophet_fc(sales, test_dates, "COGS")

# Recursive lag_365 prediction using BLEND
lgbm_r, cf_r, cs_r, ce_r = rev_final
lgbm_c, cf_c, cs_c, ce_c = cogs_final
pred_frame = feat.copy()
test_idx = pred_frame.index[pred_frame.Date.between(TEST_START, TEST_END)]
prev_rev_map, prev_cogs_map = {}, {}
rev_pr_map  = dict(zip(pd.to_datetime(test_dates), rev_pr))
cogs_pr_map = dict(zip(pd.to_datetime(test_dates), cogs_pr))

# Store per-model preds so we can try alternative blends later
per_model = {m: [] for m in ["lgbm","cat_2y","cat_6y","cat_eq","prophet"]}
per_model_c = {m: [] for m in ["lgbm","cat_2y","cat_6y","cat_eq","prophet"]}
dates_out = []

for i in test_idx:
    dt = pred_frame.at[i, "Date"]
    dates_out.append(dt)
    lagd = dt - pd.Timedelta(days=365)
    if lagd >= TEST_START:
        pred_frame.at[i, "Revenue_lag_365"] = prev_rev_map.get(lagd, np.nan)
        pred_frame.at[i, "COGS_lag_365"]    = prev_cogs_map.get(lagd, np.nan)
    Xr = pred_frame.loc[[i], REV_FEATS]
    Xc = pred_frame.loc[[i], COGS_FEATS]

    p_lr  = float(np.expm1(lgbm_r.predict(Xr)[0]))
    p_cfr = float(np.expm1(cf_r.predict(Xr)[0]))
    p_csr = float(np.expm1(cs_r.predict(Xr)[0]))
    p_cer = float(np.expm1(ce_r.predict(Xr)[0]))
    p_prr = float(rev_pr_map.get(dt, np.nan))

    p_lc  = float(np.expm1(lgbm_c.predict(Xc)[0]))
    p_cfc = float(np.expm1(cf_c.predict(Xc)[0]))
    p_csc = float(np.expm1(cs_c.predict(Xc)[0]))
    p_cec = float(np.expm1(ce_c.predict(Xc)[0]))
    p_prc = float(cogs_pr_map.get(dt, np.nan))

    per_model["lgbm"].append(p_lr); per_model_c["lgbm"].append(p_lc)
    per_model["cat_2y"].append(p_cfr); per_model_c["cat_2y"].append(p_cfc)
    per_model["cat_6y"].append(p_csr); per_model_c["cat_6y"].append(p_csc)
    per_model["cat_eq"].append(p_cer); per_model_c["cat_eq"].append(p_cec)
    per_model["prophet"].append(p_prr); per_model_c["prophet"].append(p_prc)

    # Use the val-optimal blend for the recursive lag feed
    pr = rev_w[0]*p_lr + rev_w[1]*p_cfr + rev_w[2]*p_csr + rev_w[3]*p_cer + rev_w[4]*p_prr
    pc = cogs_w[0]*p_lc + cogs_w[1]*p_cfc + cogs_w[2]*p_csc + cogs_w[3]*p_cec + cogs_w[4]*p_prc
    prev_rev_map[dt] = pr
    prev_cogs_map[dt] = pc

# Save all per-model predictions so we can pick a blend based on Kaggle feedback
out_df = pd.DataFrame({"Date": dates_out})
for m in per_model:
    out_df[f"Revenue_{m}"] = per_model[m]
    out_df[f"COGS_{m}"]    = per_model_c[m]

# Default blend from val optimization
out_df["Revenue_blend_valopt"] = (rev_w[0]*out_df["Revenue_lgbm"] + rev_w[1]*out_df["Revenue_cat_2y"]
                                  + rev_w[2]*out_df["Revenue_cat_6y"] + rev_w[3]*out_df["Revenue_cat_eq"]
                                  + rev_w[4]*out_df["Revenue_prophet"])
out_df["COGS_blend_valopt"]    = (cogs_w[0]*out_df["COGS_lgbm"] + cogs_w[1]*out_df["COGS_cat_2y"]
                                  + cogs_w[2]*out_df["COGS_cat_6y"] + cogs_w[3]*out_df["COGS_cat_eq"]
                                  + cogs_w[4]*out_df["COGS_prophet"])
# Higher-level blend (hedge upward): equal weight of cat_6y, cat_eq, prophet
out_df["Revenue_blend_higher"] = (out_df["Revenue_cat_6y"] + out_df["Revenue_cat_eq"] + out_df["Revenue_prophet"]) / 3
out_df["COGS_blend_higher"]    = (out_df["COGS_cat_6y"]    + out_df["COGS_cat_eq"]    + out_df["COGS_prophet"])    / 3
# Medium blend: 0.5 valopt + 0.5 higher
out_df["Revenue_blend_medium"] = 0.5*out_df["Revenue_blend_valopt"] + 0.5*out_df["Revenue_blend_higher"]
out_df["COGS_blend_medium"]    = 0.5*out_df["COGS_blend_valopt"]    + 0.5*out_df["COGS_blend_higher"]

out_df.to_csv(DIAG_CSV, index=False)
print(f"\n[candidates] saved to {DIAG_CSV}")

# Summarize levels
print("\nLevel summary (daily mean $M):")
for m in ["lgbm","cat_2y","cat_6y","cat_eq","prophet"]:
    print(f"  {m:10s}: Rev={out_df[f'Revenue_{m}'].mean()/1e6:.2f}M  COGS={out_df[f'COGS_{m}'].mean()/1e6:.2f}M")
print(f"  val-opt blend: Rev={out_df['Revenue_blend_valopt'].mean()/1e6:.2f}M  COGS={out_df['COGS_blend_valopt'].mean()/1e6:.2f}M")
print(f"  higher blend:  Rev={out_df['Revenue_blend_higher'].mean()/1e6:.2f}M  COGS={out_df['COGS_blend_higher'].mean()/1e6:.2f}M")
print(f"  medium blend:  Rev={out_df['Revenue_blend_medium'].mean()/1e6:.2f}M  COGS={out_df['COGS_blend_medium'].mean()/1e6:.2f}M")
print(f"  val-opt weights — Revenue: {np.round(rev_w,3).tolist()}  COGS: {np.round(cogs_w,3).tolist()}")

# Default submission = val-optimal blend (safest)
sub = sample[["Date"]].merge(
    pd.DataFrame({"Date": out_df.Date, "Revenue": out_df.Revenue_blend_valopt, "COGS": out_df.COGS_blend_valopt}),
    on="Date", how="left")
assert sub.Revenue.notna().all() and sub.COGS.notna().all()
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub.to_csv(SUB_OUT, index=False)
print(f"\n[submission] DEFAULT (val-optimal blend) -> {SUB_OUT}")
print(f"  mean Revenue={sub.Revenue.mean()/1e6:.2f}M  mean COGS={sub.COGS.mean()/1e6:.2f}M")
print(f"  sum Revenue={sub.Revenue.sum()/1e9:.3f}B  sum COGS={sub.COGS.sum()/1e9:.3f}B")
print("\n=== DONE ===")
