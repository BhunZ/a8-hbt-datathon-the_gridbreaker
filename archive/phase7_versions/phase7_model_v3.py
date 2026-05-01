"""Phase 7 v3 — adds Prophet + long-history CatBoost to widen the level prior.

Motivation: Kaggle score on v2 submission was 1,236,973 — well above our validation
RMSE (736k concat). This suggests 2023-2024 is at a materially different LEVEL than
the 2020-2022 years our v2 model was anchored on. v3 builds a more diverse blend:

  - M1: CatBoost full + LGBM full (weighted, half-life 2y)   [= v2 core]
  - M2: CatBoost with half-life 6y (includes 2015-2018 high-level years)
  - M3: CatBoost with NO recency weighting (pure historical mean)
  - M4: Prophet (linear trend, changepoints, yearly + weekly seasonality)

Then: optimize the 4-way blend on the 2021-07-01 -> 2022-12-31 validation fold.
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
import lightgbm as lgb
from catboost import CatBoostRegressor

BASE = "/sessions/lucid-relaxed-edison/mnt/Datathon"
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

SALES_CSV  = os.path.join(BASE, "sales.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
SUB_OUT    = os.path.join(OUT,  "submission.csv")

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

# 1. Load ---------------------------------------------------------------
sales  = pd.read_csv(SALES_CSV, parse_dates=["Date"])
sample = pd.read_csv(SAMPLE_SUB, parse_dates=["Date"])
spine  = pd.DataFrame({"Date": pd.date_range(sales.Date.min(), TEST_END, freq="D")})
df     = spine.merge(sales, on="Date", how="left")
print(f"[spine] {len(df):,} rows")

# 2. Feature engineering (same as v2) ------------------------------------
def build_features(df):
    d = df.copy().sort_values("Date").reset_index(drop=True)
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
    for k in range(1, 6):
        d[f"fa_sin_{k}"] = np.sin(2*np.pi*k*d.doy.values/365.25)
        d[f"fa_cos_{k}"] = np.cos(2*np.pi*k*d.doy.values/365.25)
    for k in range(1, 4):
        d[f"fw_sin_{k}"] = np.sin(2*np.pi*k*d.dow.values/7)
        d[f"fw_cos_{k}"] = np.cos(2*np.pi*k*d.dow.values/7)
    for k in range(1, 3):
        d[f"fm_sin_{k}"] = np.sin(2*np.pi*k*d.day/30.5)
        d[f"fm_cos_{k}"] = np.cos(2*np.pi*k*d.day/30.5)
    for lag in [365, 548, 700, 730, 1095, 1460]:
        d[f"Revenue_lag_{lag}"] = d["Revenue"].shift(lag)
        d[f"COGS_lag_{lag}"]    = d["COGS"].shift(lag)
    for col in ["Revenue", "COGS"]:
        grp = d.groupby(["month", "day"])[col]
        d[f"{col}_seas_expand_mean"] = grp.expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
        d[f"{col}_seas_roll3_mean"]  = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    for col in ["Revenue", "COGS"]:
        roll = d[col].rolling(365, min_periods=180).mean()
        d[f"{col}_level_548"] = roll.shift(548)
        d[f"{col}_level_730"] = roll.shift(730)
        d[f"{col}_level_913"] = roll.shift(913)
    d["rev_yoy_ratio"]  = d["Revenue_level_548"] / d["Revenue_level_913"]
    d["cogs_yoy_ratio"] = d["COGS_level_548"]    / d["COGS_level_913"]
    tet = pd.Series(d.Date.map(lambda x: TET_DATES.get(x.year, pd.NaT)))
    diff = (d.Date - tet).dt.days
    d["tet_offset"]    = diff.clip(-30, 30).fillna(99).astype(int)
    d["tet_is_window"] = (diff.abs() <= 10).fillna(False).astype(int)
    d["tet_pre"]  = ((diff >= -10) & (diff < 0)).fillna(False).astype(int)
    d["tet_post"] = ((diff > 0) & (diff <= 10)).fillna(False).astype(int)
    d["tet_day"]  = (diff == 0).fillna(False).astype(int)
    return d

feat = build_features(df)
print(f"[features] shape={feat.shape}")

BASE_FEATS = [
    "year","month","day","dow","doy","woy","is_weekend","is_month_start","is_month_end",
    "is_quarter_start","is_quarter_end","t","t2",
    *[f"fa_sin_{k}" for k in range(1,6)], *[f"fa_cos_{k}" for k in range(1,6)],
    *[f"fw_sin_{k}" for k in range(1,4)], *[f"fw_cos_{k}" for k in range(1,4)],
    *[f"fm_sin_{k}" for k in range(1,3)], *[f"fm_cos_{k}" for k in range(1,3)],
    "tet_offset","tet_is_window","tet_pre","tet_post","tet_day",
]
REV_FEATS = BASE_FEATS + [
    "Revenue_lag_365","Revenue_lag_548","Revenue_lag_700","Revenue_lag_730","Revenue_lag_1095","Revenue_lag_1460",
    "Revenue_seas_expand_mean","Revenue_seas_roll3_mean","COGS_seas_expand_mean","COGS_seas_roll3_mean",
    "Revenue_level_548","Revenue_level_730","Revenue_level_913","rev_yoy_ratio",
]
COGS_FEATS = BASE_FEATS + [
    "COGS_lag_365","COGS_lag_548","COGS_lag_700","COGS_lag_730","COGS_lag_1095","COGS_lag_1460",
    "COGS_seas_expand_mean","COGS_seas_roll3_mean","Revenue_seas_expand_mean","Revenue_seas_roll3_mean",
    "COGS_level_548","COGS_level_730","COGS_level_913","cogs_yoy_ratio",
]

LGB_PARAMS = dict(n_estimators=4000, learning_rate=0.02, num_leaves=47,
                  max_depth=-1, min_child_samples=25, feature_fraction=0.88,
                  bagging_fraction=0.9, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.2,
                  objective="regression", metric="rmse", random_state=SEED, verbose=-1)
CAT_PARAMS = dict(iterations=4000, learning_rate=0.025, depth=7, l2_leaf_reg=3.0,
                  random_seed=SEED, loss_function="RMSE", verbose=False, allow_writing_files=False)

def swt(dates, half_life_years):
    years_ago = (dates.max() - dates).dt.days / 365.25
    if half_life_years is None:
        return np.ones(len(dates))
    return (0.5 ** (years_ago / half_life_years)).values

def fit_lgb(X, y, w, X_val=None, y_val=None):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    if X_val is not None:
        m.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m.fit(X, y, sample_weight=w)
    return m

def fit_cat(X, y, w, X_val=None, y_val=None, params=None):
    p = dict(params or CAT_PARAMS)
    m = CatBoostRegressor(**p)
    if X_val is not None:
        m.fit(X, y, sample_weight=w, eval_set=(X_val, y_val),
              early_stopping_rounds=150, verbose=False)
    else:
        m.fit(X, y, sample_weight=w, verbose=False)
    return m

def mtx(y_true, y_pred):
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                R2=r2_score(y_true, y_pred))

# 3. Prophet wrapper -----------------------------------------------------
from prophet import Prophet

def prophet_forecast(sales_df, future_dates, target):
    """Fit Prophet on (Date, target) and produce predictions for future_dates."""
    dfp = sales_df[["Date", target]].rename(columns={"Date": "ds", target: "y"})
    m = Prophet(
        growth="linear",
        yearly_seasonality=20,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.08,   # allow flexible trend
        seasonality_prior_scale=10.0,
        holidays_prior_scale=5.0,
        n_changepoints=25,
    )
    # Add Tet as custom "holiday"-like event
    tet_df = pd.DataFrame({
        "holiday": "tet",
        "ds": list(TET_DATES.values()),
        "lower_window": -10, "upper_window": 10,
    })
    m.holidays = tet_df
    m.fit(dfp)
    future = pd.DataFrame({"ds": future_dates})
    fc = m.predict(future)
    out = fc[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": f"{target}_prophet"})
    return out

# 4. Validation ----------------------------------------------------------
train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START)  & (feat.Date <= VAL_END)
print("[validation] train:", train_mask.sum(), " val:", val_mask.sum())

def run_val(target, feats):
    X_tr = feat.loc[train_mask, feats].copy()
    y_tr = np.log1p(feat.loc[train_mask, target].values)
    X_va = feat.loc[val_mask, feats].copy()
    y_va_raw = feat.loc[val_mask, target].values
    y_va = np.log1p(y_va_raw)

    # Four models
    w_2y  = swt(feat.loc[train_mask, "Date"], 2.0)
    w_6y  = swt(feat.loc[train_mask, "Date"], 6.0)
    w_eq  = swt(feat.loc[train_mask, "Date"], None)

    lgbm     = fit_lgb(X_tr, y_tr, w_2y, X_va, y_va)
    cat_fast = fit_cat(X_tr, y_tr, w_2y, X_va, y_va)
    cat_slow = fit_cat(X_tr, y_tr, w_6y, X_va, y_va)
    cat_eq   = fit_cat(X_tr, y_tr, w_eq, X_va, y_va)

    p_lgbm = np.expm1(lgbm.predict(X_va))
    p_cf   = np.expm1(cat_fast.predict(X_va))
    p_cs   = np.expm1(cat_slow.predict(X_va))
    p_ce   = np.expm1(cat_eq.predict(X_va))

    # Prophet — fit on train portion only
    sales_train = sales[sales.Date < VAL_START].copy()
    val_dates = feat.loc[val_mask, "Date"].values
    pv = prophet_forecast(sales_train, val_dates, target)
    p_pr = pv[f"{target}_prophet"].values
    p_pr = np.clip(p_pr, 1.0, None)  # enforce positivity

    # Optimize blend weights (5-way)
    preds = {"lgbm": p_lgbm, "cat_fast": p_cf, "cat_slow": p_cs, "cat_eq": p_ce, "prophet": p_pr}
    # Nonneg least squares solve in log-space... we'll just do a constrained grid.
    # Try all compositions summing to 1 with step 0.1 and max 3 non-zero components.
    from itertools import product
    keys = list(preds.keys())
    best = (1e18, None)
    for a,b,c,d,e in product(np.arange(0, 1.01, 0.1), repeat=5):
        if abs(a+b+c+d+e - 1.0) > 1e-6: continue
        p = a*preds["lgbm"] + b*preds["cat_fast"] + c*preds["cat_slow"] + d*preds["cat_eq"] + e*preds["prophet"]
        r = np.sqrt(mean_squared_error(y_va_raw, p))
        if r < best[0]:
            best = (r, (round(a,1), round(b,1), round(c,1), round(d,1), round(e,1)))
    print(f"   [{target}] 5-way blend weights (lgbm, cat_fast, cat_slow, cat_eq, prophet) = {best[1]}  RMSE={best[0]:.0f}")

    w_opt = best[1]
    p_bld = (w_opt[0]*p_lgbm + w_opt[1]*p_cf + w_opt[2]*p_cs + w_opt[3]*p_ce + w_opt[4]*p_pr)

    rows = [{"target": target, "model": "lgbm",     **mtx(y_va_raw, p_lgbm)},
            {"target": target, "model": "cat_fast", **mtx(y_va_raw, p_cf)},
            {"target": target, "model": "cat_slow", **mtx(y_va_raw, p_cs)},
            {"target": target, "model": "cat_eq",   **mtx(y_va_raw, p_ce)},
            {"target": target, "model": "prophet",  **mtx(y_va_raw, p_pr)},
            {"target": target, "model": "blend",    **mtx(y_va_raw, p_bld), "w": str(w_opt)}]
    return rows, (lgbm, cat_fast, cat_slow, cat_eq), p_pr, w_opt, p_bld

print("\n--- VALIDATION: Revenue ---")
rev_rows, rev_models, rev_val_prophet, rev_w, rev_pred_va = run_val("Revenue", REV_FEATS)
print("--- VALIDATION: COGS ---")
cogs_rows, cogs_models, cogs_val_prophet, cogs_w, cogs_pred_va = run_val("COGS", COGS_FEATS)

val_df = pd.DataFrame(rev_rows + cogs_rows)
print("\n=== Validation metrics ===")
print(val_df.to_string(index=False))
val_df.to_csv(os.path.join(OUT, "phase7_validation_metrics.csv"), index=False)

# 5. Final production fit on 2013-01-01 -> 2022-12-31 --------------------
full_train_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)
test_mask       = (feat.Date >= TEST_START)  & (feat.Date <= TEST_END)
print(f"\n[final train] rows={full_train_mask.sum():,}   [test] rows={test_mask.sum():,}")

def fit_final(target, feats, val_models):
    lgbm_v, cf_v, cs_v, ce_v = val_models
    X = feat.loc[full_train_mask, feats].copy()
    y = np.log1p(feat.loc[full_train_mask, target].values)
    w2  = swt(feat.loc[full_train_mask, "Date"], 2.0)
    w6  = swt(feat.loc[full_train_mask, "Date"], 6.0)
    weq = swt(feat.loc[full_train_mask, "Date"], None)

    def fit_n(Model, params, n_iter_key, best_iter, w):
        p = dict(params); p[n_iter_key] = int(best_iter * 1.1)
        m = Model(**p); m.fit(X, y, sample_weight=w, verbose=False) if Model is CatBoostRegressor else m.fit(X, y, sample_weight=w)
        return m

    bi_l  = lgbm_v.best_iteration_ or 2000
    bi_cf = cf_v.get_best_iteration() or 2000
    bi_cs = cs_v.get_best_iteration() or 2000
    bi_ce = ce_v.get_best_iteration() or 2000

    lp = dict(LGB_PARAMS); lp["n_estimators"] = int(bi_l * 1.1)
    cp1 = dict(CAT_PARAMS); cp1["iterations"] = int(bi_cf * 1.1)
    cp2 = dict(CAT_PARAMS); cp2["iterations"] = int(bi_cs * 1.1)
    cp3 = dict(CAT_PARAMS); cp3["iterations"] = int(bi_ce * 1.1)

    lgbm     = lgb.LGBMRegressor(**lp);     lgbm.fit(X, y, sample_weight=w2)
    cat_fast = CatBoostRegressor(**cp1);    cat_fast.fit(X, y, sample_weight=w2, verbose=False)
    cat_slow = CatBoostRegressor(**cp2);    cat_slow.fit(X, y, sample_weight=w6, verbose=False)
    cat_eq   = CatBoostRegressor(**cp3);    cat_eq.fit(X, y, sample_weight=weq, verbose=False)
    return lgbm, cat_fast, cat_slow, cat_eq

print("\n[fit Revenue final]")
rev_final = fit_final("Revenue", REV_FEATS, rev_models)
print("[fit COGS final]")
cogs_final = fit_final("COGS", COGS_FEATS, cogs_models)

# Prophet final — fit on full train
test_dates = sample.Date.values
print("[fit Prophet Revenue final]")
rev_pr_final = prophet_forecast(sales, test_dates, "Revenue")
print("[fit Prophet COGS final]")
cogs_pr_final = prophet_forecast(sales, test_dates, "COGS")

# 6. Recursive prediction for the 4 tree models, then blend with Prophet -
pred_frame = feat.copy()
test_idx = pred_frame.index[pred_frame.Date.between(TEST_START, TEST_END)]

pred_rev, pred_cogs = {}, {}
rev_pr = rev_pr_final.set_index("Date")["Revenue_prophet"].to_dict()
cogs_pr = cogs_pr_final.set_index("Date")["COGS_prophet"].to_dict()
lgbm_r, cf_r, cs_r, ce_r = rev_final
lgbm_c, cf_c, cs_c, ce_c = cogs_final
rw = rev_w; cw = cogs_w

for i in test_idx:
    date = pred_frame.at[i, "Date"]
    lag_date = date - pd.Timedelta(days=365)
    if lag_date >= TEST_START:
        pred_frame.at[i, "Revenue_lag_365"] = pred_rev.get(lag_date, np.nan)
        pred_frame.at[i, "COGS_lag_365"]    = pred_cogs.get(lag_date, np.nan)
    X_r = pred_frame.loc[[i], REV_FEATS]
    X_c = pred_frame.loc[[i], COGS_FEATS]

    p_l_r  = np.expm1(lgbm_r.predict(X_r)[0])
    p_cf_r = np.expm1(cf_r.predict(X_r)[0])
    p_cs_r = np.expm1(cs_r.predict(X_r)[0])
    p_ce_r = np.expm1(ce_r.predict(X_r)[0])
    p_pr_r = max(1.0, float(rev_pr.get(date, np.nan)))

    p_l_c  = np.expm1(lgbm_c.predict(X_c)[0])
    p_cf_c = np.expm1(cf_c.predict(X_c)[0])
    p_cs_c = np.expm1(cs_c.predict(X_c)[0])
    p_ce_c = np.expm1(ce_c.predict(X_c)[0])
    p_pr_c = max(1.0, float(cogs_pr.get(date, np.nan)))

    p_rev  = rw[0]*p_l_r + rw[1]*p_cf_r + rw[2]*p_cs_r + rw[3]*p_ce_r + rw[4]*p_pr_r
    p_cogs = cw[0]*p_l_c + cw[1]*p_cf_c + cw[2]*p_cs_c + cw[3]*p_ce_c + cw[4]*p_pr_c
    pred_rev[date]  = float(p_rev)
    pred_cogs[date] = float(p_cogs)

# 7. Build submission ----------------------------------------------------
pred_df = pd.DataFrame({
    "Date":    list(pred_rev.keys()),
    "Revenue": list(pred_rev.values()),
    "COGS":    list(pred_cogs.values()),
}).sort_values("Date").reset_index(drop=True)

sub = sample[["Date"]].merge(pred_df, on="Date", how="left")
assert sub.Revenue.notna().all(); assert sub.COGS.notna().all()
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub.to_csv(SUB_OUT, index=False)
print(f"\n[submission] saved  rows={len(sub)}  file={SUB_OUT}")
print(sub.head()); print(sub.tail())
print("Sum Revenue:", sub.Revenue.sum(), "  Sum COGS:", sub.COGS.sum())
print("Mean Revenue:", sub.Revenue.mean().round(0), "  Mean COGS:", sub.COGS.mean().round(0))

# 8. Diagnostic ---------------------------------------------------------
diag = pd.DataFrame({
    "Date":           feat.loc[val_mask, "Date"].values,
    "Revenue_actual": feat.loc[val_mask, "Revenue"].values,
    "Revenue_pred":   rev_pred_va,
    "COGS_actual":    feat.loc[val_mask, "COGS"].values,
    "COGS_pred":      cogs_pred_va,
})
diag.to_csv(os.path.join(OUT, "phase7_validation_diagnostic.csv"), index=False)
print("\n=== DONE. ===")
