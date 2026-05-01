"""Phase 7 v2 — identical to phase7_model.py; re-written to force filesystem sync."""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

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
RECENT_START = pd.Timestamp("2019-01-01")

SEED = 42
np.random.seed(SEED)

TET_DATES = {y: pd.Timestamp(d) for y, d in {
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
    2025: "2025-01-29",
}.items()}

sales  = pd.read_csv(SALES_CSV,  parse_dates=["Date"])
sample = pd.read_csv(SAMPLE_SUB, parse_dates=["Date"])
spine = pd.DataFrame({"Date": pd.date_range(sales.Date.min(), TEST_END, freq="D")})
df = spine.merge(sales, on="Date", how="left")
df["is_train"] = df.Date <= TRAIN_END
df["is_test"]  = df.Date >= TEST_START
print(f"[spine] {len(df):,} rows  |  train: {df.is_train.sum():,}  test: {df.is_test.sum():,}")


def build_features(df):
    d = df.copy().sort_values("Date").reset_index(drop=True)
    dt = d.Date.dt
    d["year"]   = dt.year
    d["month"]  = dt.month
    d["day"]    = dt.day
    d["dow"]    = dt.dayofweek
    d["doy"]    = dt.dayofyear
    d["woy"]    = dt.isocalendar().week.astype(int)
    d["is_weekend"]       = (d.dow >= 5).astype(int)
    d["is_month_start"]   = dt.is_month_start.astype(int)
    d["is_month_end"]     = dt.is_month_end.astype(int)
    d["is_quarter_start"] = dt.is_quarter_start.astype(int)
    d["is_quarter_end"]   = dt.is_quarter_end.astype(int)
    d["t"]  = (d.Date - d.Date.min()).dt.days.astype(np.int32)
    d["t2"] = (d["t"] ** 2) / 1e6

    doy = d.doy.values; dow = d.dow.values
    for k in range(1, 6):
        d[f"fa_sin_{k}"] = np.sin(2*np.pi*k*doy/365.25)
        d[f"fa_cos_{k}"] = np.cos(2*np.pi*k*doy/365.25)
    for k in range(1, 4):
        d[f"fw_sin_{k}"] = np.sin(2*np.pi*k*dow/7)
        d[f"fw_cos_{k}"] = np.cos(2*np.pi*k*dow/7)
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

    tet_series = pd.Series(d.Date.map(lambda x: TET_DATES.get(x.year, pd.NaT)))
    diff = (d.Date - tet_series).dt.days
    d["tet_offset"]    = diff.clip(-30, 30).fillna(99).astype(int)
    d["tet_is_window"] = (diff.abs() <= 10).fillna(False).astype(int)
    d["tet_pre"]  = ((diff >= -10) & (diff < 0)).fillna(False).astype(int)
    d["tet_post"] = ((diff > 0) & (diff <= 10)).fillna(False).astype(int)
    d["tet_day"]  = (diff == 0).fillna(False).astype(int)
    return d

feat = build_features(df)
print(f"[features] shape={feat.shape}")

BASE_FEATS = [
    "year","month","day","dow","doy","woy",
    "is_weekend","is_month_start","is_month_end",
    "is_quarter_start","is_quarter_end",
    "t","t2",
    *[f"fa_sin_{k}" for k in range(1,6)],
    *[f"fa_cos_{k}" for k in range(1,6)],
    *[f"fw_sin_{k}" for k in range(1,4)],
    *[f"fw_cos_{k}" for k in range(1,4)],
    *[f"fm_sin_{k}" for k in range(1,3)],
    *[f"fm_cos_{k}" for k in range(1,3)],
    "tet_offset","tet_is_window","tet_pre","tet_post","tet_day",
]
REV_FEATS = BASE_FEATS + [
    "Revenue_lag_365","Revenue_lag_548","Revenue_lag_700","Revenue_lag_730",
    "Revenue_lag_1095","Revenue_lag_1460",
    "Revenue_seas_expand_mean","Revenue_seas_roll3_mean",
    "COGS_seas_expand_mean","COGS_seas_roll3_mean",
    "Revenue_level_548","Revenue_level_730","Revenue_level_913","rev_yoy_ratio",
]
COGS_FEATS = BASE_FEATS + [
    "COGS_lag_365","COGS_lag_548","COGS_lag_700","COGS_lag_730",
    "COGS_lag_1095","COGS_lag_1460",
    "COGS_seas_expand_mean","COGS_seas_roll3_mean",
    "Revenue_seas_expand_mean","Revenue_seas_roll3_mean",
    "COGS_level_548","COGS_level_730","COGS_level_913","cogs_yoy_ratio",
]

LGB_PARAMS = dict(
    n_estimators=4000, learning_rate=0.02, num_leaves=47,
    max_depth=-1, min_child_samples=25,
    feature_fraction=0.88, bagging_fraction=0.9, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.2,
    objective="regression", metric="rmse",
    random_state=SEED, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=4000, learning_rate=0.025, depth=7,
    l2_leaf_reg=3.0, random_seed=SEED,
    loss_function="RMSE", verbose=False, allow_writing_files=False,
)

def sample_weight(dates, half_life_years=2.0):
    anchor = dates.max()
    years_ago = (anchor - dates).dt.days / 365.25
    return (0.5 ** (years_ago / half_life_years)).values

def fit_lgb(X, y, w, X_val=None, y_val=None):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    if X_val is not None:
        m.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m.fit(X, y, sample_weight=w)
    return m

def fit_cat(X, y, w, X_val=None, y_val=None):
    m = CatBoostRegressor(**CAT_PARAMS)
    if X_val is not None:
        m.fit(X, y, sample_weight=w, eval_set=(X_val, y_val),
              early_stopping_rounds=150, verbose=False)
    else:
        m.fit(X, y, sample_weight=w, verbose=False)
    return m

def mtx(y_true, y_pred):
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
        R2=r2_score(y_true, y_pred),
    )

train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START)  & (feat.Date <= VAL_END)
print("[validation fold] train:", train_mask.sum(), " val:", val_mask.sum())

def run_val(target, feats):
    X_tr = feat.loc[train_mask, feats].copy()
    y_tr = np.log1p(feat.loc[train_mask, target].values)
    w_tr = sample_weight(feat.loc[train_mask, "Date"])
    X_va = feat.loc[val_mask, feats].copy()
    y_va_raw = feat.loc[val_mask, target].values
    y_va = np.log1p(y_va_raw)

    rec_mask = train_mask & (feat.Date >= RECENT_START)
    X_rec = feat.loc[rec_mask, feats].copy()
    y_rec = np.log1p(feat.loc[rec_mask, target].values)
    w_rec = np.ones(rec_mask.sum())

    lgbm    = fit_lgb(X_tr, y_tr, w_tr, X_va, y_va)
    catb    = fit_cat(X_tr, y_tr, w_tr, X_va, y_va)
    catb_rc = fit_cat(X_rec, y_rec, w_rec, X_va, y_va)

    p_lgb  = np.expm1(lgbm.predict(X_va))
    p_cat  = np.expm1(catb.predict(X_va))
    p_catr = np.expm1(catb_rc.predict(X_va))

    best = (1e18, None)
    for w1 in np.arange(0.10, 0.41, 0.05):
        for w2 in np.arange(0, 1.01 - w1 + 1e-6, 0.05):
            w3 = 1 - w1 - w2
            if w3 < -1e-6 or w3 > 0.3: continue
            p = w1*p_lgb + w2*p_cat + w3*p_catr
            r = np.sqrt(mean_squared_error(y_va_raw, p))
            if r < best[0]:
                best = (r, (round(w1,2), round(w2,2), round(w3,2)))
    w_opt = best[1]
    p_bld = w_opt[0]*p_lgb + w_opt[1]*p_cat + w_opt[2]*p_catr

    p_base = feat.loc[val_mask, f"{target}_seas_expand_mean"].values
    p_base = np.nan_to_num(p_base, nan=np.nanmean(p_base))

    rows = []
    for n, m in [("baseline_seasonal", mtx(y_va_raw, p_base)),
                 ("lgbm_full",         mtx(y_va_raw, p_lgb)),
                 ("catboost_full",     mtx(y_va_raw, p_cat)),
                 ("catboost_recent",   mtx(y_va_raw, p_catr)),
                 ("blend",             mtx(y_va_raw, p_bld))]:
        rows.append({"target": target, "model": n, **m, "blend_w": str(w_opt) if n=="blend" else ""})
    print(f"   [{target}] optimal blend (lgbm, cat_full, cat_recent) = {w_opt}  RMSE={best[0]:.0f}")
    return rows, lgbm, catb, catb_rc, p_bld, w_opt

print("\n--- VALIDATION: Revenue ---")
rev_rows, rev_lgbm, rev_cat, rev_cat_rec, rev_pred_va, rev_w = run_val("Revenue", REV_FEATS)
print("--- VALIDATION: COGS ---")
cogs_rows, cogs_lgbm, cogs_cat, cogs_cat_rec, cogs_pred_va, cogs_w = run_val("COGS", COGS_FEATS)

val_df = pd.DataFrame(rev_rows + cogs_rows)
print("\n=== Validation metrics (549 days, 2021-07-01 -> 2022-12-31) ===")
print(val_df.to_string(index=False))
val_df.to_csv(os.path.join(OUT, "phase7_validation_metrics.csv"), index=False)

full_train_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)
test_mask       = (feat.Date >= TEST_START)  & (feat.Date <= TEST_END)
print(f"\n[final train] rows={full_train_mask.sum():,}   [test] rows={test_mask.sum():,}")

def fit_final(target, feats, lgbm_val, cat_val, cat_rec_val):
    X = feat.loc[full_train_mask, feats].copy()
    y = np.log1p(feat.loc[full_train_mask, target].values)
    w = sample_weight(feat.loc[full_train_mask, "Date"])
    bi_lgb = lgbm_val.best_iteration_ if lgbm_val.best_iteration_ else 2000
    bi_cat = cat_val.get_best_iteration() if cat_val.get_best_iteration() else 2000
    bi_rec = cat_rec_val.get_best_iteration() if cat_rec_val.get_best_iteration() else 1500
    lp = dict(LGB_PARAMS); lp["n_estimators"] = int(bi_lgb * 1.10)
    cp = dict(CAT_PARAMS); cp["iterations"]   = int(bi_cat * 1.10)
    rp = dict(CAT_PARAMS); rp["iterations"]   = int(bi_rec * 1.10)

    lgbm = lgb.LGBMRegressor(**lp); lgbm.fit(X, y, sample_weight=w)
    catb = CatBoostRegressor(**cp); catb.fit(X, y, sample_weight=w, verbose=False)

    rec_mask = full_train_mask & (feat.Date >= RECENT_START)
    X_rec = feat.loc[rec_mask, feats].copy()
    y_rec = np.log1p(feat.loc[rec_mask, target].values)
    w_rec = np.ones(rec_mask.sum())
    catb_rec = CatBoostRegressor(**rp); catb_rec.fit(X_rec, y_rec, sample_weight=w_rec, verbose=False)
    return lgbm, catb, catb_rec

print("\n[fit Revenue final]")
rev_lgb_f, rev_cat_f, rev_cat_rec_f = fit_final("Revenue", REV_FEATS, rev_lgbm, rev_cat, rev_cat_rec)
print("[fit COGS final]")
cogs_lgb_f, cogs_cat_f, cogs_cat_rec_f = fit_final("COGS", COGS_FEATS, cogs_lgbm, cogs_cat, cogs_cat_rec)

pred_frame = feat.copy()
test_idx = pred_frame.index[pred_frame.Date.between(TEST_START, TEST_END)]

pred_revenue, pred_cogs = {}, {}
rw = rev_w; cw = cogs_w
for i in test_idx:
    date = pred_frame.at[i, "Date"]
    lag_date = date - pd.Timedelta(days=365)
    if lag_date >= TEST_START:
        pred_frame.at[i, "Revenue_lag_365"] = pred_revenue.get(lag_date, np.nan)
        pred_frame.at[i, "COGS_lag_365"]    = pred_cogs.get(lag_date, np.nan)
    X_rev  = pred_frame.loc[[i], REV_FEATS]
    X_cogs = pred_frame.loc[[i], COGS_FEATS]
    p_rev  = (rw[0]*np.expm1(rev_lgb_f.predict(X_rev)[0])
              + rw[1]*np.expm1(rev_cat_f.predict(X_rev)[0])
              + rw[2]*np.expm1(rev_cat_rec_f.predict(X_rev)[0]))
    p_cogs = (cw[0]*np.expm1(cogs_lgb_f.predict(X_cogs)[0])
              + cw[1]*np.expm1(cogs_cat_f.predict(X_cogs)[0])
              + cw[2]*np.expm1(cogs_cat_rec_f.predict(X_cogs)[0]))
    pred_revenue[date] = float(p_rev)
    pred_cogs[date]    = float(p_cogs)

pred_df = pd.DataFrame({
    "Date":    list(pred_revenue.keys()),
    "Revenue": list(pred_revenue.values()),
    "COGS":    list(pred_cogs.values()),
}).sort_values("Date").reset_index(drop=True)

sub = sample[["Date"]].merge(pred_df, on="Date", how="left")
assert sub.Revenue.notna().all()
assert sub.COGS.notna().all()
sub["Date"]    = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub.to_csv(SUB_OUT, index=False)
print(f"\n[submission] saved  rows={len(sub)}  file={SUB_OUT}")
print(sub.head()); print(sub.tail())
print("Sum Revenue:", sub.Revenue.sum(), "  Sum COGS:", sub.COGS.sum())

fi_rev  = pd.DataFrame({"feature": REV_FEATS,
                        "lgbm_gain_rev": rev_lgb_f.booster_.feature_importance(importance_type="gain")})
fi_cogs = pd.DataFrame({"feature": COGS_FEATS,
                        "lgbm_gain_cogs": cogs_lgb_f.booster_.feature_importance(importance_type="gain")})
fi_all = fi_rev.merge(fi_cogs, on="feature", how="outer")
fi_all.to_csv(os.path.join(OUT, "phase7_feature_importance.csv"), index=False)
print("\n=== Top features (Revenue LGBM gain) ===")
print(fi_all.sort_values("lgbm_gain_rev", ascending=False).head(15).to_string(index=False))

try:
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Xs = feat.loc[full_train_mask, REV_FEATS].sample(min(2000, full_train_mask.sum()), random_state=SEED)
    sv = shap.TreeExplainer(rev_lgb_f).shap_values(Xs)
    plt.figure(); shap.summary_plot(sv, Xs, show=False, plot_type="bar", max_display=20)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase7_shap_revenue.png"), dpi=120, bbox_inches="tight"); plt.close()
    Xs2 = feat.loc[full_train_mask, COGS_FEATS].sample(min(2000, full_train_mask.sum()), random_state=SEED)
    sv2 = shap.TreeExplainer(cogs_lgb_f).shap_values(Xs2)
    plt.figure(); shap.summary_plot(sv2, Xs2, show=False, plot_type="bar", max_display=20)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase7_shap_cogs.png"), dpi=120, bbox_inches="tight"); plt.close()
    print("[shap] saved PNGs.")
except Exception as e:
    print(f"[shap] skipped: {e}")

diag = pd.DataFrame({
    "Date":           feat.loc[val_mask, "Date"].values,
    "Revenue_actual": feat.loc[val_mask, "Revenue"].values,
    "Revenue_pred":   rev_pred_va,
    "COGS_actual":    feat.loc[val_mask, "COGS"].values,
    "COGS_pred":      cogs_pred_va,
})
diag.to_csv(os.path.join(OUT, "phase7_validation_diagnostic.csv"), index=False)
print("\n=== DONE. ===")
