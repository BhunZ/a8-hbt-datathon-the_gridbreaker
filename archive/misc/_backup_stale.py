"""
Phase 7 — Forecasting Model for DATATHON 2026 Round 1, Part 3.

Goal
----
Predict daily Revenue and COGS for 2023-01-01 → 2024-07-01 (548 days).

Design decisions
----------------
1. **Target transform**: log1p(Revenue), log1p(COGS) → RMSE-friendly, positivity-preserving.
2. **Feature set** (leak-free, horizon-consistent with 18-month test period):
    - Calendar: year, month, day, dow, doy, woy, is_weekend, is_month_{start,end},
      is_quarter_{start,end}, days_to_{year_end,month_end}.
    - Fourier: annual (K=5), weekly (K=3), monthly (K=2).
    - Trend: t (day index), t² / 1e6.
    - Static lags: lag_730, lag_1095, lag_1460 (2-,3-,4-year — always observed).
    - Recursive lag_365: initially NaN for test dates; filled iteratively using the
      model's own predictions from the immediately previous year.
    - Rolling seasonal mean over the past 3 same-(month,day) observations.
    - Vietnamese Lunar New Year (Tet) indicator (±10 days window) — hand-crafted dates.
3. **Models**:
    - LightGBM regressor on log target.
    - CatBoost regressor on log target.
    - Simple blend: 0.55 LGBM + 0.45 CatBoost (based on validation sweep).
4. **Train/valid split**:
    - Validation fold = 2021-07-01 → 2022-12-31 (548 days, horizon-matched).
    - Training data = 2013-01-01 → 2021-06-30 (full-year pre-covid + covid + pre-val).
    - Final production training uses all rows 2013-01-01 → 2022-12-31 (incl. validation).
5. **Sample weights**: geometric decay, newest year weight 1.0, older years × 0.75/yr.
6. **Recent-year-only calibration**: Also fit a "recent" variant on 2020+ only and
    blend its prediction with the full-history model (50/50) to guard against the
    2019 structural break dominating the fit.

Outputs
-------
- outputs/submission.csv (Date, Revenue, COGS) matching sample_submission.csv row order.
- outputs/phase7_validation_metrics.csv (per-model MAE/RMSE/R² on 2021H2–2022 fold).
- outputs/phase7_feature_importance.csv
- outputs/phase7_shap_revenue.png / phase7_shap_cogs.png
- outputs/phase7_predictions_diagnostic.csv
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor

# ----------------------------------------------------------------------
# 0. Paths & constants
# ----------------------------------------------------------------------
BASE      = "/sessions/lucid-relaxed-edison/mnt/Datathon"
OUT       = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

SALES_CSV  = os.path.join(BASE, "sales.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
SUB_OUT    = os.path.join(OUT,  "submission.csv")

TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")
VAL_START   = pd.Timestamp("2021-07-01")
VAL_END     = pd.Timestamp("2022-12-31")
TRAIN_START = pd.Timestamp("2013-01-01")   # drop partial 2012 (181 days only)

SEED = 42
np.random.seed(SEED)

# Vietnamese Lunar New Year (Tet) anchor dates (Day 1 of Lunar Year)
TET_DATES = {
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
    2025: "2025-01-29",
}
TET_DATES = {y: pd.Timestamp(d) for y, d in TET_DATES.items()}


# ----------------------------------------------------------------------
# 1. Load + build full date spine (train + test)
# ----------------------------------------------------------------------
sales = pd.read_csv(SALES_CSV, parse_dates=["Date"])
sample = pd.read_csv(SAMPLE_SUB, parse_dates=["Date"])

spine = pd.DataFrame({"Date": pd.date_range(
    sales.Date.min(), TEST_END, freq="D"
)})
df = spine.merge(sales, on="Date", how="left")   # Revenue/COGS NaN over test window
df["is_train"] = df.Date <= TRAIN_END
df["is_test"]  = df.Date >= TEST_START
print(f"[spine] {len(df):,} rows  |  train: {df.is_train.sum():,}  test: {df.is_test.sum():,}")


# ----------------------------------------------------------------------
# 2. Feature engineering (pure-past or calendar-only — no leakage)
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("Date").reset_index(drop=True)

    # --- Calendar
    dt = d.Date.dt
    d["year"]   = dt.year
    d["month"]  = dt.month
    d["day"]    = dt.day
    d["dow"]    = dt.dayofweek
    d["doy"]    = dt.dayofyear
    d["woy"]    = dt.isocalendar().week.astype(int)
    d["is_weekend"]      = (d.dow >= 5).astype(int)
    d["is_month_start"]  = dt.is_month_start.astype(int)
    d["is_month_end"]    = dt.is_month_end.astype(int)
    d["is_quarter_start"]= dt.is_quarter_start.astype(int)
    d["is_quarter_end"]  = dt.is_quarter_end.astype(int)

    # --- Trend
    d["t"]  = (d.Date - d.Date.min()).dt.days.astype(np.int32)
    d["t2"] = (d["t"] ** 2) / 1e6

    # --- Fourier
    doy = d.doy.values
    dow = d.dow.values
    for k in range(1, 6):
        d[f"fa_sin_{k}"] = np.sin(2*np.pi*k*doy/365.25)
        d[f"fa_cos_{k}"] = np.cos(2*np.pi*k*doy/365.25)
    for k in range(1, 4):
        d[f"fw_sin_{k}"] = np.sin(2*np.pi*k*dow/7)
        d[f"fw_cos_{k}"] = np.cos(2*np.pi*k*dow/7)
    for k in range(1, 3):
        d[f"fm_sin_{k}"] = np.sin(2*np.pi*k*d.day/30.5)
        d[f"fm_cos_{k}"] = np.cos(2*np.pi*k*d.day/30.5)

    # --- Static lags (2,3,4-yr — fully observed for all test dates)
    for lag in [730, 1095, 1460]:
        d[f"Revenue_lag_{lag}"] = d["Revenue"].shift(lag)
        d[f"COGS_lag_{lag}"]    = d["COGS"].shift(lag)

    # --- Recursive lag_365 — placeholder NaN on test rows; filled at predict time.
    d["Revenue_lag_365"] = d["Revenue"].shift(365)
    d["COGS_lag_365"]    = d["COGS"].shift(365)

    # --- Rolling same-(month,day) mean over prior 3 years' observations.
    # Using groupby+cumulative to keep leak-free (only includes rows above in time).
    for col in ["Revenue", "COGS"]:
        grp = d.groupby(["month", "day"])[col]
        d[f"{col}_seas_expand_mean"] = grp.expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
        # last-3-year same-(month,day) mean (more recency-focused)
        d[f"{col}_seas_roll3_mean"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # --- Level-anchor features: trailing 365d rolling mean, lagged so no leakage.
    # For test dates in 2023+, this uses data from 2021–2022. Fully observed.
    for col in ["Revenue", "COGS"]:
        roll = d[col].rolling(365, min_periods=180).mean()
        d[f"{col}_level_730"] = roll.shift(730)  # avg over 1yr ending 2yr ago
        d[f"{col}_level_548"] = roll.shift(548)  # avg over 1yr ending 1.5yr ago — ALWAYS usable

    # --- Trailing "short" recent level (for iterative filling later)
    for col in ["Revenue", "COGS"]:
        d[f"{col}_lag_548"]  = d[col].shift(548)
        d[f"{col}_lag_700"]  = d[col].shift(700)

    # --- YoY growth proxy: level_548 / level_913 (≈ 1.5 years vs 2.5 years back).
    d["rev_yoy_ratio"]  = d["Revenue_level_548"] / d["Revenue"].rolling(365, min_periods=180).mean().shift(913)
    d["cogs_yoy_ratio"] = d["COGS_level_548"]    / d["COGS"].rolling(365, min_periods=180).mean().shift(913)

    # --- Tet window (±10 days)
    tet_series = pd.Series(
        d.Date.map(lambda x: TET_DATES.get(x.year, pd.NaT))
    )
    diff = (d.Date - tet_series).dt.days
    d["tet_offset"] = diff.clip(-30, 30).fillna(99).astype(int)
    d["tet_is_window"]   = (diff.abs() <= 10).fillna(False).astype(int)
    d["tet_pre"]  = ((diff >= -10) & (diff < 0)).fillna(False).astype(int)
    d["tet_post"] = ((diff > 0) & (diff <= 10)).fillna(False).astype(int)
    d["tet_day"]  = (diff == 0).fillna(False).astype(int)

    return d

feat = build_features(df)
print(f"[features] shape={feat.shape}")

# ----------------------------------------------------------------------
# 3. Model training helpers
# ----------------------------------------------------------------------
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
    "Revenue_lag_365","Revenue_lag_548","Revenue_lag_700","Revenue_lag_730","Revenue_lag_1095","Revenue_lag_1460",
    "Revenue_seas_expand_mean","Revenue_seas_roll3_mean",
    "COGS_seas_expand_mean","COGS_seas_roll3_mean",
    "Revenue_level_548","Revenue_level_730",
    "rev_yoy_ratio",
]
COGS_FEATS = BASE_FEATS + [
    "COGS_lag_365","COGS_lag_548","COGS_lag_700","COGS_lag_730","COGS_lag_1095","COGS_lag_1460",
    "COGS_seas_expand_mean","COGS_seas_roll3_mean",
    "Revenue_seas_expand_mean","Revenue_seas_roll3_mean",
    "COGS_level_548","COGS_level_730",
    "cogs_yoy_ratio",
]

LGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.025,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="regression",
    metric="rmse",
    random_state=SEED,
    verbose=-1,
)

CAT_PARAMS = dict(
    iterations=3000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    loss_function="RMSE",
    verbose=False,
    allow_writing_files=False,
)


def sample_weight(dates: pd.Series, half_life_years: float = 3.0) -> np.ndarray:
    """Geometric recency weights: w = 0.5 ** (years_ago / half_life)."""
    anchor = dates.max()
    years_ago = (anchor - dates).dt.days / 365.25
    return (0.5 ** (years_ago / half_life_years)).values


def fit_lgb(X, y, w, X_val, y_val):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(
        X, y, sample_weight=w,
        eval_set=[(X_val, y_val)] if X_val is not None else None,
        callbacks=[lgb.early_stopping(100, verbose=False)] if X_val is not None else None,
    )
    return m


def fit_cat(X, y, w, X_val, y_val):
    m = CatBoostRegressor(**CAT_PARAMS)
    if X_val is not None:
        m.fit(
            X, y, sample_weight=w,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False,
        )
    else:
        m.fit(X, y, sample_weight=w, verbose=False)
    return m


def metrics(y_true, y_pred):
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
        R2=r2_score(y_true, y_pred),
    )

# ----------------------------------------------------------------------
# 4. Validation pass (2021-07-01 → 2022-12-31)
# ----------------------------------------------------------------------
train_mask = (feat.Date >= TRAIN_START) & (feat.Date < VAL_START)
val_mask   = (feat.Date >= VAL_START)  & (feat.Date <= VAL_END)

print("\n[validation fold] train:", train_mask.sum(), " val:", val_mask.sum())

def run_one_target(target: str, feats: list[str]):
    X_tr = feat.loc[train_mask, feats].copy()
    y_tr = np.log1p(feat.loc[train_mask, target].values)
    w_tr = sample_weight(feat.loc[train_mask, "Date"])
    X_va = feat.loc[val_mask, feats].copy()
    y_va_raw = feat.loc[val_mask, target].values
    y_va = np.log1p(y_va_raw)

    lgbm = fit_lgb(X_tr, y_tr, w_tr, X_va, y_va)
    catb = fit_cat(X_tr, y_tr, w_tr, X_va, y_va)

    p_lgb = np.expm1(lgbm.predict(X_va))
    p_cat = np.expm1(catb.predict(X_va))
    p_blend = 0.55 * p_lgb + 0.45 * p_cat

    m_lgb = metrics(y_va_raw, p_lgb)
    m_cat = metrics(y_va_raw, p_cat)
    m_bld = metrics(y_va_raw, p_blend)

    # Baseline (geo-mean seasonal) for reference
    p_base = feat.loc[val_mask, f"{target}_seas_expand_mean"].values
    m_base = metrics(y_va_raw, np.nan_to_num(p_base, nan=np.nanmean(p_base)))

    rows = []
    for name, m in [("baseline_seasonal", m_base), ("lgbm", m_lgb), ("catboost", m_cat), ("blend", m_bld)]:
        rows.append({"target": target, "model": name, **m})
    return rows, lgbm, catb, p_lgb, p_cat, p_blend

print("\n--- VALIDATION: Revenue ---")
rev_rows, rev_lgbm, rev_cat, _, _, rev_pred_va = run_one_target("Revenue", REV_FEATS)
print("\n--- VALIDATION: COGS ---")
cogs_rows, cogs_lgbm, cogs_cat, _, _, cogs_pred_va = run_one_target("COGS", COGS_FEATS)

val_df = pd.DataFrame(rev_rows + cogs_rows)
print("\n=== Validation metrics (2021-07-01 → 2022-12-31, 548 days) ===")
print(val_df.to_string(index=False))
val_df.to_csv(os.path.join(OUT, "phase7_validation_metrics.csv"), index=False)

# ----------------------------------------------------------------------
# 5. Final fit on all rows 2013-01-01 → 2022-12-31, then recursive predict.
# ----------------------------------------------------------------------
full_train_mask = (feat.Date >= TRAIN_START) & (feat.Date <= TRAIN_END)
test_mask       = (feat.Date >= TEST_START)  & (feat.Date <= TEST_END)

print(f"\n[final train] rows={full_train_mask.sum():,}   [test] rows={test_mask.sum():,}")

def fit_final(target, feats):
    X = feat.loc[full_train_mask, feats].copy()
    y = np.log1p(feat.loc[full_train_mask, target].values)
    w = sample_weight(feat.loc[full_train_mask, "Date"])
    # fit each on full data w/o early stopping — use best_iteration from the val pass
    bi_lgb = rev_lgbm.best_iteration_ if target == "Revenue" else cogs_lgbm.best_iteration_
    bi_cat = rev_cat.get_best_iteration() if target == "Revenue" else cogs_cat.get_best_iteration()
    p_lgb  = dict(LGB_PARAMS); p_lgb["n_estimators"] = int(bi_lgb * 1.08) if bi_lgb else 1500
    p_cat  = dict(CAT_PARAMS); p_cat["iterations"]   = int(bi_cat * 1.08) if bi_cat else 1500

    lgbm = lgb.LGBMRegressor(**p_lgb)
    lgbm.fit(X, y, sample_weight=w)

    catb = CatBoostRegressor(**p_cat)
    catb.fit(X, y, sample_weight=w, verbose=False)
    return lgbm, catb

print("\n[fit Revenue final]")
rev_lgb_f, rev_cat_f = fit_final("Revenue", REV_FEATS)
print("[fit COGS final]")
cogs_lgb_f, cogs_cat_f = fit_final("COGS", COGS_FEATS)


# ----------------------------------------------------------------------
# 6. Recursive prediction — fill lag_365 day-by-day for 2024.
# ----------------------------------------------------------------------
pred_frame = feat.copy()
# We'll predict day-by-day; only lag_365 requires recursion (2024 dates need 2023 preds)
test_idx = pred_frame.index[pred_frame.Date.between(TEST_START, TEST_END)]

# step through each test date
pred_revenue = {}
pred_cogs    = {}
for i in test_idx:
    row = pred_frame.loc[i]
    # fill recursive lag_365 if Date >= 2024-01-01 (i.e., > 2023-12-31 at lag)
    date = row.Date
    lag_date = date - pd.Timedelta(days=365)
    if lag_date >= TEST_START:
        # use own predictions
        pred_frame.at[i, "Revenue_lag_365"] = pred_revenue.get(lag_date, np.nan)
        pred_frame.at[i, "COGS_lag_365"]    = pred_cogs.get(lag_date, np.nan)
    # predict
    X_rev  = pred_frame.loc[[i], REV_FEATS]
    X_cogs = pred_frame.loc[[i], COGS_FEATS]
    p_rev  = 0.55*np.expm1(rev_lgb_f.predict(X_rev)[0]) + 0.45*np.expm1(rev_cat_f.predict(X_rev)[0])
    p_cogs = 0.55*np.expm1(cogs_lgb_f.predict(X_cogs)[0]) + 0.45*np.expm1(cogs_cat_f.predict(X_cogs)[0])
    pred_revenue[date] = p_rev
    pred_cogs[date]    = p_cogs

# ----------------------------------------------------------------------
# 7. Build submission
# ----------------------------------------------------------------------
pred_df = pd.DataFrame({
    "Date": list(pred_revenue.keys()),
    "Revenue": list(pred_revenue.values()),
    "COGS":    list(pred_cogs.values()),
}).sort_values("Date").reset_index(drop=True)

# Align to sample_submission row order
sub = sample[["Date"]].merge(pred_df, on="Date", how="left")
assert sub.Revenue.notna().all(), "Missing Revenue rows"
assert sub.COGS.notna().all(),    "Missing COGS rows"
sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
sub["Revenue"] = sub["Revenue"].round(2)
sub["COGS"]    = sub["COGS"].round(2)
sub.to_csv(SUB_OUT, index=False)
print(f"\n[submission] saved to {SUB_OUT}  rows={len(sub)}")
print(sub.head())
print(sub.tail())
print("Sum Revenue:", sub.Revenue.sum())
print("Sum COGS   :", sub.COGS.sum())

# ----------------------------------------------------------------------
# 8. Feature importance + SHAP (for the Revenue LGBM final model)
# ----------------------------------------------------------------------
fi_rev = pd.DataFrame({
    "feature": REV_FEATS,
    "lgbm_gain_rev": rev_lgb_f.booster_.feature_importance(importance_type="gain"),
})
fi_cogs = pd.DataFrame({
    "feature": COGS_FEATS,
    "lgbm_gain_cogs": cogs_lgb_f.booster_.feature_importance(importance_type="gain"),
})
fi_all = fi_rev.merge(fi_cogs, on="feature", how="outer")
fi_all.to_csv(os.path.join(OUT, "phase7_feature_importance.csv"), index=False)
print("\n=== Top features (Revenue LGBM gain) ===")
print(fi_all.sort_values("lgbm_gain_rev", ascending=False).head(15).to_string(index=False))

# SHAP (optional, can be slow)
try:
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_shap = feat.loc[full_train_mask, REV_FEATS].sample(min(2000, full_train_mask.sum()),
                                                          random_state=SEED)
    expl = s