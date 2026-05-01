"""V14 — Multi-loss LightGBM ensemble blended with V10c learners.

Built on V10c\'s 23-feature set (only the 13 data files we are allowed).
Train: 2017-01-01 -> end of train window
Validation: 2022 mirror (train -> 2021-12-31, predict 2022) all 3 metrics.

Heads:
  LGBM-L1, LGBM-Q50, LGBM-Huber, LGBM-MSE, LGBM-Tweedie  (multi-loss diversity)
  RF, ET, HGB                                            (V10c smoothing prior)

Blends tested:
  equal_all       — equal weight all 8 heads
  v10c_anchored   — 50% V10c (RF+ET+HGB) + 50% LGBM heads
  loss_balanced   — 50% MAE-aware (L1, Q50, Huber) + 50% RMSE-aware (MSE, Tweedie, RF, ET, HGB)

Acceptance rule: blend must beat V10c-original on >=2 of 3 metrics on mirror.

Modes:
  python src/v14.py validate   # train heads on 2017->2021, score 2022 mirror
  python src/v14.py submit     # train on 2017->2022, predict 2023-2024, write submission

Outputs:
  data/processed/v14_validate.json   (mirror scores per head + blends)
  submissions/submission_v14.csv     (final blended)
  submissions/submission_v14_l1.csv  (LGBM-L1 only fallback)
  docs/v14_results.md
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor)

from paths import RAW, PROCESSED, SUBMISSIONS, REFERENCE, DOCS
from v13_v10c_rebaseline import build_v10c_features, FEATS as V10C_FEATS

TRAIN_START_FULL = pd.Timestamp("2017-01-01")
MIRROR_START = pd.Timestamp("2022-01-01")
MIRROR_END   = pd.Timestamp("2022-12-31")
MIRROR_TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END   = pd.Timestamp("2024-07-01")

LGBM_PARAMS_BASE = dict(
    learning_rate=0.04, num_leaves=47, min_data_in_leaf=20,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    lambda_l2=0.1, verbosity=-1,
)
N_ROUNDS = 400


def lgbm_params(name: str) -> dict:
    p = dict(LGBM_PARAMS_BASE)
    if name == "L1":
        p.update(objective="regression_l1", metric="mae")
    elif name == "Q50":
        p.update(objective="quantile", alpha=0.5, metric="quantile")
    elif name == "Huber":
        p.update(objective="huber", alpha=0.9, metric="rmse")
    elif name == "MSE":
        p.update(objective="regression", metric="rmse")
    elif name == "Tweedie":
        p.update(objective="tweedie", tweedie_variance_power=1.5, metric="rmse")
    return p


def metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def make_features(daily_traffic, traffic_avg, dates_df):
    return build_v10c_features(dates_df[["Date"]], daily_traffic, traffic_avg, use_actual_traffic=False)


def load_inputs():
    sales   = pd.read_csv(RAW / "sales.csv", parse_dates=["Date"])
    traffic = pd.read_csv(RAW / "web_traffic.csv", parse_dates=["date"])
    sample  = pd.read_csv(REFERENCE / "sample_submission.csv", parse_dates=["Date"])
    daily_t = traffic.groupby("date").agg(
        sessions=("sessions","sum"), page_views=("page_views","sum"),
        unique_visitors=("unique_visitors","sum"),
        bounce_rate=("bounce_rate","mean"),
        avg_sess=("avg_session_duration_sec","mean")
    ).reset_index().rename(columns={"date":"Date"})
    daily_t["month"] = daily_t.Date.dt.month
    daily_t["dow"]   = daily_t.Date.dt.dayofweek
    return sales, traffic, daily_t, sample


def fit_lgbm(name, X_tr, y_tr, w_tr, use_log):
    label = np.log1p(y_tr) if use_log else y_tr
    p = lgbm_params(name)
    d = lgb.Dataset(X_tr, label=label, weight=w_tr)
    return lgb.train(p, d, num_boost_round=N_ROUNDS, callbacks=[lgb.log_evaluation(0)])


def predict_lgbm(name, booster, X, use_log):
    raw = booster.predict(X)
    if use_log:
        return np.expm1(raw)
    return np.clip(raw, 0, None)


def train_one_target(name_seeds, X_tr, y_tr, w_tr, X_te, w_year):
    """Returns dict of {head_name: predictions on X_te}."""
    out = {}
    # LightGBM heads -- raw target, except MSE which uses log1p
    for name in ["L1", "Q50", "Huber", "MSE", "Tweedie"]:
        use_log = (name == "MSE")
        booster = fit_lgbm(name, X_tr, y_tr, w_tr, use_log)
        out[f"LGBM_{name}"] = predict_lgbm(name, booster, X_te, use_log)
    # V10c learners -- log1p target
    y_log = np.log1p(y_tr)
    SEEDS = [42, 17]
    rf_preds, et_preds = [], []
    for s in SEEDS:
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=s, n_jobs=-1)
        rf.fit(X_tr, y_log, sample_weight=w_year)
        rf_preds.append(np.expm1(rf.predict(X_te)))
        et = ExtraTreesRegressor(n_estimators=250, max_depth=16, random_state=s, n_jobs=-1)
        et.fit(X_tr, y_log, sample_weight=w_year)
        et_preds.append(np.expm1(et.predict(X_te)))
    out["RF"] = np.mean(rf_preds, axis=0)
    out["ET"] = np.mean(et_preds, axis=0)
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=10, random_state=42)
    hgb.fit(X_tr, y_log, sample_weight=w_year)
    out["HGB"] = np.expm1(hgb.predict(X_te))
    return out


def build_blends(preds: dict):
    """Returns dict of blend predictions."""
    arrs = {k: np.asarray(v, dtype=float) for k, v in preds.items()}
    blends = {}
    blends["equal_all"] = np.mean([v for k,v in arrs.items() if k!="LGBM_Huber"], axis=0)
    v10c = np.mean([arrs[k] for k in ["RF", "ET", "HGB"]], axis=0)
    lgbm = np.mean([arrs[k] for k in ["LGBM_L1","LGBM_Q50","LGBM_MSE","LGBM_Tweedie"]], axis=0)
    blends["v10c_anchored"] = 0.5 * v10c + 0.5 * lgbm
    mae_aware  = np.mean([arrs[k] for k in ["LGBM_L1","LGBM_Q50"]], axis=0)
    rmse_aware = np.mean([arrs[k] for k in ["LGBM_MSE","LGBM_Tweedie","RF","ET","HGB"]], axis=0)
    blends["loss_balanced"] = 0.5 * mae_aware + 0.5 * rmse_aware
    return blends


def validate():
    t0 = time.time()
    sales, traffic, daily_t, sample = load_inputs()

    # Train cut: full 2017+ minus 2022 mirror
    train_traffic = daily_t[(daily_t.Date >= TRAIN_START_FULL) & (daily_t.Date <= MIRROR_TRAIN_END)]
    traffic_avg = (train_traffic.groupby(["month","dow"])
                   [["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]]
                   .mean().reset_index())

    train_sales  = sales[(sales.Date >= TRAIN_START_FULL) & (sales.Date <= MIRROR_TRAIN_END)].reset_index(drop=True)
    mirror_sales = sales[(sales.Date >= MIRROR_START) & (sales.Date <= MIRROR_END)].reset_index(drop=True)
    print(f"[{time.time()-t0:.1f}s] train rows={len(train_sales)}  mirror rows={len(mirror_sales)}")

    train_f  = build_v10c_features(train_sales[["Date"]],  daily_t, traffic_avg, use_actual_traffic=True)
    mirror_f = build_v10c_features(mirror_sales[["Date"]], daily_t, traffic_avg, use_actual_traffic=False)
    X_tr = train_f[V10C_FEATS].values
    X_mi = mirror_f[V10C_FEATS].values
    y_year = train_f["year"].values
    w_year_v10c = np.clip(y_year - 2016, 1.0, None) ** 1.2  # original V10c
    w_year_v14  = np.clip(y_year - 2016, 1.0, None) ** 0.7  # gentler

    rev_actual  = mirror_sales.Revenue.values
    cogs_actual = mirror_sales.COGS.values

    print(f"[{time.time()-t0:.1f}s] training Revenue heads...")
    rev_preds = train_one_target("rev", X_tr, train_sales.Revenue.values, w_year_v14, X_mi, w_year_v14)
    print(f"[{time.time()-t0:.1f}s] training COGS heads...")
    cog_preds = train_one_target("cog", X_tr, train_sales.COGS.values, w_year_v14, X_mi, w_year_v14)

    rev_blends = build_blends(rev_preds)
    cog_blends = build_blends(cog_preds)

    rows = []
    for name, p in {**rev_preds, **{f"BLEND_{k}": v for k, v in rev_blends.items()}}.items():
        mae, rmse, r2 = metrics(rev_actual, p)
        rows.append((name, "rev", mae, rmse, r2, float(np.mean(p))))
    for name, p in {**cog_preds, **{f"BLEND_{k}": v for k, v in cog_blends.items()}}.items():
        mae, rmse, r2 = metrics(cogs_actual, p)
        rows.append((name, "cogs", mae, rmse, r2, float(np.mean(p))))
    df = pd.DataFrame(rows, columns=["head","tgt","MAE","RMSE","R2","mean_pred"])

    # Original V10c reference (using w_year_v10c instead of v14)
    print(f"[{time.time()-t0:.1f}s] V10c reference fit...")
    rf_preds_v10c, et_preds_v10c = [], []
    for s in [42, 17]:
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=s, n_jobs=-1)
        rf.fit(X_tr, np.log1p(train_sales.Revenue.values), sample_weight=w_year_v10c)
        rf_preds_v10c.append(np.expm1(rf.predict(X_mi)))
        et = ExtraTreesRegressor(n_estimators=250, max_depth=16, random_state=s, n_jobs=-1)
        et.fit(X_tr, np.log1p(train_sales.Revenue.values), sample_weight=w_year_v10c)
        et_preds_v10c.append(np.expm1(et.predict(X_mi)))
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=10, random_state=42)
    hgb.fit(X_tr, np.log1p(train_sales.Revenue.values), sample_weight=w_year_v10c)
    v10c_rev = 0.30 * np.mean(rf_preds_v10c, axis=0) + 0.50 * np.mean(et_preds_v10c, axis=0) + 0.20 * np.expm1(hgb.predict(X_mi))
    rf_preds_v10c, et_preds_v10c = [], []
    for s in [42, 17]:
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=s, n_jobs=-1)
        rf.fit(X_tr, np.log1p(train_sales.COGS.values), sample_weight=w_year_v10c)
        rf_preds_v10c.append(np.expm1(rf.predict(X_mi)))
        et = ExtraTreesRegressor(n_estimators=250, max_depth=16, random_state=s, n_jobs=-1)
        et.fit(X_tr, np.log1p(train_sales.COGS.values), sample_weight=w_year_v10c)
        et_preds_v10c.append(np.expm1(et.predict(X_mi)))
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=10, random_state=42)
    hgb.fit(X_tr, np.log1p(train_sales.COGS.values), sample_weight=w_year_v10c)
    v10c_cog = 0.30 * np.mean(rf_preds_v10c, axis=0) + 0.50 * np.mean(et_preds_v10c, axis=0) + 0.20 * np.expm1(hgb.predict(X_mi))

    mae_r, rmse_r, r2_r = metrics(rev_actual, v10c_rev)
    mae_c, rmse_c, r2_c = metrics(cogs_actual, v10c_cog)
    df = pd.concat([df, pd.DataFrame([
        ("V10c_REF", "rev", mae_r, rmse_r, r2_r, float(np.mean(v10c_rev))),
        ("V10c_REF", "cogs", mae_c, rmse_c, r2_c, float(np.mean(v10c_cog))),
    ], columns=df.columns)], ignore_index=True)

    print(f"[{time.time()-t0:.1f}s] DONE")
    PROCESSED.mkdir(exist_ok=True, parents=True)
    df.to_parquet(PROCESSED / "v14_validate.parquet", index=False)
    print()
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}" if x > 1 else f"{x:.4f}")
    print(df.to_string(index=False))

    # Acceptance check vs V10c on revenue
    v10c_row = df[(df["head"]=="V10c_REF") & (df["tgt"]=="rev")].iloc[0]
    print()
    print(f"V10c_REF (revenue):  MAE={v10c_row.MAE:,.0f}  RMSE={v10c_row.RMSE:,.0f}  R2={v10c_row.R2:.4f}")
    print()
    print("Blend acceptance test (must beat V10c on >=2 of 3 metrics on rev):")
    for name in ["BLEND_equal_all", "BLEND_v10c_anchored", "BLEND_loss_balanced"]:
        r = df[(df["head"]==name) & (df["tgt"]=="rev")].iloc[0]
        wins = sum([r.MAE < v10c_row.MAE, r.RMSE < v10c_row.RMSE, r.R2 > v10c_row.R2])
        verdict = "PASS" if wins >= 2 else "FAIL"
        print(f"  {name:24s}  MAE={r.MAE:,.0f}  RMSE={r.RMSE:,.0f}  R2={r.R2:.4f}  wins={wins}/3  {verdict}")


def submit():
    """Train on full 2017-2022, predict 2023-2024 horizon, write submission."""
    t0 = time.time()
    sales, traffic, daily_t, sample = load_inputs()

    # Pick best blend from validation
    val_df = pd.read_parquet(PROCESSED / "v14_validate.parquet")
    v10c = val_df[(val_df["head"]=="V10c_REF") & (val_df["tgt"]=="rev")].iloc[0]
    best_name = None; best_score = float("inf")
    for name in ["BLEND_equal_all", "BLEND_v10c_anchored", "BLEND_loss_balanced"]:
        r = val_df[(val_df["head"]==name) & (val_df["tgt"]=="rev")].iloc[0]
        wins = sum([r.MAE < v10c.MAE, r.RMSE < v10c.RMSE, r.R2 > v10c.R2])
        if wins >= 2 and r.MAE + r.RMSE < best_score:
            best_score = r.MAE + r.RMSE; best_name = name
    print(f"Selected blend: {best_name or '(no winner -- using equal_all as default)'}")
    if best_name is None:
        best_name = "BLEND_equal_all"

    # Use full 2017-2022 train
    train_traffic = daily_t[(daily_t.Date >= TRAIN_START_FULL) & (daily_t.Date <= MIRROR_END)]
    traffic_avg = (train_traffic.groupby(["month","dow"])
                   [["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]]
                   .mean().reset_index())
    train_sales = sales[(sales.Date >= TRAIN_START_FULL) & (sales.Date <= MIRROR_END)].reset_index(drop=True)
    train_f = build_v10c_features(train_sales[["Date"]], daily_t, traffic_avg, use_actual_traffic=True)
    test_f  = build_v10c_features(sample[["Date"]],      daily_t, traffic_avg, use_actual_traffic=False)
    X_tr = train_f[V10C_FEATS].values
    X_te = test_f[V10C_FEATS].values
    w_year = np.clip(train_f["year"].values - 2016, 1.0, None) ** 0.7

    print(f"[{time.time()-t0:.1f}s] training final Revenue heads...")
    rev_preds = train_one_target("rev", X_tr, train_sales.Revenue.values, w_year, X_te, w_year)
    print(f"[{time.time()-t0:.1f}s] training final COGS heads...")
    cog_preds = train_one_target("cog", X_tr, train_sales.COGS.values, w_year, X_te, w_year)

    rev_blends = build_blends(rev_preds)
    cog_blends = build_blends(cog_preds)
    blend_key = best_name.replace("BLEND_", "")

    SUBMISSIONS.mkdir(exist_ok=True, parents=True)
    # Main blended submission
    pd.DataFrame({
        "Date": sample.Date.dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(rev_blends[blend_key], 2),
        "COGS":    np.round(cog_blends[blend_key], 2),
    }).to_csv(SUBMISSIONS / "submission_v14.csv", index=False)

    # Fallback: LGBM-L1 only (best single MAE optimizer)
    pd.DataFrame({
        "Date": sample.Date.dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(rev_preds["LGBM_L1"], 2),
        "COGS":    np.round(cog_preds["LGBM_L1"], 2),
    }).to_csv(SUBMISSIONS / "submission_v14_l1only.csv", index=False)

    print(f"[{time.time()-t0:.1f}s] DONE")
    print(f"  v14 blend: mean_rev={rev_blends[blend_key].mean()/1e6:.2f}M  "
          f"mean_cogs={cog_blends[blend_key].mean()/1e6:.2f}M")
    print(f"  v14 L1   : mean_rev={rev_preds['LGBM_L1'].mean()/1e6:.2f}M  "
          f"mean_cogs={cog_preds['LGBM_L1'].mean()/1e6:.2f}M")
    print(f"  -> {SUBMISSIONS / 'submission_v14.csv'}")
    print(f"  -> {SUBMISSIONS / 'submission_v14_l1only.csv'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "validate":
        validate()
    elif cmd == "submit":
        submit()
    else:
        print(f"Unknown command: {cmd}")
