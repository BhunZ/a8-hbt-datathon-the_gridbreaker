"""V10c-tuned: hyperparameter tune of V10c only (no architecture changes).

Same features (V10c\'s 23), same train window (2017+), same recency weights
(year-2016)^1.2, same ensemble blend (30% RF + 50% ET + 20% HGB).

Tuning knobs:
  RF: 350 -> 400 trees, depth 15 -> 17
  ET: 450 -> 500 trees, depth 16 -> 18
  Seeds: 2 -> 3
  HGB: max_iter 500 -> 700

CLI:
  python src/v10c_tuned.py mirror_rev   # mirror score for revenue
  python src/v10c_tuned.py mirror_cog   # mirror score for COGS
  python src/v10c_tuned.py submit_rev   # train on full, predict test (rev)
  python src/v10c_tuned.py submit_cog   # train on full, predict test (cogs)
  python src/v10c_tuned.py combine      # combine rev+cog into submission CSV
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, time, pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor)

from paths import RAW, PROCESSED, SUBMISSIONS, REFERENCE
from v13_v10c_rebaseline import build_v10c_features, FEATS

TRAIN_START = pd.Timestamp("2017-01-01")
MIRROR_TRAIN_END = pd.Timestamp("2021-12-31")
MIRROR_START = pd.Timestamp("2022-01-01")
MIRROR_END   = pd.Timestamp("2022-12-31")
FULL_TRAIN_END = pd.Timestamp("2022-12-31")

SEEDS = [42, 17, 5]
RF_N, RF_D = 400, 17
ET_N, ET_D = 500, 18
HGB_ITER = 700


def load_data():
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


def fit_predict(X_tr, y_tr, w_tr, X_te):
    """V10c\'s exact ensemble structure with tuned params."""
    y_log = np.log1p(y_tr)
    rf_preds, et_preds = [], []
    for s in SEEDS:
        rf = RandomForestRegressor(n_estimators=RF_N, max_depth=RF_D,
                                    random_state=s, n_jobs=-1)
        rf.fit(X_tr, y_log, sample_weight=w_tr)
        rf_preds.append(np.expm1(rf.predict(X_te)))
        et = ExtraTreesRegressor(n_estimators=ET_N, max_depth=ET_D,
                                  random_state=s, n_jobs=-1)
        et.fit(X_tr, y_log, sample_weight=w_tr)
        et_preds.append(np.expm1(et.predict(X_te)))
    hgb = HistGradientBoostingRegressor(max_iter=HGB_ITER, learning_rate=0.05,
                                         max_depth=10, random_state=42)
    hgb.fit(X_tr, y_log, sample_weight=w_tr)
    hgb_pred = np.expm1(hgb.predict(X_te))
    rf_avg = np.mean(rf_preds, axis=0)
    et_avg = np.mean(et_preds, axis=0)
    blended = 0.30 * rf_avg + 0.50 * et_avg + 0.20 * hgb_pred
    return blended, rf_avg, et_avg, hgb_pred


def metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def setup_features(train_end, test_dates_df, daily_t):
    """Build V10c features for train [TRAIN_START, train_end] + test (using projected traffic)."""
    sales = pd.read_csv(RAW / "sales.csv", parse_dates=["Date"])
    train_traffic = daily_t[(daily_t.Date >= TRAIN_START) & (daily_t.Date <= train_end)]
    traffic_avg = (train_traffic.groupby(["month","dow"])
                   [["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]]
                   .mean().reset_index())
    train_sales = sales[(sales.Date >= TRAIN_START) & (sales.Date <= train_end)].reset_index(drop=True)
    train_f = build_v10c_features(train_sales[["Date"]], daily_t, traffic_avg, use_actual_traffic=True)
    test_f  = build_v10c_features(test_dates_df[["Date"]], daily_t, traffic_avg, use_actual_traffic=False)
    return train_sales, train_f, test_f


def run_target(target: str, mode: str):
    """Fit and predict for one target. Saves preds to disk."""
    t0 = time.time()
    sales, traffic, daily_t, sample = load_data()

    if mode == "mirror":
        train_end = MIRROR_TRAIN_END
        test_dates_df = sales[(sales.Date >= MIRROR_START) & (sales.Date <= MIRROR_END)][["Date"]]
        out_label = f"mirror_{target}"
    elif mode == "submit":
        train_end = FULL_TRAIN_END
        test_dates_df = sample[["Date"]]
        out_label = f"submit_{target}"
    else:
        raise ValueError(mode)

    train_sales, train_f, test_f = setup_features(train_end, test_dates_df, daily_t)
    X_tr = train_f[FEATS].values
    X_te = test_f[FEATS].values
    w_tr = np.clip(train_f["year"].values - 2016, 1.0, None) ** 1.2
    y_tr = train_sales[target].values
    print(f"[{time.time()-t0:.1f}s] target={target} train rows={len(train_sales)} test rows={len(test_dates_df)}")

    blended, rf_avg, et_avg, hgb_pred = fit_predict(X_tr, y_tr, w_tr, X_te)
    print(f"[{time.time()-t0:.1f}s] {target} fit done. mean_pred={blended.mean()/1e6:.2f}M")

    out = pd.DataFrame({
        "Date": pd.to_datetime(test_dates_df.Date.values),
        "blended": blended,
        "rf": rf_avg, "et": et_avg, "hgb": hgb_pred,
    })
    if mode == "mirror":
        actual = sales[(sales.Date >= MIRROR_START) & (sales.Date <= MIRROR_END)][target].values
        out["actual"] = actual
        mae, rmse, r2 = metrics(actual, blended)
        print(f"  mirror metrics: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f}")

    PROCESSED.mkdir(exist_ok=True, parents=True)
    out.to_parquet(PROCESSED / f"v10c_tuned_{out_label}.parquet", index=False)
    print(f"  -> {PROCESSED / f'v10c_tuned_{out_label}.parquet'}")


def combine():
    """Combine rev + cog submission preds into final CSV."""
    rev_path = PROCESSED / "v10c_tuned_submit_Revenue.parquet"
    cog_path = PROCESSED / "v10c_tuned_submit_COGS.parquet"
    if not rev_path.exists() or not cog_path.exists():
        print(f"Missing predictions: rev={rev_path.exists()} cog={cog_path.exists()}")
        return
    rev = pd.read_parquet(rev_path)
    cog = pd.read_parquet(cog_path)
    merged = rev[["Date","blended"]].merge(cog[["Date","blended"]], on="Date", suffixes=("_rev","_cog"))
    out = pd.DataFrame({
        "Date": pd.to_datetime(merged.Date).dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(merged.blended_rev, 2),
        "COGS":    np.round(merged.blended_cog, 2),
    })
    SUBMISSIONS.mkdir(exist_ok=True, parents=True)
    out.to_csv(SUBMISSIONS / "submission_v10c_tuned.csv", index=False)
    print(f"  v10c_tuned: mean_rev={out.Revenue.mean()/1e6:.2f}M  mean_cogs={out.COGS.mean()/1e6:.2f}M")
    print(f"  -> {SUBMISSIONS / 'submission_v10c_tuned.csv'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "mirror_rev":   run_target("Revenue", "mirror")
    elif cmd == "mirror_cog": run_target("COGS",    "mirror")
    elif cmd == "submit_rev": run_target("Revenue", "submit")
    elif cmd == "submit_cog": run_target("COGS",    "submit")
    elif cmd == "combine":    combine()
    else:
        print(__doc__)
