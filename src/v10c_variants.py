"""V10c variants — two narrow tuning experiments.

variant=shallow    : RF/ET shallower (depth 12/13) to underfit more, push level up
variant=multiseed  : original V10c hyperparams but more seeds (4) for variance reduction
variant=original   : exact V10c reference

CLI:
  python src/v10c_variants.py shallow     submit  rev
  python src/v10c_variants.py shallow     submit  cog
  python src/v10c_variants.py multiseed   submit  rev
  python src/v10c_variants.py multiseed   submit  cog
  python src/v10c_variants.py combine     shallow
  python src/v10c_variants.py combine     multiseed
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, time

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor)

from paths import RAW, PROCESSED, SUBMISSIONS, REFERENCE
from v13_v10c_rebaseline import build_v10c_features, FEATS

TRAIN_START = pd.Timestamp("2017-01-01")
FULL_TRAIN_END = pd.Timestamp("2022-12-31")

VARIANTS = {
    "original":  dict(seeds=[42, 17],          rf_n=350, rf_d=15, et_n=450, et_d=16, hgb=500),
    "shallow":   dict(seeds=[42, 17],          rf_n=350, rf_d=12, et_n=450, et_d=13, hgb=500),
    "multiseed": dict(seeds=[42, 17, 5, 11],   rf_n=350, rf_d=15, et_n=450, et_d=16, hgb=500),
}


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
    return sales, daily_t, sample


def run(variant: str, target: str):
    cfg = VARIANTS[variant]
    t0 = time.time()
    sales, daily_t, sample = load_data()

    train_traffic = daily_t[(daily_t.Date >= TRAIN_START) & (daily_t.Date <= FULL_TRAIN_END)]
    traffic_avg = (train_traffic.groupby(["month","dow"])
                   [["sessions","page_views","unique_visitors","bounce_rate","avg_sess"]]
                   .mean().reset_index())
    train_sales = sales[(sales.Date >= TRAIN_START) & (sales.Date <= FULL_TRAIN_END)].reset_index(drop=True)
    train_f = build_v10c_features(train_sales[["Date"]], daily_t, traffic_avg, use_actual_traffic=True)
    test_f  = build_v10c_features(sample[["Date"]],      daily_t, traffic_avg, use_actual_traffic=False)
    X_tr = train_f[FEATS].values
    X_te = test_f[FEATS].values
    w_tr = np.clip(train_f["year"].values - 2016, 1.0, None) ** 1.2
    y_tr = train_sales[target].values
    y_log = np.log1p(y_tr)
    print(f"[{time.time()-t0:.1f}s] variant={variant} target={target} train={len(train_sales)} test={len(test_f)}")

    rf_preds, et_preds = [], []
    for s in cfg["seeds"]:
        rf = RandomForestRegressor(n_estimators=cfg["rf_n"], max_depth=cfg["rf_d"],
                                    random_state=s, n_jobs=-1)
        rf.fit(X_tr, y_log, sample_weight=w_tr)
        rf_preds.append(np.expm1(rf.predict(X_te)))
        et = ExtraTreesRegressor(n_estimators=cfg["et_n"], max_depth=cfg["et_d"],
                                  random_state=s, n_jobs=-1)
        et.fit(X_tr, y_log, sample_weight=w_tr)
        et_preds.append(np.expm1(et.predict(X_te)))
        print(f"  [{time.time()-t0:.1f}s] seed={s} done")
    hgb = HistGradientBoostingRegressor(max_iter=cfg["hgb"], learning_rate=0.05,
                                         max_depth=10, random_state=42)
    hgb.fit(X_tr, y_log, sample_weight=w_tr)
    hgb_pred = np.expm1(hgb.predict(X_te))
    rf_avg = np.mean(rf_preds, axis=0)
    et_avg = np.mean(et_preds, axis=0)
    blended = 0.30 * rf_avg + 0.50 * et_avg + 0.20 * hgb_pred
    print(f"[{time.time()-t0:.1f}s] {variant}/{target} mean_pred={blended.mean()/1e6:.2f}M")

    out = pd.DataFrame({
        "Date": pd.to_datetime(sample.Date),
        "blended": blended, "rf": rf_avg, "et": et_avg, "hgb": hgb_pred,
    })
    PROCESSED.mkdir(exist_ok=True, parents=True)
    out.to_parquet(PROCESSED / f"v10c_variant_{variant}_{target}.parquet", index=False)


def combine(variant: str):
    rev = pd.read_parquet(PROCESSED / f"v10c_variant_{variant}_Revenue.parquet")
    cog = pd.read_parquet(PROCESSED / f"v10c_variant_{variant}_COGS.parquet")
    merged = rev[["Date","blended"]].merge(cog[["Date","blended"]], on="Date", suffixes=("_rev","_cog"))
    out = pd.DataFrame({
        "Date": pd.to_datetime(merged.Date).dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(merged.blended_rev, 2),
        "COGS":    np.round(merged.blended_cog, 2),
    })
    SUBMISSIONS.mkdir(exist_ok=True, parents=True)
    fname = f"submission_v10c_{variant}.csv"
    out.to_csv(SUBMISSIONS / fname, index=False)
    print(f"  {variant}: mean_rev={out.Revenue.mean()/1e6:.2f}M  mean_cogs={out.COGS.mean()/1e6:.2f}M")
    print(f"  -> {SUBMISSIONS / fname}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "combine":
        combine(sys.argv[2])
    else:
        # cmd = variant, sys.argv[2] = mode (unused here, always submit), sys.argv[3] = target
        variant = sys.argv[1]
        target = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
        # Map "rev"/"cog" -> column names
        target_map = {"rev": "Revenue", "cog": "COGS", "Revenue": "Revenue", "COGS": "COGS"}
        run(variant, target_map[target])
