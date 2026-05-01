"""V13 FINAL Kaggle submission (slim).

Strategy: blend V10c (retrained on post-cliff window) with V13 baseline +
trend-scale.  Skips Stage-3 per-event multipliers (mirror showed only -1k
benefit, not worth the OOF cost).

Outputs (under submissions/):
  submission_v13_v10c_only.csv  -- V10c retrained component
  submission_v13_v13_only.csv   -- V13 retrained component (stage1 x trend-scale)
  submission_v13_final.csv      -- 50/50 blend (the one to upload)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor)

from paths import RAW, PROCESSED, SUBMISSIONS, REFERENCE
from v13_validation import POST_CLIFF_START, load_v13_features, feature_columns
from v13_baseline import (recency_weights, fit_ensemble, predict_ensemble,
                           DEFAULT_ITERS)
from v13_v10c_rebaseline import build_v10c_features, FEATS as V10C_FEATS

TRAIN_START = POST_CLIFF_START
TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")


def predict_v10c():
    print("=== V10c component ===")
    sales   = pd.read_csv(RAW / "sales.csv",       parse_dates=["Date"])
    traffic = pd.read_csv(RAW / "web_traffic.csv", parse_dates=["date"])
    sample  = pd.read_csv(REFERENCE / "sample_submission.csv", parse_dates=["Date"])

    daily_t = traffic.groupby("date").agg(
        sessions=("sessions", "sum"), page_views=("page_views", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_sess=("avg_session_duration_sec", "mean"),
    ).reset_index().rename(columns={"date": "Date"})
    daily_t["month"] = daily_t.Date.dt.month
    daily_t["dow"]   = daily_t.Date.dt.dayofweek
    train_traffic = daily_t[(daily_t.Date >= TRAIN_START) & (daily_t.Date <= TRAIN_END)]
    traffic_avg = (train_traffic.groupby(["month", "dow"])
                   [["sessions", "page_views", "unique_visitors", "bounce_rate", "avg_sess"]]
                   .mean().reset_index())

    train_sales = sales[(sales.Date >= TRAIN_START) & (sales.Date <= TRAIN_END)].reset_index(drop=True)
    train_f = build_v10c_features(train_sales[["Date"]], daily_t, traffic_avg, use_actual_traffic=True)
    test_f  = build_v10c_features(sample[["Date"]],      daily_t, traffic_avg, use_actual_traffic=False)

    X_tr = train_f[V10C_FEATS].values
    X_te = test_f[V10C_FEATS].values
    w_tr = np.clip(train_f["year"].values - 2018, 1.0, None) ** 1.2

    SEEDS = [42, 17]
    preds = {"Revenue": [], "COGS": []}
    for tgt in ["Revenue", "COGS"]:
        y_tr = np.log1p(train_sales[tgt].values)
        for s in SEEDS:
            rf = RandomForestRegressor(n_estimators=350, max_depth=15, random_state=s, n_jobs=-1)
            rf.fit(X_tr, y_tr, sample_weight=w_tr)
            preds[tgt].append(("rf", 0.30 / len(SEEDS), np.expm1(rf.predict(X_te))))
            et = ExtraTreesRegressor(n_estimators=450, max_depth=16, random_state=s, n_jobs=-1)
            et.fit(X_tr, y_tr, sample_weight=w_tr)
            preds[tgt].append(("et", 0.50 / len(SEEDS), np.expm1(et.predict(X_te))))
        hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=10, random_state=42)
        hgb.fit(X_tr, y_tr, sample_weight=w_tr)
        preds[tgt].append(("hgb", 0.20, np.expm1(hgb.predict(X_te))))

    p_rev  = sum(wt * p for _, wt, p in preds["Revenue"])
    p_cogs = sum(wt * p for _, wt, p in preds["COGS"])
    return sample.Date, p_rev, p_cogs


def predict_v13():
    print("=== V13 component (stage1 + trend-scale) ===")
    df = load_v13_features()
    feats = feature_columns(df)
    train_mask = (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END) & df["Revenue"].notna()
    test_mask  = (df["date"] >= TEST_START) & (df["date"] <= TEST_END)

    X_tr = df.loc[train_mask, feats].values
    X_te = df.loc[test_mask,  feats].values
    y_tr_rev = df.loc[train_mask, "Revenue"].values
    y_tr_cog = df.loc[train_mask, "COGS"].values
    w_tr = recency_weights(df.loc[train_mask, "date"])

    ens_rev = fit_ensemble(X_tr, y_tr_rev, weights=w_tr, n_round=DEFAULT_ITERS)
    ens_cog = fit_ensemble(X_tr, y_tr_cog, weights=w_tr, n_round=DEFAULT_ITERS)
    p_rev = predict_ensemble(ens_rev, X_te)
    p_cog = predict_ensemble(ens_cog, X_te)

    trend_train = df.loc[train_mask, "aov_trend_proj"].mean()
    trend_test  = df.loc[test_mask,  "aov_trend_proj"].mean()
    trend_scale = float(trend_test / trend_train)
    print(f"  Trend-scale: {trend_scale:.4f} (train={trend_train:.0f}, test={trend_test:.0f})")
    p_rev *= trend_scale
    p_cog *= trend_scale
    return df.loc[test_mask, "date"].reset_index(drop=True), p_rev, p_cog, trend_scale


def main():
    t0 = time.time()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    dates_a, rev_a, cogs_a = predict_v10c()
    print(f"  [{time.time()-t0:.1f}s] V10c done. mean rev = {rev_a.mean()/1e6:.2f}M")

    dates_b, rev_b, cogs_b, trend_scale = predict_v13()
    print(f"  [{time.time()-t0:.1f}s] V13 done.  mean rev = {rev_b.mean()/1e6:.2f}M  trend_scale={trend_scale:.4f}")

    da = pd.to_datetime(dates_a).reset_index(drop=True)
    db = pd.to_datetime(dates_b).reset_index(drop=True)
    assert (da.values == db.values).all(), "Dates do not match"

    pd.DataFrame({"Date": da.dt.strftime("%Y-%m-%d"),
                  "Revenue": np.round(rev_a, 2),
                  "COGS":    np.round(cogs_a, 2)}
                 ).to_csv(SUBMISSIONS / "submission_v13_v10c_only.csv", index=False)
    pd.DataFrame({"Date": da.dt.strftime("%Y-%m-%d"),
                  "Revenue": np.round(rev_b, 2),
                  "COGS":    np.round(cogs_b, 2)}
                 ).to_csv(SUBMISSIONS / "submission_v13_v13_only.csv", index=False)

    rev_blend  = 0.5 * rev_a  + 0.5 * rev_b
    cogs_blend = 0.5 * cogs_a + 0.5 * cogs_b
    pd.DataFrame({"Date": da.dt.strftime("%Y-%m-%d"),
                  "Revenue": np.round(rev_blend, 2),
                  "COGS":    np.round(cogs_blend, 2)}
                 ).to_csv(SUBMISSIONS / "submission_v13_final.csv", index=False)

    print()
    print("=== FINAL SUBMISSIONS ===")
    print(f"  v10c only:   mean rev={rev_a.mean()/1e6:.2f}M  sum={rev_a.sum()/1e9:.2f}B")
    print(f"  v13  only:   mean rev={rev_b.mean()/1e6:.2f}M  sum={rev_b.sum()/1e9:.2f}B")
    print(f"  blend 50/50: mean rev={rev_blend.mean()/1e6:.2f}M  sum={rev_blend.sum()/1e9:.2f}B")
    print()
    print(f"  -> {SUBMISSIONS / 'submission_v13_final.csv'}      <-- UPLOAD THIS")
    print(f"  -> {SUBMISSIONS / 'submission_v13_v10c_only.csv'}")
    print(f"  -> {SUBMISSIONS / 'submission_v13_v13_only.csv'}")
    print(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
