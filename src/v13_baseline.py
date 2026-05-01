"""V13 Step 8 -- Baseline LightGBM model.

Trains LightGBM separately on Revenue and COGS using V13 features.
Default objective: Tweedie (variance_power=1.5), the standard for spiky retail
sales forecasting.

Train: 2019-09-01 .. 2021-12-31
Mirror: 2022-01-01 .. 2022-12-31

Outputs:
  data/processed/v13_baseline_mirror_preds.parquet
  docs/v13_baseline_results.md
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from paths import PROCESSED, DOCS
from v13_validation import (POST_CLIFF_START, MIRROR_START, MIRROR_END,
                             MIRROR_TRAIN_END, ITERATION_FOLDS,
                             load_v13_features, feature_columns,
                             score_predictions, fmt_score)

V10C_FLOOR_COMBINED = 902_932


LGBM_PARAMS_TWEEDIE = {
    "objective":              "tweedie",
    "tweedie_variance_power": 1.5,
    "metric":                 "rmse",
    "learning_rate":          0.03,
    "num_leaves":             47,
    "min_data_in_leaf":       8,
    "feature_fraction":       0.85,
    "bagging_fraction":       0.85,
    "bagging_freq":           5,
    "lambda_l2":              0.1,
    "verbosity":              -1,
}

LGBM_PARAMS_LOG = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.03,
    "num_leaves":        47,
    "min_data_in_leaf":  8,
    "feature_fraction":  0.85,
    "bagging_fraction":  0.85,
    "bagging_freq":      5,
    "lambda_l2":         0.1,
    "verbosity":         -1,
}

# Toggle to switch.  Tweedie is the V13 blueprint default.
LGBM_PARAMS = LGBM_PARAMS_TWEEDIE
USE_LOG_TARGET = False
DEFAULT_ITERS = 600


def recency_weights(dates):
    """Linear ramp 1.0 -> 4.0 across the post-cliff train window."""
    t = (pd.to_datetime(dates) - POST_CLIFF_START).dt.days.values.astype(float)
    span = (MIRROR_TRAIN_END - POST_CLIFF_START).days
    return 1.0 + 3.0 * (t / span)


SEEDS = [42, 17]


def fit_one(X_tr, y_tr, weights=None, n_round=DEFAULT_ITERS, seed=42):
    label = np.log1p(y_tr) if USE_LOG_TARGET else y_tr
    p = {**LGBM_PARAMS, "seed": seed, "feature_fraction_seed": seed,
         "bagging_seed": seed, "data_random_seed": seed}
    dtrain = lgb.Dataset(X_tr, label=label, weight=weights)
    booster = lgb.train(p, dtrain, num_boost_round=n_round,
                        callbacks=[lgb.log_evaluation(0)])
    if USE_LOG_TARGET:
        sigma2 = float(np.mean((np.log1p(y_tr) - booster.predict(X_tr)) ** 2))
    else:
        sigma2 = 0.0
    return booster, sigma2


def predict_debiased(booster, X, sigma2):
    raw = booster.predict(X)
    if USE_LOG_TARGET:
        return np.expm1(raw + 0.5 * sigma2)
    return np.clip(raw, 0, None)


def fit_ensemble(X_tr, y_tr, weights=None, n_round=DEFAULT_ITERS):
    return [fit_one(X_tr, y_tr, weights=weights, n_round=n_round, seed=s) for s in SEEDS]


def predict_ensemble(boosters_sigmas, X):
    return np.mean([predict_debiased(b, X, s2) for b, s2 in boosters_sigmas], axis=0)


def main():
    t0 = time.time()
    df = load_v13_features()
    feats = feature_columns(df)
    print(f"V13 feature table: {df.shape}, {len(feats)} model features")
    print(f"Objective: {LGBM_PARAMS['objective']}, USE_LOG_TARGET={USE_LOG_TARGET}, iters={DEFAULT_ITERS}")

    # Iteration CV
    print("\n=== Iteration CV (4 folds, 90-day horizons) ===")
    fold_results = []
    for fold in ITERATION_FOLDS:
        tr_mask = (df["date"] >= POST_CLIFF_START) & (df["date"] <= fold.train_end) & df["Revenue"].notna()
        va_mask = (df["date"] >= fold.val_start) & (df["date"] <= fold.val_end) & df["Revenue"].notna()
        X_tr = df.loc[tr_mask, feats].values
        X_va = df.loc[va_mask, feats].values
        y_tr_rev = df.loc[tr_mask, "Revenue"].values
        y_va_rev = df.loc[va_mask, "Revenue"].values
        y_tr_cog = df.loc[tr_mask, "COGS"].values
        y_va_cog = df.loc[va_mask, "COGS"].values
        w_tr = recency_weights(df.loc[tr_mask, "date"])

        ens_rev = fit_ensemble(X_tr, y_tr_rev, weights=w_tr)
        p_rev = predict_ensemble(ens_rev, X_va)
        ens_cog = fit_ensemble(X_tr, y_tr_cog, weights=w_tr)
        p_cog = predict_ensemble(ens_cog, X_va)

        feat_va_df = df.loc[va_mask].reset_index(drop=True)
        score = score_predictions(
            val_dates=feat_va_df["date"],
            y_true_rev=y_va_rev, y_pred_rev=p_rev,
            y_true_cogs=y_va_cog, y_pred_cogs=p_cog,
            features_df=feat_va_df, train_y_for_pct=y_tr_rev,
        )
        score["fold"] = fold.name
        fold_results.append(score)
        print(f"\n[{fold.name}]")
        print(fmt_score(score))

    # Mirror block
    print("\n=== Mirror block (2022) ===")
    tr_mask = (df["date"] >= POST_CLIFF_START) & (df["date"] <= MIRROR_TRAIN_END) & df["Revenue"].notna()
    mi_mask = (df["date"] >= MIRROR_START) & (df["date"] <= MIRROR_END) & df["Revenue"].notna()
    X_tr = df.loc[tr_mask, feats].values
    X_mi = df.loc[mi_mask, feats].values
    y_tr_rev = df.loc[tr_mask, "Revenue"].values
    y_mi_rev = df.loc[mi_mask, "Revenue"].values
    y_tr_cog = df.loc[tr_mask, "COGS"].values
    y_mi_cog = df.loc[mi_mask, "COGS"].values
    w_tr = recency_weights(df.loc[tr_mask, "date"])

    ens_rev = fit_ensemble(X_tr, y_tr_rev, weights=w_tr)
    ens_cog = fit_ensemble(X_tr, y_tr_cog, weights=w_tr)
    print(f"Trained {DEFAULT_ITERS} rounds x {len(SEEDS)} seeds, recency-weighted (1.0->4.0)")

    p_mi_rev = predict_ensemble(ens_rev, X_mi)
    p_mi_cog = predict_ensemble(ens_cog, X_mi)

    feat_mi = df.loc[mi_mask].reset_index(drop=True)
    score_mirror = score_predictions(
        val_dates=feat_mi["date"],
        y_true_rev=y_mi_rev, y_pred_rev=p_mi_rev,
        y_true_cogs=y_mi_cog, y_pred_cogs=p_mi_cog,
        features_df=feat_mi, train_y_for_pct=y_tr_rev,
    )
    print()
    print(fmt_score(score_mirror))

    pd.DataFrame({
        "date": feat_mi["date"],
        "rev_actual":  y_mi_rev,  "rev_pred":  p_mi_rev,
        "cogs_actual": y_mi_cog,  "cogs_pred": p_mi_cog,
    }).to_parquet(PROCESSED / "v13_baseline_mirror_preds.parquet", index=False)

    delta = score_mirror["rmse_global_combined"] - V10C_FLOOR_COMBINED
    verdict = "BEATS V10c" if delta < 0 else "WORSE than V10c"
    print(f"\n=== V13 baseline vs V10c floor on mirror_2022 ===")
    print(f"  V10c floor:    {V10C_FLOOR_COMBINED:>12,d}")
    print(f"  V13 baseline:  {score_mirror['rmse_global_combined']:>12,.0f}")
    print(f"  Delta:         {delta:>+12,.0f}   ({verdict})")
    print(f"\nDONE in {time.time()-t0:.1f}s")

    # Doc
    fold_lines = [
        f"| {s['fold']} | {s['rmse_global_combined']:,.0f} | {s['rmse_global_rev']:,.0f} | "
        f"{s['rmse_peak_rev']:,.0f} | {s['rmse_log_rev']:.4f} | {s['mean_pred_rev']:,.0f} | "
        f"{s['mean_actual_rev']:,.0f} |"
        for s in fold_results
    ]
    doc = [
        "# V13 Step 8 -- Baseline LightGBM Results",
        "",
        f"- Objective: {LGBM_PARAMS['objective']}, USE_LOG_TARGET={USE_LOG_TARGET}",
        f"- DEFAULT_ITERS: {DEFAULT_ITERS}, recency-weighted",
        f"- Train: 2019-09-01 .. 2021-12-31",
        f"- Mirror: 2022-01-01 .. 2022-12-31",
        "",
        "## Iteration CV folds",
        "",
        "| Fold | rmse_combined | rmse_rev | rmse_peak_rev | rmse_log_rev | mean_pred | mean_actual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ] + fold_lines + [
        "",
        "## Mirror block",
        "",
        f"- **rmse_combined: {score_mirror['rmse_global_combined']:,.0f}**",
        f"- rmse_rev: {score_mirror['rmse_global_rev']:,.0f}   rmse_peak_rev: {score_mirror['rmse_peak_rev']:,.0f}",
        f"- rmse_cogs: {score_mirror['rmse_global_cogs']:,.0f}  rmse_peak_cogs: {score_mirror['rmse_peak_cogs']:,.0f}",
        f"- mean pred rev: {score_mirror['mean_pred_rev']:,.0f}/day vs actual: {score_mirror['mean_actual_rev']:,.0f}/day",
        "",
        "## vs V10c floor",
        "",
        "| | V10c rebaseline | V13 baseline | Delta |",
        "|---|---:|---:|---:|",
        f"| rmse_combined | {V10C_FLOOR_COMBINED:,d} | {score_mirror['rmse_global_combined']:,.0f} | {delta:+,.0f} |",
        "",
        f"**Verdict: {verdict}**",
    ]
    (DOCS / "v13_baseline_results.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"docs -> {DOCS / 'v13_baseline_results.md'}")


if __name__ == "__main__":
    main()
",
        "",
        "## vs V10c floor",
        "",
        "| | V10c rebaseline | V13 baseline | Delta |",
        "|---|---:|---:|---:|",
        f"| rmse_combined | {V10C_FLOOR_COMBINED:,d} | {score_mirror['rmse_global_combined']:,.0f} | {delta:+,.0f} |",
        "",
        f"**Verdict: {verdict}**",
    ]
    (DOCS / "v13_baseline_results.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"docs -> {DOCS / 'v13_baseline_results.md'}")


if __name__ == "__main__":
    main()
