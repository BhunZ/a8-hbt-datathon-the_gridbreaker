"""V13 Step 9 -- Peak recovery layer.

Stage 1: V13 baseline Tweedie ensemble (from src/v13_baseline.py).
Stage 2: Quantile LightGBM uplift on event-day residuals (alpha=0.85).
Stage 3: Frozen per-event multiplier table calibrated on OOF residuals.
TrendScale: Constant level scale = mean(aov_trend_proj on mirror) / mean(on train).

Outputs:
  data/processed/v13_peak_recovery_mirror_preds.parquet
  data/processed/v13_event_uplift_post_model.json
  docs/v13_peak_recovery_results.md
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json, time

import numpy as np
import pandas as pd
import lightgbm as lgb

from paths import PROCESSED, DOCS
from v13_validation import (POST_CLIFF_START, MIRROR_START, MIRROR_END,
                             MIRROR_TRAIN_END, load_v13_features,
                             feature_columns, score_predictions, fmt_score)
from v13_baseline import (LGBM_PARAMS_TWEEDIE, recency_weights, fit_ensemble,
                           predict_ensemble, DEFAULT_ITERS)

V10C_FLOOR_COMBINED = 902_932

PEAK_EVENT_FLAGS = [
    "is_singles_day", "is_twelve_twelve", "is_nine_nine", "is_ten_ten",
    "is_black_friday", "is_cyber_monday", "is_intl_women_day",
    "is_womens_day_vn", "is_teachers_day_vn", "is_independence_day_vn",
    "is_reunification_day", "is_intl_labour_day", "is_new_year_day",
    "is_valentines", "is_christmas", "is_boxing_day", "is_tet",
]

WIDE_OOF_FOLDS = [
    ("2020Q1", pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31")),
    ("2020Q2", pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-30")),
    ("2020Q3", pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2020-09-30")),
    ("2020Q4", pd.Timestamp("2020-09-30"), pd.Timestamp("2020-10-01"), pd.Timestamp("2020-12-31")),
    ("2021Q1", pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-03-31")),
    ("2021Q2", pd.Timestamp("2021-03-31"), pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30")),
    ("2021Q3", pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2021-09-30")),
    ("2021Q4", pd.Timestamp("2021-09-30"), pd.Timestamp("2021-10-01"), pd.Timestamp("2021-12-31")),
]


def compute_oof_stage1(df, feats, target_col):
    oof = np.full(len(df), np.nan)
    for name, train_end, val_start, val_end in WIDE_OOF_FOLDS:
        tr = (df["date"] >= POST_CLIFF_START) & (df["date"] <= train_end) & df[target_col].notna()
        va = (df["date"] >= val_start) & (df["date"] <= val_end) & df[target_col].notna()
        if int(tr.sum()) < 60 or int(va.sum()) == 0:
            continue
        Xtr = df.loc[tr, feats].values
        Xva = df.loc[va, feats].values
        ytr = df.loc[tr, target_col].values
        wtr = recency_weights(df.loc[tr, "date"])
        ens = fit_ensemble(Xtr, ytr, weights=wtr, n_round=DEFAULT_ITERS)
        oof[va.values] = predict_ensemble(ens, Xva)
    return oof


def calibrate_stage3(df, oof_pred, target_col, train_mask, min_obs=3):
    out = {}
    avail = train_mask & df[target_col].notna() & ~np.isnan(oof_pred)
    for flag in PEAK_EVENT_FLAGS:
        m = avail & (df[flag] == 1)
        n = int(m.sum())
        if n < min_obs:
            out[flag] = (1.0, n)
            continue
        ratio = (df.loc[m, target_col].values / oof_pred[m.values]).mean()
        out[flag] = (float(ratio), n)
    return out


def apply_stage3(df, stage1_pred, uplift_table):
    out = stage1_pred.copy()
    chosen_dev = np.zeros(len(out))
    for flag, (mult, _n) in uplift_table.items():
        if flag not in df.columns or mult == 1.0:
            continue
        mask = df[flag].values.astype(bool)
        if not mask.any():
            continue
        d = abs(np.log(mult)) if mult > 0 else 0.0
        better = mask & (d > chosen_dev)
        out[better] = stage1_pred[better] * mult
        chosen_dev[better] = d
    return out


def build_stage2(df, feats, oof_pred, target_col, train_mask, alpha=0.85):
    avail = train_mask & df[target_col].notna() & ~np.isnan(oof_pred)
    any_event = df[PEAK_EVENT_FLAGS].sum(axis=1).clip(0, 1).astype(bool)
    train_rows = avail & any_event
    n = int(train_rows.sum())
    print(f"  stage2[{target_col}] train rows: {n}")
    if n < 30:
        print(f"  stage2[{target_col}] too few rows -- skipping")
        return None
    X = df.loc[train_rows, feats].values
    rel = (df.loc[train_rows, target_col].values - oof_pred[train_rows.values]) / oof_pred[train_rows.values]
    rel = np.clip(rel, -1.0, 2.0)
    p = {**LGBM_PARAMS_TWEEDIE, "objective": "quantile", "alpha": alpha, "metric": "rmse"}
    p.pop("tweedie_variance_power", None)
    d = lgb.Dataset(X, label=rel)
    return lgb.train(p, d, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])


def apply_stage2(df, feats, stage1_pred, booster):
    if booster is None:
        return stage1_pred.copy()
    any_event = df[PEAK_EVENT_FLAGS].sum(axis=1).clip(0, 1).astype(bool).values
    out = stage1_pred.copy()
    if any_event.any():
        rel_pred = booster.predict(df[feats].values[any_event])
        out[any_event] = stage1_pred[any_event] * (1.0 + rel_pred)
    return np.clip(out, 0, None)


def main():
    t0 = time.time()
    df = load_v13_features()
    feats = feature_columns(df)
    print(f"V13 features: {df.shape}, {len(feats)} columns")

    train_mask = (df["date"] >= POST_CLIFF_START) & (df["date"] <= MIRROR_TRAIN_END) & df["Revenue"].notna()
    mirror_mask = (df["date"] >= MIRROR_START) & (df["date"] <= MIRROR_END) & df["Revenue"].notna()
    feat_mi = df.loc[mirror_mask].reset_index(drop=True)

    print("\n=== Stage 1 fit ===")
    Xtr = df.loc[train_mask, feats].values
    Xmi = df.loc[mirror_mask, feats].values
    ytr_rev = df.loc[train_mask, "Revenue"].values
    ytr_cog = df.loc[train_mask, "COGS"].values
    ymi_rev = df.loc[mirror_mask, "Revenue"].values
    ymi_cog = df.loc[mirror_mask, "COGS"].values
    wtr = recency_weights(df.loc[train_mask, "date"])

    ens_rev = fit_ensemble(Xtr, ytr_rev, weights=wtr, n_round=DEFAULT_ITERS)
    ens_cog = fit_ensemble(Xtr, ytr_cog, weights=wtr, n_round=DEFAULT_ITERS)
    p1_rev = predict_ensemble(ens_rev, Xmi)
    p1_cog = predict_ensemble(ens_cog, Xmi)
    print(f"  [{time.time()-t0:.1f}s] Stage 1 done")

    print("\n=== OOF predictions ===")
    oof_rev = compute_oof_stage1(df, feats, "Revenue")
    oof_cog = compute_oof_stage1(df, feats, "COGS")
    print(f"  [{time.time()-t0:.1f}s] OOF coverage: {int((train_mask & ~np.isnan(oof_rev)).sum())} train rows")

    print("\n=== Stage 3 per-event multipliers ===")
    uplift_rev = calibrate_stage3(df, oof_rev, "Revenue", train_mask)
    uplift_cog = calibrate_stage3(df, oof_cog, "COGS", train_mask)

    print("\n=== Stage 2 quantile uplift ===")
    s2_rev = build_stage2(df, feats, oof_rev, "Revenue", train_mask)
    s2_cog = build_stage2(df, feats, oof_cog, "COGS", train_mask)

    p_s3_rev = apply_stage3(feat_mi, p1_rev, uplift_rev)
    p_s3_cog = apply_stage3(feat_mi, p1_cog, uplift_cog)
    p_s2_rev = apply_stage2(feat_mi, feats, p1_rev, s2_rev)
    p_s2_cog = apply_stage2(feat_mi, feats, p1_cog, s2_cog)
    p_s23_rev = apply_stage3(feat_mi, p_s2_rev, uplift_rev)
    p_s23_cog = apply_stage3(feat_mi, p_s2_cog, uplift_cog)

    trend_train = df.loc[train_mask, "aov_trend_proj"].mean()
    trend_mirror = df.loc[mirror_mask, "aov_trend_proj"].mean()
    trend_scale = float(trend_mirror / trend_train)
    print(f"\nTrend-derived level scale: {trend_scale:.4f}")
    p_scl_rev = p1_rev * trend_scale
    p_scl_cog = p1_cog * trend_scale
    p_scl_s3_rev = apply_stage3(feat_mi, p_scl_rev, uplift_rev)
    p_scl_s3_cog = apply_stage3(feat_mi, p_scl_cog, uplift_cog)

    variants = {
        "stage1_only":          (p1_rev,        p1_cog),
        "stage1+stage3":        (p_s3_rev,      p_s3_cog),
        "stage1+stage2":        (p_s2_rev,      p_s2_cog),
        "stage1+stage2+stage3": (p_s23_rev,     p_s23_cog),
        "stage1+trendscale":    (p_scl_rev,     p_scl_cog),
        "stage1+trendscale+s3": (p_scl_s3_rev,  p_scl_s3_cog),
    }
    results = {}
    for name, (pr, pc) in variants.items():
        s = score_predictions(
            val_dates=feat_mi["date"], y_true_rev=ymi_rev, y_pred_rev=pr,
            y_true_cogs=ymi_cog, y_pred_cogs=pc,
            features_df=feat_mi, train_y_for_pct=ytr_rev,
        )
        results[name] = s
        d = s["rmse_global_combined"] - V10C_FLOOR_COMBINED
        print(f"\n[{name}]  delta vs floor: {d:+,.0f}")
        print(fmt_score(s))

    best = min(results.items(), key=lambda kv: kv[1]["rmse_global_combined"])
    best_name, best_score = best
    delta_best = best_score["rmse_global_combined"] - V10C_FLOOR_COMBINED
    verdict = "BEATS V10c" if delta_best < 0 else "WORSE than V10c"
    print(f"\n=== Best: {best_name}  rmse_combined={best_score['rmse_global_combined']:,.0f}  ({verdict} by {abs(delta_best):,.0f}) ===")

    pr, pc = variants[best_name]
    pd.DataFrame({
        "date": feat_mi["date"], "rev_actual": ymi_rev, "rev_pred": pr,
        "cogs_actual": ymi_cog, "cogs_pred": pc, "variant": best_name,
        "trend_scale": trend_scale,
    }).to_parquet(PROCESSED / "v13_peak_recovery_mirror_preds.parquet", index=False)

    save = {
        "trend_scale": trend_scale,
        "Revenue": {k: {"multiplier": v[0], "n_obs": v[1]} for k, v in uplift_rev.items()},
        "COGS":    {k: {"multiplier": v[0], "n_obs": v[1]} for k, v in uplift_cog.items()},
    }
    (PROCESSED / "v13_event_uplift_post_model.json").write_text(json.dumps(save, indent=2), encoding="utf-8")

    rows = []
    for n, s in results.items():
        d = s["rmse_global_combined"] - V10C_FLOOR_COMBINED
        rows.append(f"| {n} | {s['rmse_global_combined']:,.0f} | {s['rmse_global_rev']:,.0f} | "
                    f"{s['rmse_peak_rev']:,.0f} | {s['mean_pred_rev']:,.0f} | {d:+,.0f} |")
    doc = [
        "# V13 Step 9 -- Peak Recovery Results",
        "",
        f"- Trend-derived level scale (honest, train-only): {trend_scale:.4f}",
        f"- V10c floor: {V10C_FLOOR_COMBINED:,d}",
        f"- **Best variant: {best_name}**  rmse_combined={best_score['rmse_global_combined']:,.0f}  ({verdict} by {abs(delta_best):,.0f})",
        "",
        "## Variant table",
        "",
        "| Variant | rmse_combined | rmse_rev | rmse_peak_rev | mean_pred | delta_vs_floor |",
        "|---|---:|---:|---:|---:|---:|",
    ] + rows
    (DOCS / "v13_peak_recovery_results.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
