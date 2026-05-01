"""V13 Step 7 -- Rebaseline V10c on the new mirror block.

Re-runs V10c's exact 23-feature setup but on the post-cliff window:
  Train: 2019-09-01 -> 2021-12-31  (was: 2017-01-01 -> 2022-12-31 in original V10c)
  Score: 2022-01-01 -> 2022-12-31  (the V13 mirror block)

The RMSE on this mirror block IS THE FLOOR V13 must beat.

V10c's approach (faithfully reproduced):
  -- 23 features: calendar + named events + median-projected web_traffic
  -- log1p target
  -- ensemble: RandomForest (30%) + ExtraTrees (50%) + HistGB (20%), 2 seeds
  -- sample weight = clip(years - 2018, 1.0, None) ** 1.2
     (was years - 2016 in original; we shifted because we start from 2019)

Outputs:
  data/processed/v13_v10c_mirror_preds.parquet  -- per-date predictions
  docs/v13_v10c_rebaseline.md                   -- the floor number

Run: python src/v13_v10c_rebaseline.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor)

from paths import RAW, PROCESSED, DOCS
from v13_validation import (POST_CLIFF_START, MIRROR_START, MIRROR_END,
                             MIRROR_TRAIN_END, score_predictions, fmt_score)

t0 = time.time()


# --- V10c calendar / event helpers (replicated from phase7_v10c_fit.py) ----

TET = [pd.Timestamp(d) for d in [
    "2019-02-05", "2020-01-25", "2021-02-12",
    "2022-02-01", "2023-01-22", "2024-02-10"
]]


def nth_weekday_of_month(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    shift = (wd - d.weekday()) % 7
    return d + pd.Timedelta(days=shift + 7 * (n - 1))


def build_v10c_features(df, daily_traffic, traffic_avg, *, use_actual_traffic=True):
    """Replicate V10c's feature engineering."""
    df = df.copy()
    df["year"]  = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["day"]   = df.Date.dt.day
    df["dow"]   = df.Date.dt.dayofweek
    if use_actual_traffic:
        df = df.merge(daily_traffic[["Date", "sessions", "page_views",
                                      "unique_visitors", "bounce_rate", "avg_sess"]],
                      on="Date", how="left")
    else:
        df = df.merge(traffic_avg, on=["month", "dow"], how="left")
    for c in ["sessions", "page_views", "unique_visitors", "bounce_rate", "avg_sess"]:
        df[c] = df[c].fillna(0)
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df.dow / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df.dow / 7)
    df["is_payday"] = ((df.day >= 25) | (df.day <= 5)).astype(int)
    df["is_double_day"] = (df.month == df.day).astype(int)
    df["is_tet_season"] = 0
    for td in TET:
        m = (df.Date >= td - pd.Timedelta(days=21)) & (df.Date < td)
        df.loc[m, "is_tet_season"] = 1
    df["is_holiday_period"] = 0
    for mh, dh in [(1, 1), (4, 30), (5, 1), (9, 2), (12, 24), (12, 25), (12, 31)]:
        for off in range(0, 4):
            df.loc[(df.month == mh) & (df.day == (dh - off)), "is_holiday_period"] = 1
    df["is_singles_day"]   = ((df.month == 11) & (df.day == 11)).astype(int)
    df["is_nine_nine"]     = ((df.month == 9)  & (df.day == 9)).astype(int)
    df["is_twelve_twelve"] = ((df.month == 12) & (df.day == 12)).astype(int)
    df["is_black_friday"] = 0
    df["is_cyber_monday"] = 0
    df["is_mothers_day"] = 0
    for y in df.year.unique():
        bf = nth_weekday_of_month(int(y), 11, 4, 4)
        thx = nth_weekday_of_month(int(y), 11, 3, 4)
        md = nth_weekday_of_month(int(y), 5, 6, 2)
        df.loc[df.Date == bf, "is_black_friday"] = 1
        df.loc[df.Date == thx + pd.Timedelta(days=4), "is_cyber_monday"] = 1
        df.loc[df.Date == md, "is_mothers_day"] = 1
    flash = (df.is_singles_day + df.is_twelve_twelve + df.is_nine_nine
             + df.is_black_friday + df.is_cyber_monday)
    df["near_flash_event"] = 0
    for off in range(-3, 4):
        shifted = flash.shift(off).fillna(0)
        df["near_flash_event"] = np.maximum(df["near_flash_event"],
                                             (shifted > 0).astype(int))
    return df


FEATS = [
    "month_sin", "month_cos", "dow_sin", "dow_cos", "day", "month", "dow",
    "is_payday", "is_double_day", "is_tet_season", "is_holiday_period",
    "is_singles_day", "is_nine_nine", "is_twelve_twelve", "is_black_friday",
    "is_cyber_monday", "is_mothers_day", "near_flash_event",
    "sessions", "page_views", "unique_visitors", "bounce_rate", "avg_sess",
]


def main():
    sales   = pd.read_csv(RAW / "sales.csv",       parse_dates=["Date"])
    traffic = pd.read_csv(RAW / "web_traffic.csv", parse_dates=["date"])

    daily_t = traffic.groupby("date").agg(
        sessions=("sessions", "sum"), page_views=("page_views", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_sess=("avg_session_duration_sec", "mean")
    ).reset_index().rename(columns={"date": "Date"})
    daily_t["month"] = daily_t.Date.dt.month
    daily_t["dow"]   = daily_t.Date.dt.dayofweek

    # IMPORTANT: median-projected traffic is computed ONLY on the train window
    # (2019-09 -> 2021-12), so the mirror score reflects what V10c would have
    # done if we'd held out 2022.
    train_traffic = daily_t[(daily_t.Date >= POST_CLIFF_START) &
                             (daily_t.Date <= MIRROR_TRAIN_END)]
    traffic_avg = (train_traffic.groupby(["month", "dow"])
                   [["sessions", "page_views", "unique_visitors",
                     "bounce_rate", "avg_sess"]].mean().reset_index())

    train_sales = sales[(sales.Date >= POST_CLIFF_START) &
                        (sales.Date <= MIRROR_TRAIN_END)].reset_index(drop=True)
    mirror_sales = sales[(sales.Date >= MIRROR_START) &
                         (sales.Date <= MIRROR_END)].reset_index(drop=True)
    print(f"[{time.time()-t0:.1f}s] train rows={len(train_sales)}  mirror rows={len(mirror_sales)}")

    train_f  = build_v10c_features(train_sales[["Date"]],  daily_t, traffic_avg, use_actual_traffic=True)
    # On the mirror block, V10c style means projected traffic (since we pretend we don't have it):
    mirror_f = build_v10c_features(mirror_sales[["Date"]], daily_t, traffic_avg, use_actual_traffic=False)
    print(f"[{time.time()-t0:.1f}s] train_f={train_f.shape}  mirror_f={mirror_f.shape}")

    X_tr = train_f[FEATS].values
    X_mi = mirror_f[FEATS].values
    # Sample weight: years since 2018, ^1.2.  In V10c original it was years - 2016.
    w_tr = np.clip(train_f["year"].values - 2018, 1.0, None) ** 1.2

    SEEDS = [42, 17]
    preds = {"Revenue": [], "COGS": []}
    for tgt in ["Revenue", "COGS"]:
        y_tr = np.log1p(train_sales[tgt].values)
        for s in SEEDS:
            print(f"  [{tgt}] RF s={s}...", flush=True)
            rf = RandomForestRegressor(n_estimators=350, max_depth=15,
                                        random_state=s, n_jobs=-1)
            rf.fit(X_tr, y_tr, sample_weight=w_tr)
            preds[tgt].append(("rf", 0.30 / len(SEEDS), np.expm1(rf.predict(X_mi))))

            print(f"  [{tgt}] ET s={s}...", flush=True)
            et = ExtraTreesRegressor(n_estimators=450, max_depth=16,
                                      random_state=s, n_jobs=-1)
            et.fit(X_tr, y_tr, sample_weight=w_tr)
            preds[tgt].append(("et", 0.50 / len(SEEDS), np.expm1(et.predict(X_mi))))
        print(f"  [{tgt}] HGB...", flush=True)
        hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                             max_depth=10, random_state=42)
        hgb.fit(X_tr, y_tr, sample_weight=w_tr)
        preds[tgt].append(("hgb", 0.20, np.expm1(hgb.predict(X_mi))))
        print(f"  [{tgt}] done at {time.time()-t0:.1f}s")

    p_rev  = sum(wt * p for _, wt, p in preds["Revenue"])
    p_cogs = sum(wt * p for _, wt, p in preds["COGS"])

    out = pd.DataFrame({
        "date":     mirror_sales.Date,
        "rev_actual":  mirror_sales.Revenue.values,
        "rev_pred":    p_rev,
        "cogs_actual": mirror_sales.COGS.values,
        "cogs_pred":   p_cogs,
    })
    out.to_parquet(PROCESSED / "v13_v10c_mirror_preds.parquet", index=False)

    # Score using V13 validation harness
    v13_feat = pd.read_parquet(PROCESSED / "daily_features_v13.parquet")
    v13_feat["date"] = pd.to_datetime(v13_feat["date"])
    mirror_feat = v13_feat[(v13_feat["date"] >= MIRROR_START) & (v13_feat["date"] <= MIRROR_END)]
    train_y_for_pct = train_sales.Revenue.values

    score = score_predictions(
        val_dates=mirror_sales.Date,
        y_true_rev=mirror_sales.Revenue.values,
        y_pred_rev=p_rev,
        y_true_cogs=mirror_sales.COGS.values,
        y_pred_cogs=p_cogs,
        features_df=mirror_feat,
        train_y_for_pct=train_y_for_pct,
    )
    print()
    print(f"=== V10c rebaseline on mirror_2022 ({MIRROR_START.date()}..{MIRROR_END.date()}) ===")
    print(fmt_score(score))
    print(f"\nDONE in {time.time()-t0:.1f}s")

    # Write to docs
    lines = [
        "# V13 Step 7 -- V10c Rebaseline on Mirror Block",
        "",
        f"Reproduces V10c's 23-feature setup on the post-cliff train window.",
        f"Train: {POST_CLIFF_START.date()} .. {MIRROR_TRAIN_END.date()}  (rows={len(train_sales)})",
        f"Mirror: {MIRROR_START.date()} .. {MIRROR_END.date()}  (rows={len(mirror_sales)})",
        "",
        "## Floor scores (V13 must beat these)",
        "",
        f"- **rmse_combined:** {score['rmse_global_combined']:,.0f}",
        f"- rmse_rev:  {score['rmse_global_rev']:,.0f}    rmse_peak_rev:  {score['rmse_peak_rev']:,.0f}",
        f"- rmse_cogs: {score['rmse_global_cogs']:,.0f}   rmse_peak_cogs: {score['rmse_peak_cogs']:,.0f}",
        f"- rmse_log_rev: {score['rmse_log_rev']:.4f}    rmse_log_cogs: {score['rmse_log_cogs']:.4f}",
        f"- mean predicted revenue: {score['mean_pred_rev']:,.0f}/day  vs actual: {score['mean_actual_rev']:,.0f}/day",
        f"- mean predicted COGS:    {score['mean_pred_cogs']:,.0f}/day  vs actual: {score['mean_actual_cogs']:,.0f}/day",
        f"- n_rows: {score['n_rows']}    n_peak_rows: {score['n_peak_rows']}",
        "",
        "## Reading",
        "",
        "On Kaggle (full 2017+ training), V10c scored **774,898** RMSE.",
        "On this redrawn mirror block (post-cliff training only), V10c scores ",
        f"**{score['rmse_global_rev']:,.0f}** for revenue.  This is harder than Kaggle because:",
        "  1. Mirror block size is 365 days vs Kaggle's 548; volatility per day is similar.",
        "  2. Train window is 2.3 years (post-cliff only) vs Kaggle's full 6 years.",
        "  3. The model is data-starved on early train rows (lag-364 reaches pre-cliff).",
        "",
        "**This number is the V13 floor.**  If V13 scores worse than this on the same mirror, "
        "we don't submit.  If it scores better, we have a real reason to try Kaggle.",
    ]
    (DOCS / "v13_v10c_rebaseline.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"docs -> {DOCS / 'v13_v10c_rebaseline.md'}")


if __name__ == "__main__":
    main()
