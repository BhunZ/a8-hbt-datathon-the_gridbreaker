"""V13 Step 4 -- Trend features (post-cliff window).

Fits linear trends on three monthly metrics across the post-cliff train window
(2019-09-01 -> 2022-12-31) and materializes them as daily-grain projections
for the full calendar window.

Why these features exist:
  The V13 timeline audit found a clear monotonic drift inside the post-cliff
  regime: AOV +19% from 2019 to 2022, unit_price +22%, n_active_skus -10%.
  The test horizon (2023-01 .. 2024-07) lives at the FAR END of this drift.
  Without trend features the model anchors on the train-window mean, which
  systematically underestimates 2024-AOV and overestimates 2024-SKU-count.

Three metrics:
  aov_trend_proj          = mean daily AOV (Revenue/n_orders), monthly
  unit_price_trend_proj   = mean unit_price across order_items, monthly
  n_active_skus_trend_proj = unique product_id count in orders that month

Honesty rule:
  -- Slope+intercept and seasonal_means are frozen at train time.
  -- At inference, projection = seasonal_mean[month(d)] + slope * months_since_2019(d) + intercept_resid
  -- months_since_2019 is already a deterministic CAL feature in the calendar.

We DE-SEASONALIZE before fitting OLS so the trend slope captures real drift,
not month-to-month noise (e.g. December AOV dips).

Outputs:
  data/processed/v13_trends.json     -- slope+intercept+seasonal_means per metric
  data/processed/v13_trends.parquet  -- daily-grain projections (date-keyed)

Run: python src/v13_trends.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import PROCESSED, DOCS

TRAIN_START = pd.Timestamp("2019-09-01")
TRAIN_END   = pd.Timestamp("2022-12-31")
REGIME_START = pd.Timestamp("2019-09-01")  # x=0 in months_since_2019

OUT_PARQUET = PROCESSED / "v13_trends.parquet"
OUT_JSON    = PROCESSED / "v13_trends.json"


def load_inputs():
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    sales = sales.rename(columns={"Date": "date"})
    sales["date"] = pd.to_datetime(sales["date"])

    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    items = pd.read_parquet(PROCESSED / "order_items.parquet")
    items = items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    items["date"] = pd.to_datetime(items["order_date"])
    items["unit_price"] = items["unit_price"].astype(float)

    cal = pd.read_parquet(PROCESSED / "v13_calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return sales, orders, items, cal


def in_train(df, date_col):
    return df[(df[date_col] >= TRAIN_START) & (df[date_col] <= TRAIN_END)].copy()


def monthly_aggregates(sales, orders, items):
    s_train = in_train(sales, "date")
    o_train = in_train(orders, "order_date")
    i_train = in_train(items, "date")

    n_orders_daily = o_train.groupby("order_date").size().rename("n_orders").reset_index()
    n_orders_daily.columns = ["date", "n_orders"]
    daily = s_train.merge(n_orders_daily, on="date", how="inner")
    daily["month_start"] = daily["date"].values.astype("datetime64[M]")
    aov_monthly = (daily.groupby("month_start")
                        .agg(rev_total=("Revenue", "sum"),
                             orders_total=("n_orders", "sum"))
                        .reset_index())
    aov_monthly["aov"] = aov_monthly["rev_total"] / aov_monthly["orders_total"]

    i_train["month_start"] = i_train["date"].values.astype("datetime64[M]")
    up_monthly = (i_train.groupby("month_start")
                          .agg(avg_unit_price=("unit_price", "mean"),
                               n_line_items=("unit_price", "size"))
                          .reset_index())
    sku_monthly = (i_train.groupby("month_start")
                          ["product_id"].nunique()
                          .rename("n_active_skus").reset_index())

    out = (aov_monthly[["month_start", "aov"]]
           .merge(up_monthly[["month_start", "avg_unit_price"]], on="month_start")
           .merge(sku_monthly, on="month_start"))
    out["months_since_2019"] = (
        (pd.to_datetime(out["month_start"]) - REGIME_START).dt.days / 30.4375
    )
    return out


def fit_trend(monthly, value_col):
    df = monthly.copy()
    df["month_of_year"] = pd.to_datetime(df["month_start"]).dt.month
    seasonal = df.groupby("month_of_year")[value_col].mean().rename("seasonal_mean")
    df = df.merge(seasonal, on="month_of_year", how="left")
    df["resid"] = df[value_col] - df["seasonal_mean"]

    x = df["months_since_2019"].values
    y_resid = df["resid"].values
    slope, intercept_resid = np.polyfit(x, y_resid, deg=1)
    y_hat = intercept_resid + slope * x
    ss_res = np.sum((y_resid - y_hat) ** 2)
    ss_tot = np.sum((y_resid - y_resid.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    raw_slope, raw_intercept = np.polyfit(x, df[value_col].values, deg=1)

    return {
        "slope":               float(slope),
        "intercept_resid":     float(intercept_resid),
        "r2_detrended":        float(r2),
        "delta_per_year":      float(slope * 12.0),
        "seasonal_means":      {int(k): float(v) for k, v in seasonal.to_dict().items()},
        "raw_level_slope":     float(raw_slope),
        "raw_level_intercept": float(raw_intercept),
        "n":                   int(len(x)),
        "mean_y":              float(df[value_col].mean()),
    }


def main():
    sales, orders, items, cal = load_inputs()
    monthly = monthly_aggregates(sales, orders, items)
    print("Monthly aggregates (post-cliff train window):")
    print(monthly.round(2).to_string(index=False))
    print()

    fits = {
        "aov_trend":           fit_trend(monthly, "aov"),
        "unit_price_trend":    fit_trend(monthly, "avg_unit_price"),
        "n_active_skus_trend": fit_trend(monthly, "n_active_skus"),
    }
    fits["meta"] = {
        "train_window":  [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "regime_start":  str(REGIME_START.date()),
        "n_months_fit":  int(len(monthly)),
        "x_definition":  "months_since_2019 = (month_start - 2019-09-01).days / 30.4375",
        "projection":    "value(d) = seasonal_mean[month(d)] + slope * months_since_2019(d) + intercept_resid",
    }
    OUT_JSON.write_text(json.dumps(fits, indent=2), encoding="utf-8")
    print("Trend fits (de-seasonalized residual):")
    for k in ["aov_trend", "unit_price_trend", "n_active_skus_trend"]:
        f = fits[k]
        print(f"  {k:24s}  slope={f['slope']:+.4f}/mo  R^2(resid)={f['r2_detrended']:.3f}  "
              f"delta/yr={f['delta_per_year']:+.2f}  raw_slope={f['raw_level_slope']:+.4f}/mo")
    print(f"wrote {OUT_JSON}")
    print()

    cal_dates = cal[["date"]].copy()
    cal_dates["months_since_2019"] = (
        (cal_dates["date"].dt.normalize() - REGIME_START).dt.days / 30.4375
    )
    cal_dates["month"] = cal_dates["date"].dt.month

    def project(metric_key):
        f = fits[metric_key]
        season = cal_dates["month"].map(f["seasonal_means"]).astype(float).values
        return season + f["slope"] * cal_dates["months_since_2019"].values + f["intercept_resid"]

    out = pd.DataFrame({"date": cal_dates["date"]})
    out["aov_trend_proj"]           = project("aov_trend")
    out["unit_price_trend_proj"]    = project("unit_price_trend")
    out["n_active_skus_trend_proj"] = project("n_active_skus_trend")
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}  shape={out.shape}")

    print()
    print("Projections at key dates:")
    for d in ["2019-09-01", "2022-12-01", "2023-07-01", "2024-01-01", "2024-07-01"]:
        row = out[out["date"] == d]
        if len(row):
            r = row.iloc[0]
            print(f"  {d}: aov={r.aov_trend_proj:,.0f}  "
                  f"unit_price={r.unit_price_trend_proj:,.0f}  "
                  f"n_skus={r.n_active_skus_trend_proj:.1f}")


if __name__ == "__main__":
    main()
