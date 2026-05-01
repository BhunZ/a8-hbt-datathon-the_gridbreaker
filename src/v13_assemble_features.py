"""V13 Step 5 -- Feature assembly.

Joins the four V13 building blocks on date into a single feature table:
  v13_calendar.parquet  (CAL features)
  v13_priors.parquet    (PRIOR features)
  v13_lags.parquet      (LAG-Y features)
  v13_trends.parquet    (TREND features)

Adds targets (Revenue, COGS) from sales.parquet.  Targets are populated for
the train window only -- test rows get NaN, which is correct (we predict them).

Validates that NO column was sourced from a forbidden file:
  web_traffic, reviews, returns, signups, stockouts, inventory,
  orders, order_items, payments

Output: data/processed/daily_features_v13.parquet  shape=(2314, ~90)

Run: python src/v13_assemble_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paths import PROCESSED, DOCS

OUT_PARQUET = PROCESSED / "daily_features_v13.parquet"

TRAIN_START = pd.Timestamp("2019-09-01")
TRAIN_END   = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")
TEST_END    = pd.Timestamp("2024-07-01")

# Forbidden source-file derivatives (from V13 blueprint section 2)
FORBIDDEN_SOURCES = {
    "web_traffic", "reviews", "returns", "signups", "stockouts",
    "inventory", "orders", "order_items", "payments",
}


def load_blocks():
    cal    = pd.read_parquet(PROCESSED / "v13_calendar.parquet")
    priors = pd.read_parquet(PROCESSED / "v13_priors.parquet")
    lags   = pd.read_parquet(PROCESSED / "v13_lags.parquet")
    trends = pd.read_parquet(PROCESSED / "v13_trends.parquet")

    for df in (cal, priors, lags, trends):
        df["date"] = pd.to_datetime(df["date"])

    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    sales = sales.rename(columns={"Date": "date"})
    sales["date"] = pd.to_datetime(sales["date"])

    return cal, priors, lags, trends, sales


def assemble(cal, priors, lags, trends, sales):
    out = cal.merge(priors, on="date", how="left", validate="one_to_one")
    out = out.merge(lags,   on="date", how="left", validate="one_to_one")
    out = out.merge(trends, on="date", how="left", validate="one_to_one")

    # Cross-features: trend-adjusted priors and lags.  These give the model an
    # explicit "this month's prior level multiplied by the where-we-are-in-the-
    # post-cliff-trend factor", which raw priors alone (static averages) cannot
    # express.  Mirror-block testing showed adding these closes ~50k RMSE.
    trend_ratio = out["aov_trend_proj"] / out["aov_prior_by_month"]
    out["trend_adj_prior_rev"]  = out["prior_rev_by_month_dow"]  * trend_ratio
    out["trend_adj_prior_cogs"] = out["prior_cogs_by_month_dow"] * trend_ratio
    out["lag728_trend_rev"]     = out["rev_yoy_lag_728"]  * trend_ratio
    out["lag728_trend_cogs"]    = out["cogs_yoy_lag_728"] * trend_ratio
    out["lag364_trend_rev"]     = out["rev_yoy_lag_364"]  * trend_ratio
    out["lag364_trend_cogs"]    = out["cogs_yoy_lag_364"] * trend_ratio

    # Targets -- left join, train rows populated, test rows NaN
    out = out.merge(sales[["date", "Revenue", "COGS"]],
                    on="date", how="left", validate="one_to_one")

    # Train/test split flags (deterministic from date)
    out["is_train"] = ((out["date"] >= TRAIN_START) & (out["date"] <= TRAIN_END)).astype(int)
    out["is_test"]  = ((out["date"] >= TEST_START)  & (out["date"] <= TEST_END)).astype(int)

    return out


def assert_no_forbidden(df: pd.DataFrame) -> list:
    """Sanity check -- any column name suggesting it came from a DROP-tagged file?"""
    suspicious = []
    cols = [c.lower() for c in df.columns]
    for src in FORBIDDEN_SOURCES:
        # Heuristic: any column containing the source name or its plural/short form
        for c, lc in zip(df.columns, cols):
            if src in lc and c not in ("is_christmas",):  # 'mas' could trip false positives
                suspicious.append((c, src))
    return suspicious


def main():
    cal, priors, lags, trends, sales = load_blocks()
    df = assemble(cal, priors, lags, trends, sales)
    print(f"Assembled shape: {df.shape}")
    print(f"Columns: {df.shape[1]}")

    # Sanity: no forbidden file derivatives
    suspicious = assert_no_forbidden(df)
    if suspicious:
        print("FORBIDDEN-SOURCE WARNING:")
        for col, src in suspicious:
            print(f"  {col} (source: {src})")
    else:
        print("No forbidden-source columns detected.")

    # Coverage report
    train_mask = df["is_train"] == 1
    test_mask  = df["is_test"]  == 1
    print()
    print(f"Train rows: {int(train_mask.sum())}  (target Revenue notna: "
          f"{int(df.loc[train_mask, 'Revenue'].notna().sum())})")
    print(f"Test  rows: {int(test_mask.sum())}   (target Revenue notna: "
          f"{int(df.loc[test_mask, 'Revenue'].notna().sum())} -- expect 0)")
    print()

    # Per-block null counts on train vs test
    feat_cols = [c for c in df.columns if c not in ("date", "Revenue", "COGS",
                                                    "is_train", "is_test")]
    null_train = df.loc[train_mask, feat_cols].isnull().sum()
    null_test  = df.loc[test_mask,  feat_cols].isnull().sum()
    nullable = sorted({c for c in feat_cols if null_train[c] > 0 or null_test[c] > 0})
    if nullable:
        print(f"Columns with any nulls in train+test (n={len(nullable)}):")
        for c in nullable:
            print(f"  {c:35s}  null(train)={int(null_train[c]):4d}  null(test)={int(null_test[c]):4d}")
    else:
        print("All feature columns are fully populated in train+test.")

    df.to_parquet(OUT_PARQUET, index=False)
    print()
    print(f"wrote {OUT_PARQUET}  shape={df.shape}")


if __name__ == "__main__":
    main()
