"""V13 Step 3 -- Long-horizon lag features (post-cliff window).

Computes lag features from sales.csv where the LOOKUP source is restricted to
the train window 2019-09-01 -> 2022-12-31.  Where a lookup falls outside that
window (pre-cliff or test horizon), the feature is NaN.

The six features:
  Point lags (single-day lookback):
    rev_yoy_lag_364   = rev[d - 364]
    rev_yoy_lag_728   = rev[d - 728]
    cogs_yoy_lag_364  = cogs[d - 364]
    cogs_yoy_lag_728  = cogs[d - 728]

  Same-DOW averaged lags (52 prior same-dow days):
    rev_same_dow_prev_year  = mean(rev[d - 7 * k]  for k in 1..52
                                    where d - 7*k is in train window)
    cogs_same_dow_prev_year = analogous

  Same-DOM averaged lags (12 prior same-day-of-month values):
    rev_same_dom_prev_year  = mean(rev[d - DateOffset(months=k)] for k in 1..12
                                    where the lookup is in train window)
    cogs_same_dom_prev_year = analogous

Honesty rule (V13 section 0):
  -- For any date d (train or test), the lookup MUST fall in [2019-09-01, 2022-12-31].
  -- If a lookup falls before 2019-09-01 (pre-cliff regime) -> NaN
  -- If a lookup falls after 2022-12-31 (test horizon)      -> NaN

For test dates 2023-01-01 .. 2023-12-31, lag-364 is honest (lookup in 2022).
For test dates 2024-01-01 .. 2024-07-01, lag-364 is NaN (lookup falls in test).
For all test dates, lag-728 is honest (lookup falls in train).

Output: data/processed/v13_lags.parquet  keyed by date

Run: python src/v13_lags.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paths import PROCESSED, DOCS

TRAIN_START = pd.Timestamp("2019-09-01")
TRAIN_END   = pd.Timestamp("2022-12-31")

OUT_PARQUET = PROCESSED / "v13_lags.parquet"

MIN_DOW_OBS = 26   # require at least half of 52 same-dow lookups
MIN_DOM_OBS = 6    # require at least half of 12 same-dom lookups


def load_inputs():
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    sales = sales.rename(columns={"Date": "date"})
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales[["date", "Revenue", "COGS"]].sort_values("date").reset_index(drop=True)

    cal = pd.read_parquet(PROCESSED / "v13_calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return sales, cal


def build_lookup(sales: pd.DataFrame) -> dict:
    """Map date -> (Revenue, COGS) for ONLY the train window.  Lookups outside
    the train window return NaN."""
    train = sales[(sales["date"] >= TRAIN_START) & (sales["date"] <= TRAIN_END)]
    return {row.date: (row.Revenue, row.COGS) for row in train.itertuples(index=False)}


def safe_get(lookup: dict, d: pd.Timestamp):
    val = lookup.get(d)
    if val is None:
        return (np.nan, np.nan)
    return val


def compute_point_lag(dates: pd.Series, lookup: dict, days: int):
    rev = np.empty(len(dates))
    cog = np.empty(len(dates))
    rev.fill(np.nan); cog.fill(np.nan)
    for i, d in enumerate(dates):
        target = d - pd.Timedelta(days=days)
        v = lookup.get(target)
        if v is not None:
            rev[i], cog[i] = v[0], v[1]
    return rev, cog


def compute_same_dow_prev_year(dates: pd.Series, lookup: dict):
    """For each d, mean of revenue/COGS on the 52 same-dow days at d-7k for k=1..52,
    using only lookups that fall inside the train window."""
    rev = np.empty(len(dates)); cog = np.empty(len(dates))
    rev.fill(np.nan); cog.fill(np.nan)
    for i, d in enumerate(dates):
        rev_vals, cog_vals = [], []
        for k in range(1, 53):
            target = d - pd.Timedelta(days=7 * k)
            v = lookup.get(target)
            if v is not None:
                rev_vals.append(v[0])
                cog_vals.append(v[1])
        if len(rev_vals) >= MIN_DOW_OBS:
            rev[i] = np.mean(rev_vals)
        if len(cog_vals) >= MIN_DOW_OBS:
            cog[i] = np.mean(cog_vals)
    return rev, cog


def compute_same_dom_prev_year(dates: pd.Series, lookup: dict):
    """For each d, mean of revenue/COGS on the 12 same-day-of-month dates
    d - 1 month, d - 2 months, ..., d - 12 months, restricted to train window."""
    rev = np.empty(len(dates)); cog = np.empty(len(dates))
    rev.fill(np.nan); cog.fill(np.nan)
    for i, d in enumerate(dates):
        rev_vals, cog_vals = [], []
        for k in range(1, 13):
            target = d - pd.DateOffset(months=k)
            # pandas may clamp Feb 30 -> Feb 28; that's fine
            v = lookup.get(pd.Timestamp(target))
            if v is not None:
                rev_vals.append(v[0])
                cog_vals.append(v[1])
        if len(rev_vals) >= MIN_DOM_OBS:
            rev[i] = np.mean(rev_vals)
        if len(cog_vals) >= MIN_DOM_OBS:
            cog[i] = np.mean(cog_vals)
    return rev, cog


def main():
    sales, cal = load_inputs()
    lookup = build_lookup(sales)
    print(f"lookup size (train window): {len(lookup)} days")

    dates = pd.to_datetime(cal["date"])

    rev_l364,  cogs_l364  = compute_point_lag(dates, lookup, 364)
    rev_l728,  cogs_l728  = compute_point_lag(dates, lookup, 728)
    rev_dow,   cogs_dow   = compute_same_dow_prev_year(dates, lookup)
    rev_dom,   cogs_dom   = compute_same_dom_prev_year(dates, lookup)

    out = pd.DataFrame({
        "date": dates,
        "rev_yoy_lag_364":         rev_l364,
        "rev_yoy_lag_728":         rev_l728,
        "cogs_yoy_lag_364":        cogs_l364,
        "cogs_yoy_lag_728":        cogs_l728,
        "rev_same_dow_prev_year":  rev_dow,
        "cogs_same_dow_prev_year": cogs_dow,
        "rev_same_dom_prev_year":  rev_dom,
        "cogs_same_dom_prev_year": cogs_dom,
    })
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}  shape={out.shape}")

    # Summary diagnostics
    test_mask = (out["date"] >= "2023-01-01") & (out["date"] <= "2024-07-01")
    train_mask = (out["date"] >= "2019-09-01") & (out["date"] <= "2022-12-31")
    print()
    print("Coverage (% of rows with non-null lag):")
    for col in [c for c in out.columns if c != "date"]:
        cov_train = out.loc[train_mask, col].notna().mean()
        cov_test  = out.loc[test_mask, col].notna().mean()
        print(f"  {col:30s}  train: {cov_train*100:5.1f}%  test: {cov_test*100:5.1f}%")


if __name__ == "__main__":
    main()
