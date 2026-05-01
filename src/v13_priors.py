"""V13 Step 2 -- Frozen priors on the post-cliff window (2019-09-01 -> 2022-12-31).

Computes four lookup tables from train-window data and materializes them on
the full calendar window so feature assembly is just a date join.

The four priors:
  1. prior_rev_by_month_dow  -- mean daily Revenue keyed by (month, dow), 12x7 = 84 cells
  2. prior_cogs_by_month_dow -- mean daily COGS keyed by (month, dow)
  3. aov_prior_by_month      -- mean daily AOV (= Revenue / n_orders) keyed by month, 12 cells
  4. prior_rev_event_uplift / prior_cogs_event_uplift
        -- multiplicative ratio of mean( actual / prior_by_month_dow )
           on event days only.  This is a SEASON-MATCHED uplift -- the right
           definition for events that fall in non-uniform months (e.g. Black
           Friday in November, the brand's lowest-revenue month).

These satisfy V13 section 0: every value is a frozen scalar / lookup, computed
once on closed train history, joined by date components or by date directly.

Outputs:
  data/processed/v13_priors.parquet   -- daily-grain table covering the full
                                         calendar window (date + 5 prior cols)
  data/processed/v13_priors_meta.json -- the underlying lookup tables for audit

Run: python src/v13_priors.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import PROCESSED, DOCS

TRAIN_START = pd.Timestamp("2019-09-01")
TRAIN_END   = pd.Timestamp("2022-12-31")

OUT_PARQUET = PROCESSED / "v13_priors.parquet"
OUT_META    = PROCESSED / "v13_priors_meta.json"

# Event flags considered for uplift (single-day events only -- ramps like
# is_pre_tet_window are NOT event days).
EVENT_FLAGS = [
    "is_tet",
    "is_singles_day",
    "is_twelve_twelve",
    "is_nine_nine",
    "is_ten_ten",
    "is_black_friday",
    "is_cyber_monday",
    "is_intl_women_day",
    "is_womens_day_vn",
    "is_teachers_day_vn",
    "is_valentines",
    "is_christmas",
    "is_boxing_day",
    "is_new_year_day",
    "is_independence_day_vn",
    "is_reunification_day",
    "is_intl_labour_day",
]


def load_inputs():
    sales = pd.read_parquet(PROCESSED / "sales.parquet")
    sales = sales.rename(columns={"Date": "date"})
    sales["date"] = pd.to_datetime(sales["date"])

    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    n_orders = (orders.groupby("order_date").size()
                .rename("n_orders").reset_index()
                .rename(columns={"order_date": "date"}))

    cal = pd.read_parquet(PROCESSED / "v13_calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])

    return sales, n_orders, cal


def in_train(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    return df[(df[date_col] >= TRAIN_START) & (df[date_col] <= TRAIN_END)].copy()


def compute_priors(sales: pd.DataFrame, n_orders: pd.DataFrame, cal: pd.DataFrame):
    """Compute the four lookup tables on the train window."""
    cal_train = in_train(cal)
    sales_train = in_train(sales)
    sales_cal = sales_train.merge(cal_train, on="date", how="inner",
                                  validate="one_to_one")
    sales_cal = sales_cal.merge(n_orders, on="date", how="left")

    sales_cal["aov"] = sales_cal["Revenue"] / sales_cal["n_orders"]

    # 1+2. month x dow priors
    md = (sales_cal.groupby(["month", "dow"])
                   .agg(prior_rev=("Revenue", "mean"),
                        prior_cogs=("COGS", "mean"),
                        n_obs=("Revenue", "size"))
                   .reset_index())

    # 3. monthly AOV prior
    aov_m = (sales_cal.groupby("month")
                      .agg(aov_prior_by_month=("aov", "mean"),
                           n_obs=("aov", "size"))
                      .reset_index())

    # 4. event uplift -- SEASON-MATCHED (vs month-dow prior, not global mean)
    flag_cols_present = [c for c in EVENT_FLAGS if c in sales_cal.columns]
    sales_cal = sales_cal.merge(
        md[["month", "dow", "prior_rev", "prior_cogs"]].rename(
            columns={"prior_rev": "_md_rev", "prior_cogs": "_md_cogs"}),
        on=["month", "dow"], how="left", validate="many_to_one")
    sales_cal["rev_ratio"]  = sales_cal["Revenue"] / sales_cal["_md_rev"]
    sales_cal["cogs_ratio"] = sales_cal["COGS"]    / sales_cal["_md_cogs"]

    # Global no-event baseline kept only for reporting
    sales_cal["any_event"] = sales_cal[flag_cols_present].sum(axis=1).clip(0, 1)
    baseline_mask = sales_cal["any_event"] == 0
    base_rev  = float(sales_cal.loc[baseline_mask, "Revenue"].mean())
    base_cogs = float(sales_cal.loc[baseline_mask, "COGS"].mean())

    rows = []
    for flag in flag_cols_present:
        m = sales_cal[flag] == 1
        n = int(m.sum())
        if n == 0:
            rows.append((flag, 1.0, 1.0, 0))
            continue
        rev_mult  = float(sales_cal.loc[m, "rev_ratio"].mean())
        cogs_mult = float(sales_cal.loc[m, "cogs_ratio"].mean())
        rows.append((flag, rev_mult, cogs_mult, n))
    ev_uplift = pd.DataFrame(rows, columns=[
        "event", "uplift_rev", "uplift_cogs", "n_event_days"
    ])

    return md, aov_m, ev_uplift, base_rev, base_cogs


def materialize_daily(cal: pd.DataFrame,
                      md: pd.DataFrame,
                      aov_m: pd.DataFrame,
                      ev_uplift: pd.DataFrame) -> pd.DataFrame:
    """Join priors onto the full calendar window so inference is a date join."""
    out = cal[["date", "month", "dow"] + EVENT_FLAGS].copy()
    out = out.merge(md[["month", "dow", "prior_rev", "prior_cogs"]],
                    on=["month", "dow"], how="left",
                    validate="many_to_one")
    out = out.merge(aov_m[["month", "aov_prior_by_month"]],
                    on="month", how="left",
                    validate="many_to_one")
    out = out.rename(columns={"prior_rev":  "prior_rev_by_month_dow",
                              "prior_cogs": "prior_cogs_by_month_dow"})

    # When multiple event flags fire on the same day, pick the one with the
    # largest |log(multiplier)| -- i.e. the most informative event, whether
    # it's a boost (>1) or a suppression (<1).  Default 1.0 when no flag set.
    flag_to_mult_rev  = dict(zip(ev_uplift["event"], ev_uplift["uplift_rev"]))
    flag_to_mult_cogs = dict(zip(ev_uplift["event"], ev_uplift["uplift_cogs"]))

    rev_uplift  = np.ones(len(out), dtype=float)
    cogs_uplift = np.ones(len(out), dtype=float)
    rev_dev  = np.zeros(len(out), dtype=float)   # |log(mult)| of currently chosen
    cogs_dev = np.zeros(len(out), dtype=float)
    for flag in EVENT_FLAGS:
        if flag not in out.columns or flag not in flag_to_mult_rev:
            continue
        mask = out[flag].values.astype(bool)
        if not mask.any():
            continue
        m_rev  = flag_to_mult_rev[flag]
        m_cogs = flag_to_mult_cogs[flag]
        d_rev  = abs(np.log(m_rev))  if m_rev  > 0 else 0.0
        d_cogs = abs(np.log(m_cogs)) if m_cogs > 0 else 0.0
        better_rev  = mask & (d_rev  > rev_dev)
        better_cogs = mask & (d_cogs > cogs_dev)
        rev_uplift[better_rev]  = m_rev
        rev_dev[better_rev]     = d_rev
        cogs_uplift[better_cogs] = m_cogs
        cogs_dev[better_cogs]    = d_cogs
    out["prior_rev_event_uplift"]  = rev_uplift
    out["prior_cogs_event_uplift"] = cogs_uplift

    keep_cols = ["date",
                 "prior_rev_by_month_dow",
                 "prior_cogs_by_month_dow",
                 "aov_prior_by_month",
                 "prior_rev_event_uplift",
                 "prior_cogs_event_uplift"]
    return out[keep_cols].sort_values("date").reset_index(drop=True)


def main():
    sales, n_orders, cal = load_inputs()
    md, aov_m, ev_uplift, base_rev, base_cogs = compute_priors(sales, n_orders, cal)

    daily = materialize_daily(cal, md, aov_m, ev_uplift)
    daily.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}  shape={daily.shape}")

    meta = {
        "train_window": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "baseline_no_event": {
            "mean_revenue": base_rev,
            "mean_cogs":    base_cogs,
        },
        "prior_rev_by_month_dow": md.to_dict(orient="records"),
        "aov_prior_by_month":     aov_m.to_dict(orient="records"),
        "event_uplift":           ev_uplift.to_dict(orient="records"),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT_META}")

    print()
    print("Top season-matched event uplifts (sorted desc):")
    top = ev_uplift.sort_values("uplift_rev", ascending=False)
    for _, r in top.iterrows():
        print(f"  {r['event']:24s}  rev x{r['uplift_rev']:.3f}  "
              f"cogs x{r['uplift_cogs']:.3f}  n={int(r['n_event_days'])}")


if __name__ == "__main__":
    main()
