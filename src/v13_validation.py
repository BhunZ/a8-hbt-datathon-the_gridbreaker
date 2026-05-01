"""V13 Step 6 -- Validation harness.

Defines the mirror block, iteration folds, and scoring helpers used by every
V13 model fit.  This module is imported by:
  - src/v13_v10c_rebaseline.py  (Step 7: establish the floor)
  - src/v13_baseline.py         (Step 8: V13 baseline LightGBM)
  - src/v13_peak_recovery.py    (Step 9: stage-2 quantile uplift)
  - src/v13_tweedie.py          (Step 10: Tweedie head-to-head)

Locked structure (V13 blueprint section 14):

  Mirror block: hold out 2022-01-01 -> 2022-12-31.
    Train on 2019-09-01 -> 2021-12-31.
    Score this block at most once per substantive iteration.
    Includes Tet 2022, Singles Day, Black Friday -- one full post-regime year.

  Iteration folds: 4 expanding-origin folds at 90-day horizon.
    Fold 1: train <= 2020-12-31, predict 2021-Q1
    Fold 2: train <= 2021-03-31, predict 2021-Q2
    Fold 3: train <= 2021-06-30, predict 2021-Q3
    Fold 4: train <= 2021-09-30, predict 2021-Q4

Reported metrics:
  RMSE_global    -- overall RMSE on the held-out window
  RMSE_peak_days -- RMSE on dates where any event flag is set OR target >= 90th %ile
  RMSE_log       -- RMSE in log1p space (compatible with V12-style training)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from paths import PROCESSED


# --- Locked windows --------------------------------------------------------

POST_CLIFF_START = pd.Timestamp("2019-09-01")
MIRROR_START     = pd.Timestamp("2022-01-01")
MIRROR_END       = pd.Timestamp("2022-12-31")
MIRROR_TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START       = pd.Timestamp("2023-01-01")
TEST_END         = pd.Timestamp("2024-07-01")


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end:   pd.Timestamp


ITERATION_FOLDS = [
    Fold("fold1_2021Q1", pd.Timestamp("2020-12-31"),
         pd.Timestamp("2021-01-01"), pd.Timestamp("2021-03-31")),
    Fold("fold2_2021Q2", pd.Timestamp("2021-03-31"),
         pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30")),
    Fold("fold3_2021Q3", pd.Timestamp("2021-06-30"),
         pd.Timestamp("2021-07-01"), pd.Timestamp("2021-09-30")),
    Fold("fold4_2021Q4", pd.Timestamp("2021-09-30"),
         pd.Timestamp("2021-10-01"), pd.Timestamp("2021-12-31")),
]


MIRROR = Fold("mirror_2022", MIRROR_TRAIN_END, MIRROR_START, MIRROR_END)


# --- Loading helpers -------------------------------------------------------

def load_v13_features() -> pd.DataFrame:
    """Returns the assembled V13 feature table with date/Revenue/COGS/is_train/is_test."""
    df = pd.read_parquet(PROCESSED / "daily_features_v13.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    """Return the model-feature column names (everything except date, targets, and split flags)."""
    drop = {"date", "Revenue", "COGS", "is_train", "is_test"}
    return [c for c in df.columns if c not in drop]


def split_for_fold(df: pd.DataFrame, fold: Fold,
                   train_min: pd.Timestamp = POST_CLIFF_START):
    """Slice the feature table into (train_X, train_y, val_X, val_y) DataFrames keyed by date."""
    feats = feature_columns(df)
    train_mask = (df["date"] >= train_min) & (df["date"] <= fold.train_end) & df["Revenue"].notna()
    val_mask   = (df["date"] >= fold.val_start) & (df["date"] <= fold.val_end) & df["Revenue"].notna()

    train_X = df.loc[train_mask, feats].reset_index(drop=True)
    train_y = df.loc[train_mask, ["Revenue", "COGS"]].reset_index(drop=True)
    val_X   = df.loc[val_mask,   feats].reset_index(drop=True)
    val_y   = df.loc[val_mask,   ["Revenue", "COGS"]].reset_index(drop=True)
    val_dates = df.loc[val_mask, "date"].reset_index(drop=True)
    return train_X, train_y, val_X, val_y, val_dates


# --- Scoring ---------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE in log1p space.  Both inputs must be non-negative."""
    yt = np.log1p(np.clip(y_true, 0, None))
    yp = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def is_peak_day(features_row: dict, y_true: float, peak_threshold: float) -> bool:
    """Peak day if any event flag is set OR the actual target exceeds threshold."""
    if y_true >= peak_threshold:
        return True
    for c, v in features_row.items():
        if c.startswith("is_") and c not in ("is_weekend", "is_month_start", "is_month_end",
                                              "is_quarter_end", "is_year_end", "is_train",
                                              "is_test", "is_double_digit_day",
                                              "is_payday_15", "is_payday_25", "is_payday_30",
                                              "is_any_payday", "is_pre_tet_window",
                                              "is_tet_recovery_window") and v == 1:
            return True
    return False


PEAK_EVENT_FLAGS = {
    "is_singles_day", "is_twelve_twelve", "is_nine_nine", "is_ten_ten",
    "is_black_friday", "is_cyber_monday", "is_intl_women_day",
    "is_womens_day_vn", "is_teachers_day_vn", "is_independence_day_vn",
    "is_reunification_day", "is_intl_labour_day", "is_new_year_day",
    "is_valentines", "is_christmas", "is_boxing_day", "is_tet",
}


def score_predictions(val_dates: pd.Series,
                      y_true_rev: np.ndarray,  y_pred_rev: np.ndarray,
                      y_true_cogs: np.ndarray, y_pred_cogs: np.ndarray,
                      features_df: pd.DataFrame | None = None,
                      train_y_for_pct: np.ndarray | None = None) -> dict:
    """Compute the standard V13 metric set on a held-out window.

    Returns a dict with:
      rmse_global_rev, rmse_peak_rev, rmse_log_rev
      rmse_global_cogs, rmse_peak_cogs, rmse_log_cogs
      rmse_global_combined  (sqrt(mean(rev_se + cogs_se)) -- Kaggle-shaped if both targets stacked)
      mean_pred_rev, mean_actual_rev, n_rows, n_peak_rows
    """
    out = {
        "rmse_global_rev":  rmse(y_true_rev,  y_pred_rev),
        "rmse_log_rev":     rmse_log(y_true_rev, y_pred_rev),
        "rmse_global_cogs": rmse(y_true_cogs, y_pred_cogs),
        "rmse_log_cogs":    rmse_log(y_true_cogs, y_pred_cogs),
        "mean_pred_rev":    float(np.mean(y_pred_rev)),
        "mean_actual_rev":  float(np.mean(y_true_rev)),
        "mean_pred_cogs":   float(np.mean(y_pred_cogs)),
        "mean_actual_cogs": float(np.mean(y_true_cogs)),
        "n_rows":           int(len(y_true_rev)),
    }
    # Combined RMSE -- approximates Kaggle's two-target-stacked scoring
    combined_se = np.concatenate([(y_true_rev - y_pred_rev) ** 2,
                                  (y_true_cogs - y_pred_cogs) ** 2])
    out["rmse_global_combined"] = float(np.sqrt(np.mean(combined_se)))

    # Peak-day metric: dates where any major event flag is set OR target >= 90th %ile
    if features_df is not None:
        flag_cols = [c for c in PEAK_EVENT_FLAGS if c in features_df.columns]
        any_event = features_df[flag_cols].sum(axis=1).clip(0, 1).values.astype(bool)
        threshold_rev  = (np.percentile(train_y_for_pct, 90)
                          if train_y_for_pct is not None
                          else np.percentile(y_true_rev, 90))
        peak_mask_rev = any_event | (y_true_rev >= threshold_rev)
        out["n_peak_rows"]   = int(peak_mask_rev.sum())
        out["rmse_peak_rev"]  = (rmse(y_true_rev[peak_mask_rev], y_pred_rev[peak_mask_rev])
                                 if peak_mask_rev.any() else float("nan"))
        out["rmse_peak_cogs"] = (rmse(y_true_cogs[peak_mask_rev], y_pred_cogs[peak_mask_rev])
                                 if peak_mask_rev.any() else float("nan"))
    else:
        out["n_peak_rows"]   = 0
        out["rmse_peak_rev"]  = float("nan")
        out["rmse_peak_cogs"] = float("nan")
    return out


def fmt_score(s: dict) -> str:
    return (f"  rmse_combined: {s['rmse_global_combined']:>12,.0f}\n"
            f"  rmse_rev:      {s['rmse_global_rev']:>12,.0f}   "
            f"rmse_peak_rev:  {s.get('rmse_peak_rev', float('nan')):>12,.0f}   "
            f"rmse_log_rev:  {s['rmse_log_rev']:.4f}\n"
            f"  rmse_cogs:     {s['rmse_global_cogs']:>12,.0f}   "
            f"rmse_peak_cogs: {s.get('rmse_peak_cogs', float('nan')):>12,.0f}   "
            f"rmse_log_cogs: {s['rmse_log_cogs']:.4f}\n"
            f"  mean pred rev: {s['mean_pred_rev']:>12,.0f}   "
            f"mean actual rev: {s['mean_actual_rev']:>12,.0f}   "
            f"n={s['n_rows']}  n_peak={s['n_peak_rows']}")


if __name__ == "__main__":
    # Smoke test: load features and report shape per fold.
    df = load_v13_features()
    print(f"Loaded V13 features: {df.shape}")
    print()
    for f in ITERATION_FOLDS + [MIRROR]:
        train_X, train_y, val_X, val_y, val_dates = split_for_fold(df, f)
        print(f"{f.name:18s}  train={len(train_X):4d}  val={len(val_X):4d}  "
              f"val_window={f.val_start.date()}..{f.val_end.date()}")
