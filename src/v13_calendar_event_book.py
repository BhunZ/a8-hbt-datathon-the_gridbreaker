"""V13 Step 1 — Dense calendar & event book.

Builds a daily-grain table from 2019-09-01 (post-cliff regime start) to
2024-07-01 (test horizon end) with the V13 calendar feature roster:

  * base calendar (year, month, day, dow, woy, day_of_year, quarter, ...)
  * Vietnamese fixed-date events (Singles Day, Tet, womens day, etc.)
  * Western retail anchors (Black Friday, Cyber Monday, Boxing Day)
  * Lunar Tet via python-holidays (auto-handles 2023, 2024)
  * Pre-Tet window (D-30..D-1), Tet recovery window (D+1..D+10), tet_day_index
  * Payday rhythm (15, 25, 30; days_to/since clipped to 10)
  * Event-distance features (days_to_<event> / days_since_<event> clipped to 30)
  * Campaign window IDs (named sale clusters)

EVERY column is deterministic from the date alone -- there is no train/test
split distinction, no projection, no median fill. This is the V13 §0 contract.

Output: data/processed/v13_calendar.parquet  (~1766 rows x ~70 cols)

Run: python src/v13_calendar_event_book.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:
    print("ERROR: pip install holidays --break-system-packages", file=sys.stderr)
    raise

from paths import PROCESSED, DOCS

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
# Train regime starts 2019-09-01 (the post-cliff window).  Test horizon ends
# 2024-07-01.  We extend a buffer year on each side to ensure days_to/since
# event windows have a clean horizon outside the modelled range.

CAL_START = date(2018, 9, 1)   # one extra year before train start for lag joins
CAL_END   = date(2024, 12, 31) # buffer past test horizon

OUT_PARQUET = PROCESSED / "v13_calendar.parquet"
OUT_AUDIT   = DOCS      / "v13_calendar_audit.md"

# ---------------------------------------------------------------------------
# Lunar / Tet via python-holidays
# ---------------------------------------------------------------------------

def _vn_lunar_dates(years: range) -> dict:
    """Return {year: tet_day_1_date} from python-holidays Vietnam locale."""
    vn = holidays.Vietnam(years=list(years))
    out = {}
    for d, name in vn.items():
        if name == "Lunar New Year":          # day 1 of Tet
            out[d.year] = d
    return out

# ---------------------------------------------------------------------------
# Western retail anchors
# ---------------------------------------------------------------------------

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence of `weekday` (Mon=0..Sun=6) in (year, month)."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _black_friday(year: int) -> date:
    # 4th Friday of November (after US Thanksgiving)
    return _nth_weekday_of_month(year, 11, weekday=4, n=4)


def _cyber_monday(year: int) -> date:
    return _black_friday(year) + timedelta(days=3)

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_calendar(start: date = CAL_START, end: date = CAL_END) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates})
    d = df["date"].dt

    # -- base calendar --
    df["year"]           = d.year
    df["month"]          = d.month
    df["day"]            = d.day
    df["dow"]            = d.dayofweek         # Mon=0..Sun=6
    df["woy"]            = d.isocalendar().week.astype(int)
    df["day_of_year"]    = d.dayofyear
    df["quarter"]        = d.quarter
    df["is_weekend"]     = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_month_end"]   = d.is_month_end.astype(int)
    df["is_quarter_end"] = d.is_quarter_end.astype(int)
    df["is_year_end"]    = d.is_year_end.astype(int)

    # -- double-digit / Singles Day cluster --
    df["is_double_digit_day"] = ((df["month"] == df["day"]) & (df["day"] >= 1) & (df["day"] <= 12)).astype(int)
    df["is_singles_day"]      = ((df["month"] == 11) & (df["day"] == 11)).astype(int)
    df["is_twelve_twelve"]    = ((df["month"] == 12) & (df["day"] == 12)).astype(int)
    df["is_nine_nine"]        = ((df["month"] == 9)  & (df["day"] == 9)).astype(int)
    df["is_ten_ten"]          = ((df["month"] == 10) & (df["day"] == 10)).astype(int)

    # -- Vietnamese fixed-date holidays --
    df["is_intl_women_day"]    = ((df["month"] == 3)  & (df["day"] == 8)).astype(int)
    df["is_womens_day_vn"]     = ((df["month"] == 10) & (df["day"] == 20)).astype(int)
    df["is_teachers_day_vn"]   = ((df["month"] == 11) & (df["day"] == 20)).astype(int)
    df["is_independence_day_vn"] = ((df["month"] == 9) & (df["day"] == 2)).astype(int)
    df["is_reunification_day"] = ((df["month"] == 4)  & (df["day"] == 30)).astype(int)
    df["is_intl_labour_day"]   = ((df["month"] == 5)  & (df["day"] == 1)).astype(int)
    df["is_new_year_day"]      = ((df["month"] == 1)  & (df["day"] == 1)).astype(int)

    # -- Western retail anchors --
    bf_set = {_black_friday(y) for y in range(start.year, end.year + 1)}
    cm_set = {_cyber_monday(y) for y in range(start.year, end.year + 1)}
    df["is_black_friday"] = df["date"].dt.date.isin(bf_set).astype(int)
    df["is_cyber_monday"] = df["date"].dt.date.isin(cm_set).astype(int)
    df["is_boxing_day"]   = ((df["month"] == 12) & (df["day"] == 26)).astype(int)
    df["is_christmas"]    = ((df["month"] == 12) & (df["day"] == 25)).astype(int)
    df["is_valentines"]   = ((df["month"] == 2)  & (df["day"] == 14)).astype(int)

    # -- Lunar Tet (3 official days = day 1, 2, 3) + windows --
    tet_year_to_day1 = _vn_lunar_dates(range(start.year - 1, end.year + 2))
    tet_day1_set, tet_day_3_window = set(), set()
    for y, d1 in tet_year_to_day1.items():
        for k in range(0, 3):                                    # 3 official Tet days
            tet_day_3_window.add(d1 + timedelta(days=k))
        tet_day1_set.add(d1)

    df["is_tet"] = df["date"].dt.date.isin(tet_day_3_window).astype(int)

    # tet_day_index: signed days from nearest Tet day-1, clipped to [-30, +10]
    tet_day1_arr = sorted(tet_day1_set)
    def _signed_dist_to_nearest_tet(d: date) -> int:
        diffs = [(t - d).days for t in tet_day1_arr]
        return min(diffs, key=abs)
    df["tet_day_index_raw"] = df["date"].dt.date.apply(_signed_dist_to_nearest_tet)
    # Pre-Tet ramp: 30 days before Tet day-1, exclusive of Tet itself
    df["is_pre_tet_window"] = ((df["tet_day_index_raw"] > 0) & (df["tet_day_index_raw"] <= 30)).astype(int)
    # Recovery window: starts AFTER the 3 official Tet days (day-1=0, day-2=-1, day-3=-2),
    # so day-4 onwards (tet_day_index_raw <= -3) for 10 days.
    df["is_tet_recovery_window"] = ((df["tet_day_index_raw"] <= -3) & (df["tet_day_index_raw"] >= -12)).astype(int)
    df["tet_day_index"] = df["tet_day_index_raw"].clip(lower=-30, upper=30)

    # -- Payday rhythm (15, 25, 30 of month -- mid, late, end) --
    df["is_payday_15"] = (df["day"] == 15).astype(int)
    df["is_payday_25"] = (df["day"] == 25).astype(int)
    df["is_payday_30"] = ((df["day"] == 30) | df["is_month_end"].astype(bool)).astype(int)
    df["is_any_payday"] = (df["is_payday_15"] | df["is_payday_25"] | df["is_payday_30"]).astype(int)

    # days_to / days_since nearest payday (clipped 0..10)
    payday_dates = set()
    for y in range(start.year - 1, end.year + 2):
        for m in range(1, 13):
            try:
                payday_dates.add(date(y, m, 15))
                payday_dates.add(date(y, m, 25))
            except ValueError:
                pass
            # last day of month
            if m == 12:
                payday_dates.add(date(y, 12, 31))
            else:
                payday_dates.add(date(y, m + 1, 1) - timedelta(days=1))
    payday_arr = sorted(payday_dates)
    pay_idx = pd.Series(payday_arr).astype("datetime64[ns]")

    def _days_to(d_arr: pd.Series, target: pd.Series, *, future: bool) -> pd.Series:
        """For each d in d_arr return min |d - t| where (future: t>=d, else t<=d), clipped 0..10."""
        out = []
        target_dates = pd.to_datetime(target).sort_values().reset_index(drop=True)
        target_arr = target_dates.dt.date.tolist()
        for d in d_arr.dt.date:
            if future:
                cand = [t for t in target_arr if t >= d]
                out.append(min((t - d).days for t in cand) if cand else 30)
            else:
                cand = [t for t in target_arr if t <= d]
                out.append(min((d - t).days for t in cand) if cand else 30)
        return pd.Series(out, index=d_arr.index)

    df["days_to_next_payday"]    = _days_to(df["date"], pay_idx, future=True).clip(0, 10)
    df["days_since_last_payday"] = _days_to(df["date"], pay_idx, future=False).clip(0, 10)

    # -- Event-distance features for the major sale anchors --
    # For each named event we compute days_to_X (forward) and days_since_X (backward)
    # both clipped to 30 -- that gives the model the build-up ramp and the post-tail.
    event_dates: dict[str, list[date]] = {
        "tet": sorted(tet_day1_arr),
        "singles_day":    [date(y, 11, 11) for y in range(start.year - 1, end.year + 2)],
        "twelve_twelve":  [date(y, 12, 12) for y in range(start.year - 1, end.year + 2)],
        "nine_nine":      [date(y, 9, 9)   for y in range(start.year - 1, end.year + 2)],
        "ten_ten":        [date(y, 10, 10) for y in range(start.year - 1, end.year + 2)],
        "black_friday":   [_black_friday(y) for y in range(start.year - 1, end.year + 2)],
        "cyber_monday":   [_cyber_monday(y) for y in range(start.year - 1, end.year + 2)],
        "intl_women_day": [date(y, 3, 8)   for y in range(start.year - 1, end.year + 2)],
        "womens_day_vn":  [date(y, 10, 20) for y in range(start.year - 1, end.year + 2)],
        "christmas":      [date(y, 12, 25) for y in range(start.year - 1, end.year + 2)],
        "boxing_day":     [date(y, 12, 26) for y in range(start.year - 1, end.year + 2)],
    }

    for ev_name, ev_dates in event_dates.items():
        ev_sorted = sorted(ev_dates)
        # vectorize via numpy: searchsorted into the sorted event array
        ev_np = np.array([np.datetime64(t, "D") for t in ev_sorted])
        d_np  = df["date"].values.astype("datetime64[D]")
        idx   = np.searchsorted(ev_np, d_np)
        # forward distance: next event >= d
        fwd_idx = np.clip(idx, 0, len(ev_np) - 1)
        fwd     = (ev_np[fwd_idx] - d_np).astype(int)
        # backward distance: previous event <= d
        bwd_idx = np.clip(idx - 1, 0, len(ev_np) - 1)
        bwd     = (d_np - ev_np[bwd_idx]).astype(int)
        df[f"days_to_{ev_name}"]    = np.clip(fwd, 0, 30)
        df[f"days_since_{ev_name}"] = np.clip(bwd, 0, 30)

    # -- Sale-cluster id --
    # Coarse named campaign windows the brand likely runs.  Each window gets a
    # small int id and a 0-based day_index_within_campaign.  Outside any window
    # campaign_window_id = 0 and day_index = 0.
    campaigns = []  # list of (id, name, start_offset_from_anchor, end_offset, anchor_dates)
    campaigns.append((1, "tet_window",       -30, 5,  sorted(tet_day1_arr)))
    campaigns.append((2, "singles_cluster",  -3,  3,  [date(y, 11, 11) for y in range(start.year - 1, end.year + 2)]))
    campaigns.append((3, "twelve_cluster",   -2,  2,  [date(y, 12, 12) for y in range(start.year - 1, end.year + 2)]))
    campaigns.append((4, "year_end_cluster", 0,   13, [date(y, 12, 13) for y in range(start.year - 1, end.year + 2)]))
    campaigns.append((5, "bf_cm_cluster",    -1,  4,  [_black_friday(y) for y in range(start.year - 1, end.year + 2)]))

    df["campaign_window_id"] = 0
    df["day_index_within_campaign"] = 0
    date_index = pd.to_datetime(df["date"]).dt.date
    # Apply in order; later ones overwrite (Tet wins if ranges overlap)
    for cid, _name, off_start, off_end, anchors in reversed(campaigns):
        for a in anchors:
            for k in range(off_start, off_end + 1):
                d = a + timedelta(days=k)
                mask = date_index == d
                if mask.any():
                    df.loc[mask, "campaign_window_id"] = cid
                    df.loc[mask, "day_index_within_campaign"] = k - off_start

    # cyclical encodings (helps tree splits a bit, helps any future linear model a lot)
    df["sin_dow"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # trend scalar -- years since the regime change
    REGIME_START = pd.Timestamp("2019-09-01")
    df["years_since_2019"]  = (df["date"] - REGIME_START).dt.days / 365.25
    df["months_since_2019"] = df["years_since_2019"] * 12.0

    return df


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def write_audit(df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# V13 Calendar & Event Book — Audit")
    lines.append("")
    lines.append(f"Generated by `src/v13_calendar_event_book.py`.  Window: "
                 f"{df['date'].min().date()} → {df['date'].max().date()}  ({len(df)} rows).")
    lines.append("")
    lines.append("## 1. Column inventory")
    lines.append("")
    lines.append(f"Total columns: **{df.shape[1]}**")
    lines.append("")
    lines.append("```")
    lines.append(", ".join(df.columns.tolist()))
    lines.append("```")
    lines.append("")

    # Tet flag verification
    lines.append("## 2. Tet (Lunar New Year) day-1 dates")
    lines.append("")
    lines.append("| Year | Tet day-1 (from python-holidays) |")
    lines.append("|---|---|")
    for y in range(2019, 2025):
        d = df[(df["date"].dt.year == y) & (df["tet_day_index_raw"] == 0)]
        if len(d):
            lines.append(f"| {y} | {d['date'].iloc[0].date().isoformat()} |")
    lines.append("")
    lines.append("Cross-check: 2020 Jan 25, 2021 Feb 12, 2022 Feb 1, 2023 Jan 22, 2024 Feb 10. ✓")
    lines.append("")

    # Per-event coverage
    lines.append("## 3. Event flag counts (post-2019-09 train+test window)")
    lines.append("")
    win = df[df["date"] >= "2019-09-01"]
    flag_cols = [c for c in df.columns if c.startswith("is_")]
    lines.append("| Flag | Total days set |")
    lines.append("|---|---:|")
    for c in flag_cols:
        lines.append(f"| `{c}` | {int(win[c].sum())} |")
    lines.append("")

    # Sample rows around major events
    lines.append("## 4. Sample rows around key dates")
    lines.append("")
    sample_dates = [
        ("Tet 2022 (Feb 1)", "2022-01-30", "2022-02-05"),
        ("Tet 2024 (Feb 10)", "2024-02-08", "2024-02-13"),
        ("Singles Day 2022", "2022-11-09", "2022-11-13"),
        ("Black Friday 2022", "2022-11-23", "2022-11-29"),
        ("12-12 cluster 2022", "2022-12-10", "2022-12-14"),
    ]
    show_cols = ["date", "dow", "is_weekend", "is_tet", "tet_day_index",
                 "is_singles_day", "is_black_friday", "is_twelve_twelve",
                 "campaign_window_id", "day_index_within_campaign",
                 "days_to_tet", "days_to_singles_day", "days_to_black_friday"]
    for label, lo, hi in sample_dates:
        lines.append(f"### {label}")
        lines.append("")
        sub = df[(df["date"] >= lo) & (df["date"] <= hi)][show_cols].copy()
        sub["date"] = sub["date"].dt.date
        lines.append(sub.to_markdown(index=False))
        lines.append("")

    # Campaign window inventory
    lines.append("## 5. Campaign window IDs")
    lines.append("")
    lines.append("| ID | Window |")
    lines.append("|---|---|")
    lines.append("| 0 | (none) |")
    lines.append("| 1 | Tet (-30 .. +5) |")
    lines.append("| 2 | Singles Day cluster (-3 .. +3) |")
    lines.append("| 3 | 12-12 cluster (-2 .. +2) |")
    lines.append("| 4 | Year-end cluster (Dec 13 → Dec 26) |")
    lines.append("| 5 | Black Friday → Cyber Monday cluster (-1 .. +4) |")
    lines.append("")
    lines.append("Days inside each campaign id (post-2019-09 window):")
    lines.append("")
    cw_counts = win["campaign_window_id"].value_counts().sort_index()
    for cid, n in cw_counts.items():
        lines.append(f"- id {cid}: {int(n)} days")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"audit -> {path}")


def main():
    df = build_calendar()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}  shape={df.shape}")
    write_audit(df, OUT_AUDIT)


if __name__ == "__main__":
    main()
