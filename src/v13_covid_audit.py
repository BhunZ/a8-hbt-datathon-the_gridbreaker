"""V13 Step 1 — COVID disruption audit.

Goal: locate the actual COVID-related sales disruption in *our* sales.csv
(not from the public timeline) so V13's COVID flag dates are calibrated to
this dataset.

Procedure (per blueprint Section 11, verification subsection):
  1. Load sales.csv (2017-01-01 .. 2022-12-31).
  2. Plot 7-day rolling Revenue with public-timeline COVID windows annotated.
  3. Compute monthly revenue ratio per year vs 2019 baseline.
       2020 / 2019, 2021 / 2019, 2022 / 2019 by month.
  4. Identify the empirical disruption window: months where 2021/2019 < 0.7
     are 'severe disruption'; months where ratio is 0.7-0.9 are 'moderate'.
     Likewise track 2020 and 2022 deviations.
  5. Write calibrated date table to docs/v13_covid_audit.md.

Outputs:
  figures/v13_covid_disruption.png
  docs/v13_covid_audit.md
  data/processed/v13_covid_flags.parquet   (date -> covid flags for ALL train+test dates)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from paths import RAW, PROCESSED, FIGURES, DOCS

t0 = time.time()

# ---------- Load
sales = pd.read_csv(RAW / "sales.csv", parse_dates=["Date"])
sales = sales.sort_values("Date").reset_index(drop=True)
sales = sales.rename(columns={"Date": "date"})

print(f"[{time.time()-t0:5.1f}s] sales.csv loaded: "
      f"{len(sales)} rows  {sales.date.min().date()} -> {sales.date.max().date()}")

TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-07-01")
train = sales[sales.date <= TRAIN_END].copy()
train["year"] = train.date.dt.year
train["month"] = train.date.dt.month
train["rev_roll7"] = train.Revenue.rolling(7, min_periods=1).mean()

# ---------- Public-timeline COVID windows (Vietnam)
PUBLIC_COVID_WINDOWS = [
    ("2020-04-01", "2020-04-22", "Nat'l Directive 16 (3 wk lockdown)"),
    ("2020-07-28", "2020-09-05", "Da Nang outbreak"),
    ("2021-01-28", "2021-03-18", "Hai Duong / Northern wave"),
    ("2021-05-27", "2021-10-01", "DELTA wave (HCMC strict lockdown)"),
    ("2021-10-01", "2022-03-31", "Reopening / revenge spending"),
    ("2022-03-01", "2022-04-30", "Omicron (mild)"),
]

# ---------- 2. Monthly ratio table
print(f"\n[{time.time()-t0:5.1f}s] Monthly mean Revenue per (year, month)...")
monthly = train.groupby(["year", "month"])["Revenue"].mean().unstack("month")
print("\n=== Monthly mean Revenue ($M) ===")
print((monthly / 1e6).round(2).to_string())

# Use 2019 as the pre-pandemic baseline. Avoid 2017 (early-stage business).
BASELINE_YEAR = 2019
baseline = monthly.loc[BASELINE_YEAR]
ratio = monthly.div(baseline, axis=1)
print("\n=== Ratio of monthly mean Revenue vs 2019 baseline ===")
print(ratio.round(3).to_string())

# ---------- 3. Empirical disruption classification per (year, month)
def classify(r):
    if pd.isna(r):
        return "?"
    if r < 0.70:
        return "SEVERE"
    if r < 0.85:
        return "MODERATE"
    if r < 0.95:
        return "MILD"
    if r > 1.30:
        return "BOOM"
    return "NORMAL"

cls = ratio.applymap(classify)
print("\n=== Disruption classification ===")
print(cls.to_string())

# ---------- 4. Build empirical flag windows
# A month is flagged 'lockdown' if its 2021 ratio is SEVERE.
# A month is flagged 'recovery' if its 2022 ratio is BOOM (revenge spending).
# A month is flagged 'disruption' if any 2020 or 2021 ratio is SEVERE/MODERATE.
empirical_flags = []  # list of (start, end, flag_name)

for m in range(1, 13):
    label_2020 = cls.loc[2020, m] if 2020 in cls.index and m in cls.columns else "?"
    label_2021 = cls.loc[2021, m] if 2021 in cls.index and m in cls.columns else "?"
    label_2022 = cls.loc[2022, m] if 2022 in cls.index and m in cls.columns else "?"

    # 2021 SEVERE => is_covid_lockdown for that month
    if label_2021 == "SEVERE":
        start = pd.Timestamp(2021, m, 1)
        end = (start + pd.offsets.MonthEnd(0))
        empirical_flags.append((start, end, "is_covid_lockdown"))

    # 2020 SEVERE => is_covid_disruption
    if label_2020 == "SEVERE":
        start = pd.Timestamp(2020, m, 1)
        end = (start + pd.offsets.MonthEnd(0))
        empirical_flags.append((start, end, "is_covid_disruption"))

    # 2021 MODERATE/MILD => is_covid_disruption (broader window)
    if label_2021 in ("MODERATE", "MILD"):
        start = pd.Timestamp(2021, m, 1)
        end = (start + pd.offsets.MonthEnd(0))
        empirical_flags.append((start, end, "is_covid_disruption"))

    # 2022 BOOM => is_covid_recovery
    if label_2022 == "BOOM":
        start = pd.Timestamp(2022, m, 1)
        end = (start + pd.offsets.MonthEnd(0))
        empirical_flags.append((start, end, "is_covid_recovery"))

print(f"\n=== Empirical flag windows ({len(empirical_flags)}) ===")
for s, e, n in empirical_flags:
    print(f"  {n:22s}  {s.date()}  ->  {e.date()}")

# ---------- 5. Build per-day flag table
all_dates = pd.date_range("2017-01-01", TEST_END, freq="D")
flags = pd.DataFrame({"date": all_dates})
flags["is_covid_lockdown"] = 0
flags["is_covid_disruption"] = 0
flags["is_covid_recovery"] = 0
for s, e, n in empirical_flags:
    mask = (flags.date >= s) & (flags.date <= e)
    flags.loc[mask, n] = 1

# covid_severity = 1.0 if lockdown, 0.5 if disruption, 0.3 if recovery, 0 otherwise
flags["covid_severity"] = (
    1.0 * flags.is_covid_lockdown
    + 0.5 * (flags.is_covid_disruption & ~flags.is_covid_lockdown.astype(bool))
    + 0.3 * (flags.is_covid_recovery & ~flags.is_covid_lockdown.astype(bool)
                                     & ~flags.is_covid_disruption.astype(bool))
).clip(0.0, 1.0)

PROCESSED.mkdir(parents=True, exist_ok=True)
flags_path = PROCESSED / "v13_covid_flags.parquet"
flags.to_parquet(flags_path, index=False)
print(f"\n[{time.time()-t0:5.1f}s] flag table -> {flags_path}")
print(f"   train rows flagged: lockdown={flags[flags.date<=TRAIN_END].is_covid_lockdown.sum()}  "
      f"disruption={flags[flags.date<=TRAIN_END].is_covid_disruption.sum()}  "
      f"recovery={flags[flags.date<=TRAIN_END].is_covid_recovery.sum()}")
print(f"   test rows flagged (should be 0):  lockdown={flags[flags.date>=TEST_START].is_covid_lockdown.sum()}  "
      f"disruption={flags[flags.date>=TEST_START].is_covid_disruption.sum()}  "
      f"recovery={flags[flags.date>=TEST_START].is_covid_recovery.sum()}")

# ---------- 6. Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

ax = axes[0]
ax.plot(train.date, train.rev_roll7 / 1e6, color="tab:blue", lw=1.0,
        label="Revenue 7-day rolling mean")
for s, e, label in PUBLIC_COVID_WINDOWS:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="orange", alpha=0.15)
# overlay empirical lockdown windows in red
empirical_lock = [(s, e) for s, e, n in empirical_flags if n == "is_covid_lockdown"]
for s, e in empirical_lock:
    ax.axvspan(s, e, color="red", alpha=0.25)
ax.set_title("Revenue 7-day rolling mean (2017-2022)  —  orange = public COVID timeline, "
             "red = empirical lockdown months from data")
ax.set_ylabel("Revenue ($M, 7-day mean)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax = axes[1]
years = sorted(monthly.index.tolist())
months = list(range(1, 13))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(years)))
for y, c in zip(years, colors):
    ax.plot(months, ratio.loc[y].values, marker="o", label=str(y), color=c)
ax.axhline(1.0, color="k", ls=":", alpha=0.5)
ax.axhline(0.85, color="orange", ls=":", alpha=0.6, label="MODERATE threshold (0.85)")
ax.axhline(0.70, color="red", ls=":", alpha=0.6, label="SEVERE threshold (0.70)")
ax.axhline(1.30, color="green", ls=":", alpha=0.6, label="BOOM threshold (1.30)")
ax.set_xticks(months)
ax.set_xlabel("Month")
ax.set_ylabel(f"Ratio of monthly mean Revenue vs {BASELINE_YEAR}")
ax.set_title(f"Monthly Revenue ratio per year vs {BASELINE_YEAR} baseline")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", ncol=2, fontsize=8)

plt.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
fig_path = FIGURES / "v13_covid_disruption.png"
plt.savefig(fig_path, dpi=130)
print(f"\n[{time.time()-t0:5.1f}s] figure -> {fig_path}")

# ---------- 7. Markdown report
md = [
    "# V13 Step 1 — Empirical COVID Disruption Audit",
    "",
    f"Produced by `src/v13_covid_audit.py`. Baseline year: **{BASELINE_YEAR}** (last full pre-pandemic year).",
    "",
    "## 1. Monthly mean Revenue by year ($M)",
    "",
]
md.append("| Year | " + " | ".join(f"M{m:02d}" for m in months) + " |")
md.append("|---:|" + "---:|" * len(months))
for y in years:
    row = f"| {y} |"
    for m in months:
        v = monthly.loc[y, m] if m in monthly.columns else float("nan")
        row += f" {v/1e6:.2f} |" if not pd.isna(v) else " — |"
    md.append(row)

md += [
    "",
    f"## 2. Ratio vs {BASELINE_YEAR} baseline",
    "",
]
md.append("| Year | " + " | ".join(f"M{m:02d}" for m in months) + " |")
md.append("|---:|" + "---:|" * len(months))
for y in years:
    if y == BASELINE_YEAR:
        continue
    row = f"| {y} |"
    for m in months:
        v = ratio.loc[y, m] if m in ratio.columns else float("nan")
        row += f" {v:.2f} |" if not pd.isna(v) else " — |"
    md.append(row)

md += [
    "",
    "## 3. Disruption classification",
    "",
    "Thresholds: SEVERE < 0.70 · MODERATE 0.70-0.85 · MILD 0.85-0.95 · NORMAL 0.95-1.30 · BOOM > 1.30",
    "",
]
md.append("| Year | " + " | ".join(f"M{m:02d}" for m in months) + " |")
md.append("|---:|" + "---:|" * len(months))
for y in years:
    if y == BASELINE_YEAR:
        continue
    row = f"| {y} |"
    for m in months:
        c = cls.loc[y, m] if m in cls.columns else "?"
        row += f" {c} |"
    md.append(row)

md += [
    "",
    "## 4. Empirical flag windows derived from the data",
    "",
    "| Flag | Start | End | Days |",
    "|---|---|---|---:|",
]
for s, e, n in empirical_flags:
    days = (e - s).days + 1
    md.append(f"| `{n}` | {s.date()} | {e.date()} | {days} |")

md += [
    "",
    "## 5. Public Vietnam COVID timeline (for cross-reference)",
    "",
    "| Window | Description |",
    "|---|---|",
]
for s, e, label in PUBLIC_COVID_WINDOWS:
    md.append(f"| {s} → {e} | {label} |")

md += [
    "",
    "## 6. How V13 will use this",
    "",
    "1. **Features.** Each row of `daily_features_v13.parquet` carries `is_covid_lockdown`, "
    "`is_covid_disruption`, `is_covid_recovery`, and `covid_severity ∈ [0, 1]`. "
    "All four equal 0 across the 548-day test horizon.",
    "2. **Frozen priors.** `prior_rev_by_month_dow`, `prior_cogs_by_month_dow`, "
    "`aov_prior_by_month`, `prior_rev_event_uplift` are computed on the train window "
    "**excluding rows where `covid_severity > 0`**. The priors therefore reflect "
    "normal-time behaviour, which is what 2023-2024 will be.",
    "3. **Lag bypass.** `rev_same_dow_prev_year` and `rev_yoy_lag_364` skip past "
    "any candidate date whose `covid_severity > 0`, bouncing forward another 365 days "
    "until they land on a normal date.",
    "4. **Sample weighting.** When fitting a model: `sample_weight = 1 - 0.7 × covid_severity`. "
    "Lockdown rows down-weighted to 0.30, disruption rows to 0.65, recovery rows to 0.79.",
    "",
    "## 7. Sanity checks",
    "",
    f"- Train rows flagged lockdown: {flags[flags.date<=TRAIN_END].is_covid_lockdown.sum()}",
    f"- Train rows flagged disruption: {flags[flags.date<=TRAIN_END].is_covid_disruption.sum()}",
    f"- Train rows flagged recovery: {flags[flags.date<=TRAIN_END].is_covid_recovery.sum()}",
    f"- Test rows flagged any (must be 0): "
    f"{flags[flags.date>=TEST_START][['is_covid_lockdown','is_covid_disruption','is_covid_recovery']].sum().sum()}",
    "",
    "## 8. Artifacts",
    "",
    "- `data/processed/v13_covid_flags.parquet` — per-day flag table (2017-01-01 → 2024-07-01).",
    "- `figures/v13_covid_disruption.png` — Revenue trace + monthly ratio plot.",
    "",
]

DOCS.mkdir(parents=True, exist_ok=True)
md_path = DOCS / "v13_covid_audit.md"
md_path.write_text("\n".join(md), encoding="utf-8")
print(f"[{time.time()-t0:5.1f}s] report -> {md_path}")

print(f"\n=== DONE in {time.time()-t0:.1f}s ===")
