"""V12 Step 3 — Verify COGS reconstruction & fit cogs_ratio trend.

1. Load daily_features_v12.parquet (has items_cogs_recon, items_gross, Revenue, COGS).
2. Verify `items_cogs_recon ≈ sales.COGS` on 2017-2022 (expected r ≈ 1.000).
3. Verify `items_gross ≈ sales.Revenue` (expected r ≈ 0.99+).
4. Compute daily cogs_ratio = COGS / Revenue on 2017-2022. Inspect stability.
5. Fit smooth trend (rolling median + linear) and project to test horizon.
6. Save `data/processed/cogs_ratio_projection.parquet` for Stage D.

Outputs:
  data/processed/cogs_ratio_projection.parquet
  figures/v12_cogs_ratio_trend.png
  docs/v12_cogs_verification.md
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PROCESSED, FIGURES, DOCS

# ---------- Load
df = pd.read_parquet(PROCESSED / "daily_features_v12.parquet")
train = df[df.set == "train"].copy()
test  = df[df.set == "test"].copy()

TEST_START = pd.Timestamp("2023-01-01")

print("="*72)
print(" V12 Step 3 — COGS reconstruction verification")
print("="*72)

# ---------- 1. Reconstruction identity
print("\n[1] Identity check:  items_cogs_recon  vs  sales.COGS  (2017-2022)")
tr = train.dropna(subset=["COGS","items_cogs_recon"])
r_cogs = np.corrcoef(tr.items_cogs_recon, tr.COGS)[0,1]
ratio_cogs = tr.items_cogs_recon.sum() / tr.COGS.sum()
mae_cogs = (tr.items_cogs_recon - tr.COGS).abs().mean()
rmse_cogs = np.sqrt(((tr.items_cogs_recon - tr.COGS)**2).mean())
print(f"    rows considered  : {len(tr):,}")
print(f"    Pearson r        : {r_cogs:.6f}")
print(f"    Σrecon / Σactual : {ratio_cogs:.6f}")
print(f"    MAE              : {mae_cogs:,.2f}")
print(f"    RMSE             : {rmse_cogs:,.2f}")
print(f"    actual mean      : {tr.COGS.mean():,.2f}")
print(f"    recon  mean      : {tr.items_cogs_recon.mean():,.2f}")

print("\n[2] Identity check:  items_gross  vs  sales.Revenue  (2017-2022)")
tr2 = train.dropna(subset=["Revenue","items_gross"])
r_rev = np.corrcoef(tr2.items_gross, tr2.Revenue)[0,1]
ratio_rev = tr2.items_gross.sum() / tr2.Revenue.sum()
print(f"    Pearson r        : {r_rev:.6f}")
print(f"    Σgross / Σactual : {ratio_rev:.6f}")
print(f"    actual mean      : {tr2.Revenue.mean():,.2f}")
print(f"    gross  mean      : {tr2.items_gross.mean():,.2f}")

# ---------- 3. cogs_ratio trend
print("\n[3] cogs_ratio = COGS / Revenue  —  trend analysis")
tr3 = train.dropna(subset=["COGS","Revenue"]).copy()
tr3["cogs_ratio"] = tr3.COGS / tr3.Revenue
tr3 = tr3.sort_values("date").reset_index(drop=True)
tr3["year"] = tr3.date.dt.year

print("\n    Yearly mean cogs_ratio:")
yr_tbl = tr3.groupby("year")["cogs_ratio"].agg(["mean","std","count"])
print(yr_tbl.round(5).to_string())
print(f"\n    Overall mean : {tr3.cogs_ratio.mean():.5f}")
print(f"    Overall std  : {tr3.cogs_ratio.std():.5f}")
print(f"    Min / Max    : {tr3.cogs_ratio.min():.5f} / {tr3.cogs_ratio.max():.5f}")

# Rolling 90-day median to smooth
tr3["ratio_roll90"] = tr3.cogs_ratio.rolling(90, min_periods=30, center=True).median()

# ---------- 4. Fit smooth trend for projection
# Strategy: use last 180 days as anchor (recency bias), fit linear trend on rolling median
tr3["t_days"] = (tr3.date - pd.Timestamp("2017-01-01")).dt.days
recent = tr3[tr3.date >= pd.Timestamp("2021-01-01")].dropna(subset=["ratio_roll90"])
slope, intercept = np.polyfit(recent.t_days.values, recent.ratio_roll90.values, deg=1)
print(f"\n    Linear trend (2021+): slope = {slope:.3e}/day, intercept = {intercept:.5f}")

# Alternative: just use last-90-day median as flat projection
last90 = tr3.tail(90).cogs_ratio.median()
print(f"    Last-90-day median ratio: {last90:.5f}")

# Compare the two
# Use a blend: 60% flat recent-median + 40% linear trend continuation
# Why: linear trend can drift unrealistically over 548 days; the ratio is bounded.
def project_ratio(date: pd.Timestamp) -> float:
    t = (date - pd.Timestamp("2017-01-01")).days
    trend_val = intercept + slope * t
    blend = 0.6 * last90 + 0.4 * trend_val
    return float(np.clip(blend, 0.70, 0.95))  # sanity clip

# Apply
proj = test[["date"]].copy()
proj["cogs_ratio_proj"] = proj["date"].apply(project_ratio)
print(f"\n    Projected ratio range: {proj.cogs_ratio_proj.min():.5f} → {proj.cogs_ratio_proj.max():.5f}")
print(f"    Projected ratio mean : {proj.cogs_ratio_proj.mean():.5f}")

# ---------- 5. Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

ax = axes[0]
ax.plot(tr3.date, tr3.cogs_ratio, color="gray", alpha=0.3, lw=0.6, label="daily ratio")
ax.plot(tr3.date, tr3.ratio_roll90, color="tab:blue", lw=2.0, label="90-day rolling median")
trend_line = intercept + slope * tr3.t_days.values
ax.plot(tr3.date, trend_line, color="tab:orange", lw=1.5, ls="--",
        label=f"linear trend (2021+)  slope={slope:.2e}")
ax.plot(proj.date, proj.cogs_ratio_proj, color="tab:red", lw=2.5, label="Stage D projection")
ax.axvline(TEST_START, color="k", ls=":", alpha=0.4)
ax.set_title("cogs_ratio = COGS / Revenue  —  historical & projected")
ax.set_ylabel("ratio"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax.set_ylim(0.70, 0.95)

ax = axes[1]
ax.scatter(tr.items_cogs_recon/1e6, tr.COGS/1e6, s=5, alpha=0.3)
ax.plot([tr.COGS.min()/1e6, tr.COGS.max()/1e6],
        [tr.COGS.min()/1e6, tr.COGS.max()/1e6], color="tab:red", ls="--")
ax.set_title(f"items_cogs_recon vs sales.COGS  —  r={r_cogs:.5f}, ratio={ratio_cogs:.5f}")
ax.set_xlabel("items_cogs_recon ($M)"); ax.set_ylabel("sales.COGS ($M)"); ax.grid(alpha=0.3)

plt.tight_layout()
FIGURES.mkdir(exist_ok=True)
plot_path = FIGURES / "v12_cogs_ratio_trend.png"
plt.savefig(plot_path, dpi=130)
print(f"\n    figure -> {plot_path}")

# ---------- 6. Write projection parquet
proj["t_days"] = (proj.date - pd.Timestamp("2017-01-01")).dt.days
proj_path = PROCESSED / "cogs_ratio_projection.parquet"
proj.to_parquet(proj_path, index=False)
print(f"    projection -> {proj_path}")

# ---------- 7. Write verification markdown
md = [
    "# V12 Step 3 — COGS Reconstruction Verification",
    "",
    "## 1. Identity check",
    "",
    "| Identity | Pearson r | Σrecon / Σactual | Interpretation |",
    "|---|---:|---:|---|",
    f"| items_cogs_recon ≈ sales.COGS | {r_cogs:.6f} | {ratio_cogs:.6f} | "
    f"{'PERFECT — zero modelling error on historical COGS' if r_cogs > 0.9999 else 'Close but not identical'} |",
    f"| items_gross ≈ sales.Revenue    | {r_rev:.6f} | {ratio_rev:.6f} | "
    f"{'Near-perfect Revenue proxy' if r_rev > 0.99 else 'Partial Revenue proxy'} |",
    "",
    f"**MAE of COGS reconstruction**: {mae_cogs:,.2f}   "
    f"(actual mean: {tr.COGS.mean():,.2f}, so error is "
    f"{100*mae_cogs/tr.COGS.mean():.3f}% of mean)",
    "",
    "## 2. cogs_ratio yearly stability",
    "",
    "| Year | Mean ratio | Std | Days |",
    "|---:|---:|---:|---:|",
]
for y, row in yr_tbl.iterrows():
    md.append(f"| {y} | {row['mean']:.5f} | {row['std']:.5f} | {int(row['count'])} |")

md += [
    "",
    f"**Overall**: mean = {tr3.cogs_ratio.mean():.5f}, std = {tr3.cogs_ratio.std():.5f}, "
    f"range [{tr3.cogs_ratio.min():.5f}, {tr3.cogs_ratio.max():.5f}]",
    "",
    "## 3. Projection strategy for Stage D",
    "",
    "Because `cogs_ratio` is bounded (COGS can't exceed Revenue) and has drifted "
    "only modestly across 2017-2022, we blend two estimators:",
    "",
    f"- **Flat recent**: last-90-day median = {last90:.5f}",
    f"- **Linear trend (2021+)**: slope = {slope:.3e}/day, intercept = {intercept:.5f}",
    "",
    "Final projection: `ratio(t) = 0.6 × flat_recent + 0.4 × linear_trend(t)`, clipped to [0.70, 0.95].",
    "",
    f"Projected range across test horizon: "
    f"{proj.cogs_ratio_proj.min():.5f} → {proj.cogs_ratio_proj.max():.5f}, "
    f"mean {proj.cogs_ratio_proj.mean():.5f}",
    "",
    "## 4. How Stage D consumes this",
    "",
    "```python",
    "predicted_COGS = predicted_Revenue × cogs_ratio_proj",
    "```",
    "",
    "where `predicted_Revenue` comes from Stages A/B/C. This gives Stage D a deterministic, "
    "bounded COGS prediction tied to the Revenue model rather than an independent COGS model.",
    "",
    "## 5. Artifacts",
    "",
    "- `data/processed/cogs_ratio_projection.parquet` — 548-row projection for test dates.",
    "- `figures/v12_cogs_ratio_trend.png` — trend visualization.",
    "",
]
DOCS.mkdir(exist_ok=True)
md_path = DOCS / "v12_cogs_verification.md"
md_path.write_text("\n".join(md))
print(f"    report     -> {md_path}")
print("\n=== DONE ===")
