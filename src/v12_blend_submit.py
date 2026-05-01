"""V12 Step 6 — Gold blend + scaling → final submissions.

Produces the Kaggle-ready probe variants per V12_PLAN §5.2:

  submission_v12a.csv  (Probe 1)  Stages A/B/C only           [Step 4, already written]
  submission_v12b.csv  (Probe ~1.5) + Stage D COGS            [Step 5, already written]
  submission_v12c.csv  (Probe 3)  + gold-blend 0.40/0.60 with v10c
  submission_v12d.csv  (Probe 4)  + scaling Rev*0.99, COGS*0.98

Gold blend formula (V12_PLAN §4):
    pred_final = 0.40 × v12b + 0.60 × v10c_774k

v12b brings fresh 58-feature signal + Stage D COGS ratio.
v10c anchors the mean at the level that already scored 774k.
The blend should lift v12's mean from 3.66M → ~4.00M while retaining v12's improvements.

Outputs:
  submissions/submission_v12c.csv
  submissions/submission_v12d.csv
  docs/v12_probe_summary.md
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from paths import SUBMISSIONS, DOCS, REFERENCE

t0 = time.time()

# ------- load all relevant submissions
v12a = pd.read_csv(SUBMISSIONS / "submission_v12a.csv",       parse_dates=["Date"])
v12b = pd.read_csv(SUBMISSIONS / "submission_v12b.csv",       parse_dates=["Date"])
v10c = pd.read_csv(SUBMISSIONS / "submission_v10c.csv",       parse_dates=["Date"])
friend = pd.read_csv(SUBMISSIONS / "submission_weighted.csv", parse_dates=["Date"])

assert (v12b.Date.values == v10c.Date.values).all(), "date mismatch v12b vs v10c"
assert (v12b.Date.values == friend.Date.values).all(), "date mismatch v12b vs friend"

# ------- v12c: gold blend 0.40 v12b + 0.60 v10c
W_NEW, W_GOLD = 0.40, 0.60

v12c = pd.DataFrame({
    "Date":    v12b.Date.dt.strftime("%Y-%m-%d"),
    "Revenue": np.round(W_NEW * v12b.Revenue + W_GOLD * v10c.Revenue, 2),
    "COGS":    np.round(W_NEW * v12b.COGS    + W_GOLD * v10c.COGS,    2),
})
out_c = SUBMISSIONS / "submission_v12c.csv"
v12c.to_csv(out_c, index=False)
print(f"[{time.time()-t0:5.1f}s] v12c (gold blend {W_NEW}/{W_GOLD}):")
print(f"                         Rev  mean = {v12c.Revenue.mean()/1e6:.3f}M  "
      f"(v12b {v12b.Revenue.mean()/1e6:.2f}M, v10c {v10c.Revenue.mean()/1e6:.2f}M)")
print(f"                         COGS mean = {v12c.COGS.mean()/1e6:.3f}M  "
      f"(v12b {v12b.COGS.mean()/1e6:.2f}M, v10c {v10c.COGS.mean()/1e6:.2f}M)")
print(f"                         implied ratio = {v12c.COGS.mean()/v12c.Revenue.mean():.4f}")

# ------- v12d: v12c + scaling (Rev*0.99, COGS*0.98)
ALPHA_REV, BETA_COGS = 0.99, 0.98
v12d = pd.DataFrame({
    "Date":    v12c.Date,
    "Revenue": np.round(v12c.Revenue * ALPHA_REV,  2),
    "COGS":    np.round(v12c.COGS    * BETA_COGS, 2),
})
out_d = SUBMISSIONS / "submission_v12d.csv"
v12d.to_csv(out_d, index=False)
print(f"\n[{time.time()-t0:5.1f}s] v12d (v12c × scaling {ALPHA_REV}/{BETA_COGS}):")
print(f"                         Rev  mean = {v12d.Revenue.mean()/1e6:.3f}M")
print(f"                         COGS mean = {v12d.COGS.mean()/1e6:.3f}M")
print(f"                         implied ratio = {v12d.COGS.mean()/v12d.Revenue.mean():.4f}")

# ------- Sanity checks: no negatives, no explosions
for name, df in [("v12c", v12c), ("v12d", v12d)]:
    assert (df.Revenue >= 0).all(), f"{name} has negative Revenue"
    assert (df.COGS    >= 0).all(), f"{name} has negative COGS"
    # Max should be within 2x of historical max (~$15M/day)
    max_plausible_rev = 20_000_000
    if df.Revenue.max() > max_plausible_rev:
        print(f"WARNING: {name} max Revenue {df.Revenue.max()/1e6:.1f}M exceeds plausible $20M")

# ------- Cross-correlations (all six variants)
print("\n" + "="*76)
print("                      CROSS-CORRELATION MATRIX (Revenue)")
print("="*76)
variants = {
    "v12a": v12a, "v12b": v12b, "v12c": v12c, "v12d": v12d,
    "v10c": v10c, "friend": friend,
}
for a in variants:
    row = f"  {a:8s}"
    for b in variants:
        r = np.corrcoef(variants[a].Revenue, variants[b].Revenue)[0,1]
        row += f"  {r:.3f}"
    print(row)
print("           " + "   ".join(f"{n:5s}" for n in variants))

# ------- Summary markdown
probe_table = [
    ("submission_v12a.csv",  "Probe 1",  "Stages A/B/C only",
        v12a.Revenue.mean(), v12a.COGS.mean()),
    ("submission_v12b.csv",  "Probe 1b", "+ Stage D (ratio) COGS",
        v12b.Revenue.mean(), v12b.COGS.mean()),
    ("submission_v12c.csv",  "Probe 3",  "+ Gold blend 40/60 with v10c",
        v12c.Revenue.mean(), v12c.COGS.mean()),
    ("submission_v12d.csv",  "Probe 4",  "+ Scaling Rev×0.99, COGS×0.98",
        v12d.Revenue.mean(), v12d.COGS.mean()),
    ("submission_v10c.csv",  "(ref 774k)", "v10c prior best",
        v10c.Revenue.mean(), v10c.COGS.mean()),
    ("submission_weighted.csv", "(friend baseline)", "friend's weighted",
        friend.Revenue.mean(), friend.COGS.mean()),
]

md = [
    "# V12 Probe Summary",
    "",
    "Produced by `src/v12_blend_submit.py`. All files are in `submissions/`.",
    "",
    "| File | Probe | Description | Rev mean | COGS mean | Ratio |",
    "|---|---|---|---:|---:|---:|",
]
for f, p, d, rev, cogs in probe_table:
    md.append(f"| {f} | {p} | {d} | {rev/1e6:.3f}M | {cogs/1e6:.3f}M | "
              f"{cogs/rev:.4f} |")

md += [
    "",
    "## Cross-correlation — Revenue",
    "",
    "| | " + " | ".join(variants.keys()) + " |",
    "|" + "|".join(["---"] * (len(variants) + 1)) + "|",
]
for a in variants:
    row = f"| {a} |"
    for b in variants:
        r = np.corrcoef(variants[a].Revenue, variants[b].Revenue)[0,1]
        row += f" {r:.3f} |"
    md.append(row)

md += [
    "",
    "## Cross-correlation — COGS",
    "",
    "| | " + " | ".join(variants.keys()) + " |",
    "|" + "|".join(["---"] * (len(variants) + 1)) + "|",
]
for a in variants:
    row = f"| {a} |"
    for b in variants:
        r = np.corrcoef(variants[a].COGS, variants[b].COGS)[0,1]
        row += f" {r:.3f} |"
    md.append(row)

md += [
    "",
    "## Recommended upload order to Kaggle",
    "",
    "Upload budget is finite, so order by expected information gain:",
    "",
    "1. **v12d** first — full pipeline (gold blend + scaling). Most-likely best.",
    "2. **v12c** second — identifies how much scaling alone contributes (v12c→v12d delta).",
    "3. **v12a** third — identifies how much gold blend + scaling contribute vs raw model.",
    "4. v12b — only if v12a shows promise; isolates the Stage D COGS lift.",
    "",
    "Decision rules (from V12_PLAN §8):",
    "",
    "- If **v12d ≤ 728k**: beaten friend. Stop and submit.",
    "- If **v12d in 728–760k**: tune α, β via residuals from Kaggle feedback (Probe 5).",
    "- If **v12d > 774k** (worse than v10c alone): gold blend weight may need inversion (0.6 new / 0.4 gold), OR Stage D is hurting and we revert to v12a (no Stage D).",
    "",
    "## Notes",
    "",
    "- v12 ensemble trained on 58 features, with full aux-file profile projection.",
    "- v10c uses 23 features (friend's set + 7 named events + extended traffic).",
    "- Gold blend tethers v12's mean (3.66M) to v10c's mean (4.23M) → 4.00M.",
    "- Scaling applies friend's observed over-prediction correction (COGS tends higher than truth).",
    "",
]

DOCS.mkdir(exist_ok=True)
md_path = DOCS / "v12_probe_summary.md"
md_path.write_text("\n".join(md))
print(f"\n[{time.time()-t0:5.1f}s] Summary -> {md_path}")

# ------- Final console report
print("\n" + "="*76)
print("                            PROBE SUMMARY")
print("="*76)
print(f"{'File':30s}  {'Probe':10s}  {'Rev μ':>9s}  {'COGS μ':>9s}  {'Ratio':>6s}")
for f, p, d, rev, cogs in probe_table:
    print(f"{f:30s}  {p:10s}  {rev/1e6:8.3f}M  {cogs/1e6:8.3f}M  {cogs/rev:6.4f}")

print(f"\n=== DONE in {time.time()-t0:.1f}s ===")
