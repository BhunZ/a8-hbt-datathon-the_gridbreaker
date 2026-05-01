"""V12 Step 5 — Stage D: COGS via ratio route.

Per V12_PLAN §4 blend formula:
    pred_rev_final     = pred_rev_stageABC
    pred_cogs_direct   = pred_rev_stageABC × cogs_ratio_proj(t)
    pred_cogs_final    = 0.60 × pred_cogs_direct + 0.40 × pred_cogs_stageABC

Why: the cogs-ratio is bounded and stable (Step 3 showed yearly means in [0.84, 0.92]).
Tethering COGS to Revenue via the projected ratio eliminates independent COGS model error
on the 60% weight, while keeping 40% Stage-A/B/C COGS as a hedge against ratio drift.

Inputs:
  data/processed/v12a_stage_predictions.parquet      (from Step 4)
  data/processed/cogs_ratio_projection.parquet       (from Step 3)
Output:
  submissions/submission_v12b.csv                    (Stage A/B/C Rev + Stage D COGS blend)
  data/processed/v12b_stage_predictions.parquet      (diagnostic with all pred columns)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from paths import PROCESSED, SUBMISSIONS, REFERENCE

t0 = time.time()

# ------- load Step 4 stage predictions
pred = pd.read_parquet(PROCESSED / "v12a_stage_predictions.parquet")
pred["date"] = pd.to_datetime(pred.date)
print(f"[{time.time()-t0:5.1f}s] Loaded v12a stage preds: {pred.shape}")

# ------- load Step 3 cogs_ratio projection
ratio = pd.read_parquet(PROCESSED / "cogs_ratio_projection.parquet")
ratio["date"] = pd.to_datetime(ratio.date)
print(f"[{time.time()-t0:5.1f}s] Loaded cogs_ratio projection: {ratio.shape}")
print(f"                         ratio range: {ratio.cogs_ratio_proj.min():.4f} -> "
      f"{ratio.cogs_ratio_proj.max():.4f}  mean {ratio.cogs_ratio_proj.mean():.4f}")

# ------- join
pred = pred.merge(ratio[["date", "cogs_ratio_proj"]], on="date", how="left")
assert pred.cogs_ratio_proj.notna().all(), "missing cogs_ratio projection for some dates"

# ------- compute Stage D (direct = Rev × ratio)
pred["cogs_direct"] = pred.rev_blend * pred.cogs_ratio_proj

# ------- final COGS blend (0.60 direct + 0.40 ABC)
W_DIRECT, W_ABC = 0.60, 0.40
pred["cogs_final"] = W_DIRECT * pred.cogs_direct + W_ABC * pred.cogs_blend
pred["rev_final"]  = pred.rev_blend   # Rev unchanged

print(f"\n[{time.time()-t0:5.1f}s] Stage D COGS blend: {W_DIRECT} direct + {W_ABC} ABC")
print(f"                         cogs_blend  (ABC)    mean = {pred.cogs_blend.mean()/1e6:.3f}M")
print(f"                         cogs_direct (Rev×r)  mean = {pred.cogs_direct.mean()/1e6:.3f}M")
print(f"                         cogs_final  (blend)  mean = {pred.cogs_final.mean()/1e6:.3f}M")
print(f"                         rev_final            mean = {pred.rev_final.mean()/1e6:.3f}M")
print(f"                         implied ratio final       = {pred.cogs_final.mean()/pred.rev_final.mean():.4f}")

# ------- correlations between the two COGS estimators
corr_abc_d = np.corrcoef(pred.cogs_blend, pred.cogs_direct)[0, 1]
print(f"                         corr(cogs_ABC, cogs_direct) = {corr_abc_d:.4f}   "
      f"(higher = the two methods agree)")

# ------- write submission
sample = pd.read_csv(REFERENCE / "sample_submission.csv", parse_dates=["Date"])
pred_sorted = pred.sort_values("date").reset_index(drop=True)
assert (pred_sorted.date.dt.date.values == sample.Date.dt.date.values).all(), \
       "date mismatch between pred and sample_submission"

sub = pd.DataFrame({
    "Date":    sample.Date.dt.strftime("%Y-%m-%d"),
    "Revenue": np.round(pred_sorted.rev_final.values,  2),
    "COGS":    np.round(pred_sorted.cogs_final.values, 2),
})

SUBMISSIONS.mkdir(exist_ok=True)
out = SUBMISSIONS / "submission_v12b.csv"
sub.to_csv(out, index=False)
print(f"\n[{time.time()-t0:5.1f}s] WROTE {out}")
print(f"                         mean   Rev={sub.Revenue.mean()/1e6:.3f}M   COGS={sub.COGS.mean()/1e6:.3f}M")
print(f"                         total  Rev={sub.Revenue.sum()/1e9:.3f}B   COGS={sub.COGS.sum()/1e9:.3f}B")

# ------- save diagnostic
diag_path = PROCESSED / "v12b_stage_predictions.parquet"
pred_sorted.to_parquet(diag_path, index=False)
print(f"                         diagnostic -> {diag_path}")

# ------- compare to v12a (Step 4) and v10c
try:
    v12a = pd.read_csv(SUBMISSIONS / "submission_v12a.csv", parse_dates=["Date"])
    v10c = pd.read_csv(SUBMISSIONS / "submission_v10c.csv", parse_dates=["Date"])
    print(f"\n[{time.time()-t0:5.1f}s] Comparison:")
    print(f"                         v12a (ABC COGS):     Rev={v12a.Revenue.mean()/1e6:.3f}M  "
          f"COGS={v12a.COGS.mean()/1e6:.3f}M")
    print(f"                         v12b (Stage D COGS): Rev={sub.Revenue.mean()/1e6:.3f}M  "
          f"COGS={sub.COGS.mean()/1e6:.3f}M")
    print(f"                         v10c (774k Kaggle):  Rev={v10c.Revenue.mean()/1e6:.3f}M  "
          f"COGS={v10c.COGS.mean()/1e6:.3f}M")
    dcog_12a = (sub.COGS - v12a.COGS).abs().mean()
    dcog_10c = (sub.COGS - v10c.COGS).abs().mean()
    print(f"                         |COGS v12b - v12a| mean = {dcog_12a/1e6:.3f}M   "
          f"|COGS v12b - v10c| mean = {dcog_10c/1e6:.3f}M")
except Exception as e:
    print(f"(comparison skipped: {e})")

print(f"\n=== DONE in {time.time()-t0:.1f}s ===")
