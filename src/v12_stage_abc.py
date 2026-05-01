"""V12 Step 4 — Triple-stage ensemble (Stages A/B/C).

Three models × two targets (Revenue, COGS):
  Stage A — ANCHOR       : sales[year>=2016], log1p, ExtraTrees(1000, d=15), w=1
  Stage B — MODERNIST    : sales[year>=2019], log1p, RandomForest(1000, d=15),
                           w = (year-2018)^1.5
  Stage C — PEAK CATCHER : sales[year>=2018], RAW (no log!), ExtraTrees(1000, d=15), w=1

Final blend per target: 0.40 × A + 0.40 × B + 0.20 × C.

This is Probe 1 per V12_PLAN §5.2: model-only, no gold blend, no scaling.

Outputs:
  submissions/submission_v12a.csv          (the Kaggle-ready CSV)
  data/processed/v12a_stage_predictions.parquet  (per-stage preds for diagnosis)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from paths import PROCESSED, SUBMISSIONS, REFERENCE

t0 = time.time()

# ------- load
df = pd.read_parquet(PROCESSED / "daily_features_v12.parquet")

# Feature set: everything EXCEPT date/set/targets
EXCLUDE = {"date", "set", "Revenue", "COGS"}
FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"[{time.time()-t0:5.1f}s] Feature count: {len(FEATURES)}")

# Train / test splits by set tag
train_all = df[df.set == "train"].copy()
test      = df[df.set == "test"].copy()
print(f"[{time.time()-t0:5.1f}s] Train rows: {len(train_all)}  Test rows: {len(test)}\n")

# ------- sub-splits per stage (year cutoffs)
A_cut = train_all[train_all.year >= 2016].copy()   # Anchor
B_cut = train_all[train_all.year >= 2019].copy()   # Modernist
C_cut = train_all[train_all.year >= 2018].copy()   # Peak Catcher

print(f"Stage A (Anchor)      rows={len(A_cut)}  years={A_cut.year.min()}-{A_cut.year.max()}")
print(f"Stage B (Modernist)   rows={len(B_cut)}  years={B_cut.year.min()}-{B_cut.year.max()}")
print(f"Stage C (PeakCatcher) rows={len(C_cut)}  years={C_cut.year.min()}-{C_cut.year.max()}\n")

# ------- feature / target matrices
def Xy(df_sub, target):
    return df_sub[FEATURES].values.astype(float), df_sub[target].values.astype(float)

Xt = test[FEATURES].values.astype(float)

stage_preds = {"Revenue": {}, "COGS": {}}

# ------- Stage A (Anchor) — log1p, ExtraTrees(1000,15), w=1
for tgt in ("Revenue", "COGS"):
    X, y = Xy(A_cut, tgt)
    y_log = np.log1p(y)
    m = ExtraTreesRegressor(n_estimators=300, max_depth=15,
                            random_state=42, n_jobs=-1)
    m.fit(X, y_log)
    stage_preds[tgt]["A"] = np.expm1(m.predict(Xt))
    print(f"[{time.time()-t0:5.1f}s] Stage A  {tgt:8s} pred mu={stage_preds[tgt]['A'].mean()/1e6:.3f}M")

# ------- Stage B (Modernist) — log1p, RF(1000,15), w=(year-2018)^1.5
for tgt in ("Revenue", "COGS"):
    X, y = Xy(B_cut, tgt)
    y_log = np.log1p(y)
    w = (B_cut.year.values - 2018.0) ** 1.5
    m = RandomForestRegressor(n_estimators=300, max_depth=15,
                              random_state=42, n_jobs=-1)
    m.fit(X, y_log, sample_weight=w)
    stage_preds[tgt]["B"] = np.expm1(m.predict(Xt))
    print(f"[{time.time()-t0:5.1f}s] Stage B  {tgt:8s} pred mu={stage_preds[tgt]['B'].mean()/1e6:.3f}M")

# ------- Stage C (Peak Catcher) — RAW, ExtraTrees(1000,15), w=1
for tgt in ("Revenue", "COGS"):
    X, y = Xy(C_cut, tgt)
    m = ExtraTreesRegressor(n_estimators=300, max_depth=15,
                            random_state=42, n_jobs=-1)
    m.fit(X, y)   # RAW — no log
    stage_preds[tgt]["C"] = m.predict(Xt)
    print(f"[{time.time()-t0:5.1f}s] Stage C  {tgt:8s} pred mu={stage_preds[tgt]['C'].mean()/1e6:.3f}M")

# ------- Blend
WA, WB, WC = 0.40, 0.40, 0.20
def blend(p): return WA * p["A"] + WB * p["B"] + WC * p["C"]

pred_rev  = blend(stage_preds["Revenue"])
pred_cogs = blend(stage_preds["COGS"])

print(f"\n[{time.time()-t0:5.1f}s] Blend weights: A={WA} B={WB} C={WC}")
print(f"                         Revenue blended mean = {pred_rev.mean()/1e6:.3f}M  "
      f"(min {pred_rev.min()/1e6:.3f} max {pred_rev.max()/1e6:.3f})")
print(f"                         COGS    blended mean = {pred_cogs.mean()/1e6:.3f}M  "
      f"(min {pred_cogs.min()/1e6:.3f} max {pred_cogs.max()/1e6:.3f})")
print(f"                         Implied cogs_ratio   = {pred_cogs.mean()/pred_rev.mean():.4f}")

# ------- Write submission
sample = pd.read_csv(REFERENCE / "sample_submission.csv", parse_dates=["Date"])
test_sorted = test.sort_values("date").reset_index(drop=True)
assert (test_sorted.date.dt.date.values == sample.Date.dt.date.values).all(), \
       "date mismatch between test features and sample_submission"

sub = pd.DataFrame({
    "Date":    sample.Date.dt.strftime("%Y-%m-%d"),
    "Revenue": np.round(pred_rev, 2),
    "COGS":    np.round(pred_cogs, 2),
})

SUBMISSIONS.mkdir(exist_ok=True)
out = SUBMISSIONS / "submission_v12a.csv"
sub.to_csv(out, index=False)
print(f"\n[{time.time()-t0:5.1f}s] WROTE {out}")
print(f"                         mean   Rev={sub.Revenue.mean()/1e6:.3f}M   COGS={sub.COGS.mean()/1e6:.3f}M")
print(f"                         total  Rev={sub.Revenue.sum()/1e9:.3f}B   COGS={sub.COGS.sum()/1e9:.3f}B")

# ------- Also save per-stage predictions for diagnosis
diag = pd.DataFrame({
    "date":       test_sorted.date.values,
    "rev_A":      stage_preds["Revenue"]["A"],
    "rev_B":      stage_preds["Revenue"]["B"],
    "rev_C":      stage_preds["Revenue"]["C"],
    "rev_blend":  pred_rev,
    "cogs_A":     stage_preds["COGS"]["A"],
    "cogs_B":     stage_preds["COGS"]["B"],
    "cogs_C":     stage_preds["COGS"]["C"],
    "cogs_blend": pred_cogs,
})
diag_path = PROCESSED / "v12a_stage_predictions.parquet"
diag.to_parquet(diag_path, index=False)
print(f"                         per-stage preds -> {diag_path}")

# ------- Comparison to previous best
try:
    v10c = pd.read_csv(SUBMISSIONS / "submission_v10c.csv", parse_dates=["Date"])
    rev_corr = np.corrcoef(sub.Revenue, v10c.Revenue)[0,1]
    cogs_corr = np.corrcoef(sub.COGS, v10c.COGS)[0,1]
    print(f"\n[{time.time()-t0:5.1f}s] vs v10c (774k Kaggle):")
    print(f"                         Rev  corr = {rev_corr:.4f}   "
          f"dmean = {(sub.Revenue.mean()-v10c.Revenue.mean())/1e6:+.3f}M")
    print(f"                         COGS corr = {cogs_corr:.4f}   "
          f"dmean = {(sub.COGS.mean()-v10c.COGS.mean())/1e6:+.3f}M")
except Exception as e:
    print(f"(v10c comparison skipped: {e})")

try:
    fb = pd.read_csv(SUBMISSIONS / "submission_weighted.csv", parse_dates=["Date"])
    rev_corr = np.corrcoef(sub.Revenue, fb.Revenue)[0,1]
    cogs_corr = np.corrcoef(sub.COGS, fb.COGS)[0,1]
    print(f"\n[{time.time()-t0:5.1f}s] vs submission_weighted (friend baseline):")
    print(f"                         Rev  corr = {rev_corr:.4f}   "
          f"dmean = {(sub.Revenue.mean()-fb.Revenue.mean())/1e6:+.3f}M")
    print(f"                         COGS corr = {cogs_corr:.4f}   "
          f"dmean = {(sub.COGS.mean()-fb.COGS.mean())/1e6:+.3f}M")
except Exception as e:
    print(f"(friend comparison skipped: {e})")

print(f"\n=== DONE in {time.time()-t0:.1f}s ===")
