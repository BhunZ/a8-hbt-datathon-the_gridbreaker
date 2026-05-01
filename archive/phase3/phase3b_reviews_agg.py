"""
Phase 3B — Build and validate reviews_agg.

Contract (per Phase 3 preview):
  Grain / PK   : (order_id, product_id)
  Output       : outputs/parquet/reviews_agg.parquet
  Validations  : P3B_PK1, P3B_FK1, P3B_C1, P3B_B3, P3B_B6,
                 P3B_D1, P3B_D2, P3B_D3, P3B_D4

Pinned design decisions:
  - reviews is already unique on (order_id, product_id) in the source — the
    groupby is effectively a structured projection (n_review_events ≡ 1);
    we keep the groupby form anyway so the schema is future-proof (e.g.,
    amended / re-edited reviews would be absorbed cleanly).
  - No B11-style truncation: reviews are voluntary submissions, not an
    event chain. `truncation_flag_review` is constant 0.
  - has_low_rating  = 1 iff rating ≤ 2  (customer dissatisfaction flag)
    has_high_rating = 1 iff rating ≥ 4  (customer satisfaction flag)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/sessions/lucid-relaxed-edison/mnt/Datathon")

ROOT   = Path("/sessions/lucid-relaxed-edison/mnt/Datathon")
SNAP   = ROOT / "outputs" / "parquet"
OUT    = ROOT / "outputs"
AUDIT  = OUT / "phase3_audit_results.csv"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

LOW_RATING_THRESHOLD:  int = 2   # rating <= 2 → dissatisfied
HIGH_RATING_THRESHOLD: int = 4   # rating >= 4 → satisfied

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
reviews = pd.read_parquet(SNAP / "reviews.parquet")
orders  = pd.read_parquet(SNAP / "orders.parquet")[["order_id","order_date"]]
oi      = pd.read_parquet(SNAP / "order_items.parquet")

print(f"Loaded  reviews={len(reviews):,}  orders={len(orders):,}  order_items={len(oi):,}")

gkey = ["order_id", "product_id"]

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def _mode_int(s: pd.Series):
    vc = s.value_counts(dropna=True)
    return int(vc.index[0]) if len(vc) else np.nan

# Pre-compute per-event boolean flags so aggregates stay pure column reductions.
reviews_evt = reviews.assign(
    is_low_rating  = (reviews["rating"] <= LOW_RATING_THRESHOLD).astype("int8"),
    is_high_rating = (reviews["rating"] >= HIGH_RATING_THRESHOLD).astype("int8"),
)

agg = (reviews_evt
       .groupby(gkey, as_index=False, observed=True)
       .agg(
            n_review_events   = ("review_id",   "size"),
            avg_rating        = ("rating",      "mean"),
            min_rating        = ("rating",      "min"),
            max_rating        = ("rating",      "max"),
            dominant_rating   = ("rating",      _mode_int),
            first_review_date = ("review_date", "min"),
            last_review_date  = ("review_date", "max"),
            has_low_rating    = ("is_low_rating",  "max"),
            has_high_rating   = ("is_high_rating", "max"),
       ))

# time_to_first_review_days (needs orders.order_date)
agg = agg.merge(orders, on="order_id", how="left")
agg["time_to_first_review_days"] = (
    (agg["first_review_date"] - agg["order_date"]).dt.days.astype("Int32")
)
agg = agg.drop(columns=["order_date"])

# has_any_review — always 1 on this table; fills to 0 on LEFT-join downstream.
agg["has_any_review"] = np.int8(1)

# truncation_flag_review — reviews are voluntary; no source-level truncation
# signature (unlike B11). Column stays as a schema placeholder for Phase 4.
agg["truncation_flag_review"] = np.int8(0)

# Dtype compaction
agg["n_review_events"]  = agg["n_review_events"].astype("int8")
agg["avg_rating"]       = agg["avg_rating"].astype("float32")
agg["min_rating"]       = agg["min_rating"].astype("int8")
agg["max_rating"]       = agg["max_rating"].astype("int8")
agg["dominant_rating"]  = agg["dominant_rating"].astype("int8")
agg["has_low_rating"]   = agg["has_low_rating"].astype("int8")
agg["has_high_rating"]  = agg["has_high_rating"].astype("int8")

# Reorder
cols = [
    "order_id","product_id",
    "n_review_events",
    "avg_rating","min_rating","max_rating","dominant_rating",
    "first_review_date","last_review_date","time_to_first_review_days",
    "has_low_rating","has_high_rating",
    "has_any_review","truncation_flag_review",
]
reviews_agg = agg[cols].copy()

OUT_PATH = SNAP / "reviews_agg.parquet"
reviews_agg.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH}   rows={len(reviews_agg):,}   cols={len(reviews_agg.columns)}")

# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------
results: list[dict] = []
def record(rule_id: str, metric, threshold: str, verdict: str, detail: str = ""):
    results.append(dict(rule_id=rule_id, metric=metric, threshold=threshold,
                        verdict=verdict, detail=detail))

# ---- P3B_PK1 : PK uniqueness on (order_id, product_id)
dup = int(reviews_agg.duplicated(subset=gkey).sum())
record("P3B_PK1", dup, "= 0", "PASS" if dup == 0 else "FAIL",
       "reviews_agg is unique on (order_id, product_id)")

# ---- P3B_FK1 : subset of order_items keys
oi_keys = set(map(tuple, oi[gkey].drop_duplicates().itertuples(index=False, name=None)))
ra_keys = set(map(tuple, reviews_agg[gkey].itertuples(index=False, name=None)))
orphans = ra_keys - oi_keys
record("P3B_FK1", len(orphans), "= 0", "PASS" if not orphans else "FAIL",
       "reviews_agg keys must be a subset of order_items keys")

# ---- P3B_C1 : conservation of n_review_events
diff_n = int(reviews_agg["n_review_events"].sum()) - len(reviews)
record("P3B_C1", diff_n, "= 0", "PASS" if diff_n == 0 else "FAIL",
       "Σ reviews_agg.n_review_events == len(reviews)")

# ---- P3B_B3 : rating ∈ {1..5} at aggregate level
rmin = int(reviews_agg["min_rating"].min())
rmax = int(reviews_agg["max_rating"].max())
ok_b3 = (rmin >= 1) and (rmax <= 5)
record("P3B_B3", f"[{rmin}..{rmax}]", "⊆ [1..5]",
       "PASS" if ok_b3 else "FAIL",
       "aggregate min_rating and max_rating must be within [1, 5]")

# ---- P3B_B6 : first_review_date >= order_date
ra_ord = reviews_agg.merge(orders, on="order_id", how="left")
bad_b6 = ra_ord[ra_ord["first_review_date"] < ra_ord["order_date"]]
record("P3B_B6", len(bad_b6), "= 0", "PASS" if len(bad_b6) == 0 else "FAIL",
       "first_review_date must be >= orders.order_date (no retro-reviews)")

# ---- P3B_D1 : coverage vs order_items (diagnostic, LOG)
pct = round(len(reviews_agg) / oi[gkey].drop_duplicates().shape[0] * 100, 3)
record("P3B_D1", pct, "LOG", "LOG",
       f"reviews_agg touches {pct}% of distinct order_items line keys")

# ---- P3B_D2 : rating distribution (diagnostic, LOG)
rd = reviews_agg["dominant_rating"].value_counts().sort_index().to_dict()
record("P3B_D2", rd, "LOG", "LOG",
       "dominant_rating distribution across agg lines")

# ---- P3B_D3 : review latency percentiles (diagnostic, LOG)
lat = reviews_agg["time_to_first_review_days"].dropna()
q = lat.quantile([.01,.25,.5,.75,.95,.99]).round(1).to_dict()
record("P3B_D3", q, "LOG", "LOG",
       "time_to_first_review_days percentiles (expect min≈5, p99≈38)")

# ---- P3B_D4 : n_review_events collapse check (should all == 1)
nre_vc = reviews_agg["n_review_events"].value_counts().sort_index().to_dict()
record("P3B_D4", nre_vc, "{1: N}",
       "PASS" if set(nre_vc.keys()) == {1} else "LOG",
       "n_review_events expected ≡ 1 since source table is already PK-unique")

# ---------------------------------------------------------------------------
# Save audit results (append / idempotent-replace for phase 3B)
# ---------------------------------------------------------------------------
res = pd.DataFrame(results)
res["phase"] = "3B"
if AUDIT.exists():
    prev = pd.read_csv(AUDIT)
    prev = prev[prev["phase"] != "3B"]
    res = pd.concat([prev, res], ignore_index=True)
res.to_csv(AUDIT, index=False)
print(f"\nWrote {AUDIT}  ({len(res)} rows cumulative)")

print("\n" + "=" * 78)
print("Phase 3B verdict summary")
print("=" * 78)
summary_cols = ["rule_id","metric","threshold","verdict"]
print(res[res["phase"].eq("3B")][summary_cols].to_string(index=False))

print("\nHead of reviews_agg:")
print(reviews_agg.head(5).to_string(index=False))

# ---- Headline diagnostics for the console summary ------------------------
n_low  = int(reviews_agg["has_low_rating"].sum())
n_high = int(reviews_agg["has_high_rating"].sum())
n_tot  = len(reviews_agg)
print(f"\nRating-flag counts:")
print(f"  has_low_rating  (rating ≤ {LOW_RATING_THRESHOLD})  : {n_low:,}  ({n_low/n_tot*100:.2f}% of {n_tot:,} lines)")
print(f"  has_high_rating (rating ≥ {HIGH_RATING_THRESHOLD})  : {n_high:,}  ({n_high/n_tot*100:.2f}% of {n_tot:,} lines)")
print(f"  avg_rating mean across all lines         : {reviews_agg['avg_rating'].mean():.3f}")
