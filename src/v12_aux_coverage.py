"""V12 Step 1 — Audit auxiliary file date coverage.

For each auxiliary table, determine:
  - min / max date
  - % of rows in train horizon (≤ 2022-12-31)
  - % of rows in test horizon (2023-01-01 → 2024-07-01)
  - whether the table is USABLE DIRECTLY for the test horizon
    or must be PROJECTED via profile (median by month, dow, ...)

Produces:
  - stdout table
  - docs/aux_coverage.md
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd

from paths import RAW, DOCS

TRAIN_END  = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END   = pd.Timestamp("2024-07-01")

# file -> list of date columns, ordered (primary date first)
FILES = {
    "sales":        ("sales.csv",        ["Date"]),
    "web_traffic":  ("web_traffic.csv",  ["date"]),
    "orders":       ("orders.csv",       ["order_date"]),
    "order_items":  ("order_items.csv",  []),  # no date column — joins via order_id
    "payments":     ("payments.csv",     []),  # same — joins via order_id
    "shipments":    ("shipments.csv",    ["ship_date", "delivery_date"]),
    "reviews":      ("reviews.csv",      ["review_date"]),
    "returns":      ("returns.csv",      ["return_date"]),
    "customers":    ("customers.csv",    ["signup_date"]),
    "promotions":   ("promotions.csv",   ["start_date", "end_date"]),
    "inventory":    ("inventory.csv",    ["snapshot_date"]),
    "products":     ("products.csv",     []),  # static lookup
    "geography":    ("geography.csv",    []),  # static lookup
}

results = []

def analyse(name: str, fname: str, date_cols: list[str]):
    path = RAW / fname
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    if not date_cols:
        # No date column — these are either static (products, geography)
        # or need to inherit date via join (order_items, payments via order_id).
        row = {
            "table": name,
            "rows": n,
            "date_col": "-",
            "min_date": "-",
            "max_date": "-",
            "pct_train": "-",
            "pct_test": "-",
            "verdict": "STATIC / via-join" if name in ("products", "geography") else "VIA-JOIN(order_id)",
        }
        results.append(row)
        return

    primary = date_cols[0]
    dt = pd.to_datetime(df[primary], errors="coerce")
    mask_train = dt <= TRAIN_END
    mask_test  = (dt >= TEST_START) & (dt <= TEST_END)
    mask_future = dt > TEST_END

    pct_train  = 100 * mask_train.sum()  / n
    pct_test   = 100 * mask_test.sum()   / n
    pct_future = 100 * mask_future.sum() / n

    # Verdict
    if mask_test.sum() == 0:
        verdict = "PROJECT (no test-horizon coverage)"
    elif pct_test >= 20:
        verdict = "USE DIRECTLY (strong test coverage)"
    elif pct_test >= 5:
        verdict = "USE DIRECTLY (partial test coverage)"
    else:
        verdict = "PROJECT (sparse test coverage)"

    row = {
        "table": name,
        "rows": n,
        "date_col": primary,
        "min_date": str(dt.min().date()) if pd.notna(dt.min()) else "NaT",
        "max_date": str(dt.max().date()) if pd.notna(dt.max()) else "NaT",
        "pct_train": f"{pct_train:5.1f}%",
        "pct_test":  f"{pct_test:5.1f}%",
        "verdict": verdict,
    }
    results.append(row)

    # Extra analysis for multi-date tables
    if len(date_cols) > 1:
        for extra in date_cols[1:]:
            dt2 = pd.to_datetime(df[extra], errors="coerce")
            results.append({
                "table": f"  {name} (via {extra})",
                "rows": n,
                "date_col": extra,
                "min_date": str(dt2.min().date()) if pd.notna(dt2.min()) else "NaT",
                "max_date": str(dt2.max().date()) if pd.notna(dt2.max()) else "NaT",
                "pct_train": f"{100*(dt2<=TRAIN_END).sum()/n:5.1f}%",
                "pct_test":  f"{100*((dt2>=TEST_START)&(dt2<=TEST_END)).sum()/n:5.1f}%",
                "verdict": "(secondary date)",
            })


# ---------- Main
print(f"\nTest horizon: {TEST_START.date()} → {TEST_END.date()}  ({(TEST_END-TEST_START).days+1} days)")
print(f"Train cutoff: ≤ {TRAIN_END.date()}\n")

for name, (fname, cols) in FILES.items():
    print(f"Loading {fname}...", end=" ", flush=True)
    analyse(name, fname, cols)
    print("ok")

print()
res = pd.DataFrame(results)
print(res.to_string(index=False))

# ----- Join-based coverage check: order_items & payments inherit order.order_date
print("\n\n=== ORDER-ID JOIN COVERAGE CHECK ===")
orders = pd.read_csv(RAW / "orders.csv", usecols=["order_id", "order_date"],
                     parse_dates=["order_date"], low_memory=False)
for fn in ["order_items.csv", "payments.csv"]:
    aux = pd.read_csv(RAW / fn, usecols=["order_id"], low_memory=False)
    merged = aux.merge(orders, on="order_id", how="left")
    n = len(merged)
    test_mask = (merged.order_date >= TEST_START) & (merged.order_date <= TEST_END)
    print(f"  {fn:16s}  rows={n:>8}  "
          f"min={merged.order_date.min().date()}  "
          f"max={merged.order_date.max().date()}  "
          f"pct_test={100*test_mask.sum()/n:5.1f}%")


# ----- Write markdown report -----
md = [
    "# V12 Step 1 — Auxiliary File Date Coverage",
    "",
    f"**Test horizon**: {TEST_START.date()} → {TEST_END.date()}  "
    f"({(TEST_END-TEST_START).days+1} days)",
    f"**Train cutoff**: ≤ {TRAIN_END.date()}",
    "",
    "## Coverage per file",
    "",
    "| Table | Rows | Date col | Min | Max | % train | % test | Verdict |",
    "|---|---:|---|---|---|---:|---:|---|",
]
for r in results:
    md.append(
        f"| {r['table']} | {r['rows']:,} | {r['date_col']} | "
        f"{r['min_date']} | {r['max_date']} | {r['pct_train']} | "
        f"{r['pct_test']} | {r['verdict']} |"
    )

md += [
    "",
    "## Order-id join coverage (for date-less tables)",
    "",
    "`order_items.csv` and `payments.csv` inherit their date from `orders.order_date` via `order_id`.",
    "",
]

# Re-run the join check into md
orders = pd.read_csv(RAW / "orders.csv", usecols=["order_id", "order_date"],
                     parse_dates=["order_date"], low_memory=False)
md.append("| Aux file | Rows | Min (via orders) | Max (via orders) | % in test horizon |")
md.append("|---|---:|---|---|---:|")
for fn in ["order_items.csv", "payments.csv"]:
    aux = pd.read_csv(RAW / fn, usecols=["order_id"], low_memory=False)
    merged = aux.merge(orders, on="order_id", how="left")
    n = len(merged)
    test_mask = (merged.order_date >= TEST_START) & (merged.order_date <= TEST_END)
    md.append(
        f"| {fn} | {n:,} | {merged.order_date.min().date()} | "
        f"{merged.order_date.max().date()} | "
        f"{100*test_mask.sum()/n:.1f}% |"
    )

md += [
    "",
    "## Feature-strategy implications for V12",
    "",
    "### DIRECT (usable for test horizon without projection)",
    "Tables that cover the test horizon — daily rollups can be computed directly for "
    "2023-01 → 2024-07 and used as real-valued inputs, no leakage risk:",
    "",
    "### PROJECT (must be median-profiled by (month, dow, ...) )",
    "Tables that stop at or before 2022-12-31 — we must build the same (month, dow, "
    "is_dd, is_pd, is_tet) profile approach friend uses on web_traffic and apply it to "
    "these features too.",
    "",
    "### STATIC (lookup tables — join on key, no date issue)",
    "`products.csv`, `geography.csv` — used as lookups. `products.cogs` drives the exact "
    "COGS reconstruction `Σ(quantity × cogs)`.",
    "",
    "See V12_PLAN.md §3 for the full 42-feature roster that builds on this coverage map.",
    "",
]

DOCS.mkdir(exist_ok=True)
out = DOCS / "aux_coverage.md"
out.write_text("\n".join(md))
print(f"\n→ wrote {out}")
