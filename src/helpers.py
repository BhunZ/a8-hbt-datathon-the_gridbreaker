"""
DATATHON 2026 — Pipeline helpers
--------------------------------
Shared utilities for the multi-phase ETL pipeline.
Imported by pipeline.ipynb.

Contracts are defined in PIPELINE_SPEC.md.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Path constants — points at the new data/raw/ layout.
# Notebook or script callers can still override via the `data_dir` arg of
# load_csv_typed(), or by reassigning DATA_DIR before use.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema declarations — used by load_csv_typed
# ---------------------------------------------------------------------------
DATE_COLUMNS: dict[str, list[str]] = {
    "customers": ["signup_date"],
    "promotions": ["start_date", "end_date"],
    "orders": ["order_date"],
    "shipments": ["ship_date", "delivery_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "sales": ["Date"],
    "sample_submission": ["Date"],
    "inventory": ["snapshot_date"],
    "web_traffic": ["date"],
}

# Explicit dtypes (memory + correctness). `string` uses pandas StringDtype;
# `category` is only applied AFTER load because category dtype interacts badly
# with read_csv for columns that can legitimately contain NaN.
READ_DTYPES: dict[str, dict[str, str]] = {
    "products": {
        "product_id": "int32", "product_name": "string",
        "category": "string", "segment": "string",
        "size": "string", "color": "string",
        "price": "float64", "cogs": "float64",
    },
    "customers": {
        "customer_id": "int32", "zip": "int32",
        "city": "string", "gender": "string",
        "age_group": "string", "acquisition_channel": "string",
    },
    "promotions": {
        "promo_id": "string", "promo_name": "string",
        "promo_type": "string", "discount_value": "float64",
        "applicable_category": "string", "promo_channel": "string",
        "stackable_flag": "int8", "min_order_value": "float64",
    },
    "geography": {
        "zip": "int32", "city": "string",
        "region": "string", "district": "string",
    },
    "orders": {
        "order_id": "int32", "customer_id": "int32", "zip": "int32",
        "order_status": "string", "payment_method": "string",
        "device_type": "string", "order_source": "string",
    },
    "order_items": {
        "order_id": "int32", "product_id": "int32",
        "quantity": "int16", "unit_price": "float64",
        "discount_amount": "float64",
        "promo_id": "string", "promo_id_2": "string",
    },
    "payments": {
        "order_id": "int32", "payment_method": "string",
        "payment_value": "float64", "installments": "int8",
    },
    "shipments": {"order_id": "int32", "shipping_fee": "float64"},
    "returns": {
        "return_id": "string", "order_id": "int32", "product_id": "int32",
        "return_reason": "string", "return_quantity": "int16",
        "refund_amount": "float64",
    },
    "reviews": {
        "review_id": "string", "order_id": "int32", "product_id": "int32",
        "customer_id": "int32", "rating": "int8", "review_title": "string",
    },
    "sales": {"Revenue": "float64", "COGS": "float64"},
    "sample_submission": {"Revenue": "float64", "COGS": "float64"},
    "inventory": {
        "product_id": "int32", "stock_on_hand": "int32",
        "units_received": "int32", "units_sold": "int32",
        "stockout_days": "int16", "days_of_supply": "float64",
        "fill_rate": "float64", "stockout_flag": "int8",
        "overstock_flag": "int8", "reorder_flag": "int8",
        "sell_through_rate": "float64", "product_name": "string",
        "category": "string", "segment": "string",
        "year": "int16", "month": "int8",
    },
    "web_traffic": {
        "sessions": "int32", "unique_visitors": "int32",
        "page_views": "int32", "bounce_rate": "float64",
        "avg_session_duration_sec": "float64", "traffic_source": "string",
    },
}

# Columns promoted to category dtype post-load for memory.
CATEGORICAL_COLUMNS: dict[str, list[str]] = {
    "products": ["category", "segment", "size", "color"],
    "customers": ["gender", "age_group", "acquisition_channel"],
    "promotions": ["promo_type", "applicable_category", "promo_channel"],
    "geography": ["region"],
    "orders": ["order_status", "payment_method", "device_type", "order_source"],
    "payments": ["payment_method"],
    "returns": ["return_reason"],
    "inventory": ["category", "segment"],
    "web_traffic": ["traffic_source"],
}

ALL_TABLES: list[str] = [
    "customers", "geography", "inventory", "order_items", "orders",
    "payments", "products", "promotions", "returns", "reviews",
    "sales", "sample_submission", "shipments", "web_traffic",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_csv_typed(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load one CSV with explicit dtypes and parsed date columns.
    Applies categorical dtype post-load for low-cardinality string columns.
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / f"{name}.csv"
    df = pd.read_csv(
        path,
        dtype=READ_DTYPES.get(name, {}),
        parse_dates=DATE_COLUMNS.get(name, []),
        low_memory=False,
    )
    for col in CATEGORICAL_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_all(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load every table listed in ALL_TABLES into a dict."""
    return {t: load_csv_typed(t, data_dir) for t in ALL_TABLES}


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------
def profile_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-column profile: dtype, null count, null pct, unique count, min/max for numeric/date."""
    rows = []
    for c in df.columns:
        s = df[c]
        row = {
            "table": name, "column": c, "dtype": str(s.dtype),
            "n_null": int(s.isna().sum()),
            "pct_null": round(s.isna().mean() * 100, 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            row["min"] = s.min()
            row["max"] = s.max()
        else:
            row["min"] = None
            row["max"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def audit_fk(
    child: pd.DataFrame, child_key: str,
    parent: pd.DataFrame, parent_key: str,
    name: str,
) -> dict:
    """Return orphan statistics for a FK relationship."""
    cv = set(child[child_key].dropna().unique())
    pv = set(parent[parent_key].unique())
    orphans = cv - pv
    n = len(cv)
    return {
        "check_id": name,
        "child_unique_keys": n,
        "orphan_count": len(orphans),
        "orphan_pct": round(len(orphans) / max(n, 1) * 100, 4),
        "verdict": "PASS" if not orphans else "FAIL",
        "sample_orphans": list(sorted(orphans))[:5],
    }


def audit_pk_unique(df: pd.DataFrame, keys: Iterable[str], name: str) -> dict:
    """Verify one or more columns form a primary key (unique together)."""
    keys = list(keys)
    dup_count = int(df.duplicated(subset=keys).sum())
    return {
        "check_id": name,
        "pk_columns": keys,
        "rows": len(df),
        "duplicate_count": dup_count,
        "verdict": "PASS" if dup_count == 0 else "FAIL",
    }


def assert_no_fanout(
    df_before: pd.DataFrame, df_after: pd.DataFrame,
    spine_key: str, step_name: str,
) -> None:
    """Spine key must have identical unique-count before and after a join."""
    b = df_before[spine_key].nunique()
    a = df_after[spine_key].nunique()
    if b != a:
        raise AssertionError(
            f"{step_name}: spine {spine_key} dropped from {b} to {a} unique values"
        )
    if len(df_before) != len(df_after):
        print(f"[WARN {step_name}] row count changed: {len(df_before)} -> {len(df_after)}")


# ---------------------------------------------------------------------------
# Placeholders for later phases
# ---------------------------------------------------------------------------
def reconcile_revenue(*args, **kwargs):
    """Phase 5 — evaluate multiple Revenue formulas vs sales.Revenue."""
    raise NotImplementedError("Implemented in Phase 5")


def check_leakage(*args, **kwargs):
    """Phase 8 — verify no feature uses same-or-future information vs its target date."""
    raise NotImplementedError("Implemented in Phase 8")
