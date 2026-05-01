"""Central path config. Import as `from paths import RAW, PROCESSED, SUBMISSIONS, ROOT, REFERENCE`.

Works from any script in src/ regardless of where it's run from.
"""
from pathlib import Path

# src/ is one level below the project root
ROOT        = Path(__file__).resolve().parent.parent
RAW         = ROOT / "data" / "raw"
PROCESSED   = ROOT / "data" / "processed"
REFERENCE   = ROOT / "data" / "reference"
SUBMISSIONS = ROOT / "submissions"
DOCS        = ROOT / "docs"
PLANS       = ROOT / "plans"
FIGURES     = ROOT / "figures"
LOGS        = ROOT / "logs"
ARCHIVE     = ROOT / "archive"

# Quick existence check when run directly: `python src/paths.py`
if __name__ == "__main__":
    for n, p in [("ROOT", ROOT), ("RAW", RAW), ("PROCESSED", PROCESSED),
                 ("REFERENCE", REFERENCE), ("SUBMISSIONS", SUBMISSIONS),
                 ("DOCS", DOCS), ("PLANS", PLANS)]:
        print(f"{n:12s} {'OK' if p.exists() else 'MISSING':8s} {p}")
