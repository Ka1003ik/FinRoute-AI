"""Measure the deterministic text cleaner without changing the dataset."""

import json
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative


DATA_PATH = ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "text_cleaning_profile.json"
TEXT_COLUMN = "Consumer complaint narrative"
REDACTION_PATTERN = re.compile(r"\b[xX]{2,}\b")


def main() -> None:
    """Count redaction markers and resulting normalized duplicates."""
    data = pd.read_csv(
        DATA_PATH,
        usecols=[TEXT_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    narratives = data[TEXT_COLUMN].str.strip()
    cleaned = narratives.map(clean_narrative)

    placeholder_counts = narratives.map(
        lambda text: len(REDACTION_PATTERN.findall(text))
    )
    report = {
        "rows_profiled": int(len(narratives)),
        "rows_with_redaction_markers": int(placeholder_counts.gt(0).sum()),
        "redaction_marker_occurrences": int(placeholder_counts.sum()),
        "unique_narratives_before_cleaning": int(narratives.nunique()),
        "unique_narratives_after_cleaning": int(cleaned.nunique()),
        "rows_changed_by_cleaning": int(narratives.ne(cleaned).sum()),
        "empty_narratives_after_cleaning": int(cleaned.eq("").sum()),
        "cleaning_rules": {
            "unicode": "NFKC",
            "html_entities": "unescape",
            "redaction_markers": "remove standalone runs of 2 or more x characters",
            "whitespace": "collapse to one space",
            "case": "lowercase",
            "preserved": ["numbers", "punctuation", "negation", "financial terms"],
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
