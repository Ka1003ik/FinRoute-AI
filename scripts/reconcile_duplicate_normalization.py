
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative
from src.preprocessing.text_vectorizer import feature_signature


INPUT_PATH = (
    ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "duplicate_normalization_reconciliation.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


def summarize_groups(frame, key_column):

    grouped = frame.groupby(key_column, sort=False)[LABEL_COLUMN].agg(
        rows="size",
        product_count="nunique"
    )

    repeated = grouped[grouped["rows"] > 1]

    conflicting = repeated[
        repeated["product_count"] > 1
    ]

    consistent = repeated[
        repeated["product_count"] == 1
    ]

    return {
        "repeated_groups": int(len(repeated)),
        "same_label_repeated_groups": int(len(consistent)),
        "conflicting_groups": int(len(conflicting)),
        "rows_in_conflicting_groups": int(
            conflicting["rows"].sum()
        ),
        "extra_rows_in_conflicting_groups": int(
            (conflicting["rows"] - 1).sum()
        )
    }


def main():

    print("=" * 65)
    print("FINROUTE AI - DUPLICATE NORMALIZATION RECONCILIATION")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    data = pd.read_csv(
        INPUT_PATH,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    data[TEXT_COLUMN] = data[TEXT_COLUMN].str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].str.strip()

    print(f"\nOriginal rows: {len(data):,}")

    # Method 1: Original raw-text grouping.
    raw_result = summarize_groups(
        data,
        TEXT_COLUMN
    )

    # Method 2: Canonical cleaned-text grouping.
    data["_clean_text"] = data[TEXT_COLUMN].map(
        clean_narrative
    )

    nonempty = data[data["_clean_text"] != ""].copy()

    cleaned_result = summarize_groups(
        nonempty,
        "_clean_text"
    )

    # Method 3: Sorted TF-IDF feature-signature grouping.
    nonempty["_signature"] = nonempty[TEXT_COLUMN].map(
        feature_signature
    )

    signature_result = summarize_groups(
        nonempty,
        "_signature"
    )

    # Reproduce the original feature-signature conflicts.
    grouped_signatures = (
        nonempty.groupby("_signature", sort=False)[LABEL_COLUMN]
        .agg(rows="size", product_count="nunique")
    )

    conflicting_signatures = grouped_signatures[
        grouped_signatures["product_count"] > 1
    ].index

    affected = nonempty[
        nonempty["_signature"].isin(conflicting_signatures)
    ].copy()

    # Investigate the narratives inside conflicting signatures.
    cleaned_inside = (
        affected.groupby("_clean_text", sort=False)[LABEL_COLUMN]
        .agg(rows="size", product_count="nunique")
    )

    conflicting_cleaned = cleaned_inside[
        cleaned_inside["product_count"] > 1
    ]

    consistent_cleaned = cleaned_inside[
        cleaned_inside["product_count"] == 1
    ]

    report = {
        "project": "FinRoute AI",
        "source_rows": int(len(data)),

        "raw_text_grouping": raw_result,
        "canonical_cleaned_text_grouping": cleaned_result,
        "tfidf_feature_signature_grouping": signature_result,

        "conflicting_signature_groups": int(
            len(conflicting_signatures)
        ),

        "rows_in_conflicting_signature_groups": int(
            len(affected)
        ),

        "inside_conflicting_signatures": {
            "distinct_cleaned_narratives": int(
                len(cleaned_inside)
            ),
            "conflicting_cleaned_narrative_groups": int(
                len(conflicting_cleaned)
            ),
            "rows_in_conflicting_cleaned_groups": int(
                conflicting_cleaned["rows"].sum()
            ),
            "consistent_cleaned_narrative_groups": int(
                len(consistent_cleaned)
            ),
            "rows_in_consistent_cleaned_groups": int(
                consistent_cleaned["rows"].sum()
            )
        },

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "dataset_modified": False,
        "model_retrained": False
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Report already exists: {OUTPUT_PATH}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n=== COMPARISON ===")

    for method in [
        "raw_text_grouping",
        "canonical_cleaned_text_grouping",
        "tfidf_feature_signature_grouping"
    ]:
        print(f"\n{method}:")
        for key, value in report[method].items():
            print(f"  {key}: {value:,}")

    print("\n=== CONFLICTING SIGNATURE INVESTIGATION ===")

    for key, value in report[
        "inside_conflicting_signatures"
    ].items():
        print(f"{key}: {value:,}")

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo source data was modified.")
    print("RECONCILIATION COMPLETED.")


if __name__ == "__main__":
    main()
