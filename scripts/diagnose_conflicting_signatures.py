
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_vectorizer import feature_signature
from src.preprocessing.clean_text import clean_narrative


INPUT_CSV = (
    ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "conflicting_signature_diagnostic.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


def main():

    print("=" * 65)
    print("FINROUTE AI - CONFLICTING SIGNATURE DIAGNOSTIC")
    print("=" * 65)

    data = pd.read_csv(
        INPUT_CSV,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    print(f"\nOriginal records: {len(data):,}")

    # Reproduce the same cleaning and signature generation.
    data["_clean_text"] = data[TEXT_COLUMN].map(clean_narrative)

    data["_signature"] = data[TEXT_COLUMN].map(
        feature_signature
    )

    # Find signatures associated with multiple products.
    valid = data[data["_signature"] != ""]

    product_counts = (
        valid.groupby("_signature")[LABEL_COLUMN]
        .nunique()
    )

    conflicting_signatures = product_counts[
        product_counts > 1
    ].index

    conflicting = valid[
        valid["_signature"].isin(conflicting_signatures)
    ].copy()

    # Compare actual cleaned narratives inside each group.
    group_summary = (
        conflicting.groupby("_signature")
        .agg(
            rows=("_signature", "size"),
            product_count=(LABEL_COLUMN, "nunique"),
            cleaned_narrative_count=("_clean_text", "nunique")
        )
    )

    same_cleaned_text = group_summary[
        group_summary["cleaned_narrative_count"] == 1
    ]

    multiple_cleaned_texts = group_summary[
        group_summary["cleaned_narrative_count"] > 1
    ]

    # Investigate featureless records separately.
    featureless = data[data["_signature"] == ""]

    featureless_empty_text = int(
        (featureless["_clean_text"] == "").sum()
    )

    featureless_nonempty_text = int(
        (featureless["_clean_text"] != "").sum()
    )

    report = {
        "project": "FinRoute AI",
        "diagnostic_type": "Conflicting signature investigation",
        "original_records": int(len(data)),

        "conflicting_signature_groups": int(
            len(group_summary)
        ),

        "rows_in_conflicting_signature_groups": int(
            len(conflicting)
        ),

        "groups_with_one_cleaned_narrative": int(
            len(same_cleaned_text)
        ),

        "rows_in_groups_with_one_cleaned_narrative": int(
            same_cleaned_text["rows"].sum()
        ),

        "groups_with_multiple_cleaned_narratives": int(
            len(multiple_cleaned_texts)
        ),

        "rows_in_groups_with_multiple_cleaned_narratives": int(
            multiple_cleaned_texts["rows"].sum()
        ),

        "distinct_cleaned_narratives_across_conflicting_groups": int(
            group_summary["cleaned_narrative_count"].sum()
        ),

        "featureless_records": int(len(featureless)),

        "featureless_records_empty_after_cleaning": (
            featureless_empty_text
        ),

        "featureless_records_nonempty_after_cleaning": (
            featureless_nonempty_text
        ),

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "dataset_modified": False
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

    print("\n=== DIAGNOSTIC RESULTS ===")

    for key, value in report.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:,}")

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo dataset modifications performed.")
    print("DIAGNOSTIC COMPLETED.")


if __name__ == "__main__":
    main()
