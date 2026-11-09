
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_vectorizer import feature_signature
from src.preprocessing.clean_text import clean_narrative


INPUT_PATH = (
    ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "exact_text_label_conflict_diagnostic.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


def main():

    print("=" * 65)
    print("FINROUTE AI - EXACT TEXT LABEL CONFLICT DIAGNOSTIC")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    data = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    print(f"\nOriginal records: {len(data):,}")

    data["_clean_text"] = data[TEXT_COLUMN].map(clean_narrative)
    data["_signature"] = data[TEXT_COLUMN].map(feature_signature)

    valid = data[data["_signature"] != ""].copy()

    products_per_signature = (
        valid.groupby("_signature")[LABEL_COLUMN].nunique()
    )

    conflicting_signatures = products_per_signature[
        products_per_signature > 1
    ].index

    # Examine only the groups excluded by the original
    # conflicting-signature filtering strategy.
    affected = valid[
        valid["_signature"].isin(conflicting_signatures)
    ].copy()

    # Now group by actual cleaned narrative rather than
    # the sorted TF-IDF feature signature.
    exact_text_groups = (
        affected.groupby("_clean_text", sort=False)
        .agg(
            rows=(LABEL_COLUMN, "size"),
            product_count=(LABEL_COLUMN, "nunique"),
            signature_count=("_signature", "nunique")
        )
    )

    genuine_conflicts = exact_text_groups[
        exact_text_groups["product_count"] > 1
    ]

    consistent_groups = exact_text_groups[
        exact_text_groups["product_count"] == 1
    ]

    report = {
        "project": "FinRoute AI",

        "affected_signature_groups": int(
            len(conflicting_signatures)
        ),

        "affected_rows": int(len(affected)),

        "distinct_cleaned_narratives": int(
            len(exact_text_groups)
        ),

        "exact_text_groups_with_conflicting_labels": int(
            len(genuine_conflicts)
        ),

        "rows_in_exact_text_groups_with_conflicting_labels": int(
            genuine_conflicts["rows"].sum()
        ),

        "exact_text_groups_with_consistent_labels": int(
            len(consistent_groups)
        ),

        "rows_in_exact_text_groups_with_consistent_labels": int(
            consistent_groups["rows"].sum()
        ),

        "consistent_groups_sharing_a_signature": int(
            (
                consistent_groups["signature_count"] > 1
            ).sum()
        ),

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

    print("\n=== RESULTS ===")

    for key, value in report.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:,}")

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo dataset modifications performed.")
    print("DIAGNOSTIC COMPLETED.")


if __name__ == "__main__":
    main()
