
"""Prepare FinRoute AI modeling dataset v2 using canonical text grouping."""

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative
from src.preprocessing.text_vectorizer import feature_signature


INPUT_CSV = (
    ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
)

OUTPUT_CSV = (
    ROOT / "data" / "processed" / "cfpb_2026_modeling_v2.csv"
)

REPORT_PATH = (
    ROOT / "outputs" / "reports" / "modeling_dataset_prep_v2.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


# Expected counts from our completed diagnostics.
EXPECTED_INPUT_ROWS = 121393
EXPECTED_CONFLICT_GROUPS = 48
EXPECTED_CONFLICT_ROWS = 273
EXPECTED_EMPTY_ROWS = 6
EXPECTED_NONEMPTY_FEATURELESS_ROWS = 3


def main():

    print("=" * 65)
    print("FINROUTE AI - MODELING DATASET V2 PREPARATION")
    print("=" * 65)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    # Prevent accidental overwriting of the original dataset
    # or previous v2 outputs.
    for output_path in [OUTPUT_CSV, REPORT_PATH]:
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Move or rename the existing file before rerunning."
            )

    data = pd.read_csv(
        INPUT_CSV,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    print(f"\nOriginal records: {len(data):,}")

    if len(data) != EXPECTED_INPUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_ROWS:,} input rows, "
            f"found {len(data):,}. Recheck the source dataset."
        )

    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Missing text column: {TEXT_COLUMN}")

    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"Missing label column: {LABEL_COLUMN}")

    # Match the existing dataset-preparation convention.
    data[TEXT_COLUMN] = data[TEXT_COLUMN].str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].str.strip()

    if data[LABEL_COLUMN].eq("").any():
        raise ValueError("Source contains empty product labels.")

    print("Applying canonical narrative cleaning...")

    # This is the canonical grouping key.
    data["_clean_text"] = data[TEXT_COLUMN].map(
        clean_narrative
    )

    empty_mask = data["_clean_text"].eq("")
    empty_rows = int(empty_mask.sum())

    # Identify baseline-featureless records separately.
    # Non-empty records remain eligible for DistilBERT.
    print("Checking baseline feature availability...")

    signatures = data[TEXT_COLUMN].map(feature_signature)
    featureless_mask = signatures.eq("")

    nonempty_featureless_mask = (
        featureless_mask & ~empty_mask
    )

    nonempty_featureless_rows = int(
        nonempty_featureless_mask.sum()
    )

    # Determine product-label conflicts using actual
    # canonical cleaned narratives, not sorted TF-IDF features.
    print("Identifying conflicting cleaned narratives...")

    valid_rows = data.loc[~empty_mask]

    products_per_text = (
        valid_rows.groupby("_clean_text", sort=False)[LABEL_COLUMN]
        .nunique()
    )

    conflicting_texts = products_per_text[
        products_per_text > 1
    ].index

    conflicting_mask = data["_clean_text"].isin(
        conflicting_texts
    )

    conflicting_rows = int(conflicting_mask.sum())
    conflicting_groups = int(len(conflicting_texts))

    # Validate findings against the completed diagnostics.
    if conflicting_groups != EXPECTED_CONFLICT_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_CONFLICT_GROUPS} conflicting groups, "
            f"found {conflicting_groups}."
        )

    if conflicting_rows != EXPECTED_CONFLICT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CONFLICT_ROWS} conflicting rows, "
            f"found {conflicting_rows}."
        )

    if empty_rows != EXPECTED_EMPTY_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_EMPTY_ROWS} empty records, "
            f"found {empty_rows}."
        )

    if nonempty_featureless_rows != EXPECTED_NONEMPTY_FEATURELESS_ROWS:
        raise ValueError(
            "Unexpected number of non-empty featureless records: "
            f"{nonempty_featureless_rows}."
        )

    # Exclude conflicting narratives and genuinely empty narratives.
    # Retain same-label duplicates and non-empty featureless records.
    excluded_mask = conflicting_mask | empty_mask

    modeling_data = data.loc[~excluded_mask].copy()

    # Verify that no cleaned narrative has conflicting labels.
    remaining_conflicts = (
        modeling_data.groupby("_clean_text")[LABEL_COLUMN]
        .nunique()
    )

    if (remaining_conflicts > 1).any():
        raise ValueError(
            "Conflicting labels remain in the corrected dataset."
        )

    # Count repeated same-label cleaned narratives.
    repeated_groups = (
        modeling_data.groupby("_clean_text", sort=False)
        .size()
    )

    repeated_groups = repeated_groups[
        repeated_groups > 1
    ]

    same_label_extra_rows = int(
        (repeated_groups - 1).sum()
    )

    # Remove only the temporary diagnostic column.
    modeling_data = modeling_data.drop(
        columns=["_clean_text"]
    )

    actual_output_rows = len(modeling_data)

    expected_output_rows = (
        EXPECTED_INPUT_ROWS
        - EXPECTED_CONFLICT_ROWS
        - EXPECTED_EMPTY_ROWS
    )

    if actual_output_rows != expected_output_rows:
        raise ValueError(
            f"Expected {expected_output_rows:,} output rows, "
            f"found {actual_output_rows:,}."
        )

    product_counts = (
        modeling_data[LABEL_COLUMN]
        .value_counts()
        .sort_values(ascending=False)
    )

    # Save the new versioned dataset.
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    modeling_data.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "input_rows": EXPECTED_INPUT_ROWS,
        "conflicting_cleaned_narrative_groups_excluded": (
            conflicting_groups
        ),
        "rows_in_conflicting_cleaned_narrative_groups_excluded": (
            conflicting_rows
        ),
        "empty_cleaned_narratives_excluded": empty_rows,
        "nonempty_featureless_records_retained": (
            nonempty_featureless_rows
        ),
        "same_label_repeated_cleaned_text_groups_retained": (
            int(len(repeated_groups))
        ),
        "extra_rows_in_same_label_repeated_groups_retained": (
            same_label_extra_rows
        ),
        "output_rows": actual_output_rows,
        "remaining_cleaned_text_label_conflicts": 0,
        "product_counts": {
            str(label): int(count)
            for label, count in product_counts.items()
        },
        "grouping_method": (
            "Canonical clean_narrative output"
        ),
        "source_dataset_modified": False,
        "previous_modeling_dataset_modified": False,
        "model_retrained": False,
        "output_file": str(OUTPUT_CSV)
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("DATASET V2 PREPARATION RESULTS")
    print("=" * 65)

    print(f"Original records: {EXPECTED_INPUT_ROWS:,}")
    print(f"Conflicting records excluded: {conflicting_rows:,}")
    print(f"Empty records excluded: {empty_rows:,}")
    print(
        "Non-empty featureless records retained: "
        f"{nonempty_featureless_rows:,}"
    )
    print(f"Final modeling records: {actual_output_rows:,}")
    print(
        "Remaining label conflicts: 0"
    )

    print("\nProduct distribution:")

    for label, count in product_counts.items():
        print(f"{label}: {count:,}")

    print("\nDataset saved:")
    print(OUTPUT_CSV)

    print("\nReport saved:")
    print(REPORT_PATH)

    print("\nOriginal dataset remains unchanged.")
    print("DATASET V2 PREPARATION COMPLETED.")


if __name__ == "__main__":
    main()

