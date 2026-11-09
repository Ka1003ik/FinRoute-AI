
"""FinRoute AI: metadata availability and conflict investigation."""

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative


INPUT_PATH = (
    ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "metadata_investigation_v2.json"
)

TEXT = "Consumer complaint narrative"
TARGET = "Product"

# Candidate fields that may be known when a complaint arrives.
INTAKE_FIELDS = [
    "Date received",
    "Submitted via",
    "Company",
    "State",
]

# Related taxonomy fields: investigation only, not predictors.
TAXONOMY_FIELDS = [
    "Sub-product",
    "Issue",
    "Sub-issue",
]

ALL_FIELDS = [
    TEXT,
    TARGET,
    *INTAKE_FIELDS,
    *TAXONOMY_FIELDS,
]


def clean_values(series):
    return series.astype(str).str.strip()


def analyze_field(group, field):
    """Check whether a field separates labels within one text group."""

    values = clean_values(group[field])

    by_label = []

    for label in group[TARGET].unique():
        label_values = set(
            values[group[TARGET] == label]
        )
        label_values.discard("")
        by_label.append(label_values)

    complete = not values.eq("").any()

    populated = all(
        len(item) > 0 for item in by_label
    )

    overlap = False

    for i in range(len(by_label)):
        for j in range(i + 1, len(by_label)):
            if by_label[i] & by_label[j]:
                overlap = True

    return {
        "complete": complete,
        "varies": values.nunique() > 1,
        "separates_labels": (
            complete and populated and not overlap
        ),
        "overlaps": overlap,
    }


def analyze_groups(groups, field):
    totals = {
        "groups_with_complete_values": 0,
        "groups_with_missing_values": 0,
        "groups_with_variation": 0,
        "groups_separating_product_labels": 0,
        "groups_with_overlapping_values": 0,
    }

    for _, group in groups:
        result = analyze_field(group, field)

        if result["complete"]:
            totals["groups_with_complete_values"] += 1
        else:
            totals["groups_with_missing_values"] += 1

        if result["varies"]:
            totals["groups_with_variation"] += 1

        if result["separates_labels"]:
            totals["groups_separating_product_labels"] += 1

        if result["overlaps"]:
            totals["groups_with_overlapping_values"] += 1

    return totals


def analyze_combination(groups, fields):
    """Evaluate combinations without exporting actual values."""

    separated = 0
    incomplete = 0
    overlapping = 0

    for _, group in groups:

        values = group[fields].apply(clean_values)

        by_label = []

        for label in group[TARGET].unique():
            label_rows = values.loc[
                group[TARGET] == label
            ]

            tuples = set(
                map(tuple, label_rows.values.tolist())
            )

            by_label.append(tuples)

        if any(
            any(value == "" for value in item)
            for label_set in by_label
            for item in label_set
        ):
            incomplete += 1
            continue

        has_overlap = any(
            by_label[i] & by_label[j]
            for i in range(len(by_label))
            for j in range(i + 1, len(by_label))
        )

        if has_overlap:
            overlapping += 1
        else:
            separated += 1

    return {
        "separates_product_labels": separated,
        "has_overlapping_combinations": overlapping,
        "incomplete_groups": incomplete,
    }


def availability_summary(frame, fields):
    results = {}

    for field in fields:
        values = clean_values(frame[field])

        results[field] = {
            "records_with_values": int(values.ne("").sum()),
            "records_missing_values": int(values.eq("").sum()),
            "distinct_nonempty_values": int(
                values[values.ne("")].nunique()
            ),
        }

    return results


def main():

    print("=" * 65)
    print("FINROUTE AI - METADATA INVESTIGATION V2")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Report already exists: {OUTPUT_PATH}"
        )

    data = pd.read_csv(
        INPUT_PATH,
        usecols=ALL_FIELDS,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    print(f"\nSource rows: {len(data):,}")

    data["_clean_text"] = data[TEXT].map(
        clean_narrative
    )

    empty_records = data[
        data["_clean_text"].eq("")
    ].copy()

    valid_records = data[
        data["_clean_text"].ne("")
    ].copy()

    labels_per_text = (
        valid_records.groupby("_clean_text")[TARGET]
        .nunique()
    )

    conflict_texts = labels_per_text[
        labels_per_text > 1
    ].index

    conflicts = valid_records[
        valid_records["_clean_text"].isin(conflict_texts)
    ].copy()

    groups = list(
        conflicts.groupby("_clean_text", sort=False)
    )

    # Confirm against the previously completed diagnostics.
    if len(groups) != 48 or len(conflicts) != 273:
        raise ValueError(
            "Conflicting group counts differ from the "
            "verified diagnostics. Stop and investigate."
        )

    if len(empty_records) != 6:
        raise ValueError(
            "Expected six empty narratives."
        )

    print(f"Conflicting groups: {len(groups):,}")
    print(f"Conflicting records: {len(conflicts):,}")
    print(f"Empty narratives: {len(empty_records):,}")

    # Examine individual intake-time metadata fields.
    intake_analysis = {
        field: analyze_groups(groups, field)
        for field in INTAKE_FIELDS
    }

    # Taxonomy fields are strictly diagnostic.
    taxonomy_analysis = {
        field: analyze_groups(groups, field)
        for field in TAXONOMY_FIELDS
    }

    # Test combinations of potentially available metadata.
    combination_analysis = {
        "Date received + Submitted via": (
            analyze_combination(
                groups,
                ["Date received", "Submitted via"]
            )
        ),
        "Date received + Submitted via + Company": (
            analyze_combination(
                groups,
                ["Date received", "Submitted via", "Company"]
            )
        ),
        "Date received + Submitted via + Company + State": (
            analyze_combination(
                groups,
                [
                    "Date received",
                    "Submitted via",
                    "Company",
                    "State"
                ]
            )
        ),
    }

    empty_record_analysis = availability_summary(
        empty_records,
        INTAKE_FIELDS + TAXONOMY_FIELDS
    )

    full_dataset_analysis = availability_summary(
        data,
        INTAKE_FIELDS + TAXONOMY_FIELDS
    )

    report = {
        "project": "FinRoute AI",
        "source_records": int(len(data)),
        "conflicting_narrative_groups": int(len(groups)),
        "conflicting_records": int(len(conflicts)),
        "empty_narrative_records": int(len(empty_records)),

        "intake_metadata_group_analysis": intake_analysis,

        "taxonomy_group_analysis_investigation_only": (
            taxonomy_analysis
        ),

        "combined_intake_metadata_analysis": (
            combination_analysis
        ),

        "empty_narrative_metadata_availability": (
            empty_record_analysis
        ),

        "overall_metadata_availability": (
            full_dataset_analysis
        ),

        "interpretation_warning": (
            "Metadata association does not establish the correct "
            "Product label. No automatic relabelling or model "
            "feature selection was performed."
        ),

        "raw_narratives_exported": False,
        "company_names_exported": False,
        "zip_codes_exported": False,
        "complaint_ids_exported": False,
        "dataset_modified": False,
        "model_retrained": False,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n=== INTAKE METADATA ANALYSIS ===")

    for field, result in intake_analysis.items():
        print(f"\n{field}:")
        for metric, count in result.items():
            print(f"  {metric}: {count}")

    print("\n=== TAXONOMY INVESTIGATION ===")

    for field, result in taxonomy_analysis.items():
        print(f"\n{field}:")
        for metric, count in result.items():
            print(f"  {metric}: {count}")

    print("\n=== COMBINED INTAKE METADATA ===")
    print(json.dumps(combination_analysis, indent=4))

    print("\n=== EMPTY NARRATIVE METADATA ===")

    for field, result in empty_record_analysis.items():
        print(
            f"{field}: "
            f"{result['records_with_values']}/6 populated"
        )

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo raw metadata values exported.")
    print("No dataset modifications performed.")
    print("METADATA INVESTIGATION COMPLETED.")


if __name__ == "__main__":
    main()
