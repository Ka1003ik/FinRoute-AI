
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_component_assignments_v2.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_component_composition_v2.json"
)

LABEL_COLUMN = "Product"

EXPECTED_RECORDS = 121114
EXPECTED_UNIQUE_DOCUMENTS = 108902

COMPONENT_COLUMNS = {
    "0.70": "component_070",
    "0.85": "component_085",
    "0.95": "component_095",
    "0.99": "component_099",
    "1.00": "component_100",
}


def validate_data(frame):
    required = {
        "row_position",
        "unique_document_id",
        LABEL_COLUMN,
        *COMPONENT_COLUMNS.values(),
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(frame) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS:,} records, "
            f"found {len(frame):,}."
        )

    if frame["row_position"].duplicated().any():
        raise ValueError("Duplicate original row positions found.")

    positions = frame["row_position"].to_numpy()

    if set(positions) != set(range(EXPECTED_RECORDS)):
        raise ValueError(
            "Original row positions are incomplete or invalid."
        )

    unique_count = frame["unique_document_id"].nunique()

    if unique_count != EXPECTED_UNIQUE_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_DOCUMENTS:,} unique "
            f"narrative groups, found {unique_count:,}."
        )

    if frame[LABEL_COLUMN].str.strip().eq("").any():
        raise ValueError("Missing Product labels found.")

    # Identical cleaned narratives must retain one consistent
    # Product label and one component assignment per threshold.
    columns = [
        LABEL_COLUMN,
        *COMPONENT_COLUMNS.values(),
    ]

    consistency = frame.groupby(
        "unique_document_id",
        sort=False,
    )[columns].nunique()

    if (consistency > 1).any().any():
        raise ValueError(
            "A unique narrative group has inconsistent labels "
            "or component assignments."
        )

    print("Input validation passed.")
    print(f"Original complaint records: {len(frame):,}")
    print(f"Unique narrative groups: {unique_count:,}")
    print(
        f"Product categories: {frame[LABEL_COLUMN].nunique()}"
    )


def build_component_summary(frame, component_column):
    """Calculate component sizes and Product composition."""

    # One row here represents one original complaint.
    component_stats = frame.groupby(
        component_column,
        sort=False,
    ).agg(
        original_complaint_records=(
            LABEL_COLUMN,
            "size",
        ),
        unique_narrative_groups=(
            "unique_document_id",
            "nunique",
        ),
        number_of_product_labels=(
            LABEL_COLUMN,
            "nunique",
        ),
    ).reset_index()

    # Aggregate category counts within every component.
    product_breakdown = frame.groupby(
        [component_column, LABEL_COLUMN],
        sort=False,
    ).agg(
        original_complaint_records=(
            LABEL_COLUMN,
            "size",
        ),
        unique_narrative_groups=(
            "unique_document_id",
            "nunique",
        ),
    ).reset_index()

    product_lookup = {}

    for component_id, group in product_breakdown.groupby(
        component_column,
        sort=False,
    ):
        product_lookup[component_id] = [
            {
                "product": str(row[LABEL_COLUMN]),
                "original_complaint_records": int(
                    row["original_complaint_records"]
                ),
                "unique_narrative_groups": int(
                    row["unique_narrative_groups"]
                ),
            }
            for _, row in group.iterrows()
        ]

    # Summary for components with multiple Product labels.
    multi = component_stats[
        component_stats["number_of_product_labels"] > 1
    ].copy()

    composition_groups = {}

    for _, row in multi.iterrows():
        component_id = row[component_column]

        breakdown = product_lookup[component_id]

        products = tuple(sorted(
            item["product"] for item in breakdown
        ))

        if products not in composition_groups:
            composition_groups[products] = {
                "component_count": 0,
                "original_complaint_records": 0,
                "unique_narrative_groups": 0,
                "product_record_counts": {
                    product: 0 for product in products
                },
            }

        summary = composition_groups[products]

        summary["component_count"] += 1

        summary["original_complaint_records"] += int(
            row["original_complaint_records"]
        )

        summary["unique_narrative_groups"] += int(
            row["unique_narrative_groups"]
        )

        for item in breakdown:
            summary["product_record_counts"][
                item["product"]
            ] += item["original_complaint_records"]

    composition_results = []

    for products, summary in composition_groups.items():
        composition_results.append({
            "product_categories": list(products),
            **summary,
        })

    composition_results.sort(
        key=lambda item: (
            item["component_count"],
            item["original_complaint_records"],
        ),
        reverse=True,
    )

    # Largest components by original complaint record count.
    largest = component_stats.nlargest(
        20,
        "original_complaint_records",
    )

    largest_results = []

    for _, row in largest.iterrows():
        component_id = row[component_column]

        largest_results.append({
            "original_complaint_records": int(
                row["original_complaint_records"]
            ),
            "unique_narrative_groups": int(
                row["unique_narrative_groups"]
            ),
            "number_of_product_labels": int(
                row["number_of_product_labels"]
            ),
            "product_composition": (
                product_lookup[component_id]
            ),
        })

    # Overall Product representation inside multi-label components.
    multi_label_records = frame[
        frame[component_column].isin(
            multi[component_column]
        )
    ]

    category_distribution = (
        multi_label_records.groupby(LABEL_COLUMN)
        .agg(
            original_complaint_records=(
                LABEL_COLUMN,
                "size",
            ),
            unique_narrative_groups=(
                "unique_document_id",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "original_complaint_records",
            ascending=False,
        )
    )

    category_results = [
        {
            "product": str(row[LABEL_COLUMN]),
            "original_complaint_records": int(
                row["original_complaint_records"]
            ),
            "unique_narrative_groups": int(
                row["unique_narrative_groups"]
            ),
        }
        for _, row in category_distribution.iterrows()
    ]

    non_singleton = component_stats[
        component_stats["unique_narrative_groups"] > 1
    ]

    return {
        "total_components": int(len(component_stats)),
        "singleton_components": int(
            (
                component_stats["unique_narrative_groups"] == 1
            ).sum()
        ),
        "non_singleton_components": int(len(non_singleton)),
        "original_records_in_non_singleton_components": int(
            non_singleton["original_complaint_records"].sum()
        ),
        "unique_narrative_groups_in_non_singleton_components": int(
            non_singleton["unique_narrative_groups"].sum()
        ),
        "components_with_multiple_product_labels": int(len(multi)),
        "original_records_in_multiple_label_components": int(
            multi["original_complaint_records"].sum()
        ),
        "unique_narrative_groups_in_multiple_label_components": int(
            multi["unique_narrative_groups"].sum()
        ),
        "largest_component_original_record_count": int(
            component_stats[
                "original_complaint_records"
            ].max()
        ),
        "multi_product_composition_summary": composition_results,
        "product_distribution_in_multi_label_components": (
            category_results
        ),
        "largest_20_components_by_record_count": largest_results,
    }


def main():
    print("=" * 68)
    print("FINROUTE AI - COMPONENT COMPOSITION AUDIT")
    print("=" * 68)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH,
        dtype={
            "row_position": "int64",
            "unique_document_id": "int64",
            LABEL_COLUMN: str,
        },
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    validate_data(frame)

    threshold_results = {}

    for threshold, column in COMPONENT_COLUMNS.items():
        print(
            f"\nAnalysing component composition "
            f"at threshold {threshold}..."
        )

        summary = build_component_summary(
            frame,
            column,
        )

        threshold_results[threshold] = summary

        print(
            "Total components:",
            summary["total_components"],
        )

        print(
            "Multi-Product components:",
            summary["components_with_multiple_product_labels"],
        )

        print(
            "Original records in multi-Product components:",
            summary["original_records_in_multiple_label_components"],
        )

        print(
            "Largest component records:",
            summary["largest_component_original_record_count"],
        )

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "source_records": EXPECTED_RECORDS,
        "unique_narrative_groups": EXPECTED_UNIQUE_DOCUMENTS,
        "component_composition_by_threshold": threshold_results,
        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "row_positions_exported": False,
        "dataset_modified": False,
        "records_deleted_or_moved": False,
        "model_retrained": False,
        "test_set_evaluated": False,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("COMPONENT COMPOSITION AUDIT COMPLETED")
    print("=" * 68)

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nAll original complaints retained.")
    print("No source dataset modified.")


if __name__ == "__main__":
    main()
