
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_component_assignments_v2.csv"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "group_split_feasibility_v2.json"
)

LABEL_COLUMN = "Product"

EXPECTED_RECORDS = 121114
EXPECTED_UNIQUE_DOCUMENTS = 108902
EXPECTED_CLASSES = 11

TARGET_SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}

THRESHOLDS = {
    "0.99": "component_099",
    "1.00": "component_100",
}


def validate_input(frame):
    required_columns = {
        "row_position",
        "unique_document_id",
        LABEL_COLUMN,
        *THRESHOLDS.values(),
    }

    missing = required_columns - set(frame.columns)

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

    positions = frame["row_position"].to_numpy(dtype=np.int64)

    if not np.array_equal(
        np.sort(positions),
        np.arange(EXPECTED_RECORDS, dtype=np.int64),
    ):
        raise ValueError(
            "Original record positions are incomplete."
        )

    if frame["unique_document_id"].nunique() != EXPECTED_UNIQUE_DOCUMENTS:
        raise ValueError("Unexpected unique narrative group count.")

    if frame[LABEL_COLUMN].str.strip().eq("").any():
        raise ValueError("Missing Product labels found.")

    if frame[LABEL_COLUMN].nunique() != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} Product categories."
        )

    # One unique cleaned narrative must have one Product label
    # and one connected-component assignment at each threshold.
    check_columns = [
        LABEL_COLUMN,
        *THRESHOLDS.values(),
    ]

    consistency = frame.groupby(
        "unique_document_id",
        sort=False,
    )[check_columns].nunique()

    if (consistency > 1).any().any():
        raise ValueError(
            "Inconsistent labels or component assignments "
            "within a unique narrative group."
        )

    print("Input validation passed.")
    print(f"Original complaints: {len(frame):,}")
    print(
        "Unique narrative groups:",
        frame["unique_document_id"].nunique(),
    )
    print("Product categories:", frame[LABEL_COLUMN].nunique())


def size_distribution(values):
    values = np.asarray(values, dtype=np.int64)

    if len(values) == 0:
        return {}

    return {
        "minimum": int(values.min()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": int(values.max()),
    }


def build_threshold_analysis(frame, component_column):
    print(f"\nAnalysing {component_column}...")

    # One row per connected component.
    components = frame.groupby(
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
    )

    component_count = len(components)

    if int(components["original_complaint_records"].sum()) != EXPECTED_RECORDS:
        raise ValueError(
            "Component record counts do not reconcile."
        )

    if int(components["unique_narrative_groups"].sum()) != EXPECTED_UNIQUE_DOCUMENTS:
        raise ValueError(
            "Component narrative counts do not reconcile."
        )

    record_sizes = components[
        "original_complaint_records"
    ].to_numpy(dtype=np.int64)

    narrative_sizes = components[
        "unique_narrative_groups"
    ].to_numpy(dtype=np.int64)

    components_with_multiple_narratives = (
        narrative_sizes > 1
    )

    components_with_repeated_records = (
        record_sizes > 1
    )

    multi_product_components = components[
        components["number_of_product_labels"] > 1
    ]

    product_component = frame.groupby(
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

    # A component can contain several Product categories.
    # Each such component still belongs to only one eventual split.
    product_component = product_component.merge(
        components[
            ["number_of_product_labels"]
        ],
        left_on=component_column,
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    class_results = []

    for product, group in product_component.groupby(
        LABEL_COLUMN,
        sort=False,
    ):
        class_record_count = int(
            group["original_complaint_records"].sum()
        )

        class_narrative_count = int(
            group["unique_narrative_groups"].sum()
        )

        class_component_count = int(len(group))

        pure_components = int(
            (group["number_of_product_labels"] == 1).sum()
        )

        mixed_components = int(
            (group["number_of_product_labels"] > 1).sum()
        )

        class_results.append({
            "product": str(product),
            "original_complaint_records": class_record_count,
            "unique_narrative_groups": class_narrative_count,
            "components_containing_product": class_component_count,
            "single_product_components": pure_components,
            "multiple_product_components": mixed_components,
            "largest_component_records_for_product": int(
                group["original_complaint_records"].max()
            ),
            "at_least_three_components_available": (
                class_component_count >= 3
            ),
        })

    class_results.sort(
        key=lambda item: item["original_complaint_records"],
        reverse=True,
    )

    largest_components = components.nlargest(
        20,
        "original_complaint_records",
    )

    largest_component_summary = []

    for _, row in largest_components.iterrows():
        component_id = row.name

        largest_component_summary.append({
            "original_complaint_records": int(
                row["original_complaint_records"]
            ),
            "unique_narrative_groups": int(
                row["unique_narrative_groups"]
            ),
            "number_of_product_labels": int(
                row["number_of_product_labels"]
            ),
        })

    # Necessary coverage condition:
    # Each Product category must occur in at least three
    # different components to appear in all three partitions
    # without splitting any component.
    classes_with_fewer_than_three_components = [
        item["product"]
        for item in class_results
        if not item["at_least_three_components_available"]
    ]

    # Approximate split sizes for reference.
    target_counts = {
        "train": round(
            EXPECTED_RECORDS
            * TARGET_SPLIT_RATIOS["train"]
        ),
        "validation": round(
            EXPECTED_RECORDS
            * TARGET_SPLIT_RATIOS["validation"]
        ),
    }

    target_counts["test"] = (
        EXPECTED_RECORDS
        - target_counts["train"]
        - target_counts["validation"]
    )

    largest_component_records = int(record_sizes.max())

    allocation_diagnostics = {
        "target_record_counts": target_counts,
        "largest_component_records": largest_component_records,
        "largest_component_percentage_of_dataset": round(
            100 * largest_component_records / EXPECTED_RECORDS,
            4,
        ),
        "largest_component_as_percentage_of_validation_target": round(
            100 * largest_component_records
            / target_counts["validation"],
            2,
        ),
        "components_larger_than_validation_target": int(
            (record_sizes > target_counts["validation"]).sum()
        ),
        "components_larger_than_test_target": int(
            (record_sizes > target_counts["test"]).sum()
        ),
    }

    return {
        "component_count": int(component_count),

        "original_complaint_records": EXPECTED_RECORDS,
        "unique_narrative_groups": EXPECTED_UNIQUE_DOCUMENTS,

        "components_with_multiple_narratives": int(
            components_with_multiple_narratives.sum()
        ),

        "components_with_only_one_narrative_group": int(
            (~components_with_multiple_narratives).sum()
        ),

        "components_containing_multiple_original_records": int(
            components_with_repeated_records.sum()
        ),

        "components_with_multiple_product_labels": int(
            len(multi_product_components)
        ),

        "original_records_in_multiple_product_components": int(
            multi_product_components[
                "original_complaint_records"
            ].sum()
        ),

        "unique_narrative_groups_in_multiple_product_components": int(
            multi_product_components[
                "unique_narrative_groups"
            ].sum()
        ),

        "component_record_size_distribution": (
            size_distribution(record_sizes)
        ),

        "component_unique_narrative_size_distribution": (
            size_distribution(narrative_sizes)
        ),

        "product_class_component_coverage": class_results,

        "classes_with_fewer_than_three_components": (
            classes_with_fewer_than_three_components
        ),

        "largest_20_components_by_original_record_count": (
            largest_component_summary
        ),

        "allocation_diagnostics": allocation_diagnostics,
    }


def main():
    print("=" * 68)
    print("FINROUTE AI - GROUP SPLIT FEASIBILITY ANALYSIS")
    print("=" * 68)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

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

    validate_input(frame)

    threshold_results = {}

    for threshold, component_column in THRESHOLDS.items():
        threshold_results[threshold] = (
            build_threshold_analysis(
                frame,
                component_column,
            )
        )

        print("\nThreshold summary:")
        print(json.dumps(
            {
                key: value
                for key, value in threshold_results[
                    threshold
                ].items()
                if key not in {
                    "product_class_component_coverage",
                    "largest_20_components_by_original_record_count",
                }
            },
            indent=4,
        ))

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",

        "source_records": EXPECTED_RECORDS,
        "unique_narrative_groups": EXPECTED_UNIQUE_DOCUMENTS,
        "product_categories": EXPECTED_CLASSES,

        "target_split_ratios": TARGET_SPLIT_RATIOS,

        "threshold_results": threshold_results,

        "interpretation": {
            "at_least_three_components_per_class": (
                "Necessary but not sufficient for representing "
                "every Product category in all three partitions."
            ),
            "component_assignment": (
                "Every connected component must remain intact; "
                "all original complaint records belonging to "
                "its narrative groups move together."
            ),
            "allocation_status": (
                "No partition allocation has been performed."
            ),
        },

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "row_positions_exported": False,

        "dataset_modified": False,
        "records_deleted_or_moved": False,
        "new_splits_created": False,
        "model_retrained": False,
        "test_set_evaluated": False,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("GROUP SPLIT FEASIBILITY ANALYSIS COMPLETED")
    print("=" * 68)

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo partitions allocated.")
    print("No source dataset modified.")
    print("All original complaints retained.")


if __name__ == "__main__":
    main()
