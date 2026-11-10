
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "leakage_audit"
    / "full_dataset_component_assignments_v2.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "leakage_audit"

ASSIGNMENT_OUTPUT = (
    OUTPUT_DIR / "group_split_dry_run_assignments_v2.csv"
)

REPORT_OUTPUT = (
    OUTPUT_DIR / "group_split_dry_run_report_v2.json"
)

LABEL_COLUMN = "Product"
GROUP_COLUMN = "component_100"
DOCUMENT_COLUMN = "unique_document_id"

EXPECTED_RECORDS = 121114
EXPECTED_DOCUMENT_GROUPS = 108902
EXPECTED_COMPONENTS = 108369
EXPECTED_CLASSES = 11

RANDOM_STATE = 42
N_SPLITS = 10

# The ten folds will be mapped as follows:
# Fold 0: Validation
# Fold 1: Test
# Folds 2-9: Training

FOLD_TO_PARTITION = {
    0: "validation",
    1: "test",
}

PARTITION_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}

TARGET_RECORD_COUNTS = {
    "train": 96891,
    "validation": 12111,
    "test": 12112,
}


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(frame):

    required_columns = {
        LABEL_COLUMN,
        GROUP_COLUMN,
        DOCUMENT_COLUMN,
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if len(frame) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, "
            f"but found {len(frame)}."
        )

    if frame[
        [LABEL_COLUMN, GROUP_COLUMN, DOCUMENT_COLUMN]
    ].isnull().any().any():
        raise ValueError(
            "Null values detected in required columns."
        )

    if frame[LABEL_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Empty Product labels detected.")

    document_count = frame[DOCUMENT_COLUMN].nunique()
    component_count = frame[GROUP_COLUMN].nunique()
    class_count = frame[LABEL_COLUMN].nunique()

    if document_count != EXPECTED_DOCUMENT_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENT_GROUPS} unique "
            f"narrative groups, found {document_count}."
        )

    if component_count != EXPECTED_COMPONENTS:
        raise ValueError(
            f"Expected {EXPECTED_COMPONENTS} components, "
            f"found {component_count}."
        )

    if class_count != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} Product categories, "
            f"found {class_count}."
        )

    # Each unique narrative must have one Product label
    # and one component assignment.
    document_check = (
        frame.groupby(DOCUMENT_COLUMN)
        .agg(
            product_count=(LABEL_COLUMN, "nunique"),
            component_count=(GROUP_COLUMN, "nunique"),
        )
    )

    if (document_check["product_count"] != 1).any():
        raise ValueError(
            "A unique narrative group has conflicting Product labels."
        )

    if (document_check["component_count"] != 1).any():
        raise ValueError(
            "A unique narrative group maps to multiple components."
        )

    # Each category must occur in at least ten components
    # for a ten-fold stratified grouped split.
    class_component_counts = (
        frame[[LABEL_COLUMN, GROUP_COLUMN]]
        .drop_duplicates()
        .groupby(LABEL_COLUMN)[GROUP_COLUMN]
        .nunique()
    )

    insufficient = class_component_counts[
        class_component_counts < N_SPLITS
    ]

    if not insufficient.empty:
        raise ValueError(
            "Some Product categories have fewer than "
            f"{N_SPLITS} components:\n{insufficient}"
        )

    print("Input validation passed.")
    print(f"Original complaints: {len(frame)}")
    print(f"Unique narrative groups: {document_count}")
    print(f"Components: {component_count}")
    print(f"Product categories: {class_count}")


# ============================================================
# GROUP-AWARE DRY-RUN ALLOCATION
# ============================================================

def allocate_components(frame):

    labels = frame[LABEL_COLUMN].to_numpy()
    groups = frame[GROUP_COLUMN].to_numpy()

    # X is only used to provide the sample count.
    # Stratification is based on Product and component groups.
    X = np.zeros((len(frame), 1), dtype=np.uint8)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    row_folds = np.full(len(frame), -1, dtype=np.int8)

    print("\nGenerating ten stratified group folds...")

    for fold_number, (_, holdout_indices) in enumerate(
        splitter.split(X, labels, groups)
    ):

        if (row_folds[holdout_indices] != -1).any():
            raise RuntimeError(
                "Some records were assigned to multiple folds."
            )

        row_folds[holdout_indices] = fold_number

        print(
            f"Fold {fold_number}: "
            f"{len(holdout_indices):,} records"
        )

    if (row_folds == -1).any():
        raise RuntimeError(
            "Some records were not assigned to a fold."
        )

    # Map folds to final candidate partitions.
    row_partitions = np.array(
        [
            FOLD_TO_PARTITION.get(fold, "train")
            for fold in row_folds
        ],
        dtype=object,
    )

    return row_folds, row_partitions


# ============================================================
# COMPONENT INTEGRITY VALIDATION
# ============================================================

def validate_assignments(frame, row_folds, row_partitions):

    assignment_check = pd.DataFrame({
        GROUP_COLUMN: frame[GROUP_COLUMN].to_numpy(),
        "fold": row_folds,
        "partition": row_partitions,
    })

    component_check = (
        assignment_check.groupby(GROUP_COLUMN)
        .agg(
            fold_count=("fold", "nunique"),
            partition_count=("partition", "nunique"),
        )
    )

    if (component_check["fold_count"] != 1).any():
        raise RuntimeError(
            "A component was split across multiple folds."
        )

    if (component_check["partition_count"] != 1).any():
        raise RuntimeError(
            "A component was split across multiple partitions."
        )

    # Also verify that no unique narrative crosses partitions.
    document_check = pd.DataFrame({
        DOCUMENT_COLUMN: frame[DOCUMENT_COLUMN].to_numpy(),
        "partition": row_partitions,
    })

    document_partition_counts = (
        document_check.groupby(DOCUMENT_COLUMN)["partition"].nunique()
    )

    if (document_partition_counts > 1).any():
        raise RuntimeError(
            "A unique narrative appears in multiple partitions."
        )

    print("\nComponent integrity check passed.")
    print("No component crosses partitions.")
    print("No unique narrative crosses partitions.")


# ============================================================
# DIAGNOSTIC REPORT
# ============================================================

def build_report(frame, row_folds, row_partitions):

    products = sorted(frame[LABEL_COLUMN].unique())

    total_records = len(frame)

    total_class_counts = (
        frame[LABEL_COLUMN]
        .value_counts()
        .reindex(products, fill_value=0)
    )

    component_product_counts = (
        frame.groupby(GROUP_COLUMN)[LABEL_COLUMN].nunique()
    )

    mixed_component_ids = set(
        component_product_counts[
            component_product_counts > 1
        ].index
    )

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "method": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "component_column": GROUP_COLUMN,
        "source_records": total_records,
        "unique_narrative_groups": int(
            frame[DOCUMENT_COLUMN].nunique()
        ),
        "total_components": int(
            frame[GROUP_COLUMN].nunique()
        ),
        "product_categories": len(products),
        "fold_mapping": {
            "validation": [0],
            "test": [1],
            "train": list(range(2, N_SPLITS)),
        },
        "allocation": {},
        "checks": {},
    }

    for partition in ["train", "validation", "test"]:

        mask = row_partitions == partition
        subset = frame.loc[mask]

        actual_records = len(subset)
        target_records = TARGET_RECORD_COUNTS[partition]

        actual_class_counts = (
            subset[LABEL_COLUMN]
            .value_counts()
            .reindex(products, fill_value=0)
        )

        class_results = []

        for product in products:

            total_for_class = int(total_class_counts[product])
            actual_for_class = int(actual_class_counts[product])

            target_for_class = (
                total_for_class * PARTITION_RATIOS[partition]
            )

            actual_share = (
                actual_for_class / total_for_class * 100
                if total_for_class
                else 0.0
            )

            class_results.append({
                "product": product,
                "total_records": total_for_class,
                "target_records": round(target_for_class, 2),
                "actual_records": actual_for_class,
                "difference_from_target": round(
                    actual_for_class - target_for_class, 2
                ),
                "target_share_of_class_pct": round(
                    PARTITION_RATIOS[partition] * 100, 4
                ),
                "actual_share_of_class_pct": round(
                    actual_share, 4
                ),
                "share_difference_percentage_points": round(
                    actual_share
                    - PARTITION_RATIOS[partition] * 100,
                    4,
                ),
                "present": actual_for_class > 0,
            })

        split_component_ids = set(
            subset[GROUP_COLUMN].unique()
        )

        mixed_components_in_split = (
            split_component_ids & mixed_component_ids
        )

        report["allocation"][partition] = {
            "target_records": target_records,
            "actual_records": actual_records,
            "record_difference": actual_records - target_records,
            "actual_dataset_percentage": round(
                actual_records / total_records * 100, 4
            ),
            "target_dataset_percentage": round(
                PARTITION_RATIOS[partition] * 100, 4
            ),
            "component_count": int(
                subset[GROUP_COLUMN].nunique()
            ),
            "unique_narrative_groups": int(
                subset[DOCUMENT_COLUMN].nunique()
            ),
            "mixed_product_component_count": len(
                mixed_components_in_split
            ),
            "missing_product_categories": [
                item["product"]
                for item in class_results
                if not item["present"]
            ],
            "class_distribution": class_results,
        }

    report["checks"] = {
        "all_records_assigned_once": bool(
            len(row_partitions) == total_records
            and np.all(row_folds >= 0)
        ),
        "all_components_intact": True,
        "all_unique_narratives_intact": True,
        "all_product_categories_in_all_partitions": all(
            not report["allocation"][partition][
                "missing_product_categories"
            ]
            for partition in ["train", "validation", "test"]
        ),
        "mixed_product_components_kept_intact": True,
        "source_dataset_modified": False,
        "original_split_files_modified": False,
        "actual_split_files_created": False,
        "model_retrained": False,
        "test_set_evaluated": False,
    }

    return report


# ============================================================
# COMPONENT-LEVEL ASSIGNMENT EXPORT
# ============================================================

def build_component_assignments(frame, row_folds, row_partitions):

    row_assignments = pd.DataFrame({
        GROUP_COLUMN: frame[GROUP_COLUMN].to_numpy(),
        "fold": row_folds,
        "partition": row_partitions,
    })

    component_assignments = (
        row_assignments.drop_duplicates(subset=[GROUP_COLUMN])
        .sort_values(GROUP_COLUMN)
        .reset_index(drop=True)
    )

    if len(component_assignments) != EXPECTED_COMPONENTS:
        raise RuntimeError(
            "Unexpected number of component assignments."
        )

    return component_assignments


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 68)
    print("FINROUTE AI - GROUP SPLIT DRY RUN")
    print("=" * 68)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    if ASSIGNMENT_OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {ASSIGNMENT_OUTPUT}"
        )

    if REPORT_OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {REPORT_OUTPUT}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(INPUT_FILE, low_memory=False)

    validate_input(frame)

    row_folds, row_partitions = allocate_components(frame)

    validate_assignments(
        frame,
        row_folds,
        row_partitions,
    )

    report = build_report(
        frame,
        row_folds,
        row_partitions,
    )

    component_assignments = build_component_assignments(
        frame,
        row_folds,
        row_partitions,
    )

    # Save component-level mapping only.
    # No narratives, complaint IDs or row positions are exported.
    component_assignments.to_csv(
        ASSIGNMENT_OUTPUT,
        index=False,
    )

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 68)
    print("DRY-RUN ALLOCATION SUMMARY")
    print("=" * 68)

    for partition in ["train", "validation", "test"]:

        details = report["allocation"][partition]

        print(f"\n{partition.upper()}")
        print(
            f"Records: {details['actual_records']:,} "
            f"/ target {details['target_records']:,}"
        )
        print(
            f"Dataset share: "
            f"{details['actual_dataset_percentage']:.2f}%"
        )
        print(
            f"Components: {details['component_count']:,}"
        )
        print(
            f"Missing categories: "
            f"{len(details['missing_product_categories'])}"
        )

    print("\n" + "=" * 68)
    print("DRY-RUN COMPLETED")
    print("=" * 68)

    print(f"\nComponent assignments:\n{ASSIGNMENT_OUTPUT}")
    print(f"\nDiagnostic report:\n{REPORT_OUTPUT}")

    print("\nNo source dataset was modified.")
    print("No original split files were modified.")
    print("No actual train/validation/test CSV files were created.")
    print("No model was retrained.")
    print("The test set was not evaluated.")


if __name__ == "__main__":
    main()
