
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "outputs" / "leakage_audit"

SOURCE_FILE = (
    PROCESSED_DIR / "cfpb_2026_modeling_v2.csv"
)

ROW_COMPONENTS_FILE = (
    AUDIT_DIR / "full_dataset_component_assignments_v2.csv"
)

DRY_RUN_ASSIGNMENTS_FILE = (
    AUDIT_DIR / "group_split_dry_run_assignments_v2.csv"
)

OUTPUT_DIR = PROCESSED_DIR / "splits_v3"

LABEL_COLUMN = "Product"
DOCUMENT_COLUMN = "unique_document_id"
GROUP_COLUMN = "component_100"

EXPECTED_RECORDS = 121114
EXPECTED_DOCUMENTS = 108902
EXPECTED_COMPONENTS = 108369
EXPECTED_CLASSES = 11

RANDOM_STATE = 42

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

FOLD_TO_PARTITION = {
    0: "validation",
    1: "test",
    2: "train",
    3: "train",
    4: "train",
    5: "train",
    6: "train",
    7: "train",
    8: "train",
    9: "train",
}


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

def validate_inputs(
    source,
    row_components,
    component_assignments,
):

    print("\nValidating input files...")

    required_source_columns = {
        LABEL_COLUMN,
    }

    required_mapping_columns = {
        "row_position",
        DOCUMENT_COLUMN,
        LABEL_COLUMN,
        GROUP_COLUMN,
    }

    required_assignment_columns = {
        GROUP_COLUMN,
        "fold",
        "partition",
    }

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    for name, frame, required_columns in [
        (
            "Source dataset",
            source,
            required_source_columns,
        ),
        (
            "Row component mapping",
            row_components,
            required_mapping_columns,
        ),
        (
            "Dry-run component assignments",
            component_assignments,
            required_assignment_columns,
        ),
    ]:

        missing = required_columns - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # Source record count
    # --------------------------------------------------------

    if len(source) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS:,} source records, "
            f"but found {len(source):,}."
        )

    if len(row_components) != EXPECTED_RECORDS:
        raise ValueError(
            "Row component mapping does not contain "
            f"{EXPECTED_RECORDS:,} records."
        )

    # --------------------------------------------------------
    # Null-value validation
    # --------------------------------------------------------

    if source[LABEL_COLUMN].isna().any():
        raise ValueError(
            "Null Product labels found in source dataset."
        )

    if row_components[
        [DOCUMENT_COLUMN, LABEL_COLUMN, GROUP_COLUMN]
    ].isna().any().any():

        raise ValueError(
            "Null values found in row component mapping."
        )

    if component_assignments[
        [GROUP_COLUMN, "fold", "partition"]
    ].isna().any().any():

        raise ValueError(
            "Null values found in dry-run assignments."
        )

    # --------------------------------------------------------
    # Validate original row positions
    # --------------------------------------------------------

    positions = pd.to_numeric(
        row_components["row_position"],
        errors="raise",
    )

    if not np.isfinite(positions.to_numpy()).all():
        raise ValueError(
            "Non-finite row positions detected."
        )

    if not np.equal(
        positions.to_numpy(),
        np.floor(positions.to_numpy()),
    ).all():

        raise ValueError(
            "Non-integer row positions detected."
        )

    row_components["row_position"] = (
        positions.astype("int64")
    )

    row_components = (
        row_components
        .sort_values("row_position")
        .reset_index(drop=True)
    )

    expected_positions = np.arange(
        EXPECTED_RECORDS,
        dtype=np.int64,
    )

    if not np.array_equal(
        row_components["row_position"].to_numpy(),
        expected_positions,
    ):

        raise ValueError(
            "Row positions are missing, duplicated, "
            "or outside the expected range."
        )

    # --------------------------------------------------------
    # Verify Product labels against original source rows
    # --------------------------------------------------------

    source_labels = (
        source[LABEL_COLUMN]
        .astype(str)
        .to_numpy()
    )

    mapping_labels = (
        row_components[LABEL_COLUMN]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        source_labels,
        mapping_labels,
    ):

        raise ValueError(
            "Product labels do not match between "
            "the source dataset and row component mapping."
        )

    # --------------------------------------------------------
    # Normalize document and component identifiers
    # --------------------------------------------------------

    row_components["_document_key"] = (
        row_components[DOCUMENT_COLUMN]
        .astype(str)
        .str.strip()
    )

    row_components["_component_key"] = (
        row_components[GROUP_COLUMN]
        .astype(str)
        .str.strip()
    )

    component_assignments["_component_key"] = (
        component_assignments[GROUP_COLUMN]
        .astype(str)
        .str.strip()
    )

    if (
        row_components["_document_key"].eq("").any()
        or row_components["_component_key"].eq("").any()
        or component_assignments["_component_key"].eq("").any()
    ):

        raise ValueError(
            "Empty document or component identifiers detected."
        )

    # --------------------------------------------------------
    # Validate unique document and component counts
    # --------------------------------------------------------

    document_count = (
        row_components["_document_key"].nunique()
    )

    component_count = (
        row_components["_component_key"].nunique()
    )

    class_count = (
        source[LABEL_COLUMN].nunique()
    )

    if document_count != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENTS:,} unique "
            f"narrative groups, found {document_count:,}."
        )

    if component_count != EXPECTED_COMPONENTS:
        raise ValueError(
            f"Expected {EXPECTED_COMPONENTS:,} components, "
            f"found {component_count:,}."
        )

    if class_count != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} Product categories, "
            f"found {class_count}."
        )

    # --------------------------------------------------------
    # Each unique narrative must have one Product and component
    # --------------------------------------------------------

    document_check = (
        row_components
        .groupby("_document_key")
        .agg(
            product_count=(LABEL_COLUMN, "nunique"),
            component_count=("_component_key", "nunique"),
        )
    )

    if (
        document_check["product_count"] != 1
    ).any():

        raise ValueError(
            "A unique narrative has conflicting Product labels."
        )

    if (
        document_check["component_count"] != 1
    ).any():

        raise ValueError(
            "A unique narrative maps to multiple components."
        )

    # --------------------------------------------------------
    # Validate component assignment uniqueness
    # --------------------------------------------------------

    if component_assignments[
        "_component_key"
    ].duplicated().any():

        raise ValueError(
            "A component has multiple dry-run assignments."
        )

    if len(component_assignments) != EXPECTED_COMPONENTS:
        raise ValueError(
            "Unexpected number of component assignments."
        )

    mapping_components = set(
        row_components["_component_key"].unique()
    )

    assigned_components = set(
        component_assignments["_component_key"]
    )

    if mapping_components != assigned_components:
        raise ValueError(
            "Component identifiers do not match between "
            "the row mapping and dry-run assignments."
        )

    # --------------------------------------------------------
    # Validate fold and partition mapping
    # --------------------------------------------------------

    folds = pd.to_numeric(
        component_assignments["fold"],
        errors="raise",
    )

    if not np.equal(
        folds.to_numpy(),
        np.floor(folds.to_numpy()),
    ).all():

        raise ValueError(
            "Non-integer fold values detected."
        )

    component_assignments["fold"] = (
        folds.astype(int)
    )

    if not component_assignments["fold"].isin(
        FOLD_TO_PARTITION
    ).all():

        raise ValueError(
            "Unexpected fold numbers detected."
        )

    actual_partitions = (
        component_assignments["partition"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    expected_partitions = (
        component_assignments["fold"]
        .map(FOLD_TO_PARTITION)
        .to_numpy()
    )

    if not np.array_equal(
        actual_partitions,
        expected_partitions,
    ):

        raise ValueError(
            "Partition names do not match the approved "
            "fold-to-partition mapping."
        )

    component_assignments["partition"] = (
        actual_partitions
    )

    print("Input validation passed.")

    print(f"Source records: {len(source):,}")
    print(f"Unique narrative groups: {document_count:,}")
    print(f"Components: {component_count:,}")
    print(f"Product categories: {class_count}")

    # IMPORTANT:
    # Return both DataFrames after sorting and adding
    # normalized identifier columns.

    return row_components, component_assignments


# ============================================================
# 3. MAP COMPONENTS TO PARTITIONS
# ============================================================

def assign_partitions(
    row_components,
    component_assignments,
):

    print("\nMapping original complaints to partitions...")

    partition_lookup = (
        component_assignments
        .set_index("_component_key")["partition"]
    )

    row_components["_partition"] = (
        row_components["_component_key"]
        .map(partition_lookup)
    )

    if row_components["_partition"].isna().any():
        raise ValueError(
            "Some original records do not have "
            "a partition assignment."
        )

    if not row_components["_partition"].isin(
        PARTITION_RATIOS
    ).all():

        raise ValueError(
            "Unexpected partition names detected."
        )

    # --------------------------------------------------------
    # Verify that every component stays intact
    # --------------------------------------------------------

    component_check = (
        row_components
        .groupby("_component_key")["_partition"]
        .nunique()
    )

    if (component_check != 1).any():
        raise RuntimeError(
            "A component crosses partition boundaries."
        )

    # --------------------------------------------------------
    # Verify that every unique narrative stays intact
    # --------------------------------------------------------

    document_check = (
        row_components
        .groupby("_document_key")["_partition"]
        .nunique()
    )

    if (document_check != 1).any():
        raise RuntimeError(
            "A unique narrative crosses partition boundaries."
        )

    print("Partition mapping validated.")
    print("No component crosses partitions.")
    print("No unique narrative crosses partitions.")

    return row_components


# ============================================================
# 4. BUILD VALIDATION REPORT
# ============================================================

def build_partition_report(
    source,
    row_components,
):

    products = sorted(
        source[LABEL_COLUMN].unique()
    )

    total_records = len(source)

    total_class_counts = (
        source[LABEL_COLUMN]
        .value_counts()
        .reindex(products, fill_value=0)
    )

    # Identify mixed-Product components.
    product_counts_per_component = (
        row_components
        .groupby("_component_key")[LABEL_COLUMN]
        .nunique()
    )

    mixed_component_ids = set(
        product_counts_per_component[
            product_counts_per_component > 1
        ].index
    )

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v3",
        "source_dataset": "cfpb_2026_modeling_v2.csv",
        "allocation_method": "StratifiedGroupKFold",
        "random_state": RANDOM_STATE,
        "component_threshold": "Jaccard = 1.00",
        "component_column": GROUP_COLUMN,
        "target_ratios": PARTITION_RATIOS,
        "source_records": total_records,
        "unique_narrative_groups": int(
            row_components["_document_key"].nunique()
        ),
        "components": int(
            row_components["_component_key"].nunique()
        ),
        "product_categories": len(products),
        "fold_mapping": {
            "validation": [0],
            "test": [1],
            "train": list(range(2, 10)),
        },
        "partitions": {},
        "checks": {},
    }

    # --------------------------------------------------------
    # Calculate partition-level and Product-level statistics
    # --------------------------------------------------------

    for partition in ["train", "validation", "test"]:

        mask = (
            row_components["_partition"]
            .eq(partition)
            .to_numpy()
        )

        positions = np.flatnonzero(mask)

        subset = source.iloc[positions]

        subset_mapping = row_components.loc[mask]

        actual_records = len(subset)

        target_records = TARGET_RECORD_COUNTS[partition]

        actual_class_counts = (
            subset[LABEL_COLUMN]
            .value_counts()
            .reindex(products, fill_value=0)
        )

        class_distribution = []

        for product in products:

            total_for_class = int(
                total_class_counts[product]
            )

            actual_for_class = int(
                actual_class_counts[product]
            )

            target_for_class = (
                total_for_class
                * PARTITION_RATIOS[partition]
            )

            actual_share = (
                actual_for_class / total_for_class * 100
                if total_for_class > 0
                else 0.0
            )

            class_distribution.append({
                "product": product,
                "total_records": total_for_class,
                "target_records": round(
                    target_for_class, 2
                ),
                "actual_records": actual_for_class,
                "difference_from_target": round(
                    actual_for_class - target_for_class, 2
                ),
                "target_share_of_class_pct": (
                    PARTITION_RATIOS[partition] * 100
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

        split_components = set(
            subset_mapping["_component_key"].unique()
        )

        mixed_components_in_partition = (
            split_components & mixed_component_ids
        )

        report["partitions"][partition] = {
            "target_records": target_records,
            "actual_records": actual_records,
            "record_difference": (
                actual_records - target_records
            ),
            "actual_dataset_percentage": round(
                actual_records / total_records * 100,
                4,
            ),
            "target_dataset_percentage": (
                PARTITION_RATIOS[partition] * 100
            ),
            "component_count": int(
                subset_mapping["_component_key"].nunique()
            ),
            "unique_narrative_groups": int(
                subset_mapping["_document_key"].nunique()
            ),
            "mixed_product_component_count": len(
                mixed_components_in_partition
            ),
            "missing_product_categories": [
                item["product"]
                for item in class_distribution
                if not item["present"]
            ],
            "class_distribution": class_distribution,
        }

    # --------------------------------------------------------
    # Overall validation checks
    # --------------------------------------------------------

    total_partition_records = sum(
        report["partitions"][partition]["actual_records"]
        for partition in ["train", "validation", "test"]
    )

    report["checks"] = {
        "all_source_records_assigned": (
            total_partition_records == total_records
        ),
        "all_components_intact": True,
        "all_unique_narratives_intact": True,
        "all_product_categories_in_all_partitions": all(
            not report["partitions"][partition][
                "missing_product_categories"
            ]
            for partition in ["train", "validation", "test"]
        ),
        "mixed_product_components_kept_intact": True,
        "source_dataset_modified": False,
        "splits_v2_modified": False,
        "test_set_evaluated": False,
        "model_retrained": False,
    }

    return report


# ============================================================
# 5. CREATE AND VERIFY CSV FILES
# ============================================================

def create_split_files(
    source,
    row_components,
):

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{OUTPUT_DIR}\n"
            "No existing files will be overwritten."
        )

    OUTPUT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create a temporary directory on the same filesystem.
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=".splits_v3_tmp_",
            dir=OUTPUT_DIR.parent,
        )
    )

    try:

        report = build_partition_report(
            source,
            row_components,
        )

        # ----------------------------------------------------
        # Save each partition
        # ----------------------------------------------------

        for partition in ["train", "validation", "test"]:

            mask = (
                row_components["_partition"]
                .eq(partition)
                .to_numpy()
            )

            positions = np.flatnonzero(mask)

            subset = source.iloc[positions]

            if subset.empty:
                raise RuntimeError(
                    f"{partition} partition is empty."
                )

            output_file = (
                temporary_dir / f"{partition}.csv"
            )

            subset.to_csv(
                output_file,
                index=False,
                encoding="utf-8",
            )

            # ------------------------------------------------
            # Verify saved CSV header
            # ------------------------------------------------

            saved_columns = pd.read_csv(
                output_file,
                nrows=0,
            ).columns.tolist()

            if saved_columns != source.columns.tolist():
                raise RuntimeError(
                    f"{partition} CSV column mismatch."
                )

            # ------------------------------------------------
            # Verify saved row count and Product distribution
            # ------------------------------------------------

            saved_labels = pd.read_csv(
                output_file,
                usecols=[LABEL_COLUMN],
                low_memory=False,
            )[LABEL_COLUMN]

            if len(saved_labels) != len(subset):
                raise RuntimeError(
                    f"{partition} saved record count mismatch."
                )

            expected_counts = (
                subset[LABEL_COLUMN]
                .value_counts()
                .to_dict()
            )

            saved_counts = (
                saved_labels.value_counts()
                .to_dict()
            )

            if expected_counts != saved_counts:
                raise RuntimeError(
                    f"{partition} Product distribution "
                    "does not match the source partition."
                )

            print(
                f"{partition.capitalize()}: "
                f"{len(subset):,} records verified."
            )

        # ----------------------------------------------------
        # Add saved-file verification status
        # ----------------------------------------------------

        report["checks"][
            "split_files_readback_verified"
        ] = True

        # ----------------------------------------------------
        # Save validation report inside temporary directory
        # ----------------------------------------------------

        report_file = (
            temporary_dir
            / "split_validation_report_v3.json"
        )

        with open(
            report_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

        # ----------------------------------------------------
        # Publish output directory after all checks pass
        # ----------------------------------------------------

        if OUTPUT_DIR.exists():
            raise FileExistsError(
                "Output directory appeared during execution. "
                "Refusing to overwrite it."
            )

        temporary_dir.rename(OUTPUT_DIR)

        print("\n" + "=" * 68)
        print("V3 GROUP-AWARE SPLIT CREATION COMPLETED")
        print("=" * 68)

        for partition in ["train", "validation", "test"]:

            details = report["partitions"][partition]

            print(f"\n{partition.upper()}")

            print(
                f"Records: {details['actual_records']:,} "
                f"/ target {details['target_records']:,}"
            )

            print(
                f"Dataset share: "
                f"{details['actual_dataset_percentage']:.4f}%"
            )

            print(
                f"Components: "
                f"{details['component_count']:,}"
            )

            print(
                f"Unique narratives: "
                f"{details['unique_narrative_groups']:,}"
            )

            print(
                f"Missing Product categories: "
                f"{len(details['missing_product_categories'])}"
            )

        print("\n" + "=" * 68)

        print(f"\nOutput directory:\n{OUTPUT_DIR}")

        print(
            "\nValidation report:\n"
            f"{OUTPUT_DIR / 'split_validation_report_v3.json'}"
        )

        print("\nAll original complaints retained.")
        print("No component crosses partitions.")
        print("No unique narrative crosses partitions.")
        print("Original splits_v2 remains untouched.")
        print("No model retraining performed.")
        print("Test set not evaluated.")

    except Exception:

        if temporary_dir.exists():
            shutil.rmtree(
                temporary_dir,
                ignore_errors=True,
            )

        raise


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 68)
    print("FINROUTE AI - V3 GROUP-AWARE SPLIT CREATION")
    print("=" * 68)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        SOURCE_FILE,
        ROW_COMPONENTS_FILE,
        DRY_RUN_ASSIGNMENTS_FILE,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found:\n{path}"
            )

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Processed directory not found:\n{PROCESSED_DIR}"
        )

    # --------------------------------------------------------
    # Prevent overwriting existing v3 files
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        raise FileExistsError(
            f"Output directory already exists:\n{OUTPUT_DIR}\n"
            "No files will be overwritten."
        )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    print("\nLoading source and assignment files...")

    source = pd.read_csv(
        SOURCE_FILE,
        low_memory=False,
    )

    row_components = pd.read_csv(
        ROW_COMPONENTS_FILE,
        low_memory=False,
    )

    component_assignments = pd.read_csv(
        DRY_RUN_ASSIGNMENTS_FILE,
        low_memory=False,
    )

    # --------------------------------------------------------
    # Validate inputs and retain returned DataFrames
    # --------------------------------------------------------

    row_components, component_assignments = (
        validate_inputs(
            source,
            row_components,
            component_assignments,
        )
    )

    # --------------------------------------------------------
    # Assign original complaints to partitions
    # --------------------------------------------------------

    row_components = assign_partitions(
        row_components,
        component_assignments,
    )

    # --------------------------------------------------------
    # Create and verify v3 split files
    # --------------------------------------------------------

    create_split_files(
        source,
        row_components,
    )


if __name__ == "__main__":
    main()
