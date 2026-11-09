
"""Create leakage-safe stratified splits for FinRoute AI dataset v2."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = (
    ROOT / "data" / "processed" / "cfpb_2026_modeling_v2.csv"
)

SPLIT_DIR = (
    ROOT / "data" / "processed" / "splits_v2"
)

REPORT_PATH = (
    ROOT / "outputs" / "reports" / "split_summary_v2.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"

RANDOM_STATE = 42

EXPECTED_ROWS = 121114

TARGET_PROPORTIONS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def class_counts(frame: pd.DataFrame) -> dict:
    """Return stable product counts."""

    counts = frame[LABEL_COLUMN].value_counts().sort_index()

    return {
        str(label): int(count)
        for label, count in counts.items()
    }


def group_hashes(frame: pd.DataFrame) -> set:
    """Generate normalized-text fingerprints for overlap checks."""

    return {
        hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        for text in frame["_group_key"]
    }


# ============================================================
# MAIN SPLITTING FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("FINROUTE AI - DATASET V2 SPLITTING")
    print("=" * 65)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {INPUT_CSV}"
        )

    # Never overwrite previous v2 outputs.
    expected_outputs = [
        SPLIT_DIR / "train.csv",
        SPLIT_DIR / "validation.csv",
        SPLIT_DIR / "test.csv",
        REPORT_PATH,
    ]

    existing_outputs = [
        path for path in expected_outputs
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Existing v2 outputs found. Move or rename them "
            "before regenerating the splits:\n"
            + "\n".join(str(path) for path in existing_outputs)
        )

    print("\nLoading corrected modeling dataset...")

    data = pd.read_csv(
        INPUT_CSV,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    data = data.reset_index(drop=True)

    print(f"Dataset records: {len(data):,}")

    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(data):,}."
        )

    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Missing column: {TEXT_COLUMN}")

    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"Missing column: {LABEL_COLUMN}")

    if data[LABEL_COLUMN].eq("").any():
        raise ValueError("Empty product labels found.")

    # --------------------------------------------------------
    # CREATE CANONICAL TEXT GROUPS
    # --------------------------------------------------------

    print("\nCreating canonical cleaned-text groups...")

    data["_group_key"] = data[TEXT_COLUMN].map(
        clean_narrative
    )

    if data["_group_key"].eq("").any():
        raise ValueError(
            "Empty cleaned narratives found in modeling dataset."
        )

    unique_groups = data["_group_key"].nunique()

    print(f"Unique cleaned narratives: {unique_groups:,}")
    print(f"Total product classes: {data[LABEL_COLUMN].nunique()}")

    # Verify that each normalized narrative has one label.
    label_counts_per_group = (
        data.groupby("_group_key")[LABEL_COLUMN].nunique()
    )

    remaining_conflicts = int(
        (label_counts_per_group > 1).sum()
    )

    if remaining_conflicts != 0:
        raise ValueError(
            f"Found {remaining_conflicts} conflicting text groups."
        )

    all_labels = set(data[LABEL_COLUMN].unique())

    if len(all_labels) != 11:
        raise ValueError(
            f"Expected 11 product classes, found {len(all_labels)}."
        )

    # --------------------------------------------------------
    # OUTER SPLIT: 90% TRAINING + VALIDATION, 10% TEST
    # --------------------------------------------------------

    print("\nCreating outer stratified group split...")

    outer_splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_validation_indices, test_indices = next(
        outer_splitter.split(
            X=data,
            y=data[LABEL_COLUMN],
            groups=data["_group_key"],
        )
    )

    train_validation = data.iloc[
        train_validation_indices
    ].copy()

    test = data.iloc[
        test_indices
    ].copy()

    # --------------------------------------------------------
    # INNER SPLIT: APPROXIMATELY 80% TRAIN, 10% VALIDATION
    # --------------------------------------------------------

    print("Creating inner stratified group split...")

    inner_splitter = StratifiedGroupKFold(
        n_splits=9,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_indices, validation_indices = next(
        inner_splitter.split(
            X=train_validation,
            y=train_validation[LABEL_COLUMN],
            groups=train_validation["_group_key"],
        )
    )

    train = train_validation.iloc[
        train_indices
    ].copy()

    validation = train_validation.iloc[
        validation_indices
    ].copy()

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nValidating splits...")

    total_split_rows = sum(
        len(frame) for frame in partitions.values()
    )

    if total_split_rows != len(data):
        raise ValueError(
            "Split row counts do not match the modeling dataset."
        )

    for name, frame in partitions.items():

        if frame.empty:
            raise ValueError(f"{name} split is empty.")

        missing_labels = all_labels.difference(
            frame[LABEL_COLUMN].unique()
        )

        if missing_labels:
            raise ValueError(
                f"{name} is missing product classes: "
                f"{sorted(missing_labels)}"
            )

    # --------------------------------------------------------
    # CANONICAL TEXT OVERLAP CHECK
    # --------------------------------------------------------

    print("Checking cross-split normalized-text leakage...")

    partition_hashes = {
        name: group_hashes(frame)
        for name, frame in partitions.items()
    }

    overlap_counts = {
        "train_validation": len(
            partition_hashes["train"]
            & partition_hashes["validation"]
        ),
        "train_test": len(
            partition_hashes["train"]
            & partition_hashes["test"]
        ),
        "validation_test": len(
            partition_hashes["validation"]
            & partition_hashes["test"]
        ),
    }

    if any(overlap_counts.values()):
        raise ValueError(
            f"Canonical text leakage detected: {overlap_counts}"
        )

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    print("\nSaving versioned splits...")

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, frame in partitions.items():

        output_frame = frame.drop(
            columns=["_group_key"]
        )

        output_path = SPLIT_DIR / f"{name}.csv"

        output_frame.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    split_rows = {
        name: int(len(frame))
        for name, frame in partitions.items()
    }

    row_proportions = {
        name: round(
            len(frame) / len(data),
            6
        )
        for name, frame in partitions.items()
    }

    unique_groups_per_split = {
        name: int(frame["_group_key"].nunique())
        for name, frame in partitions.items()
    }

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "input_file": str(INPUT_CSV),
        "input_sha256": sha256_file(INPUT_CSV),
        "random_state": RANDOM_STATE,
        "strategy": (
            "Two-stage StratifiedGroupKFold grouped by "
            "canonical clean_narrative output"
        ),
        "target_split_proportions": TARGET_PROPORTIONS,
        "actual_row_proportions": row_proportions,
        "scikit_learn_version": sklearn.__version__,
        "total_rows": int(len(data)),
        "unique_cleaned_text_groups": int(unique_groups),
        "rows": split_rows,
        "unique_groups_by_split": unique_groups_per_split,
        "product_counts": {
            name: class_counts(frame)
            for name, frame in partitions.items()
        },
        "normalized_text_overlap_counts": overlap_counts,
        "remaining_product_label_conflicts": remaining_conflicts,
        "model_test_evaluation_performed": False,
        "raw_narratives_exported_to_report": False,
        "split_files": {
            name: str(SPLIT_DIR / f"{name}.csv")
            for name in partitions
        },
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DATASET V2 SPLIT RESULTS")
    print("=" * 65)

    for name in ["train", "validation", "test"]:

        print(
            f"{name.capitalize():12s}: "
            f"{split_rows[name]:,} rows "
            f"({row_proportions[name] * 100:.2f}%)"
        )

    print("\nCross-split normalized-text overlaps:")

    for pair, count in overlap_counts.items():
        print(f"{pair}: {count}")

    print("\nSplit files saved:")
    print(SPLIT_DIR)

    print("\nSplit report saved:")
    print(REPORT_PATH)

    print("\nTest model evaluation: NOT PERFORMED")
    print("\nALL DATASET V2 SPLIT CHECKS PASSED.")


if __name__ == "__main__":
    main()
