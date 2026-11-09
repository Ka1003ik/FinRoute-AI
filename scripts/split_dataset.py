"""Create reproducible, stratified splits without narrative overlap."""

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

from src.preprocessing.text_vectorizer import (
    BASE_VECTORIZER_DESCRIPTION,
    feature_signature,
)


INPUT_CSV = ROOT / "data" / "processed" / "cfpb_2026_modeling.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_PATH = ROOT / "outputs" / "reports" / "split_summary.json"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    """Hash the input so this split can be tied to the exact source file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Return label counts in a JSON-safe, stable order."""
    counts = frame[LABEL_COLUMN].value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def main() -> None:
    """Split rows into train, validation, and test while grouping exact text."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

    data = pd.read_csv(
        INPUT_CSV,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    data[TEXT_COLUMN] = data[TEXT_COLUMN].str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].str.strip()
    feature_groups = data[TEXT_COLUMN].map(feature_signature)

    if data[[TEXT_COLUMN, LABEL_COLUMN]].eq("").any().any():
        raise ValueError("Input contains an empty narrative or product label.")
    if feature_groups.eq("").any():
        raise ValueError("Input contains narratives with no baseline TF-IDF features.")

    all_labels = set(data[LABEL_COLUMN].unique())

    # First reserve about 10% for the untouched final test partition.
    outer_splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_validation_indices, test_indices = next(
        outer_splitter.split(
            data,
            y=data[LABEL_COLUMN],
            groups=feature_groups,
        )
    )

    train_validation = data.iloc[train_validation_indices].copy()
    test = data.iloc[test_indices].copy()

    # Reserve about 10% of the remaining data for validation (~10% overall).
    inner_splitter = StratifiedGroupKFold(
        n_splits=9,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_indices, validation_indices = next(
        inner_splitter.split(
            train_validation,
            y=train_validation[LABEL_COLUMN],
            groups=feature_groups.iloc[train_validation_indices],
        )
    )

    train = train_validation.iloc[train_indices].copy()
    validation = train_validation.iloc[validation_indices].copy()
    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Fail safely if any label is absent or exact text crosses a split boundary.
    for name, frame in partitions.items():
        missing_labels = all_labels.difference(frame[LABEL_COLUMN].unique())
        if missing_labels:
            raise ValueError(f"{name} split is missing labels: {sorted(missing_labels)}")

    feature_sets = {
        name: set(feature_groups.loc[frame.index])
        for name, frame in partitions.items()
    }
    feature_overlap_counts = {
        "train_validation": len(feature_sets["train"] & feature_sets["validation"]),
        "train_test": len(feature_sets["train"] & feature_sets["test"]),
        "validation_test": len(feature_sets["validation"] & feature_sets["test"]),
    }
    if any(feature_overlap_counts.values()):
        raise ValueError(
            f"TF-IDF feature signatures overlap across splits: {feature_overlap_counts}"
        )

    text_sets = {
        name: set(frame[TEXT_COLUMN])
        for name, frame in partitions.items()
    }
    exact_text_overlap_counts = {
        "train_validation": len(text_sets["train"] & text_sets["validation"]),
        "train_test": len(text_sets["train"] & text_sets["test"]),
        "validation_test": len(text_sets["validation"] & text_sets["test"]),
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in partitions.items():
        frame.to_csv(SPLIT_DIR / f"{name}.csv", index=False, encoding="utf-8")

    report = {
        "input_file": str(INPUT_CSV),
        "input_sha256": sha256_file(INPUT_CSV),
        "random_state": RANDOM_STATE,
        "strategy": "StratifiedGroupKFold grouped by sorted TF-IDF analyzer features",
        "baseline_vectorizer": BASE_VECTORIZER_DESCRIPTION,
        "target_split_proportions": {
            "train": 0.80,
            "validation": 0.10,
            "test": 0.10,
        },
        "scikit_learn_version": sklearn.__version__,
        "rows": {
            name: int(len(frame))
            for name, frame in partitions.items()
        },
        "row_proportions": {
            name: round(len(frame) / len(data), 4)
            for name, frame in partitions.items()
        },
        "unique_tfidf_feature_groups": {
            name: int(len(feature_sets[name]))
            for name, frame in partitions.items()
        },
        "tfidf_feature_overlap_counts": feature_overlap_counts,
        "exact_narrative_overlap_counts": exact_text_overlap_counts,
        "product_counts": {
            name: class_counts(frame)
            for name, frame in partitions.items()
        },
        "split_files": {
            name: str(SPLIT_DIR / f"{name}.csv")
            for name in partitions
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
