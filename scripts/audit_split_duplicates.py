
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# FINROUTE AI - EXACT DUPLICATE SPLIT LEAKAGE AUDIT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT_DIR / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT_DIR / "outputs" / "leakage_audit"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"

SPLITS = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

GROUP_REPORT_PATH = (
    OUTPUT_DIR / "exact_duplicate_cross_split_groups.csv"
)

JSON_REPORT_PATH = (
    OUTPUT_DIR / "exact_duplicate_audit.json"
)


def normalize_narrative(value):
    """
    Apply the same basic normalization used during
    project preprocessing.

    No narrative text is written to the output files.
    """

    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = html.unescape(text)

    # Remove standalone redaction markers.
    text = re.sub(r"\b[xX]{1,3}\b", " ", text)

    # Normalize whitespace and casing.
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def create_text_hash(text):
    """Generate a SHA-256 fingerprint of normalized text."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_split(split_name, file_path):

    print(f"\nReading {split_name} split...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        low_memory=False
    )

    records = []
    empty_narratives = 0

    for row_position, (narrative, label) in enumerate(
        zip(df[TEXT_COLUMN], df[LABEL_COLUMN])
    ):

        normalized_text = normalize_narrative(narrative)

        if not normalized_text:
            empty_narratives += 1
            continue

        if pd.isna(label) or not str(label).strip():
            raise ValueError(
                f"Missing product label in {split_name} "
                f"at row position {row_position}"
            )

        records.append({
            "split": split_name,
            "row_position": row_position,
            "text_hash": create_text_hash(normalized_text),
            "product": str(label).strip()
        })

    print(f"Rows loaded: {len(df):,}")
    print(f"Usable narratives: {len(records):,}")
    print(f"Empty narratives: {empty_narratives:,}")

    return records, {
        "rows": len(df),
        "usable_narratives": len(records),
        "empty_narratives": empty_narratives
    }


def build_duplicate_groups(records_df):

    print("\nFinding exact duplicate groups...")

    duplicate_groups = []

    group_number = 0

    for text_hash, group in records_df.groupby(
        "text_hash",
        sort=False
    ):

        present_splits = sorted(group["split"].unique())

        # We only report duplicates crossing split boundaries.
        if len(present_splits) < 2:
            continue

        group_number += 1

        labels = sorted(group["product"].unique())

        entry = {
            "duplicate_group_id": f"EXACT-{group_number:06d}",
            "number_of_splits": len(present_splits),
            "number_of_product_labels": len(labels),
            "conflicting_product_labels": len(labels) > 1,
            "product_labels": " | ".join(labels)
        }

        # Store row positions, not complaint narratives or hashes.
        for split_name in SPLITS:

            split_rows = group[
                group["split"] == split_name
            ]["row_position"].tolist()

            entry[f"{split_name}_row_count"] = len(split_rows)

            entry[f"{split_name}_row_positions"] = ";".join(
                str(position)
                for position in split_rows
            )

        duplicate_groups.append(entry)

    return pd.DataFrame(duplicate_groups)


def run_audit():

    print("=" * 65)
    print("FINROUTE AI - EXACT DUPLICATE LEAKAGE AUDIT")
    print("=" * 65)

    all_records = []
    split_statistics = {}

    for split_name, file_path in SPLITS.items():

        records, statistics = load_split(
            split_name,
            file_path
        )

        all_records.extend(records)
        split_statistics[split_name] = statistics

    if not all_records:
        raise ValueError("No usable complaint narratives found.")

    records_df = pd.DataFrame(all_records)

    print("\nBuilding unique-text statistics...")

    hash_sets = {
        split_name: set(
            records_df.loc[
                records_df["split"] == split_name,
                "text_hash"
            ]
        )
        for split_name in SPLITS
    }

    pairwise_overlap = {
        "train_validation": len(
            hash_sets["train"] & hash_sets["validation"]
        ),
        "train_test": len(
            hash_sets["train"] & hash_sets["test"]
        ),
        "validation_test": len(
            hash_sets["validation"] & hash_sets["test"]
        )
    }

    duplicate_groups_df = build_duplicate_groups(records_df)

    if duplicate_groups_df.empty:

        cross_split_rows = {
            split_name: 0
            for split_name in SPLITS
        }

        conflicting_groups = 0

    else:

        cross_split_rows = {
            split_name: int(
                duplicate_groups_df[
                    f"{split_name}_row_count"
                ].sum()
            )
            for split_name in SPLITS
        }

        conflicting_groups = int(
            duplicate_groups_df[
                "conflicting_product_labels"
            ].sum()
        )

    total_cross_split_rows = sum(cross_split_rows.values())

    # Count all repeated normalized texts, including duplicates
    # that occur only within an individual split.
    all_group_sizes = records_df.groupby(
        "text_hash"
    ).size()

    repeated_text_groups = int(
        (all_group_sizes > 1).sum()
    )

    repeated_extra_rows = int(
        (all_group_sizes - 1).clip(lower=0).sum()
    )

    report = {
        "project": "FinRoute AI",
        "audit_type": "Exact duplicate narrative leakage audit",
        "method": (
            "SHA-256 fingerprints of normalized complaint narratives"
        ),
        "normalization": (
            "Unicode NFKC, HTML entity decoding, standalone "
            "redaction-marker removal, whitespace normalization "
            "and lowercase conversion"
        ),
        "split_statistics": split_statistics,
        "unique_normalized_texts_by_split": {
            split_name: len(hashes)
            for split_name, hashes in hash_sets.items()
        },
        "pairwise_cross_split_duplicate_text_counts": (
            pairwise_overlap
        ),
        "cross_split_duplicate_groups": len(
            duplicate_groups_df
        ),
        "cross_split_duplicate_rows_by_split": cross_split_rows,
        "total_rows_in_cross_split_duplicate_groups": (
            total_cross_split_rows
        ),
        "cross_split_groups_with_conflicting_product_labels": (
            conflicting_groups
        ),
        "all_split_repeated_text_groups": repeated_text_groups,
        "all_split_repeated_extra_rows": repeated_extra_rows,
        "model_test_evaluation_performed": False,
        "raw_narratives_exported": False,
        "complaint_ids_exported": False
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Protect existing audit results from accidental overwriting.
    for output_path in [
        GROUP_REPORT_PATH,
        JSON_REPORT_PATH
    ]:

        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Rename or move the previous report before rerunning."
            )

    duplicate_groups_df.to_csv(
        GROUP_REPORT_PATH,
        index=False
    )

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("EXACT DUPLICATE AUDIT RESULTS")
    print("=" * 65)

    print(
        f"\nCross-split duplicate groups: "
        f"{report['cross_split_duplicate_groups']:,}"
    )

    print(
        f"Rows in cross-split duplicate groups: "
        f"{total_cross_split_rows:,}"
    )

    print(
        f"Groups with conflicting labels: "
        f"{conflicting_groups:,}"
    )

    print("\nPairwise overlap:")

    for pair, count in pairwise_overlap.items():
        print(f"{pair}: {count:,}")

    print("\nReports saved:")

    print(GROUP_REPORT_PATH)
    print(JSON_REPORT_PATH)

    print("\nTEST SET MODEL EVALUATION: NOT PERFORMED")

    print("\nEXACT DUPLICATE AUDIT COMPLETED.")


if __name__ == "__main__":
    run_audit()
