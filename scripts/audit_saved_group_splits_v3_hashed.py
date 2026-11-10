"""Independent, read-only audit of FinRoute AI v3 saved split files."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "outputs" / "leakage_audit"

SOURCE_FILE = PROCESSED_DIR / "cfpb_2026_modeling_v2.csv"
ROW_MAPPING_FILE = AUDIT_DIR / "full_dataset_component_assignments_v2.csv"
COMPONENT_ASSIGNMENTS_FILE = AUDIT_DIR / "group_split_dry_run_assignments_v2.csv"
SPLIT_DIR = PROCESSED_DIR / "splits_v3"
REPORT_FILE = AUDIT_DIR / "independent_saved_split_audit_v3_hashed.json"

LABEL_COLUMN = "Product"
DOCUMENT_COLUMN = "unique_document_id"
GROUP_COLUMN = "component_100"
PARTITIONS = ("train", "validation", "test")

EXPECTED_RECORDS = 121_114
EXPECTED_DOCUMENTS = 108_902
EXPECTED_COMPONENTS = 108_369
EXPECTED_CLASSES = 11
EXPECTED_PARTITION_COUNTS = {
    "train": 96_890,
    "validation": 12_112,
    "test": 12_112,
}


# -----------------------------------------------------------------------------
# CSV reading and row fingerprints
# -----------------------------------------------------------------------------
def read_mapping_csv(path: Path) -> pd.DataFrame:
    """Read identifier/mapping CSVs as text without changing identifiers."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_fields(fields: tuple[str, ...] | list[str]) -> str:
    """Hash one parsed CSV record, preserving field values and order."""
    payload = json.dumps(
        tuple(fields),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprints_from_dataframe_as_csv(frame: pd.DataFrame) -> list[str]:
    """Fingerprint records as pandas would serialize them with to_csv.

    The v3 split-creation script loads the source with normal pandas type
    inference and writes partitions with DataFrame.to_csv(index=False).
    Re-serializing the expected source rows this way reproduces that CSV
    representation, including pandas' handling of missing and numeric values.
    """
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    return [fingerprint_fields(tuple(row)) for row in csv.reader(buffer)]


def read_saved_csv_records(path: Path, expected_header: list[str]):
    """Read saved CSV records using csv.reader and return hashes/counts."""
    if not path.is_file():
        raise FileNotFoundError(f"Required split file not found: {path}")

    hashes: list[str] = []
    product_counts: Counter[str] = Counter()
    product_index = expected_header.index(LABEL_COLUMN)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            actual_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Split file is empty: {path}") from exc

        if actual_header != expected_header:
            raise ValueError(
                f"{path.name} has missing, extra, or reordered columns."
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise ValueError(
                    f"Malformed CSV row in {path.name} at line {line_number}: "
                    f"expected {len(expected_header)} fields, found {len(row)}."
                )
            hashes.append(fingerprint_fields(row))
            product_counts[row[product_index]] += 1

    return hashes, product_counts


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------
def validate_inputs(
    source: pd.DataFrame,
    row_map: pd.DataFrame,
    component_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nValidating source and assignment files...")

    required = {
        "source": (source, {LABEL_COLUMN}),
        "row mapping": (
            row_map,
            {"row_position", DOCUMENT_COLUMN, LABEL_COLUMN, GROUP_COLUMN},
        ),
        "component assignments": (
            component_assignments,
            {GROUP_COLUMN, "partition"},
        ),
    }
    for name, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    if source.columns.duplicated().any():
        raise ValueError("Duplicate column names in source dataset.")
    if len(source) != EXPECTED_RECORDS or len(row_map) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS:,} source and mapping rows; "
            f"got {len(source):,} and {len(row_map):,}."
        )
    if source[LABEL_COLUMN].isna().any():
        raise ValueError("Missing Product labels found in source dataset.")

    numeric_positions = pd.to_numeric(row_map["row_position"], errors="raise")
    numeric_array = numeric_positions.to_numpy(dtype=float)
    if not np.isfinite(numeric_array).all() or not np.equal(
        numeric_array, np.floor(numeric_array)
    ).all():
        raise ValueError("row_position contains non-finite or non-integer values.")

    row_map = row_map.copy()
    row_map["row_position"] = numeric_positions.astype("int64")
    row_map = row_map.sort_values("row_position").reset_index(drop=True)
    if not np.array_equal(
        row_map["row_position"].to_numpy(),
        np.arange(EXPECTED_RECORDS, dtype=np.int64),
    ):
        raise ValueError("row_position is missing, duplicated, or out of range.")

    if not np.array_equal(
        source[LABEL_COLUMN].astype(str).to_numpy(),
        row_map[LABEL_COLUMN].astype(str).to_numpy(),
    ):
        raise ValueError("Source Product labels do not match row mapping order.")

    row_map["_document_key"] = row_map[DOCUMENT_COLUMN].astype(str).str.strip()
    row_map["_component_key"] = row_map[GROUP_COLUMN].astype(str).str.strip()
    component_assignments = component_assignments.copy()
    component_assignments["_component_key"] = (
        component_assignments[GROUP_COLUMN].astype(str).str.strip()
    )

    for col in ("_document_key", "_component_key"):
        if row_map[col].eq("").any():
            raise ValueError(f"Blank identifiers found in {col}.")
    if component_assignments["_component_key"].eq("").any():
        raise ValueError("Blank component IDs in component assignments.")
    if component_assignments["_component_key"].duplicated().any():
        raise ValueError("A component has multiple partition assignments.")

    documents = row_map["_document_key"].nunique()
    components = row_map["_component_key"].nunique()
    classes = source[LABEL_COLUMN].nunique()
    if documents != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENTS:,} narratives; found {documents:,}."
        )
    if components != EXPECTED_COMPONENTS:
        raise ValueError(
            f"Expected {EXPECTED_COMPONENTS:,} components; found {components:,}."
        )
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} Product categories; found {classes}."
        )

    document_check = row_map.groupby("_document_key").agg(
        product_count=(LABEL_COLUMN, "nunique"),
        component_count=("_component_key", "nunique"),
    )
    if (document_check["product_count"] != 1).any():
        raise ValueError("A unique narrative maps to multiple Product labels.")
    if (document_check["component_count"] != 1).any():
        raise ValueError("A unique narrative maps to multiple components.")

    mapping_components = set(row_map["_component_key"].unique())
    assignment_components = set(component_assignments["_component_key"])
    if mapping_components != assignment_components:
        raise ValueError("Component IDs do not match assignment file.")
    if not component_assignments["partition"].isin(PARTITIONS).all():
        raise ValueError("Unexpected partition value in assignment file.")

    print("Input validation passed.")
    print(f"Source records: {len(source):,}")
    print(f"Unique narratives: {documents:,}")
    print(f"Components: {components:,}")
    print(f"Product categories: {classes}")
    return row_map, component_assignments


# -----------------------------------------------------------------------------
# Reconstruct expected partition membership from component assignments
# -----------------------------------------------------------------------------
def reconstruct_partitions(
    row_map: pd.DataFrame,
    component_assignments: pd.DataFrame,
) -> pd.DataFrame:
    print("\nReconstructing expected source-row partitions...")
    lookup = component_assignments.set_index("_component_key")["partition"]
    row_map = row_map.copy()
    row_map["_expected_partition"] = row_map["_component_key"].map(lookup)
    if row_map["_expected_partition"].isna().any():
        raise ValueError("Some source rows have no partition assignment.")

    component_partition_counts = row_map.groupby("_component_key")[
        "_expected_partition"
    ].nunique()
    if (component_partition_counts != 1).any():
        raise RuntimeError("A component crosses partitions.")

    document_partition_counts = row_map.groupby("_document_key")[
        "_expected_partition"
    ].nunique()
    if (document_partition_counts != 1).any():
        raise RuntimeError("A unique narrative crosses partitions.")

    counts = row_map["_expected_partition"].value_counts().to_dict()
    for partition in PARTITIONS:
        actual = counts.get(partition, 0)
        expected = EXPECTED_PARTITION_COUNTS[partition]
        if actual != expected:
            raise ValueError(
                f"Expected assignment count for {partition}: "
                f"{expected:,}; got {actual:,}."
            )
    print("Expected partition membership reconstructed successfully.")
    return row_map


# -----------------------------------------------------------------------------
# Audit each saved partition
# -----------------------------------------------------------------------------
def audit_one_partition(
    partition: str,
    source: pd.DataFrame,
    row_map: pd.DataFrame,
    source_hashes: list[str],
) -> tuple[Counter, dict]:
    print(f"\nAuditing saved {partition}.csv...")

    mask = row_map["_expected_partition"].eq(partition).to_numpy()
    source_positions = np.flatnonzero(mask)
    expected = source.iloc[source_positions]

    if len(expected) != EXPECTED_PARTITION_COUNTS[partition]:
        raise ValueError(
            f"Expected source assignment for {partition} has an unexpected count: "
            f"{len(expected):,}."
        )

    expected_hashes = fingerprints_from_dataframe_as_csv(expected)
    actual_hashes, actual_products = read_saved_csv_records(
        SPLIT_DIR / f"{partition}.csv",
        source.columns.tolist(),
    )

    if len(actual_hashes) != EXPECTED_PARTITION_COUNTS[partition]:
        raise ValueError(
            f"{partition} record count mismatch: expected "
            f"{EXPECTED_PARTITION_COUNTS[partition]:,}, "
            f"found {len(actual_hashes):,}."
        )

    expected_counter = Counter(expected_hashes)
    actual_counter = Counter(actual_hashes)
    missing = sum((expected_counter - actual_counter).values())
    unexpected = sum((actual_counter - expected_counter).values())
    if expected_counter != actual_counter:
        raise RuntimeError(
            f"{partition} does not match its expected source rows. "
            f"Missing occurrences={missing:,}; unexpected occurrences={unexpected:,}."
        )

    expected_products = Counter(
        str(value) for value in expected[LABEL_COLUMN].tolist()
    )
    if expected_products != actual_products:
        raise RuntimeError(f"{partition} Product distribution mismatch.")

    subset_map = row_map.loc[mask]
    result = {
        "records": len(actual_hashes),
        "dataset_percentage": round(
            len(actual_hashes) / EXPECTED_RECORDS * 100, 4
        ),
        "component_count": int(subset_map["_component_key"].nunique()),
        "unique_narrative_groups": int(subset_map["_document_key"].nunique()),
        "product_category_count": len(actual_products),
        "complete_row_sha256_match": True,
        "missing_row_occurrences": 0,
        "unexpected_row_occurrences": 0,
        "product_distribution_match": True,
        "product_distribution": {
            key: int(value) for key, value in sorted(actual_products.items())
        },
    }
    print(
        f"{len(actual_hashes):,} records verified; "
        "complete-row SHA-256 match passed."
    )
    print("Product distribution matched.")
    return actual_counter, result


# -----------------------------------------------------------------------------
# Cross-partition leakage and source conservation
# -----------------------------------------------------------------------------
def audit_cross_partition(
    row_map: pd.DataFrame,
    partition_hashes: dict[str, Counter],
) -> dict:
    print("\nChecking cross-partition separation...")
    component_sets = {}
    document_sets = {}
    row_hash_sets = {}
    for partition in PARTITIONS:
        subset = row_map[row_map["_expected_partition"].eq(partition)]
        component_sets[partition] = set(subset["_component_key"].unique())
        document_sets[partition] = set(subset["_document_key"].unique())
        row_hash_sets[partition] = set(partition_hashes[partition].keys())

    component_overlap = 0
    document_overlap = 0
    exact_row_overlap = 0
    for first, second in combinations(PARTITIONS, 2):
        component_overlap += len(component_sets[first] & component_sets[second])
        document_overlap += len(document_sets[first] & document_sets[second])
        exact_row_overlap += len(row_hash_sets[first] & row_hash_sets[second])

    if component_overlap or document_overlap or exact_row_overlap:
        raise RuntimeError(
            "Cross-partition overlap detected: "
            f"components={component_overlap}, narratives={document_overlap}, "
            f"complete_rows={exact_row_overlap}."
        )

    print("No component, unique narrative, or identical complete-row overlap.")
    return {
        "cross_partition_component_overlap_count": component_overlap,
        "cross_partition_unique_narrative_overlap_count": document_overlap,
        "cross_partition_exact_row_overlap_count": exact_row_overlap,
    }


def verify_conservation(
    source_hashes: list[str],
    partition_hashes: dict[str, Counter],
) -> dict:
    print("\nVerifying complete source-data conservation...")
    source_counter = Counter(source_hashes)
    combined = Counter()
    for counter in partition_hashes.values():
        combined.update(counter)

    missing = sum((source_counter - combined).values())
    unexpected = sum((combined - source_counter).values())
    if source_counter != combined:
        raise RuntimeError(
            f"Dataset conservation failed: missing={missing:,}; "
            f"unexpected={unexpected:,}."
        )

    saved_count = sum(sum(counter.values()) for counter in partition_hashes.values())
    print(f"Source records: {len(source_hashes):,}")
    print(f"Saved partition records: {saved_count:,}")
    print("Complete source-row conservation passed.")
    return {
        "source_records": len(source_hashes),
        "saved_partition_records": saved_count,
        "exact_source_row_conservation": True,
        "missing_row_occurrences": 0,
        "unexpected_row_occurrences": 0,
        "row_fingerprint_algorithm": "SHA-256",
        "row_fingerprint_scope": "All columns serialized using pandas to_csv conventions",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("FINROUTE AI - INDEPENDENT SAVED SPLIT AUDIT V3")
    print("=" * 72)

    for path in (SOURCE_FILE, ROW_MAPPING_FILE, COMPONENT_ASSIGNMENTS_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
    if not SPLIT_DIR.is_dir():
        raise FileNotFoundError(f"V3 split directory not found: {SPLIT_DIR}")
    for partition in PARTITIONS:
        split_path = SPLIT_DIR / f"{partition}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
    if REPORT_FILE.exists():
        raise FileExistsError(
            f"Audit report already exists: {REPORT_FILE}. Refusing to overwrite it."
        )

    source_sha256_before = sha256_file(SOURCE_FILE)
    split_sha256_before = {
        partition: sha256_file(SPLIT_DIR / f"{partition}.csv")
        for partition in PARTITIONS
    }

    print("\nLoading source and audit mappings...")
    # IMPORTANT: use the same pandas parsing behavior as the v3 creation script.
    source = pd.read_csv(SOURCE_FILE, low_memory=False)
    row_map = read_mapping_csv(ROW_MAPPING_FILE)
    assignments = read_mapping_csv(COMPONENT_ASSIGNMENTS_FILE)

    row_map, assignments = validate_inputs(source, row_map, assignments)
    row_map = reconstruct_partitions(row_map, assignments)

    print("\nCalculating expected source-row SHA-256 fingerprints...")
    source_hashes = fingerprints_from_dataframe_as_csv(source)
    if len(source_hashes) != EXPECTED_RECORDS:
        raise RuntimeError("Source fingerprint count mismatch.")
    print(f"Calculated {len(source_hashes):,} source row fingerprints.")

    partition_hashes: dict[str, Counter] = {}
    partition_reports: dict[str, dict] = {}
    for partition in PARTITIONS:
        hashes, details = audit_one_partition(
            partition, source, row_map, source_hashes
        )
        partition_hashes[partition] = hashes
        partition_reports[partition] = details

    separation = audit_cross_partition(row_map, partition_hashes)
    conservation = verify_conservation(source_hashes, partition_hashes)

    # Ensure the source and all saved partitions stayed byte-for-byte stable
    # throughout the independent audit. Record the digests for training preflight.
    source_sha256_after = sha256_file(SOURCE_FILE)
    split_sha256_after = {
        partition: sha256_file(SPLIT_DIR / f"{partition}.csv")
        for partition in PARTITIONS
    }
    if source_sha256_before != source_sha256_after:
        raise RuntimeError("Source dataset changed while the audit was running.")
    if split_sha256_before != split_sha256_after:
        raise RuntimeError("A saved split CSV changed while the audit was running.")

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v3",
        "audit_type": "Independent saved-file audit",
        "source_dataset": SOURCE_FILE.name,
        "source_dataset_sha256": source_sha256_after,
        "saved_split_sha256": split_sha256_after,
        "component_threshold": "Jaccard = 1.00",
        "component_column": GROUP_COLUMN,
        "source_records": EXPECTED_RECORDS,
        "unique_narrative_groups": EXPECTED_DOCUMENTS,
        "components": EXPECTED_COMPONENTS,
        "product_categories": EXPECTED_CLASSES,
        "partitions": partition_reports,
        "cross_partition_checks": separation,
        "dataset_conservation": conservation,
        "final_checks": {
            "all_three_saved_files_audited": True,
            "complete_rows_match_expected_source_rows": True,
            "all_product_distributions_match": True,
            "all_components_remain_intact": True,
            "all_unique_narratives_remain_intact": True,
            "no_identical_complete_rows_cross_partitions": True,
            "all_original_records_preserved": True,
            "raw_narratives_exported": False,
            "complaint_ids_exported": False,
            "source_dataset_modified": False,
            "split_files_modified": False,
            "test_set_evaluated": False,
            "model_retrained": False,
        },
    }

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwriting.
    with REPORT_FILE.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("INDEPENDENT V3 SAVED SPLIT AUDIT COMPLETED")
    print("=" * 72)
    for partition in PARTITIONS:
        print(
            f"{partition.upper()}: {partition_reports[partition]['records']:,} "
            "records - complete-row SHA-256 verified"
        )
    print("\nComponent separation: PASS")
    print("Unique narrative separation: PASS")
    print("Complete source-row conservation: PASS")
    print("Product distribution verification: PASS")
    print("\nSource dataset and split CSVs were not modified.")
    print("Test set was not evaluated. Model was not retrained.")
    print(f"\nAudit report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
