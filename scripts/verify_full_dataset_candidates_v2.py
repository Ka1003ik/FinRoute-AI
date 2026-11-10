
import gc
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative


INPUT_PATH = (
    ROOT / "data" / "processed"
    / "cfpb_2026_modeling_v2.csv"
)

CANDIDATE_DB_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_candidate_pairs_v2.sqlite"
)

DOCUMENT_MAP_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_document_map_v2.csv"
)

OUTPUT_DB_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_verified_edges_v2.sqlite"
)

OUTPUT_REPORT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_candidate_verification_v2.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"

EXPECTED_ROWS = 121114
EXPECTED_UNIQUE_DOCUMENTS = 108902
EXPECTED_CANDIDATE_PAIRS = 588734

SHINGLE_SIZE = 3
BATCH_SIZE = 10000
PRINT_INTERVAL = 50000

TOKEN_PATTERN = __import__("re").compile(r"(?u)\b\w+\b")

THRESHOLDS = {
    "0.70": 0.70,
    "0.85": 0.85,
    "0.95": 0.95,
    "0.99": 0.99,
    "1.00": 1.00,
}


def token_shingle_hashes(tokens):
    """Reproduce the existing BLAKE2b word-trigram hashes."""

    import hashlib

    hashes = []

    for index in range(len(tokens) - SHINGLE_SIZE + 1):
        shingle = "\x1f".join(
            tokens[index:index + SHINGLE_SIZE]
        )

        digest = hashlib.blake2b(
            shingle.encode("utf-8"),
            digest_size=8,
        ).digest()

        hashes.append(
            int.from_bytes(digest, byteorder="big")
        )

    if not hashes:
        return np.array([], dtype=np.uint64)

    return np.unique(
        np.asarray(hashes, dtype=np.uint64)
    )


def intersection_size(first, second):
    """Calculate intersection size of sorted unique arrays."""

    if len(first) == 0 or len(second) == 0:
        return 0

    if len(first) > len(second):
        first, second = second, first

    positions = np.searchsorted(
        second,
        first,
    )

    valid = positions < len(second)

    return int(
        np.count_nonzero(
            second[positions[valid]] == first[valid]
        )
    )


def verify_mapping(mapping, codes, labels):
    """Confirm the document map matches the source dataset."""

    if len(mapping) != len(codes):
        raise ValueError(
            "Document mapping row count does not match source."
        )

    expected_positions = np.arange(
        len(codes),
        dtype=np.int64,
    )

    if not np.array_equal(
        mapping["row_position"].to_numpy(dtype=np.int64),
        expected_positions,
    ):
        raise ValueError(
            "Document mapping row positions are inconsistent."
        )

    if not np.array_equal(
        mapping["unique_document_id"].to_numpy(dtype=np.int64),
        codes,
    ):
        raise ValueError(
            "Document IDs do not match the source factorization."
        )

    if not np.array_equal(
        mapping[LABEL_COLUMN].astype(str).to_numpy(),
        labels.astype(str),
    ):
        raise ValueError(
            "Document mapping Product labels do not match source."
        )


def main():
    print("=" * 68)
    print("FINROUTE AI - EXACT CANDIDATE SIMILARITY VERIFICATION")
    print("=" * 68)

    required_files = [
        INPUT_PATH,
        CANDIDATE_DB_PATH,
        DOCUMENT_MAP_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_DB_PATH.exists():
        raise FileExistsError(OUTPUT_DB_PATH)

    if OUTPUT_REPORT_PATH.exists():
        raise FileExistsError(OUTPUT_REPORT_PATH)

    print("\nReading source dataset...")

    frame = pd.read_csv(
        INPUT_PATH,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"found {len(frame):,}."
        )

    labels = frame[LABEL_COLUMN].to_numpy(dtype=object)

    if pd.Series(labels).str.strip().eq("").any():
        raise ValueError("Empty Product labels detected.")

    print("\nRebuilding global unique narrative IDs...")

    cleaned = frame[TEXT_COLUMN].map(clean_narrative)

    if cleaned.eq("").any():
        raise ValueError("Empty cleaned narratives detected.")

    codes, unique_texts = pd.factorize(
        cleaned,
        sort=False,
    )

    if len(unique_texts) != EXPECTED_UNIQUE_DOCUMENTS:
        raise ValueError(
            "Unique-document count does not match the "
            "candidate-generation run."
        )

    print("\nValidating original-record mapping...")

    mapping = pd.read_csv(
        DOCUMENT_MAP_PATH,
        dtype={
            "row_position": "int64",
            "unique_document_id": "int64",
            LABEL_COLUMN: str,
        },
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    verify_mapping(mapping, codes, labels)

    print("Document mapping validated.")

    del mapping
    del cleaned
    del frame
    del labels
    gc.collect()

    print("\nReconstructing shingle sets...")

    shingle_sets = []

    for document_id, clean_text in enumerate(unique_texts):
        tokens = TOKEN_PATTERN.findall(str(clean_text))

        shingle_sets.append(
            token_shingle_hashes(tokens)
        )

        if (document_id + 1) % 10000 == 0:
            print(
                f"Documents processed: {document_id + 1:,}"
            )

    del unique_texts
    del codes
    gc.collect()

    print("\nChecking candidate database integrity...")

    source_connection = sqlite3.connect(
        str(CANDIDATE_DB_PATH)
    )

    integrity = source_connection.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    if integrity != "ok":
        source_connection.close()
        raise RuntimeError(
            f"SQLite integrity check failed: {integrity}"
        )

    candidate_count = source_connection.execute(
        "SELECT COUNT(*) FROM candidate_pairs"
    ).fetchone()[0]

    if candidate_count != EXPECTED_CANDIDATE_PAIRS:
        source_connection.close()
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATE_PAIRS:,} candidates, "
            f"found {candidate_count:,}."
        )

    invalid_count = source_connection.execute(
        """
        SELECT COUNT(*)
        FROM candidate_pairs
        WHERE a < 0
           OR b < 0
           OR a >= ?
           OR b >= ?
           OR a >= b
        """,
        (
            EXPECTED_UNIQUE_DOCUMENTS,
            EXPECTED_UNIQUE_DOCUMENTS,
        ),
    ).fetchone()[0]

    if invalid_count:
        source_connection.close()
        raise ValueError(
            f"Invalid candidate pairs found: {invalid_count}"
        )

    print(f"Candidate pairs validated: {candidate_count:,}")

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="finroute_candidate_verification_",
        dir=str(OUTPUT_DB_PATH.parent),
    )

    temp_dir = Path(temporary_directory.name)

    temp_db_path = (
        temp_dir / "verified_edges.sqlite"
    )

    temp_report_path = (
        temp_dir / "verification_report.json"
    )

    target_connection = sqlite3.connect(
        str(temp_db_path)
    )

    target_connection.execute(
        "PRAGMA journal_mode=OFF"
    )

    target_connection.execute(
        "PRAGMA synchronous=OFF"
    )

    target_connection.execute(
        """
        CREATE TABLE verified_edges (
            a INTEGER NOT NULL,
            b INTEGER NOT NULL,
            jaccard_similarity REAL NOT NULL,
            containment_similarity REAL NOT NULL,
            PRIMARY KEY (a, b),
            CHECK (a < b)
        ) WITHOUT ROWID
        """
    )

    threshold_counts = {
        name: 0
        for name in THRESHOLDS
    }

    verified_count = 0
    below_review_threshold = 0

    insert_batch = []

    print("\nCalculating exact Jaccard similarities...")

    cursor = source_connection.execute(
        """
        SELECT a, b
        FROM candidate_pairs
        ORDER BY a, b
        """
    )

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)

        if not rows:
            break

        for first_id, second_id in rows:
            first_set = shingle_sets[first_id]
            second_set = shingle_sets[second_id]

            intersection = intersection_size(
                first_set,
                second_set,
            )

            union = (
                len(first_set)
                + len(second_set)
                - intersection
            )

            jaccard = (
                intersection / union
                if union else 0.0
            )

            smaller_size = min(
                len(first_set),
                len(second_set),
            )

            containment = (
                intersection / smaller_size
                if smaller_size else 0.0
            )

            verified_count += 1

            for name, threshold in THRESHOLDS.items():
                if jaccard >= threshold:
                    threshold_counts[name] += 1

            if jaccard >= 0.70:
                insert_batch.append((
                    first_id,
                    second_id,
                    jaccard,
                    containment,
                ))
            else:
                below_review_threshold += 1

            if len(insert_batch) >= BATCH_SIZE:
                target_connection.executemany(
                    """
                    INSERT INTO verified_edges
                    VALUES (?, ?, ?, ?)
                    """,
                    insert_batch,
                )

                target_connection.commit()
                insert_batch.clear()

        if verified_count % PRINT_INTERVAL < BATCH_SIZE:
            print(
                f"Candidate pairs verified: "
                f"{verified_count:,}/{candidate_count:,}"
            )

    if insert_batch:
        target_connection.executemany(
            """
            INSERT INTO verified_edges
            VALUES (?, ?, ?, ?)
            """,
            insert_batch,
        )

    target_connection.commit()

    retained_count = target_connection.execute(
        "SELECT COUNT(*) FROM verified_edges"
    ).fetchone()[0]

    if retained_count != threshold_counts["0.70"]:
        raise RuntimeError(
            "Verified edge count does not match "
            "the >= 0.70 threshold count."
        )

    target_connection.close()
    source_connection.close()

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",

        "source_records": EXPECTED_ROWS,
        "unique_documents": EXPECTED_UNIQUE_DOCUMENTS,

        "candidate_pairs_received": int(candidate_count),
        "candidate_pairs_verified": int(verified_count),

        "exact_similarity_method": (
            "Intersection and union of unique BLAKE2b "
            "word-trigram hash sets"
        ),

        "threshold_counts": threshold_counts,

        "pairs_below_0.70": int(
            below_review_threshold
        ),

        "verified_edges_retained_at_0.70": int(
            retained_count
        ),

        "candidate_mapping_validated": True,
        "candidate_database_integrity": integrity,

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "company_names_exported": False,

        "dataset_modified": False,
        "records_deleted_or_moved": False,
        "model_retrained": False,
        "test_set_evaluated": False,
    }

    temp_report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    os.replace(
        temp_db_path,
        OUTPUT_DB_PATH,
    )

    os.replace(
        temp_report_path,
        OUTPUT_REPORT_PATH,
    )

    temporary_directory.cleanup()

    print("\n" + "=" * 68)
    print("EXACT SIMILARITY VERIFICATION RESULTS")
    print("=" * 68)

    print(f"Candidate pairs verified: {verified_count:,}")

    print("\n=== VERIFIED PAIRS BY THRESHOLD ===")

    print(json.dumps(
        threshold_counts,
        indent=4,
    ))

    print(
        f"\nPairs below 0.70: "
        f"{below_review_threshold:,}"
    )

    print(
        f"Verified edges retained: "
        f"{retained_count:,}"
    )

    print("\nVerified edge database:")
    print(OUTPUT_DB_PATH)

    print("\nVerification report:")
    print(OUTPUT_REPORT_PATH)

    print("\nNo source data modified.")
    print("No records deleted or moved.")
    print("No model retrained.")
    print("EXACT SIMILARITY VERIFICATION COMPLETED.")


if __name__ == "__main__":
    main()
