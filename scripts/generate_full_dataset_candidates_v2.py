
import csv
import gc
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile

from collections import Counter, defaultdict
from itertools import combinations
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

OUTPUT_DIR = ROOT / "outputs" / "leakage_audit"

PAIR_DB_PATH = (
    OUTPUT_DIR / "full_dataset_candidate_pairs_v2.sqlite"
)

DOCUMENT_MAP_PATH = (
    OUTPUT_DIR / "full_dataset_document_map_v2.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "full_dataset_candidate_generation_v2.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"

EXPECTED_ROWS = 121114
EXPECTED_UNIQUE_DOCUMENTS = 108902

RANDOM_STATE = 42

NUM_PERMUTATIONS = 64
BANDS = 16
ROWS_PER_BAND = 4

MAX_BUCKET_SIZE_REFERENCE = 200

SHINGLE_SIZE = 3
MIN_SHINGLES = 5

BATCH_SIZE = 20000

TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def token_shingle_hashes(tokens):
    """Hash consecutive word trigrams using the existing method."""

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


def create_coefficients():
    """Generate the same reproducible MinHash coefficients."""

    rng = np.random.default_rng(RANDOM_STATE)

    a = rng.integers(
        1,
        np.iinfo(np.uint64).max,
        size=NUM_PERMUTATIONS,
        dtype=np.uint64,
    )

    b = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=NUM_PERMUTATIONS,
        dtype=np.uint64,
    )

    a |= np.uint64(1)

    return a, b


def compute_minhash(shingle_hashes, coefficients):
    """Compute MinHash using the existing uint64 implementation."""

    a, b = coefficients

    with np.errstate(over="ignore"):
        transformed = (
            a[:, np.newaxis] * shingle_hashes[np.newaxis, :]
            + b[:, np.newaxis]
        )

    return transformed.min(axis=1)


def flush_pairs(connection, batch):
    """Insert candidate pairs and deduplicate using SQLite."""

    if batch:
        connection.executemany(
            """
            INSERT OR IGNORE INTO candidate_pairs (a, b)
            VALUES (?, ?)
            """,
            batch,
        )

        batch.clear()


def add_short_document_candidates(
    connection,
    shingle_arrays,
    short_document_ids,
    eligible_document_ids,
):
    """
    Include short documents that the normal LSH process skips.

    At Jaccard >= 0.70, a document with four shingles can only
    match an eligible document with five shingles if all four
    of its shingles are shared.

    Short documents are also compared with one another.
    """

    batch = []

    # Compare every pair of short documents.
    short_pair_opportunities = 0

    for first_id, second_id in combinations(
        short_document_ids,
        2,
    ):
        a, b = sorted((first_id, second_id))
        batch.append((a, b))
        short_pair_opportunities += 1

        if len(batch) >= BATCH_SIZE:
            flush_pairs(connection, batch)

    # Build an inverted index for eligible documents containing
    # exactly five unique shingle hashes.
    inverted_five_shingle = defaultdict(list)

    for document_id in eligible_document_ids:
        hashes = shingle_arrays[document_id]

        if len(hashes) == 5:
            for value in hashes:
                inverted_five_shingle[int(value)].append(
                    document_id
                )

    short_to_eligible_candidates = 0

    for short_id in short_document_ids:
        short_hashes = shingle_arrays[short_id]

        if len(short_hashes) != 4:
            continue

        overlap_counts = Counter()

        for value in short_hashes:
            for candidate_id in inverted_five_shingle.get(
                int(value),
                [],
            ):
                overlap_counts[candidate_id] += 1

        for candidate_id, overlap in overlap_counts.items():
            if overlap == 4:
                a, b = sorted((short_id, candidate_id))
                batch.append((a, b))

                short_to_eligible_candidates += 1

                if len(batch) >= BATCH_SIZE:
                    flush_pairs(connection, batch)

    flush_pairs(connection, batch)

    return {
        "short_documents": len(short_document_ids),
        "short_to_short_pair_opportunities": (
            short_pair_opportunities
        ),
        "short_to_eligible_candidate_pairs_added": (
            short_to_eligible_candidates
        ),
    }


def main():
    print("=" * 68)
    print("FINROUTE AI - FULL DATASET CANDIDATE GENERATION")
    print("=" * 68)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    output_paths = [
        PAIR_DB_PATH,
        DOCUMENT_MAP_PATH,
        REPORT_PATH,
    ]

    for path in output_paths:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nReading the complete v2 modelling dataset...")

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

    print(f"Source records: {len(frame):,}")

    print("\nApplying canonical cleaning...")

    cleaned = frame[TEXT_COLUMN].map(clean_narrative)

    if cleaned.eq("").any():
        raise ValueError(
            "Empty cleaned narratives detected."
        )

    print("\nGrouping globally identical narratives...")

    codes, unique_texts = pd.factorize(
        cleaned,
        sort=False,
    )

    unique_count = len(unique_texts)

    if unique_count != EXPECTED_UNIQUE_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_DOCUMENTS:,} unique narratives, "
            f"found {unique_count:,}."
        )

    record_counts = np.bincount(
        codes,
        minlength=unique_count,
    )

    doc_labels = np.empty(
        unique_count,
        dtype=object,
    )

    label_seen = np.zeros(
        unique_count,
        dtype=bool,
    )

    label_conflict_groups = set()

    for position, document_id in enumerate(codes):
        label = labels[position]

        if not label_seen[document_id]:
            doc_labels[document_id] = label
            label_seen[document_id] = True

        elif doc_labels[document_id] != label:
            label_conflict_groups.add(int(document_id))

    if label_conflict_groups:
        raise ValueError(
            "Exact cleaned-text groups contain different Product labels. "
            "Stop and investigate before building the graph."
        )

    repeated_groups = int(
        np.sum(record_counts > 1)
    )

    repeated_records = int(
        record_counts.sum() - unique_count
    )

    print(f"Unique narratives: {unique_count:,}")
    print(f"Repeated groups: {repeated_groups:,}")
    print(f"Additional repeated records: {repeated_records:,}")
    print("Exact-text Product conflicts: 0")

    del cleaned
    del frame
    gc.collect()

    print("\nGenerating MinHash signatures and buckets...")

    coefficients = create_coefficients()

    buckets = [
        defaultdict(list)
        for _ in range(BANDS)
    ]

    shingle_arrays = [None] * unique_count

    short_document_ids = []
    eligible_document_ids = []

    for document_id, clean_text in enumerate(unique_texts):
        tokens = TOKEN_PATTERN.findall(str(clean_text))

        shingle_hashes = token_shingle_hashes(tokens)

        shingle_arrays[document_id] = shingle_hashes

        if len(shingle_hashes) < MIN_SHINGLES:
            short_document_ids.append(document_id)
            continue

        signature = compute_minhash(
            shingle_hashes,
            coefficients,
        )

        eligible_document_ids.append(document_id)

        for band_index in range(BANDS):
            start = band_index * ROWS_PER_BAND
            end = start + ROWS_PER_BAND

            key = tuple(
                int(value)
                for value in signature[start:end]
            )

            buckets[band_index][key].append(
                document_id
            )

        if (document_id + 1) % 10000 == 0:
            print(
                f"Documents processed: {document_id + 1:,}"
            )

    del unique_texts
    gc.collect()

    print("\nProfiling buckets before pair generation...")

    bucket_sizes = []

    oversized_buckets = 0
    oversized_memberships = 0
    total_pair_opportunities = 0

    for band in buckets:
        for members in band.values():
            size = len(members)

            if size < 2:
                continue

            bucket_sizes.append(size)

            total_pair_opportunities += (
                size * (size - 1) // 2
            )

            if size > MAX_BUCKET_SIZE_REFERENCE:
                oversized_buckets += 1
                oversized_memberships += size

    print(
        f"Buckets with at least two documents: "
        f"{len(bucket_sizes):,}"
    )

    print(
        f"Pair opportunities across all bands: "
        f"{total_pair_opportunities:,}"
    )

    print(
        f"Oversized buckets (not skipped): "
        f"{oversized_buckets:,}"
    )

    print("\nPreparing SQLite candidate database...")

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="finroute_full_dataset_candidates_",
        dir=str(OUTPUT_DIR),
    )

    temp_dir = Path(temporary_directory.name)

    temp_db_path = temp_dir / "candidate_pairs.sqlite"
    temp_map_path = temp_dir / "document_map.csv"
    temp_report_path = temp_dir / "generation_report.json"

    connection = sqlite3.connect(str(temp_db_path))

    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-100000")

        connection.execute(
            """
            CREATE TABLE candidate_pairs (
                a INTEGER NOT NULL,
                b INTEGER NOT NULL,
                PRIMARY KEY (a, b)
            ) WITHOUT ROWID
            """
        )

        batch = []

        print("\nGenerating pairs from every LSH bucket...")

        generated_pair_opportunities = 0

        connection.execute("BEGIN")

        for band_index, band in enumerate(buckets):
            for members in band.values():
                size = len(members)

                if size < 2:
                    continue

                generated_pair_opportunities += (
                    size * (size - 1) // 2
                )

                # No oversized bucket is skipped.
                # SQLite removes duplicate pairs across bands.
                for i in range(size):
                    first_id = members[i]

                    for j in range(i + 1, size):
                        second_id = members[j]

                        batch.append(
                            (first_id, second_id)
                        )

                        if len(batch) >= BATCH_SIZE:
                            flush_pairs(
                                connection,
                                batch,
                            )

            print(
                f"LSH band {band_index + 1}/{BANDS} complete."
            )

        flush_pairs(connection, batch)

        print("\nAdding short-document candidate pairs...")

        short_statistics = add_short_document_candidates(
            connection,
            shingle_arrays,
            short_document_ids,
            eligible_document_ids,
        )

        connection.commit()

        unique_candidate_pairs = connection.execute(
            "SELECT COUNT(*) FROM candidate_pairs"
        ).fetchone()[0]

        connection.execute(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        metadata = {
            "project": "FinRoute AI",
            "dataset_version": "v2",
            "source_records": EXPECTED_ROWS,
            "unique_documents": unique_count,
            "unique_candidate_pairs": unique_candidate_pairs,
            "candidate_pair_method": "16-band MinHash LSH",
        }

        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                (key, str(value))
                for key, value in metadata.items()
            ],
        )

        connection.commit()

        print("\nWriting original-record mapping...")

        with temp_map_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "row_position",
                "unique_document_id",
                "Product",
            ])

            for row_position, document_id in enumerate(codes):
                writer.writerow([
                    row_position,
                    int(document_id),
                    labels[row_position],
                ])

        report = {
            "project": "FinRoute AI",
            "dataset_version": "v2",

            "source_records": EXPECTED_ROWS,
            "unique_cleaned_narratives": unique_count,

            "exact_duplicate_statistics": {
                "repeated_narrative_groups": repeated_groups,
                "additional_repeated_records": repeated_records,
                "different_product_label_groups": 0,
            },

            "lsh_configuration": {
                "minhash_permutations": NUM_PERMUTATIONS,
                "bands": BANDS,
                "rows_per_band": ROWS_PER_BAND,
                "random_state": RANDOM_STATE,
                "shingle_type": "Consecutive word trigrams",
                "shingle_hash": "BLAKE2b 64-bit",
                "minimum_shingles": MIN_SHINGLES,
            },

            "eligible_documents": len(
                eligible_document_ids
            ),

            "short_document_statistics": short_statistics,

            "bucket_statistics": {
                "buckets_with_at_least_two_documents": len(
                    bucket_sizes
                ),
                "total_pair_opportunities_across_bands": (
                    generated_pair_opportunities
                ),
                "oversized_buckets_included": oversized_buckets,
                "oversized_bucket_memberships": (
                    oversized_memberships
                ),
            },

            "candidate_pair_statistics": {
                "unique_candidate_pairs_in_sqlite": int(
                    unique_candidate_pairs
                ),
                "candidate_pairs_are_deduplicated": True,
                "within_split_relationships_included": True,
                "cross_split_relationships_included": True,
            },

            "outputs": {
                "candidate_database": str(PAIR_DB_PATH),
                "document_mapping": str(DOCUMENT_MAP_PATH),
            },

            "raw_narratives_exported": False,
            "complaint_ids_exported": False,
            "company_names_exported": False,
            "dataset_modified": False,
            "records_deleted_or_moved": False,
            "model_retrained": False,
        }

        temp_report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

    finally:
        connection.close()

    # Publish output files only after successful completion.
    os.replace(temp_db_path, PAIR_DB_PATH)
    os.replace(temp_map_path, DOCUMENT_MAP_PATH)
    os.replace(temp_report_path, REPORT_PATH)

    temporary_directory.cleanup()

    print("\n" + "=" * 68)
    print("FULL DATASET CANDIDATE GENERATION RESULTS")
    print("=" * 68)

    print(f"Unique documents: {unique_count:,}")
    print(
        f"Eligible for LSH: {len(eligible_document_ids):,}"
    )
    print(
        f"Short documents: {len(short_document_ids):,}"
    )
    print(
        f"Pair opportunities: {generated_pair_opportunities:,}"
    )
    print(
        f"Unique candidate pairs: {unique_candidate_pairs:,}"
    )
    print(
        f"Oversized buckets included: {oversized_buckets:,}"
    )

    print("\nCandidate database:")
    print(PAIR_DB_PATH)

    print("\nDocument mapping:")
    print(DOCUMENT_MAP_PATH)

    print("\nSummary report:")
    print(REPORT_PATH)

    print("\nNo dataset changes.")
    print("No records deleted or moved.")
    print("FULL DATASET CANDIDATE GENERATION COMPLETED.")


if __name__ == "__main__":
    main()
