
import gc
import json
import re
import sys
from collections import defaultdict
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

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "full_dataset_lsh_bucket_profile_v2.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"

EXPECTED_ROWS = 121114

RANDOM_STATE = 42
NUM_PERMUTATIONS = 64
BANDS = 16
ROWS_PER_BAND = 4

MAX_BUCKET_SIZE = 200
SHINGLE_SIZE = 3
MIN_SHINGLES = 5

TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def token_shingle_hashes(tokens):
    """Hash consecutive three-word shingles."""

    hashes = []

    for index in range(
        len(tokens) - SHINGLE_SIZE + 1
    ):
        shingle = "\x1f".join(
            tokens[index:index + SHINGLE_SIZE]
        )

        digest = __import__("hashlib").blake2b(
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
    """Use exactly the same reproducible MinHash coefficients."""

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
    """Compute the MinHash signature using uint64 arithmetic."""

    a, b = coefficients

    with np.errstate(over="ignore"):
        transformed = (
            a[:, np.newaxis] * shingle_hashes[np.newaxis, :]
            + b[:, np.newaxis]
        )

    return transformed.min(axis=1)


def bucket_size_category(size):
    if size <= 5:
        return "2_to_5"
    if size <= 10:
        return "6_to_10"
    if size <= 50:
        return "11_to_50"
    if size <= 100:
        return "51_to_100"
    if size <= 200:
        return "101_to_200"

    return "above_200"


def main():
    print("=" * 68)
    print("FINROUTE AI - FULL DATASET LSH BUCKET PROFILE")
    print("=" * 68)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    print("\nReading the v2 modelling dataset...")

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

    if frame[LABEL_COLUMN].str.strip().eq("").any():
        raise ValueError("Empty Product labels detected.")

    print(f"Source records: {len(frame):,}")

    print("\nApplying canonical narrative cleaning...")

    frame["_clean_text"] = frame[TEXT_COLUMN].map(
        clean_narrative
    )

    empty_narratives = int(
        frame["_clean_text"].eq("").sum()
    )

    if empty_narratives:
        raise ValueError(
            f"Found {empty_narratives} empty narratives. "
            "Resolve the source-data discrepancy before continuing."
        )

    print("\nChecking exact cleaned-text groups...")

    group_stats = frame.groupby(
        "_clean_text",
        sort=False,
    ).agg(
        record_count=(LABEL_COLUMN, "size"),
        product_count=(LABEL_COLUMN, "nunique"),
    )

    unique_narratives = len(group_stats)

    repeated_groups = int(
        (group_stats["record_count"] > 1).sum()
    )

    repeated_records = int(
        group_stats["record_count"].sum()
        - unique_narratives
    )

    conflicting_groups = int(
        (group_stats["product_count"] > 1).sum()
    )

    conflicting_records = int(
        group_stats.loc[
            group_stats["product_count"] > 1,
            "record_count",
        ].sum()
    )

    print(f"Unique cleaned narratives: {unique_narratives:,}")
    print(f"Repeated narrative groups: {repeated_groups:,}")
    print(f"Additional repeated records: {repeated_records:,}")
    print(f"Groups with different Product labels: {conflicting_groups:,}")

    # Keep one representative for each globally unique narrative.
    # Original records and their labels remain untouched.
    unique_frame = frame.drop_duplicates(
        subset="_clean_text",
        keep="first",
    )[["_clean_text"]].copy()

    del frame
    del group_stats
    gc.collect()

    print("\nPreparing MinHash buckets...")

    coefficients = create_coefficients()

    buckets = [
        defaultdict(list)
        for _ in range(BANDS)
    ]

    skipped_short = 0
    eligible_documents = 0

    for unique_id, clean_text in enumerate(
        unique_frame["_clean_text"]
    ):
        tokens = TOKEN_PATTERN.findall(clean_text)

        shingle_hashes = token_shingle_hashes(tokens)

        if len(shingle_hashes) < MIN_SHINGLES:
            skipped_short += 1
            continue

        signature = compute_minhash(
            shingle_hashes,
            coefficients,
        )

        eligible_id = eligible_documents
        eligible_documents += 1

        for band_index in range(BANDS):
            start = band_index * ROWS_PER_BAND
            end = start + ROWS_PER_BAND

            key = tuple(
                int(value)
                for value in signature[start:end]
            )

            buckets[band_index][key].append(
                eligible_id
            )

        if (unique_id + 1) % 10000 == 0:
            print(
                f"Documents processed: {unique_id + 1:,}"
            )

    del unique_frame
    gc.collect()

    print("\nProfiling bucket sizes...")

    bucket_sizes = []

    oversized_buckets = 0
    oversized_memberships = 0
    oversized_pair_opportunities = 0

    bounded_buckets = 0
    bounded_pair_opportunities = 0

    total_pair_opportunities = 0

    category_counts = {
        "2_to_5": 0,
        "6_to_10": 0,
        "11_to_50": 0,
        "51_to_100": 0,
        "101_to_200": 0,
        "above_200": 0,
    }

    for band in buckets:
        for members in band.values():
            size = len(members)

            if size < 2:
                continue

            bucket_sizes.append(size)

            pair_opportunities = (
                size * (size - 1) // 2
            )

            total_pair_opportunities += pair_opportunities

            category_counts[
                bucket_size_category(size)
            ] += 1

            if size > MAX_BUCKET_SIZE:
                oversized_buckets += 1
                oversized_memberships += size

                oversized_pair_opportunities += (
                    pair_opportunities
                )
            else:
                bounded_buckets += 1

                bounded_pair_opportunities += (
                    pair_opportunities
                )

    if bucket_sizes:
        percentiles = {
            "p50": float(np.percentile(bucket_sizes, 50)),
            "p90": float(np.percentile(bucket_sizes, 90)),
            "p95": float(np.percentile(bucket_sizes, 95)),
            "p99": float(np.percentile(bucket_sizes, 99)),
        }

        largest_buckets = sorted(
            bucket_sizes,
            reverse=True,
        )[:20]

    else:
        percentiles = {
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }

        largest_buckets = []

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",

        "audit_type": (
            "Full-dataset MinHash/LSH bucket profiling; "
            "no candidate pairs generated"
        ),

        "source_records": EXPECTED_ROWS,
        "unique_cleaned_narratives": unique_narratives,

        "exact_duplicate_statistics": {
            "repeated_narrative_groups": repeated_groups,
            "additional_repeated_records": repeated_records,
            "different_product_label_groups": conflicting_groups,
            "records_in_different_label_groups": conflicting_records,
        },

        "documents_eligible_for_lsh": eligible_documents,
        "short_documents_skipped": skipped_short,

        "method": {
            "normalization": (
                "src.preprocessing.clean_text.clean_narrative"
            ),
            "shingle_type": "Consecutive word trigrams",
            "shingle_hash": "BLAKE2b, 64-bit",
            "minhash_permutations": NUM_PERMUTATIONS,
            "lsh_bands": BANDS,
            "rows_per_band": ROWS_PER_BAND,
            "random_state": RANDOM_STATE,
            "minimum_shingles": MIN_SHINGLES,
            "maximum_bucket_size_reference": MAX_BUCKET_SIZE,
        },

        "bucket_statistics": {
            "buckets_with_at_least_two_documents": len(bucket_sizes),
            "bucket_size_categories": category_counts,
            "bucket_size_percentiles": percentiles,
            "largest_20_bucket_sizes": largest_buckets,
        },

        "pair_opportunity_estimates": {
            "total_bucket_pair_opportunities": (
                total_pair_opportunities
            ),
            "within_size_limit_buckets": bounded_buckets,
            "within_size_limit_pair_opportunities": (
                bounded_pair_opportunities
            ),
            "oversized_buckets": oversized_buckets,
            "oversized_bucket_memberships": (
                oversized_memberships
            ),
            "oversized_bucket_pair_opportunities": (
                oversized_pair_opportunities
            ),
        },

        "important_limitation": (
            "Pair opportunities are summed across LSH bands. "
            "The same document pair may occur in multiple bands. "
            "These counts are not unique candidate-pair counts."
        ),

        "candidate_pairs_generated": False,
        "connected_components_generated": False,
        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "company_names_exported": False,
        "dataset_modified": False,
        "records_deleted_or_moved": False,
        "model_retrained": False,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n=== FULL DATASET PROFILE ===")
    print(f"Unique narratives: {unique_narratives:,}")
    print(f"Eligible for LSH: {eligible_documents:,}")
    print(f"Short documents skipped: {skipped_short:,}")

    print("\n=== BUCKET STATISTICS ===")
    print(json.dumps(
        report["bucket_statistics"],
        indent=4,
    ))

    print("\n=== PAIR OPPORTUNITY ESTIMATES ===")
    print(json.dumps(
        report["pair_opportunity_estimates"],
        indent=4,
    ))

    print("\nProfile report saved:")
    print(OUTPUT_PATH)

    print("\nNo candidate pairs or graph generated.")
    print("No dataset changes.")
    print("FULL DATASET BUCKET PROFILING COMPLETED.")


if __name__ == "__main__":
    main()
