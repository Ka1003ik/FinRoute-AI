
"""Near-duplicate leakage audit for FinRoute AI dataset v2.

Uses canonical cleaning, word-trigram MinHash/LSH candidate discovery,
and exact Jaccard verification.

Only split names, row positions and similarity scores are exported.
No source data or trained models are modified.
"""

import csv
import hashlib
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


SPLIT_DIR = ROOT / "data" / "processed" / "splits_v2"
OUTPUT_DIR = ROOT / "outputs" / "leakage_audit"

CANDIDATES_PATH = OUTPUT_DIR / "near_duplicate_candidates_v2.csv"
REPORT_PATH = OUTPUT_DIR / "near_duplicate_audit_v2.json"

TEXT_COLUMN = "Consumer complaint narrative"

SPLITS = {
    "train": {
        "path": SPLIT_DIR / "train.csv",
        "expected_rows": 96891,
    },
    "validation": {
        "path": SPLIT_DIR / "validation.csv",
        "expected_rows": 12111,
    },
    "test": {
        "path": SPLIT_DIR / "test.csv",
        "expected_rows": 12112,
    },
}

RANDOM_STATE = 42

NUM_PERMUTATIONS = 64
BANDS = 16
ROWS_PER_BAND = 4

MAX_BUCKET_SIZE = 200
MAX_CANDIDATE_PAIRS = 2_000_000

SHINGLE_SIZE = 3
MIN_SHINGLES = 5

HIGH_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70

TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")

SPLIT_ORDER = ["train", "validation", "test"]


def stable_text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def token_shingle_hashes(tokens):
    """Hash consecutive three-word shingles."""

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
    """Generate reproducible MinHash coefficients."""

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
    """Compute the MinHash signature with uint64 arithmetic."""

    a, b = coefficients

    with np.errstate(over="ignore"):
        transformed = (
            a[:, np.newaxis] * shingle_hashes[np.newaxis, :]
            + b[:, np.newaxis]
        )

    return transformed.min(axis=1)


def load_split_documents():
    """Load splits and verify exact normalized-text separation."""

    documents = []
    exact_hashes = {}
    statistics = {}

    for split_name in SPLIT_ORDER:

        config = SPLITS[split_name]
        file_path = config["path"]

        print(f"\nReading {split_name} split...")

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        frame = pd.read_csv(
            file_path,
            usecols=[TEXT_COLUMN],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

        if len(frame) != config["expected_rows"]:
            raise ValueError(
                f"{split_name}: expected "
                f"{config['expected_rows']:,} rows, "
                f"found {len(frame):,}."
            )

        frame["_row_position"] = np.arange(
            len(frame),
            dtype=np.int64,
        )

        frame["_clean_text"] = frame[TEXT_COLUMN].map(
            clean_narrative
        )

        if frame["_clean_text"].eq("").any():
            raise ValueError(
                f"Empty normalized narratives found in {split_name}."
            )

        exact_hashes[split_name] = set(
            frame["_clean_text"].map(stable_text_hash)
        )

        unique_frame = frame.drop_duplicates(
            subset="_clean_text",
            keep="first",
        )

        statistics[split_name] = {
            "rows": int(len(frame)),
            "unique_cleaned_narratives": int(len(unique_frame)),
            "repeated_rows_within_split": int(
                len(frame) - len(unique_frame)
            ),
        }

        print(f"Rows: {len(frame):,}")
        print(
            "Unique cleaned narratives:",
            f"{len(unique_frame):,}",
        )

        # Explicit column selection avoids pandas itertuples
        # renaming columns containing spaces or leading underscores.
        selected = unique_frame[
            ["_row_position", "_clean_text"]
        ]

        for row_position, clean_text in selected.itertuples(
            index=False,
            name=None,
        ):

            tokens = TOKEN_PATTERN.findall(clean_text)

            documents.append({
                "split": split_name,
                "row_position": int(row_position),
                "token_count": len(tokens),
                "tokens": tokens,
            })

        del selected
        del unique_frame
        del frame

    return documents, exact_hashes, statistics


def calculate_exact_overlaps(exact_hashes):
    return {
        "train_validation": len(
            exact_hashes["train"] & exact_hashes["validation"]
        ),
        "train_test": len(
            exact_hashes["train"] & exact_hashes["test"]
        ),
        "validation_test": len(
            exact_hashes["validation"] & exact_hashes["test"]
        ),
    }


def build_lsh_buckets(documents):
    """Build LSH buckets and retain shingle sets for verification."""

    coefficients = create_coefficients()

    buckets = [
        defaultdict(list)
        for _ in range(BANDS)
    ]

    shingle_sets = []
    eligible_documents = []
    skipped_short = 0

    print("\nGenerating MinHash signatures...")

    for document in documents:

        # Release token strings as soon as their shingles are made.
        tokens = document.pop("tokens")

        shingle_hashes = token_shingle_hashes(tokens)

        if len(shingle_hashes) < MIN_SHINGLES:
            skipped_short += 1
            continue

        document_id = len(shingle_sets)

        shingle_set = frozenset(
            int(value) for value in shingle_hashes
        )

        signature = compute_minhash(
            shingle_hashes,
            coefficients,
        )

        shingle_sets.append(shingle_set)
        eligible_documents.append(document)

        for band_index in range(BANDS):

            start = band_index * ROWS_PER_BAND
            end = start + ROWS_PER_BAND

            key = tuple(
                int(value)
                for value in signature[start:end]
            )

            buckets[band_index][key].append(document_id)

        if (document_id + 1) % 10000 == 0:
            print(
                f"Signatures generated: {document_id + 1:,}"
            )

    return buckets, shingle_sets, eligible_documents, skipped_short


def generate_candidate_pairs(buckets, eligible_documents):
    """Generate candidate pairs that cross split boundaries."""

    candidate_pairs = set()

    skipped_buckets = 0
    skipped_bucket_members = 0

    print("\nGenerating cross-split candidate pairs...")

    for band_index, band in enumerate(buckets):

        for members in band.values():

            if len(members) < 2:
                continue

            if len(members) > MAX_BUCKET_SIZE:
                skipped_buckets += 1
                skipped_bucket_members += len(members)
                continue

            by_split = {
                name: [] for name in SPLIT_ORDER
            }

            for document_id in members:
                name = eligible_documents[document_id]["split"]
                by_split[name].append(document_id)

            for i, first_split in enumerate(SPLIT_ORDER):

                for second_split in SPLIT_ORDER[i + 1:]:

                    for first_id in by_split[first_split]:
                        for second_id in by_split[second_split]:

                            candidate_pairs.add(
                                (first_id, second_id)
                            )

                            if len(candidate_pairs) > MAX_CANDIDATE_PAIRS:
                                raise RuntimeError(
                                    "Too many candidate pairs. "
                                    "Stopped to protect memory. "
                                    "A more selective candidate "
                                    "configuration is required."
                                )

        print(
            f"LSH band {band_index + 1}/{BANDS} complete; "
            f"candidate pairs: {len(candidate_pairs):,}"
        )

    return candidate_pairs, {
        "oversized_buckets_skipped": skipped_buckets,
        "bucket_members_skipped": skipped_bucket_members,
    }


def verify_candidates(
    candidate_pairs,
    shingle_sets,
    eligible_documents,
):
    """Calculate exact Jaccard and containment for candidates."""

    results = []

    high_count = 0
    review_count = 0

    print("\nVerifying candidate pairs...")

    for count, (first_id, second_id) in enumerate(
        candidate_pairs,
        start=1,
    ):

        first_set = shingle_sets[first_id]
        second_set = shingle_sets[second_id]

        intersection = len(first_set & second_set)

        union = (
            len(first_set)
            + len(second_set)
            - intersection
        )

        jaccard = intersection / union if union else 0.0

        smaller_size = min(
            len(first_set),
            len(second_set),
        )

        containment = (
            intersection / smaller_size
            if smaller_size else 0.0
        )

        if jaccard < REVIEW_THRESHOLD:
            continue

        first = eligible_documents[first_id]
        second = eligible_documents[second_id]

        if jaccard >= HIGH_THRESHOLD:
            band = "high"
            high_count += 1
        else:
            band = "review"
            review_count += 1

        results.append({
            "split_a": first["split"],
            "row_position_a": first["row_position"],
            "split_b": second["split"],
            "row_position_b": second["row_position"],
            "token_count_a": first["token_count"],
            "token_count_b": second["token_count"],
            "shingle_count_a": len(first_set),
            "shingle_count_b": len(second_set),
            "jaccard_similarity": round(jaccard, 6),
            "containment_similarity": round(containment, 6),
            "similarity_band": band,
        })

        if count % 100000 == 0:
            print(f"Candidate pairs checked: {count:,}")

    results.sort(
        key=lambda item: item["jaccard_similarity"],
        reverse=True,
    )

    return results, high_count, review_count


def main():

    print("=" * 65)
    print("FINROUTE AI - NEAR-DUPLICATE LEAKAGE AUDIT")
    print("=" * 65)

    if CANDIDATES_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "Previous audit outputs already exist. "
            "Move or rename them before rerunning."
        )

    documents, exact_hashes, split_statistics = (
        load_split_documents()
    )

    exact_overlaps = calculate_exact_overlaps(
        exact_hashes
    )

    print("\nExact normalized-text overlap:")

    for pair, count in exact_overlaps.items():
        print(f"{pair}: {count}")

    if any(exact_overlaps.values()):
        raise ValueError(
            "Exact cross-split duplicates detected. "
            "Resolve these before continuing."
        )

    (
        buckets,
        shingle_sets,
        eligible_documents,
        skipped_short,
    ) = build_lsh_buckets(documents)

    del documents

    candidate_pairs, bucket_statistics = generate_candidate_pairs(
        buckets,
        eligible_documents,
    )

    (
        results,
        high_count,
        review_count,
    ) = verify_candidates(
        candidate_pairs,
        shingle_sets,
        eligible_documents,
    )

    columns = [
        "split_a",
        "row_position_a",
        "split_b",
        "row_position_b",
        "token_count_a",
        "token_count_b",
        "shingle_count_a",
        "shingle_count_b",
        "jaccard_similarity",
        "containment_similarity",
        "similarity_band",
    ]

    with CANDIDATES_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(results)

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "audit_type": "Approximate near-duplicate cross-split audit",

        "method": {
            "normalization": (
                "src.preprocessing.clean_text.clean_narrative"
            ),
            "shingle_type": "Consecutive word trigrams",
            "minhash_permutations": NUM_PERMUTATIONS,
            "lsh_bands": BANDS,
            "rows_per_band": ROWS_PER_BAND,
            "maximum_bucket_size": MAX_BUCKET_SIZE,
            "minimum_shingles": MIN_SHINGLES,
            "random_state": RANDOM_STATE,
        },

        "similarity_thresholds": {
            "high": HIGH_THRESHOLD,
            "review": REVIEW_THRESHOLD,
        },

        "split_statistics": split_statistics,
        "unique_documents_loaded": sum(
            item["unique_cleaned_narratives"]
            for item in split_statistics.values()
        ),
        "documents_eligible_for_lsh": len(eligible_documents),
        "short_documents_skipped": skipped_short,

        "exact_normalized_text_overlap_counts": exact_overlaps,

        "candidate_pairs_generated": len(candidate_pairs),
        "candidate_pairs_scored": len(candidate_pairs),
        "high_similarity_pairs": high_count,
        "review_similarity_pairs": review_count,
        "total_pairs_exported": len(results),

        "lsh_bucket_statistics": bucket_statistics,

        "candidate_output_file": str(CANDIDATES_PATH),

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "company_names_exported": False,
        "zip_codes_exported": False,

        "datasets_modified": False,
        "records_deleted_or_moved": False,
        "model_evaluation_performed": False,

        "limitations": [
            "LSH candidate discovery is approximate, not exhaustive.",
            "Narratives below the minimum shingle count are skipped.",
            "Oversized buckets are skipped to control memory usage.",
            "High similarity is a review signal, not proof of leakage.",
            "No records should be deleted or moved based on this report alone.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("NEAR-DUPLICATE AUDIT RESULTS")
    print("=" * 65)

    print(f"Unique documents: {report['unique_documents_loaded']:,}")
    print(f"Documents eligible for LSH: {len(eligible_documents):,}")
    print(f"Short documents skipped: {skipped_short:,}")
    print(f"Candidate pairs: {len(candidate_pairs):,}")
    print(f"High similarity pairs (>= 0.85): {high_count:,}")
    print(f"Review pairs (0.70–0.849): {review_count:,}")
    print(
        "Oversized buckets skipped:",
        bucket_statistics["oversized_buckets_skipped"],
    )

    print("\nCandidate report:")
    print(CANDIDATES_PATH)

    print("\nSummary report:")
    print(REPORT_PATH)

    print("\nNo datasets modified.")
    print("No records deleted or moved.")
    print("NEAR-DUPLICATE AUDIT COMPLETED.")


if __name__ == "__main__":
    main()
