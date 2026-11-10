
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "near_duplicate_candidates_v2.csv"
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v2"

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "near_duplicate_metadata_investigation_v2.json"
)

LABEL_COLUMN = "Product"

TAXONOMY_FIELDS = [
    "Sub-product",
    "Issue",
    "Sub-issue",
]

INTAKE_FIELDS = [
    "Date received",
    "Submitted via",
    "Company",
    "State",
]

METADATA_FIELDS = TAXONOMY_FIELDS + INTAKE_FIELDS

SPLITS = ["train", "validation", "test"]

REQUIRED_CANDIDATE_COLUMNS = {
    "split_a",
    "row_position_a",
    "split_b",
    "row_position_b",
    "jaccard_similarity",
}


def normalize_value(value):
    """Normalize a metadata value for comparison without exporting it."""
    value = " ".join(str(value).strip().split())

    if not value:
        return None

    return value.casefold()


def load_split_metadata():
    """Load only Product and the seven approved metadata fields."""
    required_columns = [LABEL_COLUMN] + METADATA_FIELDS
    split_data = {}

    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        header = pd.read_csv(
            path,
            nrows=0,
            encoding="utf-8-sig",
        )

        missing = set(required_columns) - set(header.columns)

        if missing:
            raise ValueError(
                f"{split}.csv is missing columns: {sorted(missing)}"
            )

        split_data[split] = pd.read_csv(
            path,
            usecols=required_columns,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

    return split_data


def get_split_pair_name(split_a, split_b):
    order = {name: index for index, name in enumerate(SPLITS)}

    ordered = sorted(
        [split_a, split_b],
        key=lambda name: order[name],
    )

    return f"{ordered[0]}_{ordered[1]}"


def compare_values(value_a, value_b):
    value_a = normalize_value(value_a)
    value_b = normalize_value(value_b)

    if value_a is None and value_b is None:
        return "both_missing"

    if value_a is None or value_b is None:
        return "one_or_both_missing"

    if value_a == value_b:
        return "same"

    return "different"


def build_conflict_records(candidates, split_data):
    """Build internal comparison records for different-label pairs only."""
    records = []

    for row in candidates.itertuples(index=False):
        split_a = row.split_a
        split_b = row.split_b

        if split_a not in split_data or split_b not in split_data:
            raise ValueError("Unexpected split name in candidate file.")

        position_a = int(row.row_position_a)
        position_b = int(row.row_position_b)

        frame_a = split_data[split_a]
        frame_b = split_data[split_b]

        if not (0 <= position_a < len(frame_a)):
            raise IndexError("Invalid row position in split A.")

        if not (0 <= position_b < len(frame_b)):
            raise IndexError("Invalid row position in split B.")

        record_a = frame_a.iloc[position_a]
        record_b = frame_b.iloc[position_b]

        label_a = record_a[LABEL_COLUMN].strip()
        label_b = record_b[LABEL_COLUMN].strip()

        if label_a == label_b:
            continue

        if not label_a or not label_b:
            raise ValueError("An empty Product label was encountered.")

        product_pair = " <> ".join(sorted([label_a, label_b]))

        metadata_comparisons = {}

        for field in METADATA_FIELDS:
            metadata_comparisons[field] = compare_values(
                record_a[field],
                record_b[field],
            )

        records.append({
            "product_pair": product_pair,
            "split_pair": get_split_pair_name(
                split_a,
                split_b,
            ),
            "jaccard_similarity": float(
                row.jaccard_similarity
            ),
            "metadata": metadata_comparisons,
        })

    return records


def field_summary(records, field):
    statuses = [
        record["metadata"][field]
        for record in records
    ]

    total = len(statuses)
    both_present = statuses.count("same") + statuses.count("different")
    different = statuses.count("different")

    return {
        "conflicting_product_pairs": total,
        "both_values_present": both_present,
        "same_value": statuses.count("same"),
        "different_value": different,
        "one_or_both_missing": statuses.count(
            "one_or_both_missing"
        ),
        "both_missing": statuses.count("both_missing"),
        "different_percentage_among_pairs_with_both_values": round(
            100 * different / both_present,
            2,
        ) if both_present else None,
    }


def combined_field_summary(records, fields):
    """Summarize how often a group of metadata fields distinguishes pairs."""
    all_fields_comparable = 0
    at_least_one_comparable = 0
    pairs_with_any_difference = 0
    pairs_with_all_comparable_values_same = 0

    difference_count_distribution = {}

    total_comparable_field_checks = 0
    total_different_field_checks = 0

    for record in records:
        statuses = [
            record["metadata"][field]
            for field in fields
        ]

        comparable = [
            status for status in statuses
            if status in {"same", "different"}
        ]

        difference_count = comparable.count("different")

        total_comparable_field_checks += len(comparable)
        total_different_field_checks += difference_count

        difference_count_distribution[str(
            difference_count
        )] = difference_count_distribution.get(
            str(difference_count), 0
        ) + 1

        if len(comparable) == len(fields):
            all_fields_comparable += 1

        if comparable:
            at_least_one_comparable += 1

            if difference_count > 0:
                pairs_with_any_difference += 1
            else:
                pairs_with_all_comparable_values_same += 1

    return {
        "fields_examined": fields,
        "total_conflicting_product_pairs": len(records),
        "pairs_with_all_fields_comparable": all_fields_comparable,
        "pairs_with_at_least_one_comparable_field": (
            at_least_one_comparable
        ),
        "pairs_with_one_or_more_different_metadata_values": (
            pairs_with_any_difference
        ),
        "pairs_with_all_comparable_values_matching": (
            pairs_with_all_comparable_values_same
        ),
        "total_comparable_field_checks": (
            total_comparable_field_checks
        ),
        "total_different_field_checks": (
            total_different_field_checks
        ),
        "different_field_count_per_pair": (
            difference_count_distribution
        ),
    }


def summarize_by_product_pair(records):
    results = []

    product_pairs = sorted({
        record["product_pair"]
        for record in records
    })

    for product_pair in product_pairs:
        subset = [
            record for record in records
            if record["product_pair"] == product_pair
        ]

        results.append({
            "product_pair": product_pair,
            "conflicting_pairs": len(subset),
            "metadata_differences": {
                field: sum(
                    record["metadata"][field] == "different"
                    for record in subset
                )
                for field in METADATA_FIELDS
            },
        })

    results.sort(
        key=lambda item: item["conflicting_pairs"],
        reverse=True,
    )

    return results


def summarize_by_split_pair(records):
    result = {}

    for split_pair in [
        "train_validation",
        "train_test",
        "validation_test",
    ]:
        subset = [
            record for record in records
            if record["split_pair"] == split_pair
        ]

        result[split_pair] = len(subset)

    return result


def summarize_by_similarity(records):
    bands = [
        (0.70, 0.85),
        (0.85, 0.95),
        (0.95, 0.99),
        (0.99, 1.000001),
    ]

    result = {}

    for low, high in bands:
        count = sum(
            low <= record["jaccard_similarity"] < high
            for record in records
        )

        result[
            f"{low:.2f}_to_{high - 0.000001:.2f}"
        ] = count

    return result


def main():
    print("=" * 65)
    print("FINROUTE AI - NEAR-DUPLICATE METADATA INVESTIGATION")
    print("=" * 65)

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(CANDIDATES_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    candidates = pd.read_csv(CANDIDATES_PATH)

    missing_columns = (
        REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Candidate file is missing: {sorted(missing_columns)}"
        )

    split_data = load_split_metadata()

    conflict_records = build_conflict_records(
        candidates,
        split_data,
    )

    if not conflict_records:
        raise ValueError("No different-label candidate pairs found.")

    taxonomy_summary = {
        field: field_summary(conflict_records, field)
        for field in TAXONOMY_FIELDS
    }

    intake_summary = {
        field: field_summary(conflict_records, field)
        for field in INTAKE_FIELDS
    }

    all_field_summary = {
        field: field_summary(conflict_records, field)
        for field in METADATA_FIELDS
    }

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "candidate_pairs_examined": int(len(candidates)),
        "different_product_label_pairs": len(conflict_records),

        "taxonomy_field_comparison": taxonomy_summary,

        "intake_context_field_comparison": intake_summary,

        "all_metadata_field_comparison": all_field_summary,

        "combined_metadata_comparison": {
            "taxonomy_fields": combined_field_summary(
                conflict_records,
                TAXONOMY_FIELDS,
            ),
            "intake_context_fields": combined_field_summary(
                conflict_records,
                INTAKE_FIELDS,
            ),
            "all_seven_fields": combined_field_summary(
                conflict_records,
                METADATA_FIELDS,
            ),
        },

        "conflicts_by_product_pair": summarize_by_product_pair(
            conflict_records
        ),

        "conflicts_by_split_pair": summarize_by_split_pair(
            conflict_records
        ),

        "conflicts_by_similarity_range": summarize_by_similarity(
            conflict_records
        ),

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "company_names_exported": False,
        "metadata_values_exported": False,
        "row_positions_exported": False,
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

    print("\n=== TAXONOMY FIELD COMPARISON ===")
    print(json.dumps(taxonomy_summary, indent=4))

    print("\n=== INTAKE / CONTEXT FIELD COMPARISON ===")
    print(json.dumps(intake_summary, indent=4))

    print("\n=== COMBINED METADATA COMPARISON ===")
    print(json.dumps(
        report["combined_metadata_comparison"],
        indent=4,
    ))

    print("\n=== PRODUCT-PAIR BREAKDOWN ===")
    print(json.dumps(
        report["conflicts_by_product_pair"],
        indent=4,
    ))

    print("\n=== SPLIT-PAIR BREAKDOWN ===")
    print(json.dumps(
        report["conflicts_by_split_pair"],
        indent=4,
    ))

    print("\n=== SIMILARITY BREAKDOWN ===")
    print(json.dumps(
        report["conflicts_by_similarity_range"],
        indent=4,
    ))

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo raw narratives, IDs, company names, or metadata values exported.")
    print("No dataset changes or model retraining.")
    print("METADATA INVESTIGATION COMPLETED.")


if __name__ == "__main__":
    main()
