
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
    / "high_similarity_conflict_investigation_v2.json"
)

LABEL_COLUMN = "Product"

INTAKE_FIELDS = [
    "Date received",
    "Submitted via",
    "Company",
    "State",
]

SPLITS = ["train", "validation", "test"]

REQUIRED_COLUMNS = {
    "split_a",
    "row_position_a",
    "split_b",
    "row_position_b",
    "jaccard_similarity",
}

THRESHOLDS = {
    "at_least_0.95": 0.95,
    "at_least_0.99": 0.99,
    "exactly_1.00": 1.00,
}


def normalize_value(value):
    """Normalize whitespace and letter case for comparison."""
    value = " ".join(str(value).strip().split())

    if not value:
        return None

    return value.casefold()


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


def load_split_data():
    required_columns = [LABEL_COLUMN] + INTAKE_FIELDS
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
                f"{split}.csv missing columns: {sorted(missing)}"
            )

        split_data[split] = pd.read_csv(
            path,
            usecols=required_columns,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )

    return split_data


def get_split_pair(split_a, split_b):
    order = {name: i for i, name in enumerate(SPLITS)}

    ordered = sorted(
        [split_a, split_b],
        key=lambda name: order[name],
    )

    return f"{ordered[0]}_{ordered[1]}"


def build_conflict_records(candidates, split_data):
    records = []

    for row in candidates.itertuples(index=False):
        similarity = float(row.jaccard_similarity)

        # This investigation focuses only on similarity >= 0.95.
        if similarity < 0.95:
            continue

        split_a = row.split_a
        split_b = row.split_b

        if split_a not in split_data or split_b not in split_data:
            raise ValueError("Unexpected split name.")

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

        if not label_a or not label_b:
            raise ValueError("Empty Product label encountered.")

        # Keep only different-Product-label pairs.
        if label_a == label_b:
            continue

        product_pair = " <> ".join(
            sorted([label_a, label_b])
        )

        metadata_status = {
            field: compare_values(
                record_a[field],
                record_b[field],
            )
            for field in INTAKE_FIELDS
        }

        records.append({
            "similarity": similarity,
            "product_pair": product_pair,
            "split_pair": get_split_pair(
                split_a,
                split_b,
            ),
            "metadata_status": metadata_status,
        })

    return records


def select_threshold(records, threshold_name, threshold):
    if threshold_name == "exactly_1.00":
        return [
            record for record in records
            if record["similarity"] == threshold
        ]

    return [
        record for record in records
        if record["similarity"] >= threshold
    ]


def summarize_field_comparisons(records):
    summary = {}

    for field in INTAKE_FIELDS:
        statuses = [
            record["metadata_status"][field]
            for record in records
        ]

        total = len(statuses)
        comparable = statuses.count("same") + statuses.count(
            "different"
        )

        different = statuses.count("different")

        summary[field] = {
            "pairs_examined": total,
            "same": statuses.count("same"),
            "different": different,
            "one_or_both_missing": statuses.count(
                "one_or_both_missing"
            ),
            "both_missing": statuses.count("both_missing"),
            "different_percentage_among_comparable_pairs": round(
                100 * different / comparable,
                2,
            ) if comparable else None,
        }

    return summary


def summarize_combined_metadata(records):
    all_four_match = 0
    at_least_one_difference = 0
    at_least_one_missing = 0

    patterns = {}

    for record in records:
        statuses = record["metadata_status"]

        values = [
            statuses[field]
            for field in INTAKE_FIELDS
        ]

        if all(status == "same" for status in values):
            all_four_match += 1

        if any(status == "different" for status in values):
            at_least_one_difference += 1

        if any(
            status in {"one_or_both_missing", "both_missing"}
            for status in values
        ):
            at_least_one_missing += 1

        pattern = " | ".join(
            f"{field}={statuses[field]}"
            for field in INTAKE_FIELDS
        )

        patterns[pattern] = patterns.get(pattern, 0) + 1

    return {
        "pairs_examined": len(records),
        "all_four_intake_fields_match": all_four_match,
        "at_least_one_intake_field_differs": (
            at_least_one_difference
        ),
        "at_least_one_field_has_missing_values": (
            at_least_one_missing
        ),
        "metadata_comparison_patterns": [
            {
                "pattern": pattern,
                "pairs": count,
            }
            for pattern, count in sorted(
                patterns.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def summarize_product_pairs(records):
    product_pairs = {}

    for record in records:
        pair = record["product_pair"]

        if pair not in product_pairs:
            product_pairs[pair] = {
                "conflicting_pairs": 0,
                "all_four_intake_fields_match": 0,
                "at_least_one_intake_field_differs": 0,
                "metadata_field_differences": {
                    field: 0
                    for field in INTAKE_FIELDS
                },
            }

        summary = product_pairs[pair]
        summary["conflicting_pairs"] += 1

        statuses = record["metadata_status"]

        if all(
            statuses[field] == "same"
            for field in INTAKE_FIELDS
        ):
            summary["all_four_intake_fields_match"] += 1

        if any(
            statuses[field] == "different"
            for field in INTAKE_FIELDS
        ):
            summary["at_least_one_intake_field_differs"] += 1

        for field in INTAKE_FIELDS:
            if statuses[field] == "different":
                summary["metadata_field_differences"][field] += 1

    return [
        {
            "product_pair": pair,
            **summary,
        }
        for pair, summary in sorted(
            product_pairs.items(),
            key=lambda item: item[1]["conflicting_pairs"],
            reverse=True,
        )
    ]


def summarize_split_pairs(records):
    result = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }

    for record in records:
        result[record["split_pair"]] += 1

    return result


def build_report(records):
    results = {}

    for name, threshold in THRESHOLDS.items():
        selected = select_threshold(
            records,
            name,
            threshold,
        )

        results[name] = {
            "threshold": threshold,
            "conflicting_pairs": len(selected),
            "label_pair_distribution": summarize_product_pairs(
                selected
            ),
            "intake_metadata_comparison": (
                summarize_field_comparisons(selected)
            ),
            "combined_intake_metadata": (
                summarize_combined_metadata(selected)
            ),
            "split_pair_distribution": summarize_split_pairs(
                selected
            ),
        }

    return results


def main():
    print("=" * 65)
    print("FINROUTE AI - HIGH SIMILARITY CONFLICT INVESTIGATION")
    print("=" * 65)

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(CANDIDATES_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    candidates = pd.read_csv(CANDIDATES_PATH)

    missing_columns = REQUIRED_COLUMNS - set(candidates.columns)

    if missing_columns:
        raise ValueError(
            f"Candidate file missing columns: {sorted(missing_columns)}"
        )

    split_data = load_split_data()

    records = build_conflict_records(
        candidates,
        split_data,
    )

    if not records:
        raise ValueError(
            "No high-similarity different-label pairs found."
        )

    results = build_report(records)

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "investigation_scope": (
            "Different-Product-label pairs with Jaccard "
            "similarity >= 0.95"
        ),
        "high_similarity_conflicting_pairs": len(records),
        "results_by_similarity_threshold": results,
        "metadata_normalization": (
            "Trim whitespace, collapse repeated spaces, "
            "and compare case-insensitively."
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

    for name, result in results.items():
        print(f"\n=== {name.upper()} ===")

        print(
            "Different-label pairs:",
            result["conflicting_pairs"],
        )

        print("\nIntake metadata comparison:")
        print(json.dumps(
            result["intake_metadata_comparison"],
            indent=4,
        ))

        print("\nCombined intake metadata:")
        print(json.dumps(
            result["combined_intake_metadata"],
            indent=4,
        ))

        print("\nProduct-category combinations:")
        print(json.dumps(
            result["label_pair_distribution"],
            indent=4,
        ))

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo narratives, identifiers, company names, or metadata values exported.")
    print("No dataset changes or model retraining.")
    print("HIGH-SIMILARITY INVESTIGATION COMPLETED.")


if __name__ == "__main__":
    main()
