
import json
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "near_duplicate_candidates_v2.csv"
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v2"

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "near_duplicate_label_conflict_investigation_v2.json"
)

LABEL_COLUMN = "Product"
SPLITS = ["train", "validation", "test"]

REQUIRED_COLUMNS = {
    "split_a",
    "row_position_a",
    "split_b",
    "row_position_b",
    "jaccard_similarity",
}

THRESHOLDS = [0.70, 0.85, 0.95, 0.99, 1.00]


def load_labels():
    labels = {}

    for split in SPLITS:
        path = SPLIT_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        labels[split] = pd.read_csv(
            path,
            usecols=[LABEL_COLUMN],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )[LABEL_COLUMN]

    return labels


def split_pair_name(first, second):
    ordered = sorted(
        [first, second],
        key=SPLITS.index,
    )
    return f"{ordered[0]}_{ordered[1]}"


def build_label_information(candidates, split_labels):
    label_a_values = []
    label_b_values = []

    for row in candidates.itertuples(index=False):
        split_a = row.split_a
        split_b = row.split_b

        if split_a not in split_labels or split_b not in split_labels:
            raise ValueError("Unexpected split name in candidate file.")

        position_a = int(row.row_position_a)
        position_b = int(row.row_position_b)

        if not (0 <= position_a < len(split_labels[split_a])):
            raise IndexError("Invalid row position in split A.")

        if not (0 <= position_b < len(split_labels[split_b])):
            raise IndexError("Invalid row position in split B.")

        label_a_values.append(
            split_labels[split_a].iloc[position_a]
        )

        label_b_values.append(
            split_labels[split_b].iloc[position_b]
        )

    candidates = candidates.copy()
    candidates["label_a"] = label_a_values
    candidates["label_b"] = label_b_values

    candidates["different_label"] = (
        candidates["label_a"] != candidates["label_b"]
    )

    return candidates


def summarize_conflicts(frame):
    total = len(frame)

    return {
        "different_label_pairs": int(total),
        "percentage_of_all_candidate_pairs": round(
            100 * total / len(candidates), 2
        ) if len(candidates) else 0,
    }


def connected_component_conflicts(data, threshold):
    selected = data[
        data["jaccard_similarity"] >= threshold
    ]

    graph = nx.Graph()

    for row in selected.itertuples(index=False):
        node_a = (row.split_a, int(row.row_position_a))
        node_b = (row.split_b, int(row.row_position_b))

        graph.add_edge(
            node_a,
            node_b,
            different_label=bool(row.different_label),
        )

    conflict_components = []
    directly_conflicting_records = set()

    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)

        conflict_edges = [
            (u, v)
            for u, v, attributes in subgraph.edges(data=True)
            if attributes["different_label"]
        ]

        if not conflict_edges:
            continue

        for u, v in conflict_edges:
            directly_conflicting_records.add(u)
            directly_conflicting_records.add(v)

        conflict_components.append({
            "component_size": len(component),
            "different_label_pairs": len(conflict_edges),
        })

    conflict_components.sort(
        key=lambda item: item["different_label_pairs"],
        reverse=True,
    )

    return {
        "threshold": threshold,
        "all_candidate_edges": int(graph.number_of_edges()),
        "components_with_label_disagreements": len(
            conflict_components
        ),
        "different_label_pairs": sum(
            item["different_label_pairs"]
            for item in conflict_components
        ),
        "unique_records_directly_in_conflicting_pairs": len(
            directly_conflicting_records
        ),
        "largest_conflict_component_size": max(
            (
                item["component_size"]
                for item in conflict_components
            ),
            default=0,
        ),
        "largest_number_of_conflicts_in_one_component": max(
            (
                item["different_label_pairs"]
                for item in conflict_components
            ),
            default=0,
        ),
        "top_10_conflict_components": conflict_components[:10],
    }


def main():
    global candidates

    print("=" * 65)
    print("FINROUTE AI - LABEL CONFLICT INVESTIGATION")
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
            f"Missing candidate columns: {sorted(missing_columns)}"
        )

    split_labels = load_labels()

    candidates = build_label_information(
        candidates,
        split_labels,
    )

    conflicts = candidates[
        candidates["different_label"]
    ].copy()

    total_pairs = len(candidates)
    conflict_count = len(conflicts)

    print(f"\nTotal candidate pairs: {total_pairs}")
    print(f"Different-label pairs: {conflict_count}")

    # Product-category combinations.
    product_pairs = []

    for (label_a, label_b), group in conflicts.groupby(
        conflicts.apply(
            lambda row: tuple(sorted([
                row["label_a"],
                row["label_b"],
            ])),
            axis=1,
        )
    ):
        product_pairs.append({
            "product_a": label_a,
            "product_b": label_b,
            "different_label_pairs": int(len(group)),
        })

    product_pairs.sort(
        key=lambda item: item["different_label_pairs"],
        reverse=True,
    )

    # Similarity ranges.
    similarity_bands = {}

    bands = [
        (0.70, 0.85),
        (0.85, 0.95),
        (0.95, 0.99),
        (0.99, 1.000001),
    ]

    for low, high in bands:
        subset = conflicts[
            (conflicts["jaccard_similarity"] >= low)
            & (conflicts["jaccard_similarity"] < high)
        ]

        similarity_bands[
            f"{low:.2f}_to_{high - 0.000001:.2f}"
        ] = int(len(subset))

    # Split-pair distribution.
    split_pair_counts = {}

    for name in [
        "train_validation",
        "train_test",
        "validation_test",
    ]:
        subset = conflicts[
            conflicts.apply(
                lambda row: split_pair_name(
                    row["split_a"],
                    row["split_b"],
                ) == name,
                axis=1,
            )
        ]

        split_pair_counts[name] = int(len(subset))

    # Connected-component investigation.
    component_results = {
        str(threshold): connected_component_conflicts(
            candidates,
            threshold,
        )
        for threshold in THRESHOLDS
    }

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",
        "total_candidate_pairs": int(total_pairs),
        "different_label_pairs": int(conflict_count),
        "different_label_percentage": round(
            100 * conflict_count / total_pairs,
            2,
        ) if total_pairs else 0,
        "conflicts_by_split_pair": split_pair_counts,
        "conflicts_by_similarity_range": similarity_bands,
        "conflicts_by_product_category_pair": product_pairs,
        "connected_component_conflict_analysis": component_results,
        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
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

    print("\n=== CONFLICTS BY PRODUCT CATEGORY ===")
    print(json.dumps(product_pairs, indent=4))

    print("\n=== CONFLICTS BY SIMILARITY RANGE ===")
    print(json.dumps(similarity_bands, indent=4))

    print("\n=== CONFLICTS BY SPLIT PAIR ===")
    print(json.dumps(split_pair_counts, indent=4))

    print("\n=== CONNECTED COMPONENT CONFLICTS ===")
    print(json.dumps(component_results, indent=4))

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo narratives or complaint IDs exported.")
    print("No dataset changes or model retraining.")
    print("INVESTIGATION COMPLETED.")


if __name__ == "__main__":
    main()
