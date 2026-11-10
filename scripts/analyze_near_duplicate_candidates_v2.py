
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "near_duplicate_candidates_v2.csv"
)

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v2"

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "near_duplicate_label_analysis_v2.json"
)

LABEL_COLUMN = "Product"

SPLITS = ["train", "validation", "test"]

THRESHOLDS = [0.70, 0.85, 0.95, 0.99, 1.00]


def component_analysis(data, threshold):

    graph = nx.Graph()

    selected = data[
        data["jaccard_similarity"] >= threshold
    ]

    for row in selected.itertuples(index=False):
        graph.add_edge(
            (row.split_a, int(row.row_position_a)),
            (row.split_b, int(row.row_position_b))
        )

    components = list(nx.connected_components(graph))

    sizes = [len(component) for component in components]

    return {
        "edges": int(graph.number_of_edges()),
        "unique_records_involved": int(graph.number_of_nodes()),
        "connected_components": len(components),
        "components_with_more_than_two_records": sum(
            size > 2 for size in sizes
        ),
        "largest_component_size": max(sizes, default=0),
        "records_in_components_larger_than_two": sum(
            size for size in sizes if size > 2
        ),
    }


def main():

    print("=" * 65)
    print("FINROUTE AI - NEAR DUPLICATE LABEL ANALYSIS")
    print("=" * 65)

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(CANDIDATES_PATH)

    candidates = pd.read_csv(CANDIDATES_PATH)

    split_labels = {}

    # Load only Product labels. No narratives or identifiers exported.
    for split in SPLITS:

        path = SPLIT_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        split_labels[split] = pd.read_csv(
            path,
            usecols=[LABEL_COLUMN],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig"
        )[LABEL_COLUMN]

    same_label = []
    different_label = []

    for row in candidates.itertuples(index=False):

        label_a = split_labels[row.split_a].iloc[
            int(row.row_position_a)
        ]

        label_b = split_labels[row.split_b].iloc[
            int(row.row_position_b)
        ]

        same_label.append(label_a == label_b)
        different_label.append(label_a != label_b)

    candidates["same_product_label"] = same_label
    candidates["different_product_label"] = different_label

    def agreement_summary(frame):

        return {
            "pairs": int(len(frame)),
            "same_product_label": int(
                frame["same_product_label"].sum()
            ),
            "different_product_label": int(
                frame["different_product_label"].sum()
            ),
            "same_label_percentage": round(
                100 * frame["same_product_label"].mean(), 2
            ) if len(frame) else None,
        }

    by_split_pair = {}

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test")
    ]:

        mask = (
            (candidates["split_a"] == first)
            & (candidates["split_b"] == second)
        ) | (
            (candidates["split_a"] == second)
            & (candidates["split_b"] == first)
        )

        by_split_pair[f"{first}_{second}"] = (
            agreement_summary(candidates[mask])
        )

    by_similarity_band = {}

    for low, high in [
        (0.70, 0.85),
        (0.85, 0.95),
        (0.95, 0.99),
        (0.99, 1.01)
    ]:

        mask = (
            (candidates["jaccard_similarity"] >= low)
            & (candidates["jaccard_similarity"] < high)
        )

        by_similarity_band[
            f"{low:.2f}_to_{high - 0.000001:.2f}"
        ] = agreement_summary(candidates[mask])

    component_results = {
        str(threshold): component_analysis(
            candidates,
            threshold
        )
        for threshold in THRESHOLDS
    }

    # Count distinct candidate endpoints by split.
    endpoint_counts = {}

    for threshold in THRESHOLDS:

        subset = candidates[
            candidates["jaccard_similarity"] >= threshold
        ]

        endpoints = set()

        for row in subset.itertuples(index=False):
            endpoints.add((row.split_a, int(row.row_position_a)))
            endpoints.add((row.split_b, int(row.row_position_b)))

        endpoint_counts[str(threshold)] = {
            split: sum(
                endpoint[0] == split
                for endpoint in endpoints
            )
            for split in SPLITS
        }

    # Highest similarity pairs, with no complaint content.
    top_pairs = []

    for row in candidates.nlargest(
        25, "jaccard_similarity"
    ).itertuples(index=False):

        top_pairs.append({
            "split_a": row.split_a,
            "row_position_a": int(row.row_position_a),
            "split_b": row.split_b,
            "row_position_b": int(row.row_position_b),
            "jaccard_similarity": float(row.jaccard_similarity),
            "containment_similarity": float(
                row.containment_similarity
            ),
            "same_product_label": bool(
                row.same_product_label
            )
        })

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",

        "total_candidate_pairs": int(len(candidates)),

        "overall_label_agreement": agreement_summary(
            candidates
        ),

        "label_agreement_by_split_pair": by_split_pair,

        "label_agreement_by_similarity": by_similarity_band,

        "connected_component_analysis": component_results,

        "unique_candidate_endpoints_by_split": endpoint_counts,

        "top_25_pairs": top_pairs,

        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "dataset_modified": False,
        "records_deleted_or_moved": False,
        "model_retrained": False
    }

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n=== OVERALL LABEL AGREEMENT ===")
    print(json.dumps(
        report["overall_label_agreement"],
        indent=4
    ))

    print("\n=== BY SPLIT PAIR ===")
    print(json.dumps(by_split_pair, indent=4))

    print("\n=== BY SIMILARITY RANGE ===")
    print(json.dumps(by_similarity_band, indent=4))

    print("\n=== CONNECTED COMPONENTS ===")
    print(json.dumps(component_results, indent=4))

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo narratives or IDs exported.")
    print("No data modified.")
    print("LABEL ANALYSIS COMPLETED.")


if __name__ == "__main__":
    main()
