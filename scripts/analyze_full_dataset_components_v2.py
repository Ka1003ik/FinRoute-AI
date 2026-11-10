
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "leakage_audit"

EDGE_DB_PATH = (
    OUTPUT_DIR / "full_dataset_verified_edges_v2.sqlite"
)

DOCUMENT_MAP_PATH = (
    OUTPUT_DIR / "full_dataset_document_map_v2.csv"
)

ASSIGNMENT_PATH = (
    OUTPUT_DIR / "full_dataset_component_assignments_v2.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "full_dataset_component_analysis_v2.json"
)

LABEL_COLUMN = "Product"

EXPECTED_SOURCE_ROWS = 121114
EXPECTED_UNIQUE_DOCUMENTS = 108902
EXPECTED_VERIFIED_EDGES = 474172

THRESHOLDS = {
    "0.70": {
        "value": 0.70,
        "expected_edges": 474172,
        "column": "component_070",
    },
    "0.85": {
        "value": 0.85,
        "expected_edges": 136293,
        "column": "component_085",
    },
    "0.95": {
        "value": 0.95,
        "expected_edges": 9163,
        "column": "component_095",
    },
    "0.99": {
        "value": 0.99,
        "expected_edges": 1168,
        "column": "component_099",
    },
    "1.00": {
        "value": 1.00,
        "expected_edges": 906,
        "column": "component_100",
    },
}


def validate_inputs():
    for path in [
        EDGE_DB_PATH,
        DOCUMENT_MAP_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    for path in [
        ASSIGNMENT_PATH,
        REPORT_PATH,
    ]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )


def load_document_information():
    print("\nLoading original-record mapping...")

    mapping = pd.read_csv(
        DOCUMENT_MAP_PATH,
        usecols=[
            "row_position",
            "unique_document_id",
            LABEL_COLUMN,
        ],
        dtype={
            "row_position": "int64",
            "unique_document_id": "int64",
            LABEL_COLUMN: str,
        },
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    if len(mapping) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS:,} original records, "
            f"found {len(mapping):,}."
        )

    expected_positions = np.arange(
        EXPECTED_SOURCE_ROWS,
        dtype=np.int64,
    )

    if not np.array_equal(
        mapping["row_position"].to_numpy(),
        expected_positions,
    ):
        raise ValueError(
            "Original row positions are missing or out of order."
        )

    print("Building unique-document label statistics...")

    document_summary = mapping.groupby(
        "unique_document_id",
        sort=True,
    ).agg(
        Product=(LABEL_COLUMN, "first"),
        product_count=(LABEL_COLUMN, "nunique"),
        record_count=("row_position", "size"),
    )

    if len(document_summary) != EXPECTED_UNIQUE_DOCUMENTS:
        raise ValueError(
            "Unique-document count does not match expected value."
        )

    expected_ids = np.arange(
        EXPECTED_UNIQUE_DOCUMENTS,
        dtype=np.int64,
    )

    if not np.array_equal(
        document_summary.index.to_numpy(),
        expected_ids,
    ):
        raise ValueError(
            "Unique document IDs are not continuous."
        )

    if (document_summary["product_count"] != 1).any():
        raise ValueError(
            "An exact cleaned-text group has conflicting Product labels."
        )

    doc_labels = document_summary["Product"].to_numpy(
        dtype=object
    )

    doc_record_counts = document_summary["record_count"].to_numpy(
        dtype=np.int64
    )

    if int(doc_record_counts.sum()) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            "Original record counts do not reconcile."
        )

    return mapping, doc_labels, doc_record_counts


def load_verified_edges():
    print("\nChecking verified-edge database...")

    connection = sqlite3.connect(
        f"file:{EDGE_DB_PATH.as_posix()}?mode=ro",
        uri=True,
    )

    try:
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(
                f"SQLite integrity check failed: {integrity}"
            )

        edge_count = connection.execute(
            "SELECT COUNT(*) FROM verified_edges"
        ).fetchone()[0]

        if edge_count != EXPECTED_VERIFIED_EDGES:
            raise ValueError(
                f"Expected {EXPECTED_VERIFIED_EDGES:,} verified edges, "
                f"found {edge_count:,}."
            )

        print(f"Verified edges: {edge_count:,}")

        edges = pd.read_sql_query(
            """
            SELECT a, b, jaccard_similarity
            FROM verified_edges
            ORDER BY a, b
            """,
            connection,
        )

    finally:
        connection.close()

    edge_a = edges["a"].to_numpy(dtype=np.int64)
    edge_b = edges["b"].to_numpy(dtype=np.int64)
    similarities = edges[
        "jaccard_similarity"
    ].to_numpy(dtype=np.float64)

    if (
        (edge_a < 0).any()
        or (edge_b < 0).any()
        or (edge_a >= EXPECTED_UNIQUE_DOCUMENTS).any()
        or (edge_b >= EXPECTED_UNIQUE_DOCUMENTS).any()
        or (edge_a >= edge_b).any()
    ):
        raise ValueError(
            "Invalid document IDs or edge ordering detected."
        )

    if not np.isfinite(similarities).all():
        raise ValueError(
            "Non-finite similarity values found."
        )

    if (
        (similarities < 0.70).any()
        or (similarities > 1.0).any()
    ):
        raise ValueError(
            "Unexpected similarity values found."
        )

    return edge_a, edge_b, similarities, integrity


def create_stable_component_ids(raw_labels, number_of_components):
    """
    Create reproducible component IDs based on the smallest
    unique-document ID in each component.
    """

    number_of_nodes = len(raw_labels)

    minimum_document_id = np.full(
        number_of_components,
        number_of_nodes,
        dtype=np.int64,
    )

    np.minimum.at(
        minimum_document_id,
        raw_labels,
        np.arange(number_of_nodes, dtype=np.int64),
    )

    component_order = np.argsort(
        minimum_document_id
    )

    remapping = np.empty(
        number_of_components,
        dtype=np.int64,
    )

    remapping[component_order] = np.arange(
        number_of_components,
        dtype=np.int64,
    )

    return remapping[raw_labels]


def analyze_threshold(
    threshold_name,
    configuration,
    edge_a,
    edge_b,
    similarities,
    doc_labels,
    doc_record_counts,
):
    threshold = configuration["value"]

    selected = similarities >= threshold

    selected_a = edge_a[selected]
    selected_b = edge_b[selected]

    edge_count = len(selected_a)

    if edge_count != configuration["expected_edges"]:
        raise ValueError(
            f"Threshold {threshold} expected "
            f"{configuration['expected_edges']:,} edges, "
            f"found {edge_count:,}."
        )

    print(
        f"\nAnalyzing threshold {threshold:.2f}: "
        f"{edge_count:,} edges..."
    )

    # Make the adjacency matrix explicitly symmetric.
    rows = np.concatenate([
        selected_a,
        selected_b,
    ])

    columns = np.concatenate([
        selected_b,
        selected_a,
    ])

    values = np.ones(
        len(rows),
        dtype=np.uint8,
    )

    graph = coo_matrix(
        (
            values,
            (rows, columns),
        ),
        shape=(
            EXPECTED_UNIQUE_DOCUMENTS,
            EXPECTED_UNIQUE_DOCUMENTS,
        ),
    ).tocsr()

    number_of_components, raw_labels = connected_components(
        csgraph=graph,
        directed=False,
        return_labels=True,
    )

    component_ids = create_stable_component_ids(
        raw_labels,
        number_of_components,
    )

    component_sizes = np.bincount(
        component_ids,
        minlength=number_of_components,
    )

    component_record_counts = np.bincount(
        component_ids,
        weights=doc_record_counts,
        minlength=number_of_components,
    ).astype(np.int64)

    # Each edge belongs to the component of either endpoint.
    component_edge_counts = np.bincount(
        component_ids[selected_a],
        minlength=number_of_components,
    )

    # Find components containing more than one Product label.
    label_frame = pd.DataFrame({
        "component_id": component_ids,
        "Product": doc_labels,
    })

    product_diversity = label_frame.groupby(
        "component_id"
    )["Product"].nunique()

    mixed_component = np.zeros(
        number_of_components,
        dtype=bool,
    )

    mixed_component[
        product_diversity.index.to_numpy(dtype=np.int64)
    ] = (
        product_diversity.to_numpy() > 1
    )

    singleton = component_sizes == 1
    size_two = component_sizes == 2
    size_three_plus = component_sizes >= 3
    non_singleton = component_sizes > 1

    non_singleton_sizes = component_sizes[non_singleton]

    largest_component_ids = np.argsort(
        component_sizes
    )[::-1][:20]

    largest_components = []

    for component_id in largest_component_ids:
        size = int(component_sizes[component_id])

        if size <= 1:
            continue

        largest_components.append({
            "component_id": int(component_id),
            "unique_narrative_groups": size,
            "original_complaint_records": int(
                component_record_counts[component_id]
            ),
            "verified_edges": int(
                component_edge_counts[component_id]
            ),
            "number_of_product_labels": int(
                product_diversity.loc[component_id]
            ),
        })

    statistics = {
        "threshold": threshold,
        "verified_edges": int(edge_count),

        "unique_narrative_groups": (
            EXPECTED_UNIQUE_DOCUMENTS
        ),

        "connected_components_including_singletons": int(
            number_of_components
        ),

        "singleton_components": int(
            singleton.sum()
        ),

        "components_with_two_narrative_groups": int(
            size_two.sum()
        ),

        "components_with_three_or_more_groups": int(
            size_three_plus.sum()
        ),

        "non_singleton_components": int(
            non_singleton.sum()
        ),

        "unique_narrative_groups_in_non_singleton_components": int(
            component_sizes[non_singleton].sum()
        ),

        "original_complaint_records_in_non_singleton_components": int(
            component_record_counts[non_singleton].sum()
        ),

        "largest_component_size": int(
            component_sizes.max()
        ),

        "largest_component_original_record_count": int(
            component_record_counts[
                np.argmax(component_sizes)
            ]
        ),

        "non_singleton_component_size_percentiles": (
            {
                f"p{percentile}": float(
                    np.percentile(
                        non_singleton_sizes,
                        percentile,
                    )
                )
                for percentile in [50, 90, 95, 99]
            }
            if len(non_singleton_sizes)
            else {}
        ),

        "components_with_multiple_product_labels": int(
            mixed_component.sum()
        ),

        "unique_narrative_groups_in_multiple_label_components": int(
            component_sizes[mixed_component].sum()
        ),

        "original_complaint_records_in_multiple_label_components": int(
            component_record_counts[mixed_component].sum()
        ),

        "largest_20_components": largest_components,
    }

    return component_ids, statistics


def main():
    print("=" * 68)
    print("FINROUTE AI - FULL DATASET CONNECTED COMPONENT ANALYSIS")
    print("=" * 68)

    validate_inputs()

    mapping, doc_labels, doc_record_counts = (
        load_document_information()
    )

    edge_a, edge_b, similarities, integrity = (
        load_verified_edges()
    )

    results = {}
    assignments = {}

    for name, configuration in THRESHOLDS.items():
        component_ids, statistics = analyze_threshold(
            name,
            configuration,
            edge_a,
            edge_b,
            similarities,
            doc_labels,
            doc_record_counts,
        )

        results[name] = statistics

        assignments[
            configuration["column"]
        ] = component_ids

        print("\nThreshold summary:")
        print(json.dumps(
            statistics,
            indent=4,
        ))

    print("\nPreparing original complaint assignments...")

    document_ids = mapping[
        "unique_document_id"
    ].to_numpy(dtype=np.int64)

    assignment_frame = mapping[
        [
            "row_position",
            "unique_document_id",
            LABEL_COLUMN,
        ]
    ].copy()

    for column_name, component_ids in assignments.items():
        assignment_frame[column_name] = (
            component_ids[document_ids]
        )

    assignment_frame.to_csv(
        ASSIGNMENT_PATH,
        index=False,
        encoding="utf-8",
    )

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v2",

        "source_complaint_records": EXPECTED_SOURCE_ROWS,
        "unique_cleaned_narrative_groups": (
            EXPECTED_UNIQUE_DOCUMENTS
        ),

        "verified_edge_database_integrity": integrity,

        "graph_method": (
            "Undirected connected components using SciPy sparse graphs"
        ),

        "component_id_method": (
            "Stable ordering by the smallest unique-document ID "
            "within each component"
        ),

        "threshold_results": results,

        "assignment_file": str(ASSIGNMENT_PATH),

        "assignment_columns": [
            "component_070",
            "component_085",
            "component_095",
            "component_099",
            "component_100",
        ],

        "all_original_complaint_records_preserved": True,
        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "company_names_exported": False,

        "dataset_modified": False,
        "records_deleted_or_moved": False,
        "model_retrained": False,
        "test_set_evaluated": False,

        "limitation": (
            "Connected components reflect the approximate "
            "LSH candidate-discovery process. They are not "
            "proof that every near-duplicate relationship "
            "in the complete corpus has been found."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("CONNECTED COMPONENT ANALYSIS COMPLETED")
    print("=" * 68)

    print("\nComponent assignment file:")
    print(ASSIGNMENT_PATH)

    print("\nSummary report:")
    print(REPORT_PATH)

    print("\nAll original complaint records are retained.")
    print("No source dataset modified.")
    print("No model retrained.")
    print("No test-set evaluation performed.")


if __name__ == "__main__":
    main()
