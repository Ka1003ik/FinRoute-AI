"""Prepare a single-label dataset while retaining repeated same-label cases."""

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_vectorizer import (
    BASE_VECTORIZER_DESCRIPTION,
    feature_signature,
)


INPUT_CSV = ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "cfpb_2026_modeling.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "modeling_dataset_prep.json"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


def main() -> None:
    """Exclude only exact narratives associated with conflicting products."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

    data = pd.read_csv(
        INPUT_CSV,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    data[TEXT_COLUMN] = data[TEXT_COLUMN].str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].str.strip()

    data["_feature_signature"] = data[TEXT_COLUMN].map(feature_signature)
    featureless_mask = data["_feature_signature"].eq("")

    products_per_signature = data.loc[~featureless_mask].groupby(
        "_feature_signature", sort=False
    )[LABEL_COLUMN].nunique()
    conflicting_signatures = products_per_signature[
        products_per_signature > 1
    ].index

    conflicting_mask = data["_feature_signature"].isin(conflicting_signatures)
    conflicting_rows = data.loc[conflicting_mask]
    excluded_mask = conflicting_mask | featureless_mask
    modeling_rows = data.loc[~excluded_mask].copy()

    remaining_label_conflicts = (
        modeling_rows.groupby("_feature_signature")[LABEL_COLUMN].nunique() > 1
    ).any()
    if remaining_label_conflicts:
        raise ValueError("Conflicting product labels remain after filtering.")

    equivalent_rows_retained = int(
        modeling_rows.duplicated(subset="_feature_signature").sum()
    )
    modeling_data = modeling_rows.drop(columns="_feature_signature")
    product_counts = modeling_data[LABEL_COLUMN].value_counts().sort_values(
        ascending=False
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    modeling_data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    report = {
        "input_rows": int(len(data)),
        "conflicting_tfidf_feature_groups_excluded": int(
            len(conflicting_signatures)
        ),
        "rows_in_conflicting_groups_excluded": int(len(conflicting_rows)),
        "rows_without_baseline_features_excluded": int(featureless_mask.sum()),
        "same_label_equivalent_rows_retained": equivalent_rows_retained,
        "output_rows": int(len(modeling_data)),
        "product_counts": {
            str(label): int(count)
            for label, count in product_counts.items()
        },
        "baseline_vectorizer": BASE_VECTORIZER_DESCRIPTION,
        "output_file": str(OUTPUT_CSV),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
