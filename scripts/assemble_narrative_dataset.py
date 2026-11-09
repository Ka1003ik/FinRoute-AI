from collections import Counter
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "interim" / "cfpb_2026"
OUTPUT_CSV = ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "cfpb_2026_data_quality.json"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
ID_COLUMN = "Complaint ID"


def main() -> None:
    csv_paths = sorted(INPUT_DIR.rglob("*.csv"))
    if len(csv_paths) != 6:
        raise ValueError(f"Expected 6 CSV files under {INPUT_DIR}, found {len(csv_paths)}")

    expected_columns = None
    temp_output = OUTPUT_CSV.with_suffix(".csv.tmp")
    temp_output.unlink(missing_ok=True)

    total_rows = 0
    rows_with_text = 0
    rows_saved = 0
    missing_product_for_text = 0
    missing_ids = 0
    duplicate_ids = 0
    seen_ids = set()
    product_counts = Counter()
    per_file = {}

    for csv_path in csv_paths:
        file_rows = 0
        file_text_rows = 0
        file_saved_rows = 0

        for chunk in pd.read_csv(
            csv_path,
            chunksize=100_000,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        ):
            columns = list(chunk.columns)

            if expected_columns is None:
                expected_columns = columns
                required = {TEXT_COLUMN, LABEL_COLUMN, ID_COLUMN}
                missing = required.difference(columns)
                if missing:
                    raise ValueError(f"Required columns missing: {sorted(missing)}")
            elif set(columns) != set(expected_columns):
                raise ValueError(f"Column mismatch in {csv_path.name}")

            chunk = chunk[expected_columns]
            text = chunk[TEXT_COLUMN].str.strip()
            labels = chunk[LABEL_COLUMN].str.strip()
            has_text = text.ne("")
            keep = has_text & labels.ne("")
            selected = chunk.loc[keep].copy()

            total_rows += len(chunk)
            file_rows += len(chunk)
            rows_with_text += int(has_text.sum())
            file_text_rows += int(has_text.sum())
            missing_product_for_text += int((has_text & labels.eq("")).sum())
            rows_saved += len(selected)
            file_saved_rows += len(selected)
            product_counts.update(labels[keep].value_counts().to_dict())

            ids = selected[ID_COLUMN].str.strip()
            missing_ids += int(ids.eq("").sum())

            for complaint_id in ids[ids.ne("")]:
                if complaint_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(complaint_id)

            if not selected.empty:
                selected.to_csv(
                    temp_output,
                    mode="a",
                    header=not temp_output.exists(),
                    index=False,
                    encoding="utf-8",
                )

        per_file[csv_path.name] = {
            "rows": file_rows,
            "rows_with_narrative": file_text_rows,
            "rows_saved": file_saved_rows,
        }

    if not temp_output.exists():
        raise ValueError("No rows with both narrative text and product labels were found.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temp_output.replace(OUTPUT_CSV)

    report = {
        "source_files": per_file,
        "total_source_rows": total_rows,
        "rows_with_narrative": rows_with_text,
        "rows_saved_for_modeling": rows_saved,
        "rows_with_narrative_but_missing_product": missing_product_for_text,
        "missing_complaint_ids_in_saved_rows": missing_ids,
        "duplicate_complaint_ids_in_saved_rows": duplicate_ids,
        "saved_rows_by_product": dict(sorted(product_counts.items())),
        "output_file": str(OUTPUT_CSV),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()