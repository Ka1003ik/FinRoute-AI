from collections import Counter
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "interim" / "cfpb_archive" / "CCDB_Export_21_August_2026.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "august_export_data_quality.json"


def normalize(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    total_rows = 0
    narrative_rows = 0
    text_chars = 0
    min_text_length = None
    max_text_length = 0
    missing_product_rows = 0
    missing_id_rows = 0
    duplicate_id_rows = 0
    invalid_date_rows = 0
    product_counts = Counter()
    product_narrative_counts = Counter()
    seen_ids = set()
    first_date = None
    last_date = None

    reader = pd.read_csv(
        CSV_PATH,
        chunksize=100_000,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    for chunk in reader:
        columns = {normalize(name): name for name in chunk.columns}
        needed = {
            "narrative": columns.get("consumercomplaintnarrative"),
            "product": columns.get("product"),
            "complaint_id": columns.get("complaintid"),
            "date": columns.get("datereceived"),
        }
        missing = [name for name, column in needed.items() if column is None]
        if missing:
            raise ValueError(f"Required columns not found: {missing}")

        narrative = chunk[needed["narrative"]].str.strip()
        product = chunk[needed["product"]].str.strip()
        ids = chunk[needed["complaint_id"]].str.strip()

        has_narrative = narrative.ne("")
        total_rows += len(chunk)
        narrative_rows += int(has_narrative.sum())
        missing_product_rows += int(product.eq("").sum())
        product_counts.update(product[product.ne("")].value_counts().to_dict())
        product_narrative_counts.update(
            product[has_narrative & product.ne("")].value_counts().to_dict()
        )

        lengths = narrative[has_narrative].str.len()
        if not lengths.empty:
            text_chars += int(lengths.sum())
            chunk_min = int(lengths.min())
            min_text_length = chunk_min if min_text_length is None else min(min_text_length, chunk_min)
            max_text_length = max(max_text_length, int(lengths.max()))

        missing_id_rows += int(ids.eq("").sum())
        valid_ids = ids[ids.ne("")]
        within_chunk_duplicates = valid_ids.duplicated()
        duplicate_id_rows += int(within_chunk_duplicates.sum())
        for complaint_id in valid_ids[~within_chunk_duplicates]:
            if complaint_id in seen_ids:
                duplicate_id_rows += 1
            else:
                seen_ids.add(complaint_id)

        dates = pd.to_datetime(chunk[needed["date"]], errors="coerce", format="mixed")
        invalid_date_rows += int(dates.isna().sum())
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            chunk_first, chunk_last = valid_dates.min(), valid_dates.max()
            first_date = chunk_first if first_date is None else min(first_date, chunk_first)
            last_date = chunk_last if last_date is None else max(last_date, chunk_last)

    report = {
        "file": CSV_PATH.name,
        "file_size_bytes": CSV_PATH.stat().st_size,
        "total_rows": total_rows,
        "narratives": {
            "non_empty_rows": narrative_rows,
            "empty_rows": total_rows - narrative_rows,
            "non_empty_percent": (
                round(narrative_rows / total_rows * 100, 6)
                if total_rows
                else 0
            ),
            "mean_characters_when_present": round(text_chars / narrative_rows, 1) if narrative_rows else 0,
            "minimum_characters": min_text_length,
            "maximum_characters": max_text_length,
        },
        "products": dict(sorted(product_counts.items())),
        "narratives_by_product": dict(sorted(product_narrative_counts.items())),
        "missing_product_rows": missing_product_rows,
        "complaint_ids": {
            "missing_rows": missing_id_rows,
            "duplicate_rows": duplicate_id_rows,
        },
        "dates_received": {
            "earliest": str(first_date.date()) if first_date is not None else None,
            "latest": str(last_date.date()) if last_date is not None else None,
            "missing_or_unparsed_rows": invalid_date_rows,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
