from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cfpb_archive"
    / "CCDB_Export_21_August_2026.csv"
)


def normalize(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    sample = pd.read_csv(CSV_PATH, nrows=5, dtype=str, keep_default_na=False)
    columns = {normalize(name): name for name in sample.columns}

    print(f"File size: {CSV_PATH.stat().st_size:,} bytes")
    print("\nColumns:")
    for column in sample.columns:
        print(f"- {column}")

    narrative_column = next(
        (
            columns[name]
            for name in (
                "consumercomplaintnarrative",
                "complaintwhathappened",
                "narrative",
            )
            if name in columns
        ),
        None,
    )
    product_column = columns.get("product")

    print(f"\nNarrative column: {narrative_column or 'NOT FOUND'}")
    print(f"Product label column: {product_column or 'NOT FOUND'}")

    if narrative_column:
        lengths = sample[narrative_column].str.len()
        print(f"Non-empty narratives in sample: {(lengths > 0).sum()} of {len(sample)}")
        print(f"Sample narrative lengths: {lengths.tolist()}")

    if product_column:
        print("\nProduct labels in sample:")
        print(sample[product_column].value_counts().to_string())


if __name__ == "__main__":
    main()