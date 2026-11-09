
import json
import re
from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_narrative
from src.preprocessing.text_vectorizer import feature_signature


INPUT_PATH = (
    ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
)

MODEL_DIR = (
    ROOT / "models" / "transformer"
    / "distilbert_product_classifier_3epoch"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "leakage_audit"
    / "featureless_records_diagnostic.json"
)

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


def main():

    print("=" * 65)
    print("FINROUTE AI - FEATURELESS RECORD DIAGNOSTIC")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(MODEL_DIR)

    data = pd.read_csv(
        INPUT_PATH,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    # Identify records excluded by the original
    # baseline feature-signature preparation.
    data["_signature"] = data[TEXT_COLUMN].map(
        feature_signature
    )

    featureless = data[
        data["_signature"] == ""
    ].copy()

    print(f"\nOriginal dataset rows: {len(data):,}")
    print(f"Featureless records: {len(featureless):,}")

    print("\nLoading saved DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    results = []

    for row_position, row in featureless.iterrows():

        raw_text = row[TEXT_COLUMN]
        cleaned_text = clean_narrative(raw_text)

        word_like_tokens = re.findall(
            r"(?u)\b\w\w+\b",
            cleaned_text
        )

        encoded = tokenizer(
            cleaned_text,
            add_special_tokens=False,
            truncation=False
        )

        bert_token_count = len(encoded["input_ids"])

        results.append({
            "source_row_position": int(row_position),
            "product": row[LABEL_COLUMN],
            "raw_character_count": len(raw_text),
            "cleaned_character_count": len(cleaned_text),
            "word_like_token_count": len(word_like_tokens),
            "distilbert_token_count": bert_token_count,
            "has_alphanumeric_content": any(
                character.isalnum()
                for character in cleaned_text
            )
        })

    empty_count = sum(
        item["cleaned_character_count"] == 0
        for item in results
    )

    nonempty_count = len(results) - empty_count

    bert_usable_count = sum(
        item["distilbert_token_count"] > 0
        for item in results
    )

    report = {
        "project": "FinRoute AI",
        "total_featureless_records": len(results),
        "empty_after_canonical_cleaning": empty_count,
        "nonempty_after_canonical_cleaning": nonempty_count,
        "records_with_distilbert_tokens": bert_usable_count,
        "records_without_distilbert_tokens": (
            len(results) - bert_usable_count
        ),
        "product_distribution": (
            featureless[LABEL_COLUMN]
            .value_counts()
            .to_dict()
        ),
        "record_diagnostics": results,
        "raw_narratives_exported": False,
        "complaint_ids_exported": False,
        "dataset_modified": False
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Report already exists: {OUTPUT_PATH}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n=== RESULTS ===")
    print(f"Empty after cleaning: {empty_count}")
    print(f"Non-empty after cleaning: {nonempty_count}")
    print(f"DistilBERT-tokenizable: {bert_usable_count}")
    print(
        "Without DistilBERT tokens:",
        len(results) - bert_usable_count
    )

    print("\nIndividual record diagnostics:")

    for item in results:
        print(
            f"Row {item['source_row_position']}: "
            f"Product={item['product']}, "
            f"Clean characters={item['cleaned_character_count']}, "
            f"TF-IDF tokens={item['word_like_token_count']}, "
            f"DistilBERT tokens={item['distilbert_token_count']}"
        )

    print("\nReport saved:")
    print(OUTPUT_PATH)

    print("\nNo narrative text or complaint IDs exported.")
    print("No dataset modifications performed.")
    print("DIAGNOSTIC COMPLETED.")


if __name__ == "__main__":
    main()
