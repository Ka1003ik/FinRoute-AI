"""Profile token lengths on train and validation without reading the test split."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_DIR = ROOT / "models" / "transformer"
os.environ.setdefault("HF_HOME", str(TRANSFORMER_DIR / "hf_home"))
os.environ.setdefault("HF_XET_CACHE", str(TRANSFORMER_DIR / "hf_home" / "xet"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.bert_tokenizer import (
    DISTILBERT_MODEL_ID,
    DISTILBERT_REVISION,
    MAX_TOKEN_LENGTH,
    load_distilbert_tokenizer,
)


SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_PATH = ROOT / "outputs" / "reports" / "distilbert_token_length_profile.json"
TEXT_COLUMN = "Consumer complaint narrative"
BATCH_SIZE = 128
LENGTH_LIMITS = (128, 256, 384, 512)


def profile_partition(tokenizer, split_name):
    """Count untruncated token lengths in bounded CSV chunks."""
    path = SPLIT_DIR / f"{split_name}.csv"
    lengths = []
    for chunk in pd.read_csv(
        path,
        usecols=[TEXT_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=2_000,
    ):
        texts = chunk[TEXT_COLUMN].tolist()
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(batch, padding=False, truncation=False)
            lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
        print(f"{split_name}: tokenized {len(lengths):,} rows", flush=True)

    token_lengths = np.asarray(lengths, dtype=np.int32)
    over_limit = int((token_lengths > MAX_TOKEN_LENGTH).sum())
    return {
        "rows": int(len(token_lengths)),
        "mean_tokens": round(float(token_lengths.mean()), 2),
        "median_tokens": float(np.median(token_lengths)),
        "p90_tokens": float(np.percentile(token_lengths, 90)),
        "p95_tokens": float(np.percentile(token_lengths, 95)),
        "p99_tokens": float(np.percentile(token_lengths, 99)),
        "maximum_tokens": int(token_lengths.max()),
        "rows_truncated_at_configured_max_length": over_limit,
        "share_truncated": round(over_limit / len(token_lengths), 6),
        "truncation_by_limit": {
            str(limit): {
                "rows": int((token_lengths > limit).sum()),
                "share": round(float((token_lengths > limit).mean()), 6),
            }
            for limit in LENGTH_LIMITS
        },
    }


def main() -> None:
    """Write aggregate token-length statistics, never complaint text."""
    tokenizer = load_distilbert_tokenizer(TRANSFORMER_DIR / "cache")
    report = {
        "model_id": DISTILBERT_MODEL_ID,
        "model_revision": DISTILBERT_REVISION,
        "max_token_length": MAX_TOKEN_LENGTH,
        "token_lengths_include_special_tokens": True,
        "partitions": {
            "train": profile_partition(tokenizer, "train"),
            "validation": profile_partition(tokenizer, "validation"),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved token profile to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
