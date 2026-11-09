"""Pinned DistilBERT tokenizer and dynamic batch-collation utilities."""

import torch
from transformers import AutoTokenizer


DISTILBERT_MODEL_ID = "distilbert/distilbert-base-uncased"
DISTILBERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MAX_TOKEN_LENGTH = 512


def load_distilbert_tokenizer(cache_dir):
    """Load the fast tokenizer from a pinned model revision."""
    return AutoTokenizer.from_pretrained(
        DISTILBERT_MODEL_ID,
        revision=DISTILBERT_REVISION,
        cache_dir=str(cache_dir),
        use_fast=True,
    )


class TextBatchCollator:
    """Tokenize a batch and pad only to that batch's longest text."""

    def __init__(self, tokenizer, max_length=MAX_TOKEN_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        encoded = self.tokenizer(
            [example["text"] for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [example["labels"] for example in examples],
            dtype=torch.long,
        )
        return encoded
