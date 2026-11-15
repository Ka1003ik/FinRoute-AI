"""Evaluate the saved FinRoute AI DistilBERT v3 model on the untouched test split.

This script does not train, fine-tune, or modify the model or source datasets.
It verifies the test CSV against the SHA-256 recorded during the training run,
then saves one final test report and confusion matrix.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits_v3"
TEST_PATH = SPLIT_DIR / "test.csv"
MODEL_DIR = ROOT / "models" / "transformer" / "distilbert_product_classifier_v3"
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "v3" / "distilbert"
VALIDATION_REPORT_PATH = OUTPUT_DIR / "validation.json"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"
TEST_REPORT_PATH = OUTPUT_DIR / "test.json"
TEST_CONFUSION_PATH = OUTPUT_DIR / "test_confusion.csv"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
BATCH_SIZE = 8

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_from_predictions  # noqa: E402
from src.features.bert_tokenizer import MAX_TOKEN_LENGTH, TextBatchCollator  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComplaintTestDataset(Dataset):
    def __init__(self, texts: list[str], encoded_labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = np.asarray(encoded_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.texts[index], "labels": int(self.labels[index])}


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    # Refuse to overwrite prior test results. This protects the final evaluation.
    if TEST_REPORT_PATH.exists() or TEST_CONFUSION_PATH.exists():
        raise FileExistsError(
            "A V3 test report or test confusion matrix already exists. "
            "Refusing to overwrite or silently repeat the final evaluation."
        )

    require_file(VALIDATION_REPORT_PATH, "DistilBERT V3 validation report")
    require_file(MODEL_DIR / "config.json", "Saved DistilBERT model config")
    require_file(LABEL_MAPPING_PATH, "Saved model label mapping")
    require_file(TEST_PATH, "V3 test split")

    validation_report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    if validation_report.get("project") != "FinRoute AI" or validation_report.get("dataset_version") != "v3":
        raise ValueError("The validation report is not for FinRoute AI dataset V3.")
    if validation_report.get("test_rows_loaded") != 0 or validation_report.get("test_rows_evaluated") != 0:
        raise ValueError("The validation report does not confirm an untouched test set.")

    expected_test_hash = (
        validation_report.get("input_sha256", {}).get("test_csv_hashed_but_not_loaded")
    )
    if not isinstance(expected_test_hash, str) or len(expected_test_hash) != 64:
        raise ValueError("The validation report has no valid saved SHA-256 for test.csv.")

    test_hash_before = sha256_file(TEST_PATH)
    if test_hash_before != expected_test_hash:
        raise RuntimeError(
            "The current test.csv does not match the SHA-256 recorded during V3 training. "
            "Evaluation cancelled; do not replace or edit the test file."
        )
    print("V3 test.csv SHA-256 verification: PASS", flush=True)

    saved_mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    labels = saved_mapping.get("labels")
    label_to_id = saved_mapping.get("label_to_id")
    if not isinstance(labels, list) or len(labels) != 11:
        raise ValueError("Saved label_mapping.json must contain exactly 11 labels.")
    expected_mapping = {str(label): index for index, label in enumerate(labels)}
    if label_to_id != expected_mapping:
        raise ValueError("Saved label_to_id does not match the order of saved labels.")
    if labels != validation_report.get("product_categories"):
        raise ValueError("Saved model labels differ from the V3 validation report.")

    # Load the fine-tuned model/tokenizer from the local final model directory.
    # Do not call load_distilbert_classifier(): that factory initializes a fresh head.
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True, use_safetensors=True
    )
    if int(model.config.num_labels) != len(labels):
        raise ValueError("Saved model output dimension does not match the 11 labels.")
    saved_id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    expected_id2label = {index: str(label) for index, label in enumerate(labels)}
    if saved_id2label != expected_id2label:
        raise ValueError("Saved model config label order differs from label_mapping.json.")

    print("Saved fine-tuned model and tokenizer loaded locally: PASS", flush=True)

    # Read only the test columns needed for the final classification evaluation.
    test = pd.read_csv(
        TEST_PATH,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if test.empty:
        raise ValueError("V3 test split is empty.")
    unknown_labels = sorted(set(test[LABEL_COLUMN]) - set(label_to_id))
    if unknown_labels:
        raise ValueError(f"Test split contains labels absent from saved mapping: {unknown_labels}")
    missing_labels = sorted(set(labels) - set(test[LABEL_COLUMN]))
    if missing_labels:
        raise ValueError(f"Test split is missing expected Product categories: {missing_labels}")

    y_true = test[LABEL_COLUMN].tolist()
    encoded_labels = np.asarray([label_to_id[label] for label in y_true], dtype=np.int64)
    dataset = ComplaintTestDataset(test[TEXT_COLUMN].tolist(), encoded_labels)
    collator = TextBatchCollator(tokenizer, max_length=MAX_TOKEN_LENGTH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )

    all_logits: list[np.ndarray] = []
    all_predictions: list[int] = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            batch.pop("labels")  # Gold labels are retained separately for scoring.
            model_inputs = {
                name: tensor.to(device, non_blocking=(device.type == "cuda"))
                for name, tensor in batch.items()
            }
            logits = model(**model_inputs).logits
            all_logits.append(logits.float().cpu().numpy())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            if batch_number % 200 == 0:
                print(f"Evaluated {min(batch_number * BATCH_SIZE, len(dataset)):,}/{len(dataset):,} rows", flush=True)

    logits_array = np.concatenate(all_logits, axis=0)
    y_pred = [labels[index] for index in all_predictions]
    metrics, confusion = metrics_from_predictions(
        y_true,
        y_pred,
        labels,
        rankings=logits_array,
    )

    # Confirm that the test bytes did not change while evaluation was running.
    test_hash_after = sha256_file(TEST_PATH)
    if test_hash_after != test_hash_before:
        raise RuntimeError("test.csv changed during evaluation; results were not saved.")

    validation_report_hash = sha256_file(VALIDATION_REPORT_PATH)
    report = {
        "project": "FinRoute AI",
        "dataset_version": "v3",
        "experiment": "final_distilbert_test_evaluation",
        "model_directory": str(MODEL_DIR),
        "test_rows_evaluated": int(len(test)),
        "product_categories": labels,
        "results": metrics,
        "input_sha256": {
            "test_csv": test_hash_after,
            "training_validation_report": validation_report_hash,
        },
        "evaluation": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "max_token_length": MAX_TOKEN_LENGTH,
            "model_loaded_from_saved_local_directory": True,
            "model_retrained_or_finetuned": False,
            "test_hash_verified_before_and_after_evaluation": True,
        },
        "artifacts": {
            "test_report": str(TEST_REPORT_PATH),
            "test_confusion_matrix": str(TEST_CONFUSION_PATH),
        },
    }

    pd.DataFrame(confusion, index=labels, columns=labels).to_csv(
        TEST_CONFUSION_PATH, encoding="utf-8"
    )
    with TEST_REPORT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\nDISTILBERT V3 FINAL TEST EVALUATION COMPLETED", flush=True)
    print(f"Test rows: {len(test):,}", flush=True)
    print(f"Accuracy: {metrics['accuracy']:.4f}", flush=True)
    print(f"Macro-F1: {metrics['macro_f1']:.4f}", flush=True)
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}", flush=True)
    print(f"Top-2 accuracy: {metrics.get('top_2_accuracy', float('nan')):.4f}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Test report: {TEST_REPORT_PATH}", flush=True)
    print(f"Test confusion matrix: {TEST_CONFUSION_PATH}", flush=True)


if __name__ == "__main__":
    main()
