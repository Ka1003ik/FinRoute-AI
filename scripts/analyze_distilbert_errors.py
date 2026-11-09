
"""Generate privacy-preserving DistilBERT validation error analysis."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.bert_tokenizer import (
    TextBatchCollator,
    MAX_TOKEN_LENGTH,
)

from src.evaluation.metrics import metrics_from_predictions


# Input files
VALIDATION_PATH = (
    ROOT / "data" / "processed" / "splits" / "validation.csv"
)

MODEL_DIR = (
    ROOT / "models" / "transformer"
    / "distilbert_product_classifier_3epoch"
)

# Output directory
OUTPUT_DIR = ROOT / "outputs" / "error_analysis"

ERRORS_PATH = OUTPUT_DIR / "distilbert_validation_errors.csv"
SUMMARY_PATH = OUTPUT_DIR / "distilbert_prediction_summary.csv"
REPORT_PATH = OUTPUT_DIR / "distilbert_error_analysis.json"

# Dataset configuration
TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"

BATCH_SIZE = 8
MAX_LENGTH = MAX_TOKEN_LENGTH
EXPECTED_CLASSES = 11


# ============================================================
# 2. VALIDATION DATASET
# ============================================================

class ValidationComplaintDataset(Dataset):
    """Prepare validation complaints for batch inference."""

    def __init__(self, dataframe, label_to_id):
        self.texts = dataframe[TEXT_COLUMN].astype(str).tolist()

        if any(not text.strip() for text in self.texts):
            raise ValueError(
                "Validation dataset contains empty complaint narratives."
            )

        encoded_labels = dataframe[LABEL_COLUMN].map(label_to_id)

        if encoded_labels.isna().any():
            raise ValueError(
                "Validation dataset contains unknown product labels."
            )

        self.labels = encoded_labels.to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "labels": int(self.labels[index]),
        }


# ============================================================
# 3. LOAD MODEL, TOKENIZER AND VALIDATION DATA
# ============================================================

def load_validation_assets():
    """Load the saved model, tokenizer, labels and validation dataset."""

    required_paths = [
        VALIDATION_PATH,
        MODEL_DIR / "config.json",
        MODEL_DIR / "label_mapping.json",
        MODEL_DIR / "tokenizer_config.json",
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print("Loading validation dataset...", flush=True)

    validation_df = pd.read_csv(
        VALIDATION_PATH,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    print(
        f"Validation records: {len(validation_df):,}",
        flush=True,
    )

    # Load the saved training label mapping.
    mapping = json.loads(
        (MODEL_DIR / "label_mapping.json").read_text(
            encoding="utf-8"
        )
    )

    labels = mapping["labels"]
    label_to_id = mapping["label_to_id"]

    if len(labels) != EXPECTED_CLASSES:
        raise ValueError("Expected exactly 11 product categories.")

    if len(set(labels)) != EXPECTED_CLASSES:
        raise ValueError("Duplicate labels in saved label mapping.")

    if set(labels) != set(label_to_id):
        raise ValueError("Saved label mapping is inconsistent.")

    if any(
        label_to_id[label] != index
        for index, label in enumerate(labels)
    ):
        raise ValueError("Label IDs do not match saved label order.")

    unknown_labels = (
        set(validation_df[LABEL_COLUMN].unique()) - set(labels)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown validation categories: {unknown_labels}"
        )

    print("Loading trained three-epoch model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    if model.config.num_labels != EXPECTED_CLASSES:
        raise ValueError(
            "Model output dimension does not match label mapping."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Device: {device}", flush=True)
    print(f"Model classes: {model.config.num_labels}", flush=True)

    dataset = ValidationComplaintDataset(
        validation_df,
        label_to_id,
    )

    collator = TextBatchCollator(
        tokenizer,
        max_length=MAX_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    return validation_df, labels, model, dataloader, device


# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, dataloader, device):
    """Run batched inference and collect softmax scores."""

    all_probabilities = []
    all_true_labels = []

    model.eval()

    print("\nGenerating validation predictions...", flush=True)

    with torch.inference_mode():

        for batch in dataloader:

            true_labels = batch.pop("labels")

            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**inputs)

            # Convert logits to FP32 before softmax.
            logits = outputs.logits.float()

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_true_labels.append(
                true_labels.numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    true_label_ids = np.concatenate(
        all_true_labels,
        axis=0,
    )

    if len(probabilities) != len(true_label_ids):
        raise ValueError("Prediction and label counts differ.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite prediction scores detected.")

    if probabilities.shape[1] != EXPECTED_CLASSES:
        raise ValueError("Unexpected prediction output dimension.")

    print(
        f"Predictions generated: {len(probabilities):,}",
        flush=True,
    )

    return true_label_ids, probabilities


# ============================================================
# 5. ANALYZE TOP-1 AND TOP-2 PREDICTIONS
# ============================================================

def analyze_predictions(
    validation_df,
    labels,
    true_label_ids,
    probabilities,
):
    """Calculate predictions, ranking scores and classification errors."""

    # Highest-ranked class first.
    ranked_indices = np.argsort(
        -probabilities,
        axis=1,
    )

    top1_indices = ranked_indices[:, 0]
    top2_indices = ranked_indices[:, 1]

    top1_scores = probabilities[
        np.arange(len(probabilities)),
        top1_indices,
    ]

    top2_scores = probabilities[
        np.arange(len(probabilities)),
        top2_indices,
    ]

    score_margins = top1_scores - top2_scores

    true_labels = np.asarray(labels)[true_label_ids]
    predicted_labels = np.asarray(labels)[top1_indices]

    errors_mask = true_label_ids != top1_indices
    error_indices = np.flatnonzero(errors_mask)

    # Only incorrect predictions are exported.
    # Complaint narratives and complaint IDs are excluded.
    errors_df = pd.DataFrame({
        "validation_row_index": error_indices,
        "actual_product": true_labels[errors_mask],
        "predicted_product": predicted_labels[errors_mask],
        "top1_softmax_score": top1_scores[errors_mask],
        "top2_product": np.asarray(labels)[
            top2_indices[errors_mask]
        ],
        "top2_softmax_score": top2_scores[errors_mask],
        "top1_top2_margin": score_margins[errors_mask],
    })

    # Calculate consistent classification metrics.
    metrics, confusion = metrics_from_predictions(
        true_labels,
        predicted_labels,
        labels,
        rankings=probabilities,
    )

    return (
        errors_df,
        metrics,
        confusion,
        true_labels,
        predicted_labels,
        top1_scores,
        top2_indices,
        top2_scores,
        score_margins,
    )


# ============================================================
# 6. CREATE PER-CLASS SUMMARY
# ============================================================

def create_class_summary(
    labels,
    metrics,
    confusion,
    predicted_labels,
):
    """Create a per-product classification summary."""

    class_report = metrics["classification_report"]

    predicted_counts = {
        label: int(np.sum(predicted_labels == label))
        for label in labels
    }

    records = []

    for index, label in enumerate(labels):

        class_metrics = class_report[label]

        support = int(class_metrics["support"])
        correct = int(confusion[index, index])

        records.append({
            "product": label,
            "support": support,
            "predicted_count": predicted_counts[label],
            "correct_predictions": correct,
            "misclassified_count": support - correct,
            "precision": class_metrics["precision"],
            "recall": class_metrics["recall"],
            "f1_score": class_metrics["f1-score"],
        })

    return pd.DataFrame(records)


# ============================================================
# 7. SAVE PRIVACY-PRESERVING REPORTS
# ============================================================

def save_analysis_outputs(
    errors_df,
    summary_df,
    metrics,
    confusion,
    labels,
    total_records,
):
    """Save error analysis without exporting complaint narratives."""

    # Protect existing analysis outputs from accidental overwriting.
    existing_outputs = [
        path
        for path in (ERRORS_PATH, SUMMARY_PATH, REPORT_PATH)
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Existing analysis files would be overwritten. "
            f"Please preserve or rename them first: {existing_outputs}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors_df.to_csv(
        ERRORS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "project": "FinRoute AI",
        "model": str(MODEL_DIR.relative_to(ROOT)),
        "evaluation_partition": "validation",
        "test_rows_used": 0,
        "total_validation_records": int(total_records),
        "correct_predictions": int(
            total_records - len(errors_df)
        ),
        "incorrect_predictions": int(len(errors_df)),
        "error_rate": float(
            len(errors_df) / total_records
        ),
        "metrics": metrics,
        "labels": labels,
        "confusion_matrix": confusion.tolist(),
        "per_class_summary": summary_df.to_dict(
            orient="records"
        ),
        "score_interpretation": (
            "Top-1 and top-2 scores are softmax outputs. "
            "They are ranking scores and must not be interpreted "
            "as calibrated probabilities."
        ),
        "privacy_note": (
            "Complaint narratives and complaint IDs are not "
            "included in exported analysis files."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nAnalysis files saved:", flush=True)
    print(f"Errors: {ERRORS_PATH}", flush=True)
    print(f"Summary: {SUMMARY_PATH}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():
    """Execute privacy-preserving validation error analysis."""

    print("=" * 60)
    print("FINROUTE AI - DISTILBERT ERROR ANALYSIS")
    print("=" * 60)

    (
        validation_df,
        labels,
        model,
        dataloader,
        device,
    ) = load_validation_assets()

    true_label_ids, probabilities = generate_predictions(
        model,
        dataloader,
        device,
    )

    (
        errors_df,
        metrics,
        confusion,
        true_labels,
        predicted_labels,
        top1_scores,
        top2_indices,
        top2_scores,
        score_margins,
    ) = analyze_predictions(
        validation_df,
        labels,
        true_label_ids,
        probabilities,
    )

    if len(true_labels) != len(validation_df):
        raise ValueError("Validation row alignment check failed.")

    if confusion.sum() != len(validation_df):
        raise ValueError("Confusion matrix record count is incorrect.")

    summary_df = create_class_summary(
        labels,
        metrics,
        confusion,
        predicted_labels,
    )

    save_analysis_outputs(
        errors_df,
        summary_df,
        metrics,
        confusion,
        labels,
        len(validation_df),
    )

    print("\n=== OVERALL VALIDATION RESULTS ===")

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top_2_accuracy",
    ):
        print(
            f"{metric}: {metrics[metric]:.4f}"
        )

    print("\n=== ERROR ANALYSIS ===")
    print(f"Total records: {len(validation_df):,}")
    print(f"Correct: {len(validation_df) - len(errors_df):,}")
    print(f"Incorrect: {len(errors_df):,}")

    print("\n=== LOWEST PER-CLASS F1 ===")

    print(
        summary_df.sort_values(
            "f1_score",
            ascending=True,
        )[
            [
                "product",
                "support",
                "precision",
                "recall",
                "f1_score",
            ]
        ].to_string(index=False)
    )

    print("\nALL ERROR ANALYSIS CHECKS COMPLETED.")


if __name__ == "__main__":
    main()
