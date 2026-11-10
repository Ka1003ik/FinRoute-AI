"""Train a fresh DistilBERT classifier on FinRoute AI's v3 group-aware split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_DIR = ROOT / "models" / "transformer"
CACHE_DIR = TRANSFORMER_DIR / "cache"
HF_HOME = TRANSFORMER_DIR / "hf_home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_XET_CACHE", str(HF_HOME / "xet"))

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_from_predictions
from src.features.bert_tokenizer import (
    DISTILBERT_MODEL_ID,
    DISTILBERT_REVISION,
    MAX_TOKEN_LENGTH,
    TextBatchCollator,
    load_distilbert_tokenizer,
)
from src.models.distilbert import load_distilbert_classifier


# Inputs: v3 train/validation data and the independent audit report.
SPLIT_DIR = ROOT / "data" / "processed" / "splits_v3"
AUDIT_PATH = (
    ROOT
    / "outputs"
    / "leakage_audit"
    / "independent_saved_split_audit_v3_hashed.json"
)

# All v3 DistilBERT artifacts are kept separate from the previous experiment.
CHECKPOINT_DIR = TRANSFORMER_DIR / "distilbert_v3_checkpoints"
MODEL_DIR = TRANSFORMER_DIR / "distilbert_product_classifier_v3"
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "v3" / "distilbert"
REPORT_PATH = OUTPUT_DIR / "validation.json"
CONFUSION_PATH = OUTPUT_DIR / "validation_confusion.csv"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
RANDOM_STATE = 42
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 6
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2


class ComplaintDataset(Dataset):
    """Keep narratives on CPU and tokenize each batch in the data collator."""

    def __init__(self, texts: pd.Series, labels: np.ndarray) -> None:
        self.texts = texts.tolist()
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.texts[index], "labels": int(self.labels[index])}


class ClassWeightedTrainer(Trainer):
    """Use training-set class-balanced weights in the cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs) -> None:
        self.class_weights = class_weights
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits.float(),
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required v3 split not found: {path}")
    return pd.read_csv(
        path,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def verify_audit_and_hashes() -> tuple[dict, dict[str, str]]:
    """Require the successful hashed audit and unchanged saved split files."""
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(
            f"Hashed v3 audit report not found: {AUDIT_PATH}. "
            "Run audit_saved_group_splits_v3_hashed.py first."
        )

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("project") != "FinRoute AI"
        or audit.get("dataset_version") != "v3"
        or audit.get("component_threshold") != "Jaccard = 1.00"
    ):
        raise ValueError("Audit report does not describe the expected FinRoute AI v3 split.")

    required_checks = (
        "all_three_saved_files_audited",
        "complete_rows_match_expected_source_rows",
        "all_product_distributions_match",
        "all_components_remain_intact",
        "all_unique_narratives_remain_intact",
        "no_identical_complete_rows_cross_partitions",
        "all_original_records_preserved",
    )
    checks = audit.get("final_checks", {})
    failed = [name for name in required_checks if checks.get(name) is not True]
    if failed:
        raise ValueError(f"Required independent audit checks did not pass: {failed}")

    separation = audit.get("cross_partition_checks", {})
    for name in (
        "cross_partition_component_overlap_count",
        "cross_partition_unique_narrative_overlap_count",
        "cross_partition_exact_row_overlap_count",
    ):
        if separation.get(name) != 0:
            raise ValueError(f"Audit reports cross-partition overlap: {name}={separation.get(name)}")

    conservation = audit.get("dataset_conservation", {})
    if conservation.get("exact_source_row_conservation") is not True:
        raise ValueError("Audit did not confirm complete source-row conservation.")

    expected_hashes = audit.get("saved_split_sha256", {})
    actual_hashes = {}
    for name in ("train", "validation", "test"):
        path = SPLIT_DIR / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required v3 split not found: {path}")
        expected = expected_hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Audit report has no valid SHA-256 for {name}.csv.")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name}.csv no longer matches the independent audit SHA-256.")
        actual_hashes[name] = actual

    print("Hashed independent v3 split audit: PASS", flush=True)
    print("Train, validation and test file SHA-256 checks: PASS", flush=True)
    return audit, actual_hashes


def validate_split_contents(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    audit: dict,
) -> list[str]:
    """Cross-check row counts and Product distributions against the audit."""
    for name, frame in (("train", train), ("validation", validation)):
        expected = audit["partitions"][name]
        if len(frame) != expected["records"]:
            raise ValueError(
                f"{name} row count mismatch: expected {expected['records']:,}, "
                f"found {len(frame):,}."
            )
        actual_distribution = {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().items()
        }
        expected_distribution = {
            str(label): int(count)
            for label, count in expected["product_distribution"].items()
        }
        if actual_distribution != expected_distribution:
            raise ValueError(f"{name} Product distribution differs from the audit report.")

    train_labels = set(train[LABEL_COLUMN].unique())
    validation_labels = set(validation[LABEL_COLUMN].unique())
    if train_labels != validation_labels or len(train_labels) != 11:
        raise ValueError("Train and validation must contain the same 11 Product categories.")
    return sorted(train_labels)


def checkpoint_metadata(input_hashes: dict[str, str], labels: list[str]) -> dict:
    return {
        "dataset_version": "v3",
        "audit_report_sha256": sha256_file(AUDIT_PATH),
        "split_sha256": input_hashes,
        "model_id": DISTILBERT_MODEL_ID,
        "model_revision": DISTILBERT_REVISION,
        "labels": labels,
        "random_state": RANDOM_STATE,
        "epochs": NUM_TRAIN_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_token_length": MAX_TOKEN_LENGTH,
        "class_weighting": "balanced cross-entropy from v3 train labels",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "optimizer": "adamw_torch",
    }


def prepare_checkpoint_directory(metadata: dict, create: bool = False) -> None:
    """Prevent accidental resumption from a checkpoint made for other data."""
    if create:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_DIR.exists():
        return

    metadata_path = CHECKPOINT_DIR / "run_metadata.json"
    existing_items = list(CHECKPOINT_DIR.iterdir())
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved_metadata != metadata:
            raise RuntimeError(
                "Existing DistilBERT v3 checkpoint metadata differs from this run. "
                "Refusing to resume from an incompatible checkpoint."
            )
    elif existing_items:
        raise RuntimeError(
            f"Checkpoint directory contains files but no run metadata: {CHECKPOINT_DIR}. "
            "Refusing to reuse an unverified checkpoint directory."
        )
    elif create:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or preflight DistilBERT v3.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the audit, split files, class distributions, CUDA and output paths without training.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured DistilBERT v3 run.")
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"Final model directory already exists; refusing to overwrite it: {MODEL_DIR}"
        )
    if REPORT_PATH.exists() or CONFUSION_PATH.exists():
        raise FileExistsError(
            "A DistilBERT v3 report or confusion matrix already exists. "
            "Refusing to overwrite previous experiment results."
        )

    set_seed(RANDOM_STATE)
    audit, input_hashes = verify_audit_and_hashes()
    train = read_split(SPLIT_DIR / "train.csv")
    validation = read_split(SPLIT_DIR / "validation.csv")
    labels = validate_split_contents(train, validation, audit)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train[LABEL_COLUMN].to_numpy())
    y_validation = label_encoder.transform(validation[LABEL_COLUMN].to_numpy())
    labels = [str(label) for label in label_encoder.classes_]
    class_indices = np.arange(len(labels))

    metadata = checkpoint_metadata(input_hashes, labels)
    prepare_checkpoint_directory(metadata, create=args.preflight_only is False)

    if args.preflight_only:
        print("Train/validation row counts and Product distributions: PASS")
        print(f"Product categories: {len(labels)}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Pinned model: {DISTILBERT_MODEL_ID}@{DISTILBERT_REVISION}")
        print(f"Output checkpoint directory: {CHECKPOINT_DIR}")
        print(f"Final model directory: {MODEL_DIR}")
        print("Preflight completed. No model was downloaded or trained.")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not (CHECKPOINT_DIR / "run_metadata.json").exists():
        (CHECKPOINT_DIR / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_distilbert_tokenizer(CACHE_DIR)
    # The classification head is freshly initialized; no old fine-tuned checkpoint is loaded.
    model = load_distilbert_classifier(labels, CACHE_DIR)
    train_dataset = ComplaintDataset(train[TEXT_COLUMN], y_train)
    validation_dataset = ComplaintDataset(validation[TEXT_COLUMN], y_validation)
    collator = TextBatchCollator(tokenizer, max_length=MAX_TOKEN_LENGTH)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=y_train,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    updates_per_epoch = math.ceil(
        len(train_dataset) / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    warmup_steps = math.ceil(
        updates_per_epoch * NUM_TRAIN_EPOCHS * WARMUP_RATIO
    )

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits)
        predictions = logits.argmax(axis=1)
        values, _ = metrics_from_predictions(
            eval_prediction.label_ids,
            predictions,
            class_indices,
            rankings=logits,
        )
        return {
            key: float(values[key])
            for key in (
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "top_2_accuracy",
            )
        }

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        fp16_full_eval=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=200,
        logging_first_step=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
        report_to="none",
        restore_callback_states_from_checkpoint=False,
    )
    trainer = ClassWeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    resume_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR))
    if resume_checkpoint:
        print(f"Resuming this v3 experiment from: {resume_checkpoint}", flush=True)
    else:
        print("Starting a fresh v3 fine-tuning run from pinned pretrained weights.", flush=True)

    torch.cuda.reset_peak_memory_stats()
    print(
        f"Model: {DISTILBERT_MODEL_ID}@{DISTILBERT_REVISION}\n"
        f"Train rows: {len(train_dataset):,} | Validation rows: {len(validation_dataset):,}\n"
        f"Classes: {len(labels)} | Device: {torch.cuda.get_device_name(0)}\n"
        f"Epochs: {NUM_TRAIN_EPOCHS} | Batch: {TRAIN_BATCH_SIZE} | "
        f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}",
        flush=True,
    )

    start_time = time.perf_counter()
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    training_seconds = time.perf_counter() - start_time

    # Recheck all split bytes after training; test.csv is hashed only, never parsed or evaluated.
    for name, expected_hash in input_hashes.items():
        actual_hash = sha256_file(SPLIT_DIR / f"{name}.csv")
        if actual_hash != expected_hash:
            raise RuntimeError(f"{name}.csv changed during DistilBERT training.")

    prediction_output = trainer.predict(validation_dataset)
    logits = np.asarray(prediction_output.predictions)
    encoded_predictions = logits.argmax(axis=1)
    y_true = label_encoder.inverse_transform(y_validation)
    y_pred = label_encoder.inverse_transform(encoded_predictions)
    metrics, confusion = metrics_from_predictions(
        y_true,
        y_pred,
        labels,
        rankings=logits,
    )

    metrics.update(
        {
            "training_seconds": round(training_seconds, 2),
            "epochs_completed": float(trainer.state.epoch or 0),
            "best_validation_macro_f1": float(trainer.state.best_metric or 0),
            "peak_gpu_memory_mb": round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            ),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "training_history": trainer.state.log_history,
        }
    )

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "label_mapping.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "label_to_id": {label: index for index, label in enumerate(labels)},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(confusion, index=labels, columns=labels).to_csv(
        CONFUSION_PATH,
        encoding="utf-8",
    )

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v3",
        "experiment": "fresh_distilbert_finetuning",
        "grouping": audit["component_threshold"],
        "model_id": DISTILBERT_MODEL_ID,
        "model_revision": DISTILBERT_REVISION,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_loaded": 0,
        "test_rows_evaluated": 0,
        "product_categories": labels,
        "train_class_distribution": {
            str(label): int(count)
            for label, count in train[LABEL_COLUMN].value_counts().items()
        },
        "validation_class_distribution": {
            str(label): int(count)
            for label, count in validation[LABEL_COLUMN].value_counts().items()
        },
        "parameters": {
            "max_token_length": MAX_TOKEN_LENGTH,
            "epochs_requested": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
            "updates_per_epoch": updates_per_epoch,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_train_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "class_weighting": "balanced cross-entropy weights from v3 train labels",
            "class_weight_values": {
                label: float(weight)
                for label, weight in zip(labels, class_weights, strict=True)
            },
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection_metric": "validation_macro_f1",
            "random_state": RANDOM_STATE,
        },
        "results": metrics,
        "input_sha256": {
            "audit_report": sha256_file(AUDIT_PATH),
            "source_dataset": audit["source_dataset_sha256"],
            "train_csv": input_hashes["train"],
            "validation_csv": input_hashes["validation"],
            "test_csv_hashed_but_not_loaded": input_hashes["test"],
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "artifacts": {
            "model": str(MODEL_DIR),
            "checkpoints": str(CHECKPOINT_DIR),
            "validation_confusion_matrix": str(CONFUSION_PATH),
        },
        "test_policy": "The test CSV was SHA-256 verified only; it was not parsed, loaded, predicted on, or evaluated.",
    }
    with REPORT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\nDISTILBERT V3 TRAINING COMPLETED", flush=True)
    print(
        f"Validation macro-F1={metrics['macro_f1']:.4f} | "
        f"macro-recall={metrics['macro_recall']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f}",
        flush=True,
    )
    print(f"Model: {MODEL_DIR}", flush=True)
    print(f"Validation report: {REPORT_PATH}", flush=True)
    print("Test data was not loaded or evaluated.", flush=True)


if __name__ == "__main__":
    main()
