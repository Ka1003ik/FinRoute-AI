"""Fine-tune DistilBERT on train narratives and select by validation macro F1."""

import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_DIR = ROOT / "models" / "transformer"
CACHE_DIR = TRANSFORMER_DIR / "cache"
HF_HOME = TRANSFORMER_DIR / "hf_home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_XET_CACHE", str(HF_HOME / "xet"))

import joblib
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


SPLIT_DIR = ROOT / "data" / "processed" / "splits"

# Preserve the original baseline report.
BASELINE_REPORT_PATH = ROOT / "outputs" / "reports" / "baseline_validation.json"

# Separate report for the extended experiment.
REPORT_PATH = ROOT / "outputs" / "reports" / "distilbert_3epoch_validation.json"

CONFUSION_DIR = ROOT / "outputs" / "comparisons"
CONFUSION_FILENAME = "distilbert_3epoch_validation_confusion.csv"

CLASSICAL_MODEL_PATH = ROOT / "models" / "baseline_best.joblib"

# Original completed checkpoint: read-only source for continuation.
ORIGINAL_CHECKPOINT_DIR = TRANSFORMER_DIR / "distilbert_checkpoints"
ORIGINAL_CHECKPOINT_PATH = ORIGINAL_CHECKPOINT_DIR / "checkpoint-8071"

# New continuation checkpoint directory.
CHECKPOINT_DIR = TRANSFORMER_DIR / "distilbert_continuation_checkpoints"

# Separate final model directory.
MODEL_DIR = TRANSFORMER_DIR / "distilbert_product_classifier_3epoch"
TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
RANDOM_STATE = 42
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 6
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_RATIO = 0.1


class ComplaintDataset(Dataset):
    """Keep raw texts on CPU and tokenize only when forming each batch."""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"text": self.texts[index], "labels": int(self.labels[index])}


class ClassWeightedTrainer(Trainer):
    """Apply class-balanced weights to cross-entropy during fine-tuning."""

    def __init__(self, class_weights, *args, **kwargs):
        self.class_weights = class_weights
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
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


def main() -> None:
    """Train on the saved training split and evaluate on validation only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured DistilBERT run.")
    if not BASELINE_REPORT_PATH.exists() or not CLASSICAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Run the classical baseline comparison before this script."
        )

    # Seed before model loading so the newly initialized classifier head is reproducible.
    set_seed(RANDOM_STATE)
    report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    classical_bundle = joblib.load(CLASSICAL_MODEL_PATH)
    train = pd.read_csv(
        SPLIT_DIR / "train.csv",
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    validation = pd.read_csv(
        SPLIT_DIR / "validation.csv",
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if len(train) != report["train_rows"] or len(validation) != report["validation_rows"]:
        raise ValueError("Split sizes differ from the saved comparison report.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train[LABEL_COLUMN].to_numpy())
    if set(label_encoder.classes_) != set(classical_bundle["classifier"].classes_):
        raise ValueError("Product labels differ from the saved classical comparison.")
    y_validation = label_encoder.transform(validation[LABEL_COLUMN].to_numpy())
    labels = [str(value) for value in label_encoder.classes_]
    class_indices = np.arange(len(labels))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = load_distilbert_tokenizer(CACHE_DIR)
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
        predictions = np.asarray(logits).argmax(axis=1)
        values, _ = metrics_from_predictions(
            eval_prediction.label_ids,
            predictions,
            class_indices,
            rankings=np.asarray(logits),
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
        learning_rate=2e-5,
        weight_decay=0.01,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    torch.cuda.reset_peak_memory_stats()
    print(
        f"Fine-tuning {DISTILBERT_MODEL_ID} on {len(train_dataset):,} training rows; "
        f"validation rows={len(validation_dataset):,}; device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    start_time = time.perf_counter()

    # Prefer the latest continuation checkpoint when resuming an interrupted run.
    resume_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR))

    # On the first continuation run, use the verified epoch-one checkpoint.
    if resume_checkpoint is None:
        resume_checkpoint = str(ORIGINAL_CHECKPOINT_PATH)

    checkpoint_path = Path(resume_checkpoint)

    # Verify that this is a complete resumable checkpoint.
    required_files = [
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (checkpoint_path / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Incomplete checkpoint {checkpoint_path}. "
            f"Missing files: {missing_files}"
        )

    # Verify the saved training progress.
    checkpoint_state = json.loads(
        (checkpoint_path / "trainer_state.json").read_text(encoding="utf-8")
    )

    if checkpoint_state["global_step"] < 8071:
        raise ValueError(
            "Checkpoint is earlier than the verified epoch-one checkpoint."
        )

    print(
        f"Resuming from: {checkpoint_path}\n"
        f"Global step: {checkpoint_state['global_step']}\n"
        f"Saved epoch: {checkpoint_state['epoch']}",
        flush=True,
    )

    trainer.train(resume_from_checkpoint=str(checkpoint_path))
    fit_seconds = time.perf_counter() - start_time

    prediction_output = trainer.predict(validation_dataset)
    logits = np.asarray(prediction_output.predictions)
    encoded_predictions = logits.argmax(axis=1)
    y_true = label_encoder.inverse_transform(y_validation)
    y_pred = label_encoder.inverse_transform(encoded_predictions)
    values, matrix = metrics_from_predictions(
        y_true,
        y_pred,
        labels,
        rankings=logits,
    )
    values["fit_seconds"] = round(fit_seconds, 2)
    values["training_warnings"] = []
    values["epochs_completed"] = float(trainer.state.epoch or 0)
    values["best_validation_macro_f1"] = float(trainer.state.best_metric or 0)
    values["peak_gpu_memory_mb"] = round(
        torch.cuda.max_memory_allocated() / (1024**2),
        2,
    )
    values["training_history"] = trainer.state.log_history
    values["best_checkpoint"] = trainer.state.best_model_checkpoint

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "label_mapping.json").write_text(
        json.dumps(
            {"labels": labels, "label_to_id": {label: index for index, label in enumerate(labels)}},
            indent=2,
        ),
        encoding="utf-8",
    )
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        CONFUSION_DIR / CONFUSION_FILENAME ,
        encoding="utf-8",
    )

    report["results"]["distilbert"] = values
    report["candidate_parameters"]["distilbert"] = {
        "model_id": DISTILBERT_MODEL_ID,
        "model_revision": DISTILBERT_REVISION,
        "max_token_length": MAX_TOKEN_LENGTH,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": warmup_steps,
        "updates_per_epoch": updates_per_epoch,
        "epochs_requested": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "class_weighting": "balanced cross-entropy weights from train labels",
        "class_weight_values": {
            label: float(weight)
            for label, weight in zip(labels, class_weights, strict=True)
        },
        "selection_metric": "validation macro_f1",
        "random_state": RANDOM_STATE,
    }
    report["torch_version"] = torch.__version__
    report["transformers_version"] = transformers.__version__
    report["cuda_version"] = torch.version.cuda
    report["cuda_device"] = torch.cuda.get_device_name(0)
    report.setdefault("candidate_artifacts", {})["distilbert"] = str(MODEL_DIR)
    best_name = max(
        report["results"],
        key=lambda name: report["results"][name]["macro_f1"],
    )
    report["selected_model"] = best_name
    report["selection_metric"] = "macro_f1"
    report["selected_model_artifact"] = (
        str(MODEL_DIR) if best_name == "distilbert" else str(CLASSICAL_MODEL_PATH)
    )
    report["test_rows_used"] = 0
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"DistilBERT validation macro-F1={values['macro_f1']:.4f}; "
        f"selected model={best_name}; fit-seconds={values['fit_seconds']:.2f}; "
        f"peak-GPU-MB={values['peak_gpu_memory_mb']:.0f}"
    )
    print(f"Saved model to: {MODEL_DIR}")
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
