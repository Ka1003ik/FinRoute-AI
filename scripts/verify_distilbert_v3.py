"""Read-only verification of the saved FinRoute AI DistilBERT v3 artifact.

Checks the on-disk classification-head tensors, verifies the saved label map,
reloads the model and tokenizer from the local artifact directory, and runs a
small inference smoke test. This script does not train or modify any files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "transformer" / "distilbert_product_classifier_v3"
VALIDATION_REPORT = PROJECT_ROOT / "outputs" / "experiments" / "v3" / "distilbert" / "validation.json"
EXPECTED_HEAD_KEYS = {
    "pre_classifier.weight",
    "pre_classifier.bias",
    "classifier.weight",
    "classifier.bias",
}
SMOKE_TEST_TEXTS = [
    "I noticed an unauthorized withdrawal from my checking account.",
    "My credit card statement contains a charge I do not recognize.",
    "A debt collector keeps contacting me about a debt I believe I already paid.",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def inspect_saved_safetensors(model_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read tensor names and shapes directly from saved safetensors files."""
    weight_files = sorted(model_dir.glob("model*.safetensors"))
    if not weight_files:
        fail(f"No model*.safetensors file found in: {model_dir}")

    found: dict[str, tuple[int, ...]] = {}
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in EXPECTED_HEAD_KEYS:
                    found[key] = tuple(handle.get_slice(key).get_shape())

    missing = EXPECTED_HEAD_KEYS - found.keys()
    if missing:
        fail(f"Classification-head tensors missing from saved weights: {sorted(missing)}")
    return found


def main() -> int:
    print("=" * 68)
    print("FINROUTE AI | DISTILBERT V3 SAVED-MODEL VERIFICATION")
    print("=" * 68)

    if not MODEL_DIR.is_dir():
        fail(f"Saved model directory does not exist: {MODEL_DIR}")
    if not VALIDATION_REPORT.is_file():
        fail(f"Validation report does not exist: {VALIDATION_REPORT}")

    with VALIDATION_REPORT.open("r", encoding="utf-8") as file:
        report = json.load(file)
    expected_labels = report.get("product_categories")
    if not isinstance(expected_labels, list) or len(expected_labels) != 11:
        fail("Validation report does not contain the expected 11 Product categories.")

    print(f"Saved model directory: {MODEL_DIR}")
    print("\n1. Inspecting saved weight files directly...")
    head_shapes = inspect_saved_safetensors(MODEL_DIR)
    for key in sorted(head_shapes):
        print(f"   PASS  {key}: {head_shapes[key]}")

    if head_shapes["classifier.weight"][0] != 11:
        fail(f"Classifier output dimension is {head_shapes['classifier.weight'][0]}, expected 11.")
    if head_shapes["classifier.bias"] != (11,):
        fail(f"Classifier bias shape is {head_shapes['classifier.bias']}, expected (11,).")
    print("   PASS  Saved classification head has 11 output classes.")

    print("\n2. Checking saved model configuration and label mapping...")
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    if config.num_labels != 11:
        fail(f"Saved config num_labels={config.num_labels}, expected 11.")
    id2label = {int(index): label for index, label in config.id2label.items()}
    saved_labels = [id2label.get(index) for index in range(config.num_labels)]
    if saved_labels != expected_labels:
        fail(
            "Saved id2label order does not match validation report Product categories.\n"
            f"Saved: {saved_labels}\nExpected: {expected_labels}"
        )
    print("   PASS  num_labels = 11")
    print("   PASS  id2label matches all 11 validation Product categories in order.")

    print("\n3. Reloading model and tokenizer from the local saved directory...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=True,
    )
    print("   PASS  Model reloaded from the saved v3 directory.")
    print("   PASS  Tokenizer reloaded from the saved v3 directory.")

    print("\n4. Running a small inference smoke test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(
        SMOKE_TEST_TEXTS,
        padding=True,
        truncation=True,
        max_length=512,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted_ids = probabilities.argmax(dim=-1).cpu().tolist()

    if tuple(logits.shape) != (len(SMOKE_TEST_TEXTS), 11):
        fail(f"Unexpected inference logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        fail("Inference produced non-finite logits.")
    if not torch.isfinite(probabilities).all().item():
        fail("Inference produced non-finite probabilities.")

    print(f"   PASS  Device: {device}")
    print(f"   PASS  Logits shape: {tuple(logits.shape)}")
    for text, class_id in zip(SMOKE_TEST_TEXTS, predicted_ids):
        print(f"   Prediction: {expected_labels[class_id]}")
        print(f"      Input: {text}")

    print("\n" + "=" * 68)
    print("DISTILBERT V3 SAVED-MODEL VERIFICATION PASSED")
    print("No training was performed. No model or dataset files were modified.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nVERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
