"""Train the SGDClassifier candidate and add it to validation results."""

import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_for_model
from src.models.sgd import make_sgd_classifier


SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_PATH = ROOT / "outputs" / "reports" / "baseline_validation.json"
CONFUSION_DIR = ROOT / "outputs" / "comparisons"
MODEL_PATH = ROOT / "models" / "baseline_best.joblib"
TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
RANDOM_STATE = 42


def main() -> None:
    """Fit on training rows, score on validation, and update the comparison."""
    if not REPORT_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Run a successful baseline comparison before this script."
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(MODEL_PATH)
    vectorizer = bundle["vectorizer"]
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
        raise ValueError("Split sizes differ from the saved baseline report.")

    print("Transforming train and validation text with the saved TF-IDF model...", flush=True)
    x_train = vectorizer.transform(train[TEXT_COLUMN])
    x_validation = vectorizer.transform(validation[TEXT_COLUMN])
    y_train = train[LABEL_COLUMN].to_numpy()
    y_validation = validation[LABEL_COLUMN].to_numpy()
    class_labels = np.sort(np.unique(y_train))
    if set(class_labels) != set(bundle["classifier"].classes_):
        raise ValueError("Product labels differ from the saved baseline model.")
    if list(x_train.shape) != report["train_tfidf_shape"]:
        raise ValueError("Training TF-IDF shape differs from the saved report.")
    if list(x_validation.shape) != report["validation_tfidf_shape"]:
        raise ValueError("Validation TF-IDF shape differs from the saved report.")

    model = make_sgd_classifier(random_state=RANDOM_STATE)
    print("Training SGDClassifier with logistic loss...", flush=True)
    start_time = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start_time

    result, matrix = metrics_for_model(
        model,
        x_validation,
        y_validation,
        class_labels,
    )
    result["fit_seconds"] = round(fit_seconds, 2)
    result["convergence_warnings"] = [
        str(item.message)
        for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    report["results"]["sgd_classifier"] = result
    report["candidate_parameters"]["sgd_classifier"] = {
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 1e-5,
        "class_weight": "balanced",
        "max_iter": 1000,
        "tol": 1e-3,
        "random_state": RANDOM_STATE,
    }
    report["scikit_learn_version"] = sklearn.__version__

    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=class_labels, columns=class_labels).to_csv(
        CONFUSION_DIR / "sgd_classifier_validation_confusion.csv",
        encoding="utf-8",
    )
    best_name = max(
        report["results"],
        key=lambda name: report["results"][name]["macro_f1"],
    )
    report["selected_model"] = best_name
    report["selection_metric"] = "macro_f1"
    if best_name == "sgd_classifier":
        bundle["classifier"] = model
        bundle["model_name"] = best_name
        joblib.dump(bundle, MODEL_PATH)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"SGDClassifier validation macro-F1={result['macro_f1']:.4f}; "
        f"selected model={best_name}; fit-seconds={result['fit_seconds']:.2f}"
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
