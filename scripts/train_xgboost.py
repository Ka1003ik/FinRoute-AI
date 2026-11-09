"""Train the sparse XGBoost candidate and update validation results."""

import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_for_model
from src.models.xgboost_model import XGBoostTextClassifier


SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_PATH = ROOT / "outputs" / "reports" / "baseline_validation.json"
CONFUSION_DIR = ROOT / "outputs" / "comparisons"
MODEL_PATH = ROOT / "models" / "baseline_best.joblib"
TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
RANDOM_STATE = 42
SELECTED_FEATURES = 10_000


def main() -> None:
    """Fit on training rows, score validation rows, and record the candidate."""
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
    if SELECTED_FEATURES > x_train.shape[1]:
        raise ValueError("Requested feature count exceeds the fitted vocabulary.")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBoostTextClassifier(
        n_features=SELECTED_FEATURES,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_weight=3.0,
        reg_lambda=1.0,
        max_bin=128,
        n_jobs=4,
        random_state=RANDOM_STATE,
    )
    print(
        f"Training XGBoost with {SELECTED_FEATURES:,} train-selected features...",
        flush=True,
    )
    start_time = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(x_train, y_train, sample_weight=sample_weights)
    fit_seconds = time.perf_counter() - start_time

    result, matrix = metrics_for_model(
        model,
        x_validation,
        y_validation,
        class_labels,
    )
    result["fit_seconds"] = round(fit_seconds, 2)
    result["training_warnings"] = [str(item.message) for item in caught_warnings]
    report["results"]["xgboost"] = result
    report["candidate_parameters"]["xgboost"] = {
        "feature_selector": "SelectKBest(chi2)",
        "selected_features": SELECTED_FEATURES,
        "selector_fit_partition": "train",
        "objective": "multi:softprob",
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.3,
        "min_child_weight": 3.0,
        "reg_lambda": 1.0,
        "max_bin": 128,
        "class_weighting": "balanced sample weights from train labels",
        "n_jobs": 4,
        "random_state": RANDOM_STATE,
    }
    report["xgboost_version"] = xgboost.__version__
    report["scikit_learn_version"] = sklearn.__version__

    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=class_labels, columns=class_labels).to_csv(
        CONFUSION_DIR / "xgboost_validation_confusion.csv",
        encoding="utf-8",
    )
    best_name = max(
        report["results"],
        key=lambda name: report["results"][name]["macro_f1"],
    )
    report["selected_model"] = best_name
    report["selection_metric"] = "macro_f1"
    if best_name == "xgboost":
        bundle["classifier"] = model
        bundle["model_name"] = best_name
        joblib.dump(bundle, MODEL_PATH)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"XGBoost validation macro-F1={result['macro_f1']:.4f}; "
        f"selected model={best_name}; fit-seconds={result['fit_seconds']:.2f}"
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
