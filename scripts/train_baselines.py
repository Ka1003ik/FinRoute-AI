"""Train TF-IDF baselines on train data and select using validation only."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_vectorizer import (
    BASE_VECTORIZER_DESCRIPTION,
    make_base_vectorizer,
)
from src.evaluation.metrics import metrics_for_model


SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_PATH = ROOT / "outputs" / "reports" / "baseline_validation.json"
CONFUSION_DIR = ROOT / "outputs" / "comparisons"
MODEL_PATH = ROOT / "models" / "baseline_best.joblib"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
RANDOM_STATE = 42


def main() -> None:
    """Fit candidate classifiers using train rows and compare on validation."""
    print("Loading train and validation splits...", flush=True)
    train_path = SPLIT_DIR / "train.csv"
    validation_path = SPLIT_DIR / "validation.csv"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Required split file not found: {path}")

    train = pd.read_csv(
        train_path,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    validation = pd.read_csv(
        validation_path,
        usecols=[TEXT_COLUMN, LABEL_COLUMN],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    y_train = train[LABEL_COLUMN].to_numpy()
    y_validation = validation[LABEL_COLUMN].to_numpy()
    class_labels = np.sort(np.unique(y_train))
    if set(class_labels) != set(np.unique(y_validation)):
        raise ValueError("Train and validation splits do not contain the same labels.")

    # Fit vocabulary and IDF weights only on train; validation is transform-only.
    print("Fitting TF-IDF on training narratives...", flush=True)
    vectorizer_start = time.perf_counter()
    vectorizer = make_base_vectorizer()
    x_train = vectorizer.fit_transform(train[TEXT_COLUMN])
    x_validation = vectorizer.transform(validation[TEXT_COLUMN])
    empty_validation_rows = int((x_validation.getnnz(axis=1) == 0).sum())
    print(
        f"TF-IDF ready: train={x_train.shape}, validation={x_validation.shape}, "
        f"seconds={time.perf_counter() - vectorizer_start:.2f}",
        flush=True,
    )

    candidates = {
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "complement_nb": ComplementNB(alpha=1.0),
        "linear_svc": LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "logistic_regression_ovr": OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                random_state=RANDOM_STATE,
                solver="liblinear",
                tol=1e-3,
            ),
            n_jobs=1,
        ),
    }

    results = {}
    matrices = {}
    fitted_models = {}
    for name, model in candidates.items():
        print(f"Training {name}...", flush=True)
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
        results[name] = result
        matrices[name] = matrix
        fitted_models[name] = model
        print(
            f"{name}: validation macro-F1={result['macro_f1']:.4f}, "
            f"macro-recall={result['macro_recall']:.4f}, "
            f"fit-seconds={result['fit_seconds']:.2f}",
            flush=True,
        )

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        pd.DataFrame(
            matrix,
            index=class_labels,
            columns=class_labels,
        ).to_csv(CONFUSION_DIR / f"{name}_validation_confusion.csv", encoding="utf-8")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": fitted_models[best_name],
            "selected_by": "validation_macro_f1",
            "model_name": best_name,
        },
        MODEL_PATH,
    )

    report = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows_used": 0,
        "train_tfidf_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "validation_tfidf_shape": [
            int(x_validation.shape[0]),
            int(x_validation.shape[1]),
        ],
        "validation_rows_with_no_train_vocabulary_features": empty_validation_rows,
        "random_state": RANDOM_STATE,
        "selection_metric": "macro_f1",
        "vectorizer": BASE_VECTORIZER_DESCRIPTION,
        "candidate_parameters": {
            "multinomial_nb": {"alpha": 1.0},
            "complement_nb": {"alpha": 1.0},
            "linear_svc": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 5000,
                "random_state": RANDOM_STATE,
            },
            "logistic_regression_ovr": {
                "wrapper": "OneVsRestClassifier",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 500,
                "solver": "liblinear",
                "tol": 1e-3,
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        },
        "scikit_learn_version": sklearn.__version__,
        "results": results,
        "selected_model": best_name,
        "saved_candidate_artifact": str(MODEL_PATH),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSelected by validation macro-F1: {best_name}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
