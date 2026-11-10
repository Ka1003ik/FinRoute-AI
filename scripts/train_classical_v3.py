"""Train and compare FinRoute AI classical models on the audited v3 split."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_for_model
from src.models.xgboost_model import XGBoostTextClassifier
from src.preprocessing.text_vectorizer import (
    BASE_VECTORIZER_DESCRIPTION,
    make_base_vectorizer,
)


SPLIT_DIR = ROOT / "data" / "processed" / "splits_v3"
AUDIT_PATH = ROOT / "outputs" / "leakage_audit" / "independent_saved_split_audit_v3_hashed.json"
RUN_DIR = ROOT / "outputs" / "experiments" / "v3" / "classical"

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"
EXPECTED_ROWS = {"train": 96_890, "validation": 12_112, "test": 12_112}
EXPECTED_SOURCE_ROWS = 121_114
EXPECTED_CLASSES = 11
RANDOM_STATE = 42

# Single source of truth for all candidate-model hyperparameters.
MODEL_CONFIGS = {
    "multinomial_nb": {"alpha": 1.0},
    "complement_nb": {"alpha": 1.0},
    "logistic_regression_ovr": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 500,
        "solver": "liblinear",
        "tol": 1e-3,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    },
    "linear_svc": {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    },
    "sgd_classifier": {
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 1e-5,
        "class_weight": "balanced",
        "max_iter": 1000,
        "tol": 1e-3,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "selected_features": 10_000,
        "n_estimators": 100,
        "max_depth": 24,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": 2,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_features": 10_000,
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.3,
        "min_child_weight": 3.0,
        "reg_lambda": 1.0,
        "max_bin": 128,
        "n_jobs": 4,
        "random_state": RANDOM_STATE,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit_and_inputs() -> tuple[dict, dict[str, str]]:
    required = [AUDIT_PATH, *(SPLIT_DIR / f"{name}.csv" for name in EXPECTED_ROWS)]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required v3 input(s) missing:\n" + "\n".join(map(str, missing)))
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {RUN_DIR}")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("dataset_version") != "v3" or audit.get("component_threshold") != "Jaccard = 1.00":
        raise ValueError("The supplied independent audit is not for the expected v3 grouping.")

    flags = (
        "all_three_saved_files_audited",
        "complete_rows_match_expected_source_rows",
        "all_product_distributions_match",
        "all_components_remain_intact",
        "all_unique_narratives_remain_intact",
        "no_identical_complete_rows_cross_partitions",
        "all_original_records_preserved",
    )
    checks = audit.get("final_checks", {})
    failed = [flag for flag in flags if checks.get(flag) is not True]
    if failed:
        raise RuntimeError("Independent split audit failed: " + ", ".join(failed))
    if any(checks.get(key) is not False for key in (
        "source_dataset_modified", "split_files_modified", "test_set_evaluated"
    )):
        raise RuntimeError("Audit does not confirm that source/splits stayed unchanged and test stayed unused.")

    conservation = audit.get("dataset_conservation", {})
    if (
        conservation.get("exact_source_row_conservation") is not True
        or conservation.get("source_records") != EXPECTED_SOURCE_ROWS
        or conservation.get("saved_partition_records") != EXPECTED_SOURCE_ROWS
        or conservation.get("missing_row_occurrences") != 0
        or conservation.get("unexpected_row_occurrences") != 0
    ):
        raise RuntimeError("Independent audit does not confirm complete source-row conservation.")

    overlap = audit.get("cross_partition_checks", {})
    overlap_fields = (
        "cross_partition_component_overlap_count",
        "cross_partition_unique_narrative_overlap_count",
        "cross_partition_exact_row_overlap_count",
    )
    if any(overlap.get(field) != 0 for field in overlap_fields):
        raise RuntimeError("Independent audit reports cross-partition leakage.")

    recorded_hashes = audit.get("saved_split_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("Audit report has no saved_split_sha256; use the new hashed audit script.")
    actual_hashes = {
        name: sha256_file(SPLIT_DIR / f"{name}.csv") for name in EXPECTED_ROWS
    }
    if actual_hashes != recorded_hashes:
        raise RuntimeError("A saved v3 split CSV does not match the audited file hash.")
    if not audit.get("source_dataset_sha256"):
        raise ValueError("Hashed audit report is missing the source dataset SHA-256.")

    for name, expected_rows in EXPECTED_ROWS.items():
        partition = audit.get("partitions", {}).get(name, {})
        if partition.get("records") != expected_rows:
            raise ValueError(f"Audit row count for {name} does not match the expected v3 size.")
        if partition.get("complete_row_sha256_match") is not True or partition.get("product_distribution_match") is not True:
            raise ValueError(f"Audit verification failed for {name}.")

    print("Hashed independent v3 split audit: PASS", flush=True)
    return audit, actual_hashes


def load_train_validation(audit: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    read_options = {
        "usecols": [TEXT_COLUMN, LABEL_COLUMN],
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8-sig",
    }
    train = pd.read_csv(SPLIT_DIR / "train.csv", **read_options)
    validation = pd.read_csv(SPLIT_DIR / "validation.csv", **read_options)

    for name, frame in (("train", train), ("validation", validation)):
        if len(frame) != EXPECTED_ROWS[name]:
            raise ValueError(f"Unexpected {name} row count: {len(frame):,}.")
        if frame[[TEXT_COLUMN, LABEL_COLUMN]].eq("").any().any():
            raise ValueError(f"Empty narrative or Product value found in {name}.")
        expected = audit["partitions"][name]["product_distribution"]
        actual = {str(label): int(count) for label, count in frame[LABEL_COLUMN].value_counts().items()}
        if actual != expected:
            raise ValueError(f"{name} Product distribution differs from the audit report.")

    classes = np.sort(train[LABEL_COLUMN].unique())
    if len(classes) != EXPECTED_CLASSES or set(classes) != set(validation[LABEL_COLUMN].unique()):
        raise ValueError("Unexpected Product labels or inconsistent train/validation classes.")
    return train, validation, classes


def make_candidates() -> list[tuple[str, object]]:
    cfg = MODEL_CONFIGS
    lr_cfg = {key: value for key, value in cfg["logistic_regression_ovr"].items() if key != "n_jobs"}
    rf_cfg = dict(cfg["random_forest"])
    selected_features = rf_cfg.pop("selected_features")
    rf = Pipeline([
        ("select_features", SelectKBest(chi2, k=selected_features)),
        ("classifier", RandomForestClassifier(**rf_cfg)),
    ])
    return [
        ("multinomial_nb", MultinomialNB(**cfg["multinomial_nb"])),
        ("complement_nb", ComplementNB(**cfg["complement_nb"])),
        ("logistic_regression_ovr", OneVsRestClassifier(
            LogisticRegression(**lr_cfg), n_jobs=cfg["logistic_regression_ovr"]["n_jobs"]
        )),
        ("linear_svc", LinearSVC(**cfg["linear_svc"])),
        ("sgd_classifier", SGDClassifier(**cfg["sgd_classifier"])),
        ("random_forest", rf),
        ("xgboost", XGBoostTextClassifier(**cfg["xgboost"])),
    ]


def train_candidates(train: pd.DataFrame, validation: pd.DataFrame, classes: np.ndarray, stage: Path) -> dict:
    print("\nFitting TF-IDF on train narratives only...", flush=True)
    vectorizer_start = time.perf_counter()
    vectorizer = make_base_vectorizer()
    x_train = vectorizer.fit_transform(train[TEXT_COLUMN])
    x_validation = vectorizer.transform(validation[TEXT_COLUMN])
    vectorizer_seconds = time.perf_counter() - vectorizer_start

    if x_train.shape[1] < 10_000:
        raise ValueError(f"TF-IDF created only {x_train.shape[1]:,} features; RF/XGBoost require 10,000.")
    stage.mkdir(parents=True, exist_ok=True)
    y_train = train[LABEL_COLUMN].to_numpy()
    y_validation = validation[LABEL_COLUMN].to_numpy()
    xgb_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    results: dict[str, dict] = {}
    best_name = None
    best_model = None
    best_score = float("-inf")
    fit_total = 0.0

    for name, model in make_candidates():
        print(f"\nTraining {name}...", flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name == "xgboost":
                model.fit(x_train, y_train, sample_weight=xgb_weights)
            else:
                model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        fit_total += fit_seconds

        learned_classes = set(map(str, model.classes_))
        if learned_classes != set(map(str, classes)):
            raise RuntimeError(f"{name} did not learn the full expected Product label set.")
        metrics, matrix = metrics_for_model(model, x_validation, y_validation, classes)
        metrics["fit_seconds"] = round(fit_seconds, 2)
        metrics["warnings"] = [str(item.message) for item in caught]
        metrics["convergence_warnings"] = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        results[name] = metrics
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
            stage / f"{name}_validation_confusion.csv", encoding="utf-8"
        )
        print(
            f"{name}: macro-F1={metrics['macro_f1']:.4f} | "
            f"macro-recall={metrics['macro_recall']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | fit={fit_seconds:.1f}s",
            flush=True,
        )
        if metrics["macro_f1"] > best_score:
            best_name, best_model, best_score = name, model, metrics["macro_f1"]

    return {
        "vectorizer": vectorizer,
        "best_model": best_model,
        "best_name": best_name,
        "results": results,
        "tfidf_train_shape": list(x_train.shape),
        "tfidf_validation_shape": list(x_validation.shape),
        "train_empty_feature_rows": int((x_train.getnnz(axis=1) == 0).sum()),
        "validation_empty_feature_rows": int((x_validation.getnnz(axis=1) == 0).sum()),
        "vectorizer_seconds": round(vectorizer_seconds, 2),
        "model_fit_seconds": round(fit_total, 2),
    }


def write_artifacts(stage: Path, audit: dict, audit_hash: str, split_hashes: dict[str, str],
                    train: pd.DataFrame, validation: pd.DataFrame, classes: np.ndarray,
                    trained: dict, run_seconds: float) -> None:
    model_dir = stage / "model"
    report_dir = stage / "report"
    confusion_dir = stage / "confusion_matrices"
    model_dir.mkdir()
    report_dir.mkdir()
    confusion_dir.mkdir()

    joblib.dump({
        "vectorizer": trained["vectorizer"],
        "classifier": trained["best_model"],
        "model_name": trained["best_name"],
        "selected_by": "validation_macro_f1",
        "dataset_version": "v3",
        "class_labels": [str(label) for label in classes],
        "random_state": RANDOM_STATE,
    }, model_dir / "classical_best.joblib")

    # Move confusion matrices into the final bundle after all candidates finish.
    for path in stage.glob("*_validation_confusion.csv"):
        path.replace(confusion_dir / path.name)

    report = {
        "project": "FinRoute AI",
        "dataset_version": "v3",
        "grouping": audit["component_threshold"],
        "allocation_method": "StratifiedGroupKFold",
        "random_state": RANDOM_STATE,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_loaded": 0,
        "test_rows_evaluated": 0,
        "product_categories": [str(label) for label in classes],
        "train_class_distribution": {str(k): int(v) for k, v in train[LABEL_COLUMN].value_counts().items()},
        "validation_class_distribution": {str(k): int(v) for k, v in validation[LABEL_COLUMN].value_counts().items()},
        "tfidf_train_shape": trained["tfidf_train_shape"],
        "tfidf_validation_shape": trained["tfidf_validation_shape"],
        "train_rows_without_tfidf_features": trained["train_empty_feature_rows"],
        "validation_rows_without_tfidf_features": trained["validation_empty_feature_rows"],
        "vectorizer": BASE_VECTORIZER_DESCRIPTION,
        "candidate_parameters": MODEL_CONFIGS,
        "results": trained["results"],
        "selected_model": trained["best_name"],
        "selection_metric": "validation_macro_f1",
        "timing_seconds": {
            "vectorizer_fit": trained["vectorizer_seconds"],
            "candidate_model_fits_total": trained["model_fit_seconds"],
            "whole_run": round(run_seconds, 2),
        },
        "input_sha256": {
            "audit_report": audit_hash,
            "source_dataset": audit["source_dataset_sha256"],
            "train_csv": split_hashes["train"],
            "validation_csv": split_hashes["validation"],
            "test_csv_hashed_but_not_loaded": split_hashes["test"],
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "test_policy": "The test CSV was hash-verified by the independent audit but was not opened, loaded, predicted on, or evaluated.",
    }
    (report_dir / "classical_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    print("=" * 68)
    print("FINROUTE AI | CLASSICAL V3 TRAINING")
    print("=" * 68)

    audit, split_hashes = validate_audit_and_inputs()
    audit_hash = sha256_file(AUDIT_PATH)
    source_path = ROOT / "data" / "processed" / audit["source_dataset"]
    if not source_path.is_file() or sha256_file(source_path) != audit["source_dataset_sha256"]:
        raise RuntimeError("The source dataset does not match the SHA-256 recorded by the audit.")

    train, validation, classes = load_train_validation(audit)
    # Confirm none of the audited inputs changed while the CSVs were being loaded.
    if sha256_file(AUDIT_PATH) != audit_hash:
        raise RuntimeError("The independent audit report changed during input loading.")
    if any(sha256_file(SPLIT_DIR / f"{name}.csv") != digest for name, digest in split_hashes.items()):
        raise RuntimeError("A saved split CSV changed during input loading.")

    stage = Path(tempfile.mkdtemp(prefix=".finroute_classical_v3_", dir=ROOT))
    try:
        trained = train_candidates(train, validation, classes, stage)
        # The test CSV is only hashed, never opened or loaded. Recheck all inputs
        # before publishing so the report identifies the exact audited files.
        if sha256_file(AUDIT_PATH) != audit_hash:
            raise RuntimeError("The independent audit report changed during training.")
        if any(sha256_file(SPLIT_DIR / f"{name}.csv") != digest for name, digest in split_hashes.items()):
            raise RuntimeError("A saved split CSV changed during training.")
        if sha256_file(source_path) != audit["source_dataset_sha256"]:
            raise RuntimeError("The source dataset changed during training.")
        write_artifacts(stage, audit, audit_hash, split_hashes, train, validation, classes, trained,
                        time.perf_counter() - started)
        RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
        if RUN_DIR.exists():
            raise FileExistsError(f"Refusing to overwrite {RUN_DIR}")
        stage.rename(RUN_DIR)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    selected = trained["results"][trained["best_name"]]
    print("\nCLASSICAL V3 TRAINING COMPLETED")
    print(f"Selected model: {trained['best_name']}")
    print(f"Validation macro-F1: {selected['macro_f1']:.4f}")
    print(f"Artifacts: {RUN_DIR}")
    print("Test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
