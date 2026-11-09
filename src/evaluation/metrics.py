"""Consistent classification metrics for validation and test evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def metrics_from_predictions(labels, predictions, class_labels, rankings=None):
    """Calculate shared metrics from labels, predictions, and optional class scores."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=class_labels,
            average="weighted",
            zero_division=0,
        )
    )

    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    if rankings is not None and rankings.ndim == 2 and rankings.shape[1] >= 2:
        top_two_indices = np.argpartition(rankings, -2, axis=1)[:, -2:]
        top_two_labels = np.asarray(class_labels)[top_two_indices]
        result["top_2_accuracy"] = float(
            np.any(top_two_labels == np.asarray(labels)[:, None], axis=1).mean()
        )

    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    return result, matrix


def metrics_for_model(model, features, labels, class_labels):
    """Return aggregate/per-class metrics and a label-ordered confusion matrix."""
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        rankings = model.predict_proba(features)
    elif hasattr(model, "decision_function"):
        rankings = model.decision_function(features)
    else:
        rankings = None
    return metrics_from_predictions(
        labels,
        predictions,
        class_labels,
        rankings=rankings,
    )
