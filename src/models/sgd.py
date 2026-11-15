"""SGD-based linear classifier configuration for sparse TF-IDF features."""

from sklearn.linear_model import SGDClassifier


def make_sgd_classifier(random_state: int = 42) -> SGDClassifier:
    """Create the reproducible logistic-loss SGD baseline."""
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
    )
