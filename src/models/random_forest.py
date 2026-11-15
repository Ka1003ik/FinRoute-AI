"""Resource-bounded Random Forest pipeline for sparse TF-IDF inputs."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline


def make_random_forest_pipeline(
    n_features: int = 10_000,
    random_state: int = 42,
) -> Pipeline:
    """Select train-fitted chi-square features, then fit a balanced forest."""
    return Pipeline(
        steps=[
            ("select_features", SelectKBest(score_func=chi2, k=n_features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=24,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=2,
                ),
            ),
        ]
    )
