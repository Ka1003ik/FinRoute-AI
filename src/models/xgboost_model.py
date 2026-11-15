"""Sparse-text XGBoost classifier with target-label encoding."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoostTextClassifier(ClassifierMixin, BaseEstimator):
    """Select sparse text features and preserve original string product labels."""

    def __init__(
        self,
        n_features: int = 10_000,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.3,
        min_child_weight: float = 3.0,
        reg_lambda: float = 1.0,
        max_bin: int = 128,
        n_jobs: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, features, labels, sample_weight=None):
        """Fit feature selection and XGBoost on the provided training rows."""
        self.label_encoder_ = LabelEncoder()
        encoded_labels = self.label_encoder_.fit_transform(labels)
        if self.n_features > features.shape[1]:
            raise ValueError("n_features exceeds the input feature count.")

        self.pipeline_ = Pipeline(
            steps=[
                ("select_features", SelectKBest(score_func=chi2, k=self.n_features)),
                (
                    "classifier",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(self.label_encoder_.classes_),
                        eval_metric="mlogloss",
                        tree_method="hist",
                        device="cpu",
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
                        min_child_weight=self.min_child_weight,
                        reg_lambda=self.reg_lambda,
                        max_bin=self.max_bin,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        fit_parameters = {}
        if sample_weight is not None:
            fit_parameters["classifier__sample_weight"] = np.asarray(sample_weight)
        self.pipeline_.fit(features, encoded_labels, **fit_parameters)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, features):
        """Return predictions using the original product labels."""
        encoded_predictions = self.pipeline_.predict(features).astype(int)
        return self.label_encoder_.inverse_transform(encoded_predictions)

    def predict_proba(self, features):
        """Return class scores in the label-encoder class order."""
        return self.pipeline_.predict_proba(features)
