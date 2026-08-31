from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import EmailFeatureExtractor


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", EmailFeatureExtractor()),
            (
                "classifier",
                LogisticRegression(
                    C=3.0,
                    max_iter=1_000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

