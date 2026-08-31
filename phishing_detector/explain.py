from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def explain_prediction(model: Pipeline, email: dict, limit: int = 8) -> dict:
    frame = pd.DataFrame([email])
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    matrix = features.transform(frame)
    names = np.asarray(features.get_feature_names_out())
    contributions = matrix.multiply(classifier.coef_[0]).toarray()[0]
    active = np.flatnonzero(matrix.toarray()[0])
    positive = sorted((active[contributions[active] > 0]), key=lambda index: contributions[index], reverse=True)[:limit]
    negative = sorted((active[contributions[active] < 0]), key=lambda index: contributions[index])[:limit]
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "label": "MALICIOUS" if probability >= 0.5 else "LEGITIMATE",
        "probability": probability,
        "positive_features": [{"feature": names[index], "contribution": float(contributions[index])} for index in positive],
        "negative_features": [{"feature": names[index], "contribution": float(contributions[index])} for index in negative],
    }

