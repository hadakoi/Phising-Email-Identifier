from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .model import build_model


def train(data_path: str | Path, model_path: str | Path, metrics_path: str | Path, test_size: float = 0.2, random_state: int = 42) -> dict:
    dataset = load_dataset(data_path)
    train_frame, test_frame = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["label"],
    )

    model = build_model()
    model.fit(train_frame, train_frame["label"])
    predictions = model.predict(test_frame)
    probabilities = model.predict_proba(test_frame)[:, 1]
    metrics = {
        "rows": int(len(dataset)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "class_balance": {str(key): int(value) for key, value in dataset["label"].value_counts().sort_index().items()},
        "accuracy": round(float(accuracy_score(test_frame["label"], predictions)), 4),
        "precision": round(float(precision_score(test_frame["label"], predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(test_frame["label"], predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(test_frame["label"], predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(test_frame["label"], probabilities)), 4),
        "confusion_matrix": confusion_matrix(test_frame["label"], predictions).tolist(),
        "classification_report": classification_report(test_frame["label"], predictions, output_dict=True, zero_division=0),
    }
    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phishing email detector")
    parser.add_argument("--data", default="data", help="CSV file or directory containing CSV files")
    parser.add_argument("--model", default="models/phishing_model.joblib")
    parser.add_argument("--metrics", default="models/metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metrics = train(args.data, args.model, args.metrics, args.test_size, args.random_state)
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()

