from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "subject": {"subject", "emailsubject", "subjectline"},
    "body": {
        "body",
        "emailbody",
        "emailtext",
        "textcombined",
        "message",
        "mailbody",
        "content",
        "text",
    },
    "label": {"label", "class", "category", "emailtype", "target", "isphishing"},
    "sender": {"sender", "from", "fromemail", "senderemail", "senderaddress"},
    "recipient": {"recipient", "receiver", "to", "toemail", "recipientemail"},
    "date": {"date", "sentdate", "emaildate", "timestamp"},
    "attachment": {"attachment", "attachments", "hasattachment", "attachmentindicator"},
}

MALICIOUS_LABELS = {"1", "true", "spam", "phishing", "malicious", "fraud", "scam", "bad"}
BENIGN_LABELS = {"0", "false", "ham", "legitimate", "benign", "safe", "normal", "good"}


def normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _column_for(frame: pd.DataFrame, name: str) -> str | None:
    aliases = COLUMN_ALIASES[name]
    for column in frame.columns:
        if normalize_column(column) in aliases:
            return column
    return None


def _text_series(frame: pd.DataFrame) -> pd.Series:
    subject_column = _column_for(frame, "subject")
    body_column = _column_for(frame, "body")
    subject = frame[subject_column].fillna("").astype(str) if subject_column else ""
    body = frame[body_column].fillna("").astype(str) if body_column else ""
    return (subject + "\n" + body).str.strip()


def normalize_label(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in MALICIOUS_LABELS:
        return 1
    if normalized in BENIGN_LABELS:
        return 0
    try:
        numeric = float(normalized)
    except ValueError:
        return None
    if numeric in (0.0, 1.0):
        return int(numeric)
    return None


def standardize_frame(frame: pd.DataFrame, source: str = "") -> pd.DataFrame:
    label_column = _column_for(frame, "label")
    if not label_column:
        raise ValueError("Could not find a label column")

    standardized = pd.DataFrame(index=frame.index)
    standardized["text"] = _text_series(frame)
    for field in ("subject", "body", "sender", "recipient", "date", "attachment"):
        column = _column_for(frame, field)
        standardized[field] = frame[column].fillna("").astype(str) if column else ""
    standardized["label"] = frame[label_column].map(normalize_label)
    standardized["source"] = source
    standardized = standardized.dropna(subset=["label"])
    standardized["label"] = standardized["label"].astype(int)
    standardized = standardized[standardized["text"].str.len() > 0]
    return standardized.reset_index(drop=True)


def _csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    files = sorted(path.glob("*.csv"))
    combined = [file for file in files if file.stem.lower() in {"phishing_email", "phishingemail"}]
    return combined or files


def load_dataset(path: str | Path) -> pd.DataFrame:
    csv_files = _csv_files(Path(path))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for csv_file in csv_files:
        try:
            raw = pd.read_csv(csv_file, low_memory=False, encoding_errors="replace")
            frames.append(standardize_frame(raw, source=csv_file.name))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            errors.append(f"{csv_file.name}: {error}")

    if not frames:
        details = "; ".join(errors) or "no readable files"
        raise ValueError(f"Could not load a labeled dataset: {details}")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if dataset["label"].nunique() < 2:
        raise ValueError("The dataset must contain both benign and malicious labels")
    return dataset
