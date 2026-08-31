from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@([\w.-]+\.[a-z]{2,})", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-z]{3,}")
URGENCY_WORDS = {"urgent", "immediately", "suspended", "warning", "expire", "action"}
CREDENTIAL_WORDS = {"password", "verify", "login", "credential", "authenticate"}
MONEY_WORDS = {"payment", "invoice", "refund", "prize", "money", "transfer"}


def _as_frame(data: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    for column in ("text", "subject", "body", "sender", "recipient", "attachment"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    if frame["text"].str.len().eq(0).all():
        frame["text"] = (frame["subject"] + "\n" + frame["body"]).str.strip()
    return frame


def _domain(value: str) -> str:
    match = EMAIL_PATTERN.search(value)
    return match.group(1).lower() if match else "unknown"


def _keyword_count(words: set[str], text: str) -> int:
    return sum(1 for word in WORD_PATTERN.findall(text.lower()) if word in words)


def metadata_records(data: pd.DataFrame | Iterable[dict]) -> list[dict]:
    frame = _as_frame(data)
    records = []
    for row in frame.itertuples(index=False):
        text = f"{row.subject}\n{row.body}\n{row.text}".strip()
        letters = [char for char in text if char.isalpha()]
        uppercase_ratio = sum(char.isupper() for char in letters) / max(len(letters), 1)
        records.append(
            {
                "url_count": len(URL_PATTERN.findall(text)),
                "html_present": int(bool(re.search(r"<\s*html|<\s*a\b|</\s*body", text, re.I))),
                "attachment_indicator": int(bool(row.attachment.strip()) or bool(re.search(r"attach(?:ed|ment)", text, re.I))),
                "message_length": len(text),
                "word_count": len(WORD_PATTERN.findall(text)),
                "uppercase_ratio": round(uppercase_ratio, 4),
                "exclamation_count": text.count("!"),
                "urgency_keyword_count": _keyword_count(URGENCY_WORDS, text),
                "credential_keyword_count": _keyword_count(CREDENTIAL_WORDS, text),
                "money_keyword_count": _keyword_count(MONEY_WORDS, text),
                "sender_domain": _domain(row.sender),
                "recipient_domain": _domain(row.recipient),
            }
        )
    return records


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features: int = 100_000, char_max_features: int = 50_000, min_df: int = 2):
        self.max_features = max_features
        self.char_max_features = char_max_features
        self.min_df = min_df

    def fit(self, X: pd.DataFrame, y=None):
        frame = _as_frame(X)
        min_df = self.min_df if len(frame) >= 20 else 1
        self.text_vectorizer_ = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=self.max_features,
            sublinear_tf=True,
        ).fit(frame["text"])
        self.char_vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            max_features=self.char_max_features,
            sublinear_tf=True,
        ).fit(frame["text"])
        self.metadata_vectorizer_ = DictVectorizer(sparse=True).fit(metadata_records(frame))
        return self

    def transform(self, X: pd.DataFrame) -> csr_matrix:
        frame = _as_frame(X)
        text_features = self.text_vectorizer_.transform(frame["text"])
        char_features = self.char_vectorizer_.transform(frame["text"])
        metadata_features = self.metadata_vectorizer_.transform(metadata_records(frame))
        return hstack([text_features, char_features, metadata_features], format="csr")

    def get_feature_names_out(self, input_features=None):
        text_names = [f"text__{name}" for name in self.text_vectorizer_.get_feature_names_out()]
        char_names = [f"char__{name}" for name in self.char_vectorizer_.get_feature_names_out()]
        metadata_names = [f"metadata__{name}" for name in self.metadata_vectorizer_.get_feature_names_out()]
        return text_names + char_names + metadata_names
