from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from phishing_detector.explain import explain_prediction


MODEL_PATH = Path("models/phishing_model.joblib")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")
st.title("Phishing Email Detector")
st.caption("Word and character TF-IDF + metadata features + Logistic Regression")

if not MODEL_PATH.exists():
    st.info("Train a model first: `python -m phishing_detector.train --data data`")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Email details")
    subject = st.text_input("Subject", "Urgent: verify your account")
    sender = st.text_input("Sender", "security@example.com")
    recipient = st.text_input("Recipient", "employee@company.com")
    attachment = st.text_input("Attachment", "")
    body = st.text_area("Body", height=280, placeholder="Paste the email body here...")
    analyze = st.button("Analyze email", type="primary", use_container_width=True)

email = {"subject": subject, "body": body, "sender": sender, "recipient": recipient, "attachment": attachment}
if analyze or body or subject:
    result = explain_prediction(model, email)
    probability = result["probability"]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Malicious probability", f"{probability:.1%}")
        if result["label"] == "MALICIOUS":
            st.error("MALICIOUS / PHISHING")
        else:
            st.success("LIKELY LEGITIMATE")
    with right:
        st.subheader("Why the model decided this")
        explanation = pd.DataFrame(result["positive_features"])
        if explanation.empty:
            st.write("No strong malicious signals were active in this message.")
        else:
            explanation["contribution"] = explanation["contribution"].round(3)
            st.dataframe(explanation, hide_index=True, use_container_width=True)

    with st.expander("Features pushing toward legitimate"):
        st.dataframe(pd.DataFrame(result["negative_features"]), hide_index=True, use_container_width=True)

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.divider()
    st.subheader("Validation snapshot")
    cols = st.columns(5)
    for column, key in zip(cols, ("accuracy", "precision", "recall", "f1", "roc_auc")):
        column.metric(key.replace("_", " ").title(), f"{metrics[key]:.1%}")
