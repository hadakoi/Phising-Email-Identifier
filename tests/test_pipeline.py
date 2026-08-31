import pandas as pd

from phishing_detector.data import standardize_frame
from phishing_detector.explain import explain_prediction
from phishing_detector.model import build_model


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Email Subject": [
                "Team lunch next Friday",
                "Please review the project notes",
                "Urgent verify your password now",
                "Your account is suspended click http://bad.example",
            ],
            "Email Text": [
                "Let me know if you can join us.",
                "The notes are attached for review.",
                "Login immediately to avoid suspension.",
                "Send your credentials to restore access.",
            ],
            "Email Type": ["ham", "legitimate", "spam", "phishing"],
            "Sender": ["colleague@company.com", "lead@company.com", "alert@unknown.com", "support@unknown.com"],
        }
    )


def test_standardize_frame_maps_common_dataset_headers():
    result = standardize_frame(sample_data())
    assert list(result["label"]) == [0, 0, 1, 1]
    assert result.loc[2, "text"].startswith("Urgent verify")


def test_standardize_frame_supports_combined_dataset_header():
    result = standardize_frame(pd.DataFrame({"text_combined": ["normal note", "verify password"], "label": [0, 1]}))
    assert list(result["label"]) == [0, 1]
    assert result.loc[1, "text"] == "verify password"


def test_model_predicts_and_explains():
    data = standardize_frame(sample_data())
    model = build_model().fit(data, data["label"])
    result = explain_prediction(model, data.iloc[2].to_dict())
    assert 0 <= result["probability"] <= 1
    assert result["label"] in {"MALICIOUS", "LEGITIMATE"}
    assert result["positive_features"]
