# Phishing Email Detector

An interpretable malicious-email classifier built around the combined Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin dataset. The model uses word and character TF-IDF features, lightweight email metadata, and Logistic Regression.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the Kaggle dataset from [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) and place the CSV files in `data/`. If the combined `phishing_email.csv` is present, it is used automatically. Otherwise, all CSV files in `data/` are loaded and combined.

## Train

```bash
python -m phishing_detector.train --data data
```

The command writes `models/phishing_model.joblib` and `models/metrics.json`.

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard shows malicious probability and the highest-weight active features behind the prediction. It is intended for demonstration and analysis, not as a production mail gateway.

## Modeling approach: Logistic Regression

The classifier converts each email into a sparse feature vector containing word-level TF-IDF, character-level TF-IDF, and metadata signals such as URL counts, message length, HTML presence, and suspicious keyword counts. Logistic Regression learns a weight for each feature and combines them into a linear score. A sigmoid function converts that score into the probability that the email is malicious. The dashboard uses this probability for the final classification and displays the highest-contributing active features, making the prediction easier to interpret.

## Current results

The saved model was trained and evaluated on 82,077 emails from the combined dataset using an 80/20 stratified split:

| Split | Emails |
| --- | ---: |
| Training | 65,661 |
| Test | 16,416 |

| Metric | Score |
| --- | ---: |
| Accuracy | 96.98% |
| Precision | 96.57% |
| Recall | 97.69% |
| F1 score | 97.13% |
| ROC-AUC | 0.9949 |

The test confusion matrix is `[[7550, 297], [198, 8371]]`, using legitimate as class `0` and malicious/phishing as class `1`. Compared with the original word-only baseline, the character TF-IDF features improved accuracy from 95.24% to 96.98% and F1 from 95.42% to 97.13%.

## Project layout

```text
phishing_detector/
├── data.py       # CSV loading and label normalization
├── features.py   # TF-IDF and metadata feature extraction
├── model.py      # Logistic Regression pipeline
├── train.py      # training and evaluation CLI
└── explain.py    # per-email feature contributions
app.py            # Streamlit dashboard
```
