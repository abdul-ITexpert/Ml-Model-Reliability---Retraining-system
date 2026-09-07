import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(
    page_title="ML Reliability Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title(
    "Production ML Model Reliability Dashboard"
)

st.caption(
    "Monitoring • Model Versioning • Retraining • Reliability"
)

with open(
    r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\models\model_registry.json",
    "r"
) as f:

    registry = json.load(f)

active_model = registry[
    "active_model"
]

st.metric(
    "Active Model",
    active_model
)

models = registry["models"]

for version, info in models.items():

    st.write(
        version,
        "→",
        info["status"]
    )

LOG_PATH = (
    r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\data\processed\prediction_logs.csv"
)


if os.path.exists(LOG_PATH):

    logs = pd.read_csv(LOG_PATH)
    logs["prediction_probability"] = pd.to_numeric(
    logs["prediction_probability"],
    errors="coerce"
)

else:

    logs = pd.DataFrame()

if not logs.empty:

    total_predictions = len(logs)

    high_risk = (
        logs["prediction_probability"] >= 0.7
    ).sum()

    average_probability = (
        logs["prediction_probability"].mean()
    )

else:

    total_predictions = 0
    high_risk = 0
    average_probability = 0

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Predictions",
    total_predictions
)

col2.metric(
    "High Risk Predictions",
    high_risk
)

col3.metric(
    "Average Risk",
    f"{average_probability:.2%}"
)

if not logs.empty:

    prediction_counts = (
        logs["prediction"]
        .value_counts()
        .sort_index()
    )

    st.subheader(
        "Prediction Distribution"
    )

    st.bar_chart(
        prediction_counts
    )

if not logs.empty:

    risk_counts = (
        logs["risk_level"]
        .value_counts()
    )

    st.subheader(
        "Risk Distribution"
    )

    st.bar_chart(
        risk_counts
    )

if not logs.empty:

    version_counts = (
        logs["model_version"]
        .value_counts()
    )

    st.subheader(
        "Predictions by Model Version"
    )

    st.bar_chart(
        version_counts
    )

if not logs.empty:

    st.subheader(
        "Recent Predictions"
    )

    st.dataframe(
        logs.tail(20),
        use_container_width=True
    )

with open(
    r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\data\processed\retraining_report.json",
    "r"
) as f:

    retraining_report = json.load(f)

st.subheader(
    "Retraining Status"
)

st.write(
    "Previous Model:",
    retraining_report[
        "previous_model"
    ]
)

st.write(
    "Candidate Model:",
    retraining_report[
        "candidate_model"
    ]
)

st.write(
    "Model Promoted:",
    retraining_report[
        "model_promoted"
    ]
)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ],

    "Model v1": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1,
        baseline_roc_auc
    ],

    "Model v2": [
        accuracy_v2,
        precision_v2,
        recall_v2,
        f1_v2,
        roc_auc_v2
    ]
})

st.subheader(
    "Model Performance Comparison"
)

st.dataframe(
    comparison,
    use_container_width=True
)