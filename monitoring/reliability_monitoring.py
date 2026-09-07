import os
import json
import pandas as pd
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")



# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR = (
#     r"C:\Users\ah266\OneDrive\Documents"
#     r"\production-ml-reliability"
# )

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOG_PATH = PROCESSED_DIR / "prediction_logs.csv"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


# REGISTRY_PATH = os.path.join(
#     BASE_DIR,
#     "models",
#     "model_registry.json"
# )


# LOG_PATH = os.path.join(
#     BASE_DIR,
#     "data",
#     "processed",
#     "prediction_logs.csv"
# )


# ============================================================
# 2. RELIABILITY THRESHOLDS
# ============================================================

MIN_F1 = 0.60

MIN_ROC_AUC = 0.82


# ============================================================
# 3. LOAD MODEL REGISTRY
# ============================================================

if not os.path.exists(REGISTRY_PATH):

    raise FileNotFoundError(
        "model_registry.json not found."
    )


with open(
    REGISTRY_PATH,
    "r"
) as f:

    registry = json.load(f)


active_model = registry[
    "active_model"
]


print(
    "=========================================="
)

print(
    "PRODUCTION MODEL RELIABILITY MONITOR"
)

print(
    "=========================================="
)

print(
    "Active model:",
    active_model
)


# ============================================================
# 4. CHECK PRODUCTION LOGS
# ============================================================

if not os.path.exists(LOG_PATH):

    print(
        "\nPrediction log not found."
    )

    print(
        "No production monitoring data available."
    )

    raise SystemExit


logs = pd.read_csv(
    LOG_PATH
)

# print("\nLOG PATH:")
# print(LOG_PATH)

# print("\nLAST 10 ROWS:")
# print(logs.tail(10).to_string())

# print("\nMODEL VERSIONS:")
# print(logs["model_version"].value_counts(dropna=False))


print(
    "\nTotal production predictions:",
    len(logs)
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "timestamp",
    "prediction",
    "prediction_probability",
    "model_version"

]


missing_columns = [

    column

    for column in required_columns

    if column not in logs.columns

]


if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 6. CLEAN PROBABILITY COLUMN
# ============================================================

logs["prediction_probability"] = pd.to_numeric(

    logs["prediction_probability"],

    errors="coerce"

)


logs = logs.dropna(
    subset=["prediction_probability"]
)


# ============================================================
# 7. ACTIVE MODEL LOGS
# ============================================================

active_logs = logs[
    logs["model_version"]
    ==
    active_model
]


print(
    "Predictions from active model:",
    len(active_logs)
)


# ============================================================
# 8. BASIC PRODUCTION ANALYSIS
# ============================================================

if len(active_logs) == 0:

    print(
        "\nNo predictions found for active model."
    )

    raise SystemExit


high_risk = (

    active_logs["prediction_probability"]
    >=
    0.70

).sum()


low_risk = (

    active_logs["prediction_probability"]
    <
    0.70

).sum()


high_risk_percentage = (

    high_risk
    /
    len(active_logs)
    *
    100

)


print(
    "\n=========================================="
)

print(
    "PRODUCTION PREDICTION ANALYSIS"
)

print(
    "=========================================="
)

print(
    "High-risk predictions:",
    high_risk
)

print(
    "Low-risk predictions:",
    low_risk
)

print(
    "High-risk percentage:",
    round(
        high_risk_percentage,
        2
    ),
    "%"
)


# ============================================================
# 9. RETRAINING DECISION
# ============================================================

# We use the latest known production metrics.
#
# These can later be replaced by metrics
# calculated automatically from newly labelled
# production data.

current_f1 = 0.6403

current_roc_auc = 0.8777


print(
    "\n=========================================="
)

print(
    "RELIABILITY CHECK"
)

print(
    "=========================================="
)

print(
    "Current F1:",
    current_f1
)

print(
    "Required minimum F1:",
    MIN_F1
)

print(
    "Current ROC-AUC:",
    current_roc_auc
)

print(
    "Required minimum ROC-AUC:",
    MIN_ROC_AUC
)


# ============================================================
# 10. DECISION
# ============================================================

if (

    current_f1 < MIN_F1

    or

    current_roc_auc < MIN_ROC_AUC

):

    retraining_required = True

else:

    retraining_required = False


# ============================================================
# 11. FINAL DECISION
# ============================================================

print(
    "\n=========================================="
)

print(
    "AUTONOMOUS DECISION"
)

print(
    "=========================================="
)


if retraining_required:

    print(
        "⚠️ MODEL PERFORMANCE DEGRADATION DETECTED"
    )

    print(
        "Retraining is required."
    )

    print(
        "Action: START RETRAINING PIPELINE"
    )

else:

    print(
        "✅ MODEL IS RELIABLE"
    )

    print(
        "Retraining is not required."
    )

    print(
        "Action: KEEP CURRENT MODEL"
    )