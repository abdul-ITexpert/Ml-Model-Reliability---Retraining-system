# ============================================================
# PRODUCTION ML MODEL RELIABILITY SYSTEM
# STEP 4 - MODEL EVALUATION, PROMOTION & ROLLBACK
# ============================================================

import os
import json
import joblib

import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTION_DIR = DATA_DIR / "production"

MODELS_DIR = BASE_DIR / "models"

LOG_PATH = PROCESSED_DIR / "prediction_logs.csv"

REGISTRY_PATH = MODELS_DIR / "model_registry.json"

REFERENCE_DATA_PATH = PROCESSED_DIR / "reference_data.csv"

PRODUCTION_PATH = PROCESSED_DIR / "production_data.csv"
# BASE_DIR = (
#     r"C:\Users\ah266\OneDrive\Documents"
#     r"\production-ml-reliability"
# )

# MODELS_DIR = BASE_DIR / "models"
# MODELS_DIR = os.path.join(
#     BASE_DIR,
#     "models"
# )

# REGISTRY_PATH = MODELS_DIR / "model_registry.json"
# REGISTRY_PATH = os.path.join(
#     MODELS_DIR,
#     "model_registry.json"
# )


# PRODUCTION_PATH = os.path.join(
#     BASE_DIR,
#     "data",
#     "processed",
#     "production_data.csv"
# )

# PRODUCTION_DIR = DATA_DIR / "production"
# PRODUCTION_DIR = os.path.join(
#     BASE_DIR,
#     "data",
#     "production"
# )


# ============================================================
# 2. PROMOTION CONFIGURATION
# ============================================================

MIN_F1_IMPROVEMENT = 0.01

MAX_ROC_AUC_DROP = 0.02


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


active_version = registry[
    "active_model"
]


print(
    "Current active model:",
    active_version
)


# ============================================================
# 4. FIND CANDIDATE MODEL
# ============================================================

model_versions = list(
    registry["models"].keys()
)


version_numbers = []


for version in model_versions:

    try:

        number = int(
            version.replace(
                "v",
                ""
            )
        )

        version_numbers.append(
            number
        )

    except ValueError:

        pass


if len(version_numbers) == 0:

    raise ValueError(
        "No model versions found."
    )


latest_version_number = max(
    version_numbers
)


candidate_version = (
    f"v{latest_version_number}"
)


if candidate_version == active_version:

    if latest_version_number == 1:

        print(
            "No candidate model available yet."
        )

        raise SystemExit

    candidate_version = (
        f"v{latest_version_number - 1}"
    )


print(
    "Candidate model:",
    candidate_version
)


# ============================================================
# 5. LOAD PRODUCTION DATA
# ============================================================

if not os.path.exists(
    PRODUCTION_PATH
):

    raise FileNotFoundError(
        "Production data not found."
    )


X_production = pd.read_csv(
    PRODUCTION_PATH
)


# ============================================================
# 6. LOAD PRODUCTION LABELS
# ============================================================

production_labels = []


for i in range(1, 6):

    label_path = os.path.join(

        PRODUCTION_DIR,

        f"labels_{i:03d}.csv"

    )


    if os.path.exists(label_path):

        y_batch = pd.read_csv(
            label_path
        )

        y_batch = y_batch.iloc[:, 0]

        production_labels.append(
            y_batch
        )


if len(production_labels) == 0:

    raise ValueError(
        "Production labels not found."
    )


y_production = pd.concat(

    production_labels,

    ignore_index=True

)


# ============================================================
# 7. CHECK DATA SIZE
# ============================================================

if len(X_production) != len(y_production):

    raise ValueError(

        "Production data and labels "
        "have different sizes."

    )


# ============================================================
# 8. LOAD CURRENT MODEL
# ============================================================

current_model_path = os.path.join(

    MODELS_DIR,

    active_version,

    "model.pkl"

)


if not os.path.exists(
    current_model_path
):

    raise FileNotFoundError(

        f"Model not found: "
        f"{current_model_path}"

    )


current_model = joblib.load(
    current_model_path
)


# ============================================================
# 9. LOAD CANDIDATE MODEL
# ============================================================

candidate_model_path = os.path.join(

    MODELS_DIR,

    candidate_version,

    "model.pkl"

)


if not os.path.exists(
    candidate_model_path
):

    raise FileNotFoundError(

        f"Candidate model not found: "
        f"{candidate_model_path}"

    )


candidate_model = joblib.load(
    candidate_model_path
)


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y
):

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )


    metrics = {

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        )

    }


    return metrics


# ============================================================
# 11. EVALUATE CURRENT MODEL
# ============================================================

current_metrics = evaluate_model(

    current_model,

    X_production,

    y_production

)


# ============================================================
# 12. EVALUATE CANDIDATE MODEL
# ============================================================

candidate_metrics = evaluate_model(

    candidate_model,

    X_production,

    y_production

)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "CURRENT MODEL"
)

print(
    "=========================================="
)

for key, value in current_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


print(
    "\n=========================================="
)

print(
    "CANDIDATE MODEL"
)

print(
    "=========================================="
)

for key, value in candidate_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# 14. CALCULATE IMPROVEMENT
# ============================================================

f1_improvement = (

    candidate_metrics["f1"]

    -

    current_metrics["f1"]

)


roc_auc_change = (

    candidate_metrics["roc_auc"]

    -

    current_metrics["roc_auc"]

)


accuracy_change = (

    candidate_metrics["accuracy"]

    -

    current_metrics["accuracy"]

)


print(
    "\n=========================================="
)

print(
    "MODEL COMPARISON"
)

print(
    "=========================================="
)

print(
    "F1 improvement:",
    round(f1_improvement, 4)
)

print(
    "ROC-AUC change:",
    round(roc_auc_change, 4)
)

print(
    "Accuracy change:",
    round(accuracy_change, 4)
)


# ============================================================
# 15. PROMOTION RULE
# ============================================================

f1_requirement = (

    f1_improvement
    >=
    MIN_F1_IMPROVEMENT

)


roc_auc_requirement = (

    roc_auc_change
    >=
    -MAX_ROC_AUC_DROP

)


promote_candidate = (

    f1_requirement
    and
    roc_auc_requirement

)


# ============================================================
# 16. PROMOTION DECISION
# ============================================================

print(
    "\n=========================================="
)

print(
    "PROMOTION DECISION"
)

print(
    "=========================================="
)


if promote_candidate:

    print(
        "Candidate model PASSED evaluation."
    )

    print(
        f"{candidate_version} "
        "will become production."
    )


    # Archive current model

    registry[
        "models"
    ][
        active_version
    ][
        "status"
    ] = "archived"


    # Update candidate metadata

    registry[
        "models"
    ][
        candidate_version
    ][
        "status"
    ] = "production"


    registry[
        "models"
    ][
        candidate_version
    ][
        "production_metrics"
    ] = candidate_metrics


    # Store previous model

    registry[
        "models"
    ][
        candidate_version
    ][
        "previous_model"
    ] = active_version


    # Change active model

    registry[
        "active_model"
    ] = candidate_version


    # Save registry

    with open(

        REGISTRY_PATH,

        "w"

    ) as f:

        json.dump(

            registry,

            f,

            indent=4

        )


    print(
        "\nModel promoted successfully."
    )

    print(
        "Active model:",
        candidate_version
    )


else:

    print(
        "Candidate model FAILED evaluation."
    )

    print(
        f"{active_version} "
        "will remain production."
    )


    registry[
        "models"
    ][
        candidate_version
    ][
        "status"
    ] = "rejected"


    registry[
        "models"
    ][
        candidate_version
    ][
        "evaluation_metrics"
    ] = candidate_metrics


    with open(

        REGISTRY_PATH,

        "w"

    ) as f:

        json.dump(

            registry,

            f,

            indent=4

        )


    print(
        "\nCandidate rejected."
    )

    print(
        "Active model:",
        active_version
    )


# ============================================================
# 17. FINAL STATUS
# ============================================================

print(
    "\n=========================================="
)

print(
    "FINAL MODEL STATUS"
)

print(
    "=========================================="
)

print(
    "Active model:",
    registry["active_model"]
)