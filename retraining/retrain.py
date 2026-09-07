# ============================================================
# PRODUCTION ML MODEL RELIABILITY SYSTEM
# STEP 3 - AUTONOMOUS RETRAINING ENGINE
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import shutil
import joblib

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier
from pathlib import Path


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTION_DIR = DATA_DIR / "production"

MODELS_DIR = BASE_DIR / "models"

LOG_PATH = PROCESSED_DIR / "prediction_logs.csv"

REGISTRY_PATH = MODELS_DIR / "model_registry.json"

REFERENCE_PATH = PROCESSED_DIR / "reference_data.csv"

PRODUCTION_PATH = PRODUCTION_DIR / "production_data.csv"


# BASE_DIR = (
#     r"C:\Users\ah266\OneDrive\Documents"
#     r"\production-ml-reliability"
# )


# REFERENCE_PATH = os.path.join(
#     BASE_DIR,
#     "data",
#     "processed",
#     "reference_data.csv"
# )


# PRODUCTION_PATH = os.path.join(
#     BASE_DIR,
#     "data",
#     "processed",
#     "production_data.csv"
# )


# REGISTRY_PATH = os.path.join(
#     BASE_DIR,
#     "models",
#     "model_registry.json"
# )


# MODELS_DIR = os.path.join(
#     BASE_DIR,
#     "models"
# )


# PRODUCTION_DIR = os.path.join(
#     BASE_DIR,
#     "data",
#     "production"
# )


# ============================================================
# 3. RETRAINING CONFIGURATION
# ============================================================

F1_DROP_THRESHOLD = 0.03

MINIMUM_IMPROVEMENT = 0.01


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = [

    REFERENCE_PATH,

    PRODUCTION_PATH,

    REGISTRY_PATH

]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


print("\nAll required files found.")


# ============================================================
# 5. LOAD REFERENCE DATA
# ============================================================

reference_data = pd.read_csv(
    REFERENCE_PATH
)


print(
    "\nReference data shape:",
    reference_data.shape
)


# ============================================================
# 6. LOAD PRODUCTION DATA
# ============================================================

production_data = pd.read_csv(
    PRODUCTION_PATH
)


print(
    "Production data shape:",
    production_data.shape
)


# ============================================================
# 7. LOAD PRODUCTION LABELS
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


        # Convert DataFrame to Series

        y_batch = y_batch.iloc[:, 0]


        production_labels.append(
            y_batch
        )


        print(
            f"Batch {i} labels:",
            len(y_batch)
        )


    else:

        print(
            f"WARNING: File not found: "
            f"{label_path}"
        )


# ============================================================
# 8. COMBINE PRODUCTION LABELS
# ============================================================

if len(production_labels) == 0:

    raise ValueError(
        "No production label files were found."
    )


y_production = pd.concat(

    production_labels,

    ignore_index=True

)


print(
    "\nTotal production labels:",
    len(y_production)
)


# ============================================================
# 9. VERIFY PRODUCTION DATA AND LABELS
# ============================================================

if len(production_data) != len(y_production):

    raise ValueError(

        "Production data and production labels "
        "have different numbers of rows.\n"

        f"Production data: "
        f"{len(production_data)}\n"

        f"Production labels: "
        f"{len(y_production)}"

    )


X_production = production_data.copy()


# ============================================================
# 10. LOAD MODEL REGISTRY
# ============================================================

with open(

    REGISTRY_PATH,

    "r"

) as f:

    registry = json.load(f)


active_version = registry[
    "active_model"
]


print(
    "\nActive model:",
    active_version
)


# ============================================================
# 11. LOAD CURRENT PRODUCTION MODEL
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

        f"Current model not found: "
        f"{current_model_path}"

    )


current_model = joblib.load(

    current_model_path

)


# ============================================================
# 12. PREDICT PRODUCTION DATA USING CURRENT MODEL
# ============================================================

production_predictions = (

    current_model.predict(
        X_production
    )

)


production_probabilities = (

    current_model.predict_proba(
        X_production
    )[:, 1]

)


# ============================================================
# 13. CALCULATE PRODUCTION METRICS
# ============================================================

production_accuracy = (

    accuracy_score(

        y_production,

        production_predictions

    )

)


production_precision = (

    precision_score(

        y_production,

        production_predictions,

        zero_division=0

    )

)


production_recall = (

    recall_score(

        y_production,

        production_predictions,

        zero_division=0

    )

)


production_f1 = (

    f1_score(

        y_production,

        production_predictions,

        zero_division=0

    )

)


production_roc_auc = (

    roc_auc_score(

        y_production,

        production_probabilities

    )

)


# ============================================================
# 14. DISPLAY PRODUCTION PERFORMANCE
# ============================================================

print(
    "\n=========================================="
)

print(
    "CURRENT PRODUCTION PERFORMANCE"
)

print(
    "=========================================="
)

print(
    "Accuracy:",
    round(production_accuracy, 4)
)

print(
    "Precision:",
    round(production_precision, 4)
)

print(
    "Recall:",
    round(production_recall, 4)
)

print(
    "F1:",
    round(production_f1, 4)
)

print(
    "ROC-AUC:",
    round(production_roc_auc, 4)
)


# ============================================================
# 15. GET BASELINE METRICS
# ============================================================

baseline_info = registry[
    "models"
][
    active_version
]


baseline_f1 = float(
    baseline_info["f1"]
)


baseline_roc_auc = float(
    baseline_info["roc_auc"]
)


baseline_accuracy = float(
    baseline_info["accuracy"]
)


# ============================================================
# 16. CALCULATE PERFORMANCE DEGRADATION
# ============================================================

f1_drop = (

    baseline_f1
    - production_f1

)


roc_auc_drop = (

    baseline_roc_auc
    - production_roc_auc

)


accuracy_drop = (

    baseline_accuracy
    - production_accuracy

)


print(
    "\n=========================================="
)

print(
    "BASELINE VS PRODUCTION"
)

print(
    "=========================================="
)

print(
    "Baseline F1:",
    round(baseline_f1, 4)
)

print(
    "Production F1:",
    round(production_f1, 4)
)

print(
    "F1 drop:",
    round(f1_drop, 4)
)

print(
    "Baseline ROC-AUC:",
    round(baseline_roc_auc, 4)
)

print(
    "Production ROC-AUC:",
    round(production_roc_auc, 4)
)

print(
    "ROC-AUC drop:",
    round(roc_auc_drop, 4)
)


# ============================================================
# 17. RETRAINING DECISION
# ============================================================

retraining_required = (

    f1_drop >= F1_DROP_THRESHOLD

)


print(
    "\n=========================================="
)

print(
    "RETRAINING DECISION"
)

print(
    "=========================================="
)


if retraining_required:

    print(
        "WARNING: Performance degradation detected."
    )

    print(
        "Retraining is REQUIRED."
    )

else:

    print(
        "Current model performance is acceptable."
    )

    print(
        "Retraining is NOT required."
    )


# ============================================================
# 18. STOP IF RETRAINING IS NOT REQUIRED
# ============================================================

if not retraining_required:

    print(
        "\nNo retraining performed."
    )

    print(
        "Keeping model:",
        active_version
    )

    raise SystemExit


# ============================================================
# 19. PREPARE TRAINING DATA
# ============================================================

X_train = reference_data.copy()


# We need the training labels.
#
# Your reference_data.csv contains features only.
#
# Therefore we load the original dataset again
# for retraining.
#
# CHANGE THIS PATH to your original dataset.

ORIGINAL_DATA_PATH = os.path.join(

    BASE_DIR,

    "data",

    "raw",

    "Telco_customer.csv"

)


if not os.path.exists(
    ORIGINAL_DATA_PATH
):

    raise FileNotFoundError(

        "Original training dataset was not found.\n"

        "Expected path:\n"

        f"{ORIGINAL_DATA_PATH}\n\n"

        "Change ORIGINAL_DATA_PATH in retrain.py "
        "to your actual original dataset."

    )


df = pd.read_csv(
    ORIGINAL_DATA_PATH
)


# ============================================================
# 20. PREPROCESS ORIGINAL DATA
# ============================================================

df["Gender"] = df["Gender"].map({

    "Male": 1,

    "Female": 0

})


df["Senior Citizen"] = df[
    "Senior Citizen"
].map({

    "Yes": 1,

    "No": 0

})


df["Partner"] = df[
    "Partner"
].map({

    "Yes": 1,

    "No": 0

})


df["Dependents"] = df[
    "Dependents"
].map({

    "Yes": 1,

    "No": 0

})


df["Phone Service"] = df[
    "Phone Service"
].map({

    "Yes": 1,

    "No": 0

})


df["Paperless Billing"] = df[
    "Paperless Billing"
].map({

    "Yes": 1,

    "No": 0

})


# ============================================================
# 21. PREPARE TOTAL CHARGES
# ============================================================

df["Total Charges"] = pd.to_numeric(

    df["Total Charges"],

    errors="coerce"

)
df['Total Charges'] = df['Total Charges'].fillna(0)


# df = df.dropna(
#     subset=["Total Charges"]
# )


# ============================================================
# 22. PREPARE TARGET
# ============================================================

# df["Churn Value"] = df[
#     "Churn Value"
# ].map({

#     "Yes": 1,

#     "No": 0

# })


# ============================================================
# 23. REMOVE CUSTOMER ID
# ============================================================

if "customerID" in df.columns:

    df = df.drop(
        columns=["customerID"]
    )


# ============================================================
# 24. SELECT FEATURES
# ============================================================

feature_columns = [

    "Gender",

    "Senior Citizen",

    "Partner",

    "Dependents",

    "Tenure Months",

    "Phone Service",

    "Multiple Lines",

    "Internet Service",

    "Online Security",

    "Online Backup",

    "Device Protection",

    "Tech Support",

    "Streaming TV",

    "Streaming Movies",

    "Contract",

    "Paperless Billing",

    "Payment Method",

    "Monthly Charges",

    "Total Charges"

]


X = df[
    feature_columns
]


y = df[
    "Churn Value"
]


# ============================================================
# 25. CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical = [

    "Multiple Lines",

    "Internet Service",

    "Online Security",

    "Online Backup",

    "Device Protection",

    "Tech Support",

    "Streaming TV",

    "Streaming Movies",

    "Contract",

    "Payment Method"

]


numeric = [

    "Monthly Charges",

    "Total Charges",

    "Tenure Months"

]


# ============================================================
# 26. CREATE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            OneHotEncoder(
                handle_unknown="ignore" , 
                drop='first'
            ),

            categorical

        ),

        (

            "num",

            "passthrough",

            numeric

        )

    ],

    remainder="passthrough"

)


# ============================================================
# 27. CREATE CANDIDATE MODEL
# ============================================================

candidate_model = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "smote",

            SMOTE(

                sampling_strategy=0.6,

                random_state=42

            )

        ),

        (

            "cat",

            CatBoostClassifier(

                verbose=0,

                random_state=42,

                max_depth=5,

                learning_rate=0.03,

                n_estimators=300

            )

        )

    ]

)


# ============================================================
# 28. TRAIN CANDIDATE MODEL
# ============================================================

print(
    "\n=========================================="
)

print(
    "TRAINING CANDIDATE MODEL"
)

print(
    "=========================================="
)


candidate_model.fit(

    X,

    y

)


print(
    "Candidate model training completed."
)


# ============================================================
# 29. EVALUATE CANDIDATE MODEL
# ============================================================

candidate_predictions = (

    candidate_model.predict(
        X_production
    )

)


candidate_probabilities = (

    candidate_model.predict_proba(
        X_production
    )[:, 1]

)


candidate_accuracy = (

    accuracy_score(

        y_production,

        candidate_predictions

    )

)


candidate_precision = (

    precision_score(

        y_production,

        candidate_predictions,

        zero_division=0

    )

)


candidate_recall = (

    recall_score(

        y_production,

        candidate_predictions,

        zero_division=0

    )

)


candidate_f1 = (

    f1_score(

        y_production,

        candidate_predictions,

        zero_division=0

    )

)


candidate_roc_auc = (

    roc_auc_score(

        y_production,

        candidate_probabilities

    )

)


# ============================================================
# 30. DISPLAY CANDIDATE PERFORMANCE
# ============================================================

print(
    "\n=========================================="
)

print(
    "CANDIDATE MODEL PERFORMANCE"
)

print(
    "=========================================="
)

print(
    "Accuracy:",
    round(candidate_accuracy, 4)
)

print(
    "Precision:",
    round(candidate_precision, 4)
)

print(
    "Recall:",
    round(candidate_recall, 4)
)

print(
    "F1:",
    round(candidate_f1, 4)
)

print(
    "ROC-AUC:",
    round(candidate_roc_auc, 4)
)


# ============================================================
# 31. COMPARE CURRENT MODEL VS CANDIDATE
# ============================================================

f1_improvement = (

    candidate_f1
    - production_f1

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
    "Current v1 production F1:",
    round(production_f1, 4)
)

print(
    "Candidate F1:",
    round(candidate_f1, 4)
)

print(
    "Improvement:",
    round(f1_improvement, 4)
)


# ============================================================
# 32. DECIDE WHETHER TO PROMOTE CANDIDATE
# ============================================================

candidate_is_better = (

    candidate_f1
    >=
    production_f1
    + MINIMUM_IMPROVEMENT

)


# ============================================================
# 33. CREATE NEXT VERSION
# ============================================================

current_number = int(

    active_version.replace(
        "v",
        ""
    )

)


next_version = (

    f"v{current_number + 1}"

)


candidate_dir = os.path.join(

    MODELS_DIR,

    next_version

)


# ============================================================
# 34. PROMOTE OR REJECT
# ============================================================

if candidate_is_better:

    print(
        "\n=========================================="
    )

    print(
        "PROMOTION DECISION"
    )

    print(
        "=========================================="
    )

    print(
        f"Candidate {next_version} "
        "performed better."
    )

    print(
        f"Promoting {next_version} "
        "to production."
    )


    # Create candidate directory

    os.makedirs(

        candidate_dir,

        exist_ok=True

    )


    # Save model

    model_path = os.path.join(

        candidate_dir,

        "model.pkl"

    )


    joblib.dump(

        candidate_model,

        model_path

    )


    # ========================================================
    # 35. SAVE CANDIDATE METADATA
    # ========================================================

    metadata = {

        "version": next_version,

        "model_type":
            "CatBoostClassifier",

        "accuracy":
            float(candidate_accuracy),

        "precision":
            float(candidate_precision),

        "recall":
            float(candidate_recall),

        "f1":
            float(candidate_f1),

        "roc_auc":
            float(candidate_roc_auc),

        "status":
            "production",

        "created_at":
            pd.Timestamp.now().isoformat(),

        "previous_model":
            active_version,

        "training_reason":
            "Production F1 degradation",

        "production_f1_before_retraining":
            float(production_f1)

    }


    metadata_path = os.path.join(

        candidate_dir,

        "metadata.json"

    )


    with open(

        metadata_path,

        "w"

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4

        )


    # ========================================================
    # 36. UPDATE OLD MODEL STATUS
    # ========================================================

    registry[
        "models"
    ][
        active_version
    ][
        "status"
    ] = "archived"


    # ========================================================
    # 37. ADD NEW MODEL TO REGISTRY
    # ========================================================

    registry[
        "models"
    ][
        next_version
    ] = metadata


    # ========================================================
    # 38. CHANGE ACTIVE MODEL
    # ========================================================

    registry[
        "active_model"
    ] = next_version


    # ========================================================
    # 39. SAVE REGISTRY
    # ========================================================

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
        "\nModel registry updated."
    )

    print(
        "Active model:",
        next_version
    )

    print(
        "\nRetraining completed successfully."
    )


else:

    print(
        "\n=========================================="
    )

    print(
        "RETRAINING REJECTED"
    )

    print(
        "=========================================="
    )

    print(
        f"Candidate {next_version} "
        "did not improve enough."
    )

    print(
        "Keeping current model:",
        active_version
    )

    print(
        "Candidate was NOT promoted."
    )