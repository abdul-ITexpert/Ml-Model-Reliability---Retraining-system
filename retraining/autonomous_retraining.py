import os
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

REGISTRY_PATH = MODELS_DIR / "model_registry.json"

RETRAIN_SCRIPT = BASE_DIR / "retraining" / "retrain.py"
# BASE_DIR = (
#     r"C:\Users\ah266\OneDrive\Documents"
#     r"\production-ml-reliability"
# )


# REGISTRY_PATH = os.path.join(
#     BASE_DIR,
#     "models",
#     "model_registry.json"
# )


# RETRAIN_SCRIPT = os.path.join(
#     BASE_DIR,
#     "retraining",
#     "retrain.py"
# )


# ============================================================
# LOAD MODEL REGISTRY
# ============================================================

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
    "AUTONOMOUS RETRAINING CONTROLLER"
)

print(
    "=========================================="
)

print(
    "Active production model:",
    active_model
)


# ============================================================
# CURRENT RELIABILITY METRICS
# ============================================================

current_f1 = 0.6403

current_roc_auc = 0.8777


# ============================================================
# RELIABILITY THRESHOLDS
# ============================================================

MIN_F1 = 0.60

MIN_ROC_AUC = 0.82


print(
    "\nCurrent F1:",
    current_f1
)

print(
    "Current ROC-AUC:",
    current_roc_auc
)


# ============================================================
# CHECK MODEL RELIABILITY
# ============================================================

performance_degraded = (

    current_f1 < MIN_F1

    or

    current_roc_auc < MIN_ROC_AUC

)


# ============================================================
# DECISION
# ============================================================

print(
    "\n=========================================="
)

print(
    "RETRAINING DECISION"
)

print(
    "=========================================="
)


if not performance_degraded:

    print(
        "✅ Model performance is acceptable."
    )

    print(
        "Retraining is NOT required."
    )

    print(
        "Keeping:",
        active_model
    )

    raise SystemExit


# ============================================================
# START RETRAINING
# ============================================================

print(
    "⚠️ Performance degradation detected."
)

print(
    "Starting autonomous retraining..."
)


if not os.path.exists(RETRAIN_SCRIPT):

    raise FileNotFoundError(
        f"Retraining script not found: {RETRAIN_SCRIPT}"
    )


result = subprocess.run(

    [
        "python",
        RETRAIN_SCRIPT
    ],

    cwd=BASE_DIR,

    capture_output=True,

    text=True

)


# ============================================================
# DISPLAY RETRAINING OUTPUT
# ============================================================

print(
    "\n=========================================="
)

print(
    "RETRAINING OUTPUT"
)

print(
    "=========================================="
)

print(
    result.stdout
)


if result.stderr:

    print(
        "\nRETRAINING ERRORS:"
    )

    print(
        result.stderr
    )


# ============================================================
# CHECK RESULT
# ============================================================

if result.returncode == 0:

    print(
        "\n=========================================="
    )

    print(
        "RETRAINING COMPLETED"
    )

    print(
        "=========================================="
    )

    print(
        "Candidate model generated."
    )

    print(
        "Candidate must now be evaluated."
    )

else:

    print(
        "\n=========================================="
    )

    print(
        "RETRAINING FAILED"
    )

    print(
        "=========================================="
    )

    print(
        "Current production model remains:",
        active_model
    )