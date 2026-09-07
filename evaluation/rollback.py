import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR = (
#     r"C:\Users\ah266\OneDrive\Documents"
#     r"\production-ml-reliability"
# )
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"



# REGISTRY_PATH = os.path.join(
#     BASE_DIR,
#     "models",
#     "model_registry.json"
# )


with open(
    REGISTRY_PATH,
    "r"
) as f:

    registry = json.load(f)


active_model = registry[
    "active_model"
]


active_info = registry[
    "models"
][
    active_model
]


previous_model = active_info.get(
    "previous_model"
)


if previous_model is None:

    print(
        "No previous model available."
    )

    raise SystemExit


print(
    "Current model:",
    active_model
)

print(
    "Rollback target:",
    previous_model
)


# Archive current model

registry[
    "models"
][
    active_model
][
    "status"
] = "rolled_back"


# Restore previous model

registry[
    "models"
][
    previous_model
][
    "status"
] = "production"


registry[
    "active_model"
] = previous_model


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
    "\nRollback completed."
)

print(
    "Active model:",
    previous_model
)