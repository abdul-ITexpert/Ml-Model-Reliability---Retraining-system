from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd 
import joblib
import json
import os
from datetime import datetime
from scipy.stats import ks_2samp
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTION_DIR = DATA_DIR / "production"

FRONTEND_DIR = BASE_DIR / "ml-reliability-frontend"

MODELS_DIR = BASE_DIR / "models"

LOG_PATH = PROCESSED_DIR / "prediction_logs.csv"

REGISTRY_PATH = MODELS_DIR / "model_registry.json"

REFERENCE_DATA_PATH = PROCESSED_DIR / "reference_data.csv"

PRODUCTION_DATA_PATH = PRODUCTION_DIR / "production_data.csv"

app = FastAPI(
    title="Production ML Reliability System",
    description="ML model serving and model version management API",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# LOG_PATH = r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\data\processed\prediction_logs.csv"
# REGISTRY_PATH = r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\models\model_registry.json"
# REFERENCE_DATA_PATH = (
#     r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\data\processed\reference_data.csv"
# )

# PRODUCTION_DATA_PATH = (
#     r"C:\Users\ah266\OneDrive\Documents\production-ml-reliability\data\processed\production_data.csv"
# )
BASELINE_METRICS = {
    "accuracy": 0.8260869565217391,
    "precision": 0.69,
    "recall": 0.64,
    "f1": 0.66,
    "roc_auc": 0.8668892281841242
}
def load_registry():

    with open(
        REGISTRY_PATH,
        "r"
    ) as f:

        return json.load(f)

def log_prediction(
    prediction,
    probability,
    model_version,
    actual_label=None,
    batch_id="frontend"
):

    log = pd.DataFrame([{

        "batch_id": batch_id,

        "model_version": model_version,

        "prediction": int(prediction),

        "prediction_probability": float(probability),

        "actual_label": actual_label,

        "timestamp": datetime.now().isoformat()

    }])

    if os.path.exists(LOG_PATH):

        log.to_csv(
            LOG_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        log.to_csv(
            LOG_PATH,
            index=False
        )

def load_active_model():

    registry = load_registry()

    active_version = registry["active_model"]

    model_path = MODELS_DIR / active_version / "model.pkl"

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model, active_version

model, active_version = load_active_model()

def calculate_numeric_drift(
    reference,
    production
):

    reference = pd.to_numeric(
        reference,
        errors="coerce"
    ).dropna()

    production = pd.to_numeric(
        production,
        errors="coerce"
    ).dropna()


    if len(reference) == 0 or len(production) == 0:

        return {
            "drift_score": None,
            "status": "Unknown"
        }


    statistic, p_value = ks_2samp(
        reference,
        production
    )


    status = (
        "Drift Detected"
        if p_value < 0.05
        else "Normal"
    )


    return {
        "drift_score": round(
            float(statistic),
            4
        ),

        "p_value": round(
            float(p_value),
            6
        ),

        "status": status,

        "reference_mean": round(
            float(reference.mean()),
            2
        ),

        "production_mean": round(
            float(production.mean()),
            2
        )
    }

def calculate_categorical_drift(
    reference,
    production
):

    reference = reference.astype(str)
    production = production.astype(str)


    reference_dist = (
        reference
        .value_counts(normalize=True)
    )


    production_dist = (
        production
        .value_counts(normalize=True)
    )


    all_categories = set(
        reference_dist.index
    ).union(
        set(production_dist.index)
    )


    difference = 0


    for category in all_categories:

        ref_value = reference_dist.get(
            category,
            0
        )

        prod_value = production_dist.get(
            category,
            0
        )


        difference += abs(
            ref_value - prod_value
        )


    drift_score = difference / 2


    status = (
        "Drift Detected"
        if drift_score >= 0.20
        else "Normal"
    )


    return {

        "drift_score": round(
            float(drift_score),
            4
        ),

        "status": status,

        "reference_top_value":
            str(
                reference.mode().iloc[0]
            )
            if len(reference) > 0
            else None,

        "production_top_value":
            str(
                production.mode().iloc[0]
            )
            if len(production) > 0
            else None
    }

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "service": "Production ML Reliability System"
    }

@app.get("/")
def frontend():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )

@app.get("/model")
def model_status():

    registry = load_registry()

    active_version = registry[
        "active_model"
    ]

    return {
        "active_model": active_version
    }

@app.get("/models")
def get_models():

    registry = load_registry()

    return {
        "active_model": registry["active_model"],
        "models": registry["models"]
    }
@app.get("/models/{version}")
def get_model_version(version: str):

    registry = load_registry()

    if version not in registry["models"]:

        raise HTTPException(
            status_code=404,
            detail=f"Model version '{version}' not found"
        )

    return registry["models"][version]

@app.get("/models/compare")
def compare_models():

    registry = load_registry()

    models = registry["models"]

    comparison = []

    for version, info in models.items():

        comparison.append({

            "version": version,

            "status": info.get(
                "status"
            ),

            "accuracy": info.get(
                "accuracy"
            ),

            "precision": info.get(
                "precision"
            ),

            "recall": info.get(
                "recall"
            ),

            "f1": info.get(
                "f1"
            ),

            "roc_auc": info.get(
                "roc_auc"
            )

        })

    return {
        "active_model":
            registry["active_model"],

        "models":
            comparison
    }

class CustomerData(BaseModel):

    Gender: int
    Senior_Citizen: int
    Partner: int
    Dependents: int

    Tenure_Months: float

    Phone_Service: int
    Multiple_Lines: str

    Internet_Service: str

    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str

    Streaming_TV: str
    Streaming_Movies: str

    Contract: str

    Paperless_Billing: int

    Payment_Method: str

    Monthly_Charges: float
    Total_Charges: float

@app.post("/predict")
def predict(data: CustomerData):

    global model
    global active_version

    input_data = {
        "Gender": data.Gender,
        "Senior Citizen": data.Senior_Citizen,
        "Partner": data.Partner,
        "Dependents": data.Dependents,
        "Tenure Months": data.Tenure_Months,
        "Phone Service": data.Phone_Service,
        "Multiple Lines": data.Multiple_Lines,
        "Internet Service": data.Internet_Service,
        "Online Security": data.Online_Security,
        "Online Backup": data.Online_Backup,
        "Device Protection": data.Device_Protection,
        "Tech Support": data.Tech_Support,
        "Streaming TV": data.Streaming_TV,
        "Streaming Movies": data.Streaming_Movies,
        "Contract": data.Contract,
        "Paperless Billing": data.Paperless_Billing,
        "Payment Method": data.Payment_Method,
        "Monthly Charges": data.Monthly_Charges,
        "Total Charges": data.Total_Charges
    }

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    log_prediction(
    prediction,
    probability,
    active_version,
    batch_id="frontend"
    )

    return {
    "prediction": int(prediction),
    "churn_probability": float(probability),
    "model_version": active_version
    }

@app.get("/logs")
def get_logs():

    if not os.path.exists(LOG_PATH):
        return []

    logs = pd.read_csv(LOG_PATH)

    logs = logs.fillna("")

    return logs.to_dict(
        orient="records"
    )

@app.get("/monitoring")
def monitoring():

    if not os.path.exists(
        REFERENCE_DATA_PATH
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Reference data file not found. "
                "Check REFERENCE_DATA_PATH."
            )
        )


    if not os.path.exists(
        PRODUCTION_DATA_PATH
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Production data file not found. "
                "Check PRODUCTION_DATA_PATH."
            )
        )


    reference_df = pd.read_csv(
        REFERENCE_DATA_PATH
    )


    production_df = pd.read_csv(
        PRODUCTION_DATA_PATH
    )


    numeric_features = [

        "Monthly Charges",

        "Tenure Months",

        "Total Charges"

    ]


    categorical_features = [

        "Contract",

        "Internet Service",

        "Payment Method"

    ]


    feature_results = {}


    for feature in numeric_features:

        if (
            feature in reference_df.columns
            and feature in production_df.columns
        ):

            feature_results[feature] = (
                calculate_numeric_drift(

                    reference_df[feature],

                    production_df[feature]

                )
            )


    for feature in categorical_features:

        if (
            feature in reference_df.columns
            and feature in production_df.columns
        ):

            feature_results[feature] = (
                calculate_categorical_drift(

                    reference_df[feature],

                    production_df[feature]

                )
            )


    drift_detected = any(

        result["status"] == "Drift Detected"

        for result in feature_results.values()

    )


    data_drift_status = (

        "Drift Detected"

        if drift_detected

        else "Normal"

    )


    retraining_status = (

        "Review Required"

        if drift_detected

        else "Not Required"

    )


    return {

        "data_drift": data_drift_status,

        "baseline_metrics": BASELINE_METRICS,

        "retraining": retraining_status,

        "features": feature_results,

        "reference_rows":
            len(reference_df),

        "production_rows":
            len(production_df)

    }

@app.post("/reload-model")
def reload_model():

    global model
    global active_version

    model, active_version = load_active_model()

    return {
        "message": "Model reloaded successfully",
        "active_model": active_version
    }

@app.get("/model/metadata")
def model_metadata():

    registry = load_registry()

    active_version = registry[
        "active_model"
    ]

    metadata_path = (
    MODELS_DIR / active_version / "metadata.json"
    )

    if not metadata_path.exists():

        return {
        "model_version": active_version,
        "metadata": "Not available"
    }

    with open(metadata_path, "r") as f:

        metadata = json.load(f)

    return metadata

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend"
)

app.mount(
    "/css",
    StaticFiles(directory=FRONTEND_DIR / "css"),
    name="css"
)

app.mount(
    "/js",
    StaticFiles(directory=FRONTEND_DIR / "js"),
    name="js"
)