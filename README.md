
# 🚀 Production ML Model Reliability & Autonomous Retraining System

An end-to-end **Machine Learning Engineering and MLOps project** designed to deploy, monitor, evaluate, version, and reliably manage a production Machine Learning model.

The project uses a **Telco Customer Churn** dataset and demonstrates how an ML model can be managed throughout its production lifecycle rather than stopping after model training.

---

## 🎥 Demo Video

[▶ Watch the Demo](https://github.com/user-attachments/assets/9f6bc937-11b9-4995-9dd0-9a052898b825)

## 🌐 Live Demo

🔗 **Try the Application Here:**

https://ml-model-reliability-retraining-system.onrender.com/

## 📌 Project Overview

A Machine Learning model may perform well during development but degrade after deployment because production data can change over time.

This project addresses that problem by building a complete ML reliability workflow:

```text
Data
 ↓
Preprocessing
 ↓
Model Training
 ↓
Evaluation
 ↓
Model Versioning
 ↓
Production Deployment
 ↓
Prediction Logging
 ↓
Production Monitoring
 ↓
Candidate Retraining
 ↓
Model Evaluation
 ↓
Accept / Reject Candidate
 ↓
Reliable Production Model
````

The system currently keeps **Model v2** in production because the candidate **Model v1** did not provide sufficient improvement.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Build a customer churn classification model
* Handle class imbalance using SMOTE
* Build a reusable preprocessing pipeline
* Deploy the ML model using FastAPI
* Create REST API endpoints for prediction
* Track model versions
* Maintain an active production model
* Log production predictions
* Monitor production model performance
* Monitor production data
* Train candidate models
* Compare candidate and production models
* Automatically reject weaker candidate models
* Provide a monitoring dashboard
* Prepare the system for cloud deployment

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │    Web Dashboard     │
                         │    HTML/CSS/JS       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     FastAPI API      │
                         │   Model Serving      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │   Active Model  │   │ Prediction Logs │
                │       v2        │   │      CSV        │
                └────────┬────────┘   └────────┬────────┘
                         │                     │
                         │                     ▼
                         │            ┌─────────────────┐
                         │            │    Monitoring   │
                         │            └────────┬────────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Model Evaluation &   │
                         │ Retraining Pipeline  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Candidate Model v1   │
                         └──────────┬───────────┘
                                    │
                            ┌───────┴────────┐
                            ▼                ▼
                         ACCEPT            REJECT
                            │                │
                            ▼                ▼
                       Production       Keep v2
```

---

# 📁 Project Structure

```text
production-ml-reliability/
│
├── main.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── model_registry.json
│   │
│   ├── v1/
│   │   ├── model.pkl
│   │   └── metadata.json
│   │
│   └── v2/
│       ├── model.pkl
│       └── metadata.json
│
├── data/
│   ├── processed/
│   │   └── prediction_logs.csv
│   │
│   └── production/
│       ├── batch_001.csv
│       ├── batch_002.csv
│       ├── batch_003.csv
│       ├── batch_004.csv
│       ├── batch_005.csv
│       │
│       ├── labels_001.csv
│       ├── labels_002.csv
│       ├── labels_003.csv
│       ├── labels_004.csv
│       └── labels_005.csv
│
├── monitoring/
│   └── ...
│
├── retraining/
│   └── ...
│
└── frontend/
    ├── index.html
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

# 🧠 Machine Learning Problem

The project solves a **binary classification problem**:

> Predict whether a telecommunications customer is likely to churn.

### Target

```text
0 → Customer will not churn
1 → Customer will churn
```

---

# 📊 Dataset Features

The model uses customer information such as:

```text
Gender
Senior Citizen
Partner
Dependents
Tenure Months
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
Contract
Paperless Billing
Payment Method
Monthly Charges
Total Charges
```

---

# 🔄 Machine Learning Pipeline

The ML pipeline consists of:

```text
Raw Dataset
      ↓
Data Cleaning
      ↓
Feature / Target Separation
      ↓
Train/Test Split
      ↓
Preprocessing
      │
      ├── Numerical Features
      │       ↓
      │    Scaling
      │
      └── Categorical Features
              ↓
         One-Hot Encoding
      ↓
SMOTE
      ↓
CatBoost Classifier
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Model Registry
```

---

# ⚖️ Handling Class Imbalance

The churn dataset contains an imbalance between churn and non-churn customers.

To address this, the project uses:

### SMOTE

**Synthetic Minority Over-sampling Technique**

Example:

```python
SMOTE(
    sampling_strategy=0.6,
    random_state=42
)
```

SMOTE generates synthetic minority-class examples during training instead of simply duplicating existing samples.

---

# 🤖 Model

The main production classifier is:

### CatBoost Classifier

Example configuration:

```python
CatBoostClassifier(
    verbose=0,
    random_state=42,
    max_depth=5,
    learning_rate=0.03,
    n_estimators=300
)
```

The preprocessing and model are stored together as a pipeline where appropriate so that the API can receive raw customer feature values.

---

# 📈 Model Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

F1 and ROC-AUC are particularly important because accuracy alone can be misleading when dealing with imbalanced classification data.

---

# 🏆 Final Model Performance

## Model v1 — Candidate

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **80.13%** |
| Precision | **62.43%** |
| Recall    | **63.10%** |
| F1 Score  | **62.77%** |
| ROC-AUC   | **85.77%** |

---

## Model v2 — Production

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **81.90%** |
| Precision | **67.76%** |
| Recall    | **60.70%** |
| F1 Score  | **64.03%** |
| ROC-AUC   | **87.77%** |

---

# 🔍 Candidate Model Evaluation

The system compares a candidate model against the active production model before promotion.

Comparison:

```text
F1 improvement:   -0.0127
ROC-AUC change:   -0.0200
Accuracy change:  -0.0177
```

The candidate model failed the evaluation criteria.

Therefore:

```text
Candidate Model → v1
Decision         → REJECTED

Production Model → v2
Status            → ACTIVE
```

This prevents a newly trained but weaker model from automatically replacing the production model.

---

# 🗂️ Model Registry

The project maintains multiple model versions.

```text
models/
│
├── v1/
│   ├── model.pkl
│   └── metadata.json
│
└── v2/
    ├── model.pkl
    └── metadata.json
```

The registry identifies the active version.

Example:

```json
{
    "active_model": "v2"
}
```

### Current Registry

```text
v2 → PRODUCTION
v1 → REJECTED / ARCHIVED
```

---

# 🌐 FastAPI Backend

The backend is built with **FastAPI**.

It provides the ML model as a REST API.

### Main responsibilities

* Load the active model
* Serve predictions
* Return prediction probabilities
* Return model information
* Return model metadata
* Log predictions
* Reload the active model
* Provide monitoring data
* Serve the frontend

---

# 🔌 API Endpoints

| Endpoint          | Method | Description                    |
| ----------------- | ------ | ------------------------------ |
| `/`               | GET    | API health check               |
| `/model`          | GET    | Get active model               |
| `/model/metadata` | GET    | Get model metadata             |
| `/predict`        | POST   | Generate churn prediction      |
| `/reload-model`   | POST   | Reload active model            |
| `/logs`           | GET    | Get production prediction logs |
| `/docs`           | GET    | FastAPI Swagger documentation  |

---

# 🔮 Prediction Workflow

```text
User
 ↓
Customer Information
 ↓
Frontend Form
 ↓
POST /predict
 ↓
FastAPI
 ↓
Active Model v2
 ↓
Prediction + Probability
 ↓
Prediction Logging
 ↓
Dashboard
```

Example API response:

```json
{
    "prediction": 1,
    "probability": 0.78,
    "model_version": "v2"
}
```

---

# 📝 Prediction Logging

Every production prediction can be recorded for monitoring.

Typical fields:

```text
timestamp
prediction
probability
model_version
```

Example:

```text
2026-08-31T21:57:15,0,0.4224,v2
2026-08-31T22:00:08,0,0.0919,v2
```

This allows the monitoring system to determine:

* How many predictions were made
* Which model generated the prediction
* Prediction probability distribution
* Whether the active model is serving traffic

---

# 📦 Production Data Batches

Production data is simulated using multiple batches:

```text
batch_001.csv
batch_002.csv
batch_003.csv
batch_004.csv
batch_005.csv
```

Production labels:

```text
labels_001.csv
labels_002.csv
labels_003.csv
labels_004.csv
labels_005.csv
```

This allows the project to simulate data arriving over time.

The batches can later be used for production model evaluation and retraining.

---

# 📈 Production Monitoring

The monitoring system checks several aspects of model reliability.

## Data Drift

Current status:

```text
Normal
```

## Model Performance

Current active model:

```text
Model: v2
F1: 0.6403
ROC-AUC: 0.8777
```

## Feature Monitoring

Important monitored features include:

```text
Contract
Monthly Charges
Tenure Months
Total Charges
```

Current dashboard status:

```text
Normal
Within baseline
```

> Numerical drift values should only be shown when they are actually calculated by the monitoring pipeline.

---

# 🔁 Autonomous Retraining Workflow

The retraining workflow follows an evaluation gate.

```text
Production Data
       ↓
Production Labels
       ↓
Train Candidate Model
       ↓
Evaluate Candidate
       ↓
Compare Candidate vs Production
       ↓
      ┌───────────────┐
      │ Evaluation    │
      │    Gate       │
      └───────┬───────┘
              │
       ┌──────┴──────┐
       ↓             ↓
     PASS           FAIL
       ↓             ↓
   Promote        Reject
    Model          Model
       ↓             ↓
 Production      Keep v2
```

### Final decision

```text
Current Model:     v2
Candidate Model:   v1
Trigger:           Candidate failed evaluation
Decision:          REJECTED
```

---

# 🖥️ Web Dashboard

The frontend is built using:

* HTML
* CSS
* JavaScript

### Dashboard sections

```text
Dashboard
Make Prediction
Monitoring
Models & Retraining
Alerts
```

---

## 📊 Dashboard

Displays:

```text
Active Model: v2

Accuracy:  81.90%
Precision: 67.76%
Recall:    60.70%
F1:        64.03%
ROC-AUC:   87.77%
```

---

## 🔮 Make Prediction

Users can enter customer information through a web form.

The frontend sends the data to:

```text
POST /predict
```

The prediction and probability are then displayed to the user.

---

## 📈 Monitoring

The monitoring page displays:

* Data drift
* Model F1
* ROC-AUC
* Retraining status
* Feature monitoring
* Model performance
* Reliability checks

---

## 🤖 Models & Retraining

Displays:

```text
Production Model: v2
Candidate Model:  v1

Trigger:
Candidate failed evaluation

Decision:
Rejected
```

---

## 🚨 Alerts

The dashboard can display:

* Model health
* Monitoring status
* Retraining status
* Candidate model decisions

---

# 🔧 Portable File Paths

The original local application used Windows-specific paths such as:

```text
C:\Users\...\production-ml-reliability\...
```

These paths are not suitable for cloud deployment.

The project should use portable paths with `pathlib`.

Example:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTION_DIR = DATA_DIR / "production"
MODELS_DIR = BASE_DIR / "models"

LOG_PATH = PROCESSED_DIR / "prediction_logs.csv"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
```

Model path:

```python
model_path = MODELS_DIR / active_version / "model.pkl"
```

Metadata path:

```python
metadata_path = MODELS_DIR / active_version / "metadata.json"
```

This makes the project portable across:

```text
Windows
Linux
macOS
Docker
Cloud Servers
Render
Railway
```

---

# 🔗 Frontend API Configuration

For local development, it may be tempting to use:

```javascript
const API_BASE = "http://127.0.0.1:8000";
```

However, this should not be used for a public deployment.

When the frontend is served by the same FastAPI application, use:

```javascript
const API_BASE = "";
```

Then:

```javascript
fetch("/predict")
fetch("/model")
fetch("/logs")
```

will automatically use the current domain.

---

# 🚀 Local Installation

## 1. Clone Repository

```bash
git clone https://github.com/abdul-ITexpert/Ml-Model-Reliability---Retraining-system.git
cd production-ml-reliability
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start FastAPI

```bash
uvicorn main:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📦 Requirements

Typical dependencies:

```text
fastapi
uvicorn[standard]
pydantic
pandas
numpy
scikit-learn
imbalanced-learn
catboost
joblib
```

For deployment, package versions should be compatible with the versions used when the model was trained and serialized.

You can generate the environment requirements with:

```bash
pip freeze > requirements.txt
```

---

# ☁️ Deployment

The application can be deployed as a public FastAPI web service.

Recommended architecture:

```text
                 Internet
                    │
                    ▼
             Public Web URL
                    │
                    ▼
              FastAPI Server
              ┌─────┴─────┐
              ▼           ▼
          Frontend      ML API
                           │
                           ▼
                        Model v2
```

A cloud platform such as Render can run the FastAPI application.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

After deployment, the generated public URL can be shared with users.

---

# ⚠️ Production Storage

CSV files are useful for this academic project and demonstration.

However, cloud containers may have temporary filesystems.

For a real production system, use persistent storage for:

* Prediction logs
* Model artifacts
* Model registry
* Retraining results

Possible options:

```text
PostgreSQL
Cloud Object Storage
Persistent Disk
Managed Model Registry
```

---

# 🧪 Testing Checklist

Before deployment, test:

### API Health

```text
GET /
```

### Active Model

```text
GET /model
```

Expected:

```text
v2
```

### Model Metadata

```text
GET /model/metadata
```

### Prediction

```text
POST /predict
```

Verify:

* Input validation
* Prediction
* Probability
* Model version
* Prediction logging

### Logs

```text
GET /logs
```

### Swagger

```text
GET /docs
```

---

# 🐛 Troubleshooting

## 1. HTTP 422

If `/predict` returns:

```text
422 Unprocessable Entity
```

check:

* Field names
* Required fields
* Data types
* Missing values
* Categorical values

If the preprocessing pipeline already contains `OneHotEncoder`, send the original categorical values rather than manually one-hot encoding them.

---

## 2. Feature Name Error

If you see:

```text
Feature names must match fit
```

verify:

* Feature names are identical
* Feature order is correct
* The same preprocessing pipeline is used
* Required features are present

---

## 3. NaN Labels

If retraining produces:

```text
ValueError: Input y contains NaN
```

remove invalid target rows:

```python
valid = y.notna()

X = X.loc[valid]
y = y.loc[valid]
```

---

## 4. Active Model Has Zero Predictions

If monitoring reports:

```text
Active model: v2
Predictions from active model: 0
```

while prediction logs exist, inspect the CSV schema.

Use consistent fields:

```text
timestamp
prediction
probability
model_version
```

---

# 🔒 Security Considerations

For a real public deployment, add:

* Authentication
* Authorization
* Rate limiting
* Strict CORS configuration
* HTTPS
* Secure environment variables
* Database-backed logging
* Protected model-management endpoints
* Error monitoring

Never commit secrets to GitHub.

Do not commit:

```text
.env
API keys
Passwords
Private tokens
Cloud credentials
```

---

# 📊 Final Project Status

| Component                    | Status      |
| ---------------------------- | ----------- |
| ML Model                     | ✅ Completed |
| Preprocessing                | ✅ Completed |
| SMOTE                        | ✅ Completed |
| CatBoost                     | ✅ Completed |
| Model Evaluation             | ✅ Completed |
| Model Versioning             | ✅ Completed |
| Model Registry               | ✅ Completed |
| FastAPI Backend              | ✅ Completed |
| Prediction Logging           | ✅ Completed |
| Production Monitoring        | ✅ Completed |
| Retraining Workflow          | ✅ Completed |
| Candidate Evaluation         | ✅ Completed |
| Web Dashboard                | ✅ Completed |
| Portable Paths               | ✅ Completed |
| Cloud Deployment Preparation | ✅ Completed |

---

# 🏆 Current Production Model

```text
┌───────────────────────────────┐
│       PRODUCTION MODEL        │
├───────────────────────────────┤
│ Model Version: v2             │
│                               │
│ Accuracy:   81.90%            │
│ Precision:  67.76%             │
│ Recall:     60.70%            │
│ F1 Score:   64.03%            │
│ ROC-AUC:    87.77%            │
└───────────────────────────────┘
```

Candidate:

```text
v1 → REJECTED
```

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

* Machine Learning classification
* Customer churn prediction
* Data preprocessing
* Feature scaling
* One-Hot Encoding
* SMOTE
* CatBoost
* Cross-validation
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* FastAPI
* REST APIs
* Pydantic
* Joblib
* Model serialization
* Model versioning
* Model registries
* Production prediction logging
* Data drift monitoring
* Model performance monitoring
* Candidate model evaluation
* Autonomous retraining concepts
* Model promotion and rejection
* HTML
* CSS
* JavaScript
* Cloud deployment
* MLOps
* Production ML reliability

---

# 🔮 Future Improvements

The system can be extended with:

### 1. Real Statistical Drift Detection

Implement:

* PSI
* KS Test
* Distribution comparison
* Feature-level drift thresholds

### 2. Automated Retraining Scheduler

Trigger retraining when:

```text
Drift > Threshold
OR
F1 < Threshold
OR
ROC-AUC < Threshold
```

### 3. Database Logging

Replace:

```text
prediction_logs.csv
```

with:

```text
PostgreSQL
```

### 4. Cloud Model Storage

Store model artifacts using:

```text
AWS S3
Google Cloud Storage
Azure Blob Storage
```

### 5. Authentication

Protect model-management and retraining endpoints.

### 6. Alerting

Integrate:

```text
Email
Slack
Discord
Monitoring Platforms
```

### 7. CI/CD

Automate:

```text
Git Push
   ↓
Tests
   ↓
Model Validation
   ↓
Build
   ↓
Deployment
```

### 8. Docker

Containerize the complete application for reproducible deployment.

### 9. Experiment Tracking

Integrate an experiment/model tracking platform such as MLflow.

---

# 💡 Why This Project Matters

Traditional ML projects often end here:

```text
Train Model
     ↓
Calculate Accuracy
     ↓
Done
```

This project goes further:

```text
Train
  ↓
Evaluate
  ↓
Version
  ↓
Deploy
  ↓
Predict
  ↓
Log
  ↓
Monitor
  ↓
Evaluate Production Data
  ↓
Retrain
  ↓
Compare
  ↓
Accept / Reject
  ↓
Maintain Reliable Production Model
```

This makes the project an example of **production-oriented Machine Learning and MLOps**, rather than only a model-training project.

---

# 👨‍💻 Author

**Abdul Hanan**

### Focus Areas

```text
AI / ML Engineering
MLOps
Generative AI
Python
FastAPI
Machine Learning
WordPress Development
```

---

# ⭐ Project Conclusion

The **Production ML Model Reliability & Autonomous Retraining System** demonstrates how a Machine Learning model can be continuously managed after deployment.

The final system keeps **Model v2** active because the candidate **Model v1** failed the evaluation gate.

The key principle is:

> **A new model should never replace a production model simply because it is newly trained. It should first prove that it is reliable and better according to predefined evaluation criteria.**

---

## ⭐ If you found this project useful

Give the repository a ⭐ and feel free to explore the implementation.

```

**GitHub tip:** save this as exactly **`README.md`** in the root of your repository.
```
