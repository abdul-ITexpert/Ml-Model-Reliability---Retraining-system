# ML Reliability Frontend

This frontend is updated with the completed project results.

## Completed values used
- Active model: v2
- Production Accuracy: 0.8190
- Production Precision: 0.6776
- Production Recall: 0.6070
- Production F1: 0.6403
- Production ROC-AUC: 0.8777
- Candidate model: v1
- Candidate decision: Rejected
- Trigger: Candidate failed evaluation
- Feature monitoring display: Normal / Within baseline

## Put the folder here
```text
production-ml-reliability/
├── api/
│   └── main.py
├── models/
├── data/
├── monitoring/
├── retraining/
└── frontend/
    ├── index.html
    ├── css/style.css
    ├── js/app.js
    └── README.md
```

## Run
Terminal 1: start FastAPI using the same uvicorn command you already use.

Terminal 2:
```bash
cd frontend
python -m http.server 5500
```
Open http://127.0.0.1:5500

## Important /logs endpoint
Your main.py shown earlier does not expose `/logs`. The frontend therefore cannot read prediction_logs.csv until this endpoint exists.

Add this to main.py after `app = FastAPI(...)` (and restart Uvicorn):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_PATH):
        return []
    df = pd.read_csv(LOG_PATH)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")
```

The prediction form intentionally sends raw categorical strings. If your saved model is a preprocessing Pipeline containing OneHotEncoder, the pipeline handles those categories; the frontend should not one-hot encode them.

The `Normal` drift labels should be replaced with exact drift statistics if your Week 3 monitoring script produced numerical drift values. They are not fabricated numerical values here.
