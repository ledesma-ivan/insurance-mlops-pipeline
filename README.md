# Insurance Fraud Detection - MLOps Pipeline

End-to-end MLOps pipeline for insurance claim fraud detection. Built with **XGBoost**, **MLflow**, **FastAPI**, **Docker**, **Kubeflow**, and **Vertex AI**.

---

## Architecture

```
Raw Data (3 CSVs)
        |
        v
Feature Engineering --> Feature Store (Parquet)
        |
        v
Training (XGBoost) --> MLflow Tracking + Model Registry
        |
        v
Evaluation (Precision, Recall, F1, ROC-AUC)
        |
        v
Serving (FastAPI REST API) --> Foundry / Consumers
        |
        v
Monitoring (PSI Drift Detection) --> Automatic Retraining
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | XGBoost |
| Experiment Tracking | MLflow |
| Feature Store | Parquet / PyArrow |
| API | FastAPI |
| Pipeline Orchestration | Kubeflow SDK + Vertex AI |
| Containerization | Docker + Docker Compose |
| Local Orchestration | Docker Compose |
| CI/CD | GitHub Actions |
| Model Monitoring | PSI (Population Stability Index) Custom Scripts |
| Language | Python 3.11 |

---

## Quick Start

### Prerequisites
- Python 3.9+
- Docker Desktop
- Git

### 1. Clone and setup

```bash
git clone https://github.com/YOUR_USER/insurance-mlops-pipeline.git
cd insurance-mlops-pipeline
python -m venv .venv
.venv\Scripts\Activate  # Windows
pip install -r requirements/requirements.txt
pip install -e .
```

### 2. Train model

```bash
python -m training.train
```

### 3. Run API locally

```bash
python -m serving.app
# Open http://localhost:8000/docs
```

### 4. Run with Docker

```bash
docker-compose up --build
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

### 5. Run tests

```bash
pip install -r requirements/requirements-dev.txt
python -m pytest tests/ -v
```

### 6. Run drift detection

```bash
python -m monitoring.drift
```

### 7. Run automatic retraining

```bash
python -m monitoring.alerts
```

---

## API Usage

### Health Check

```bash
GET http://localhost:8000/health
```

### Predict

```bash
POST http://localhost:8000/predict
Content-Type: application/json
```

```json
{
  "premium_amount": 150.0,
  "claim_amount": 50000,
  "age": 35,
  "tenure": 24,
  "no_of_family_members": 3,
  "incident_hour_of_the_day": 2,
  "incident_severity": "Major Loss",
  "any_injury": 0,
  "police_report_available": 0,
  "risk_segmentation": "H",
  "loss_dt": "2024-01-15",
  "report_dt": "2024-01-25",
  "txn_date_time": "2024-01-28",
  "policy_eff_dt": "2020-06-01"
}
```

### Response

```json
{
  "prediction": 0,
  "probability": 0.0044,
  "risk_level": "LOW"
}
```

---

## Project Structure

```
insurance-mlops-pipeline/
├── feature_store/
│   ├── feature_engineering.py   # Feature creation + joins
│   └── store.py                 # Feature Store (Parquet)
├── training/
│   └── train.py                 # XGBoost + MLflow tracking + registry
├── monitoring/
│   ├── drift.py                 # PSI drift detection
│   └── alerts.py                # Alerts + automatic retraining
├── serving/
│   ├── app.py                   # FastAPI REST endpoint
│   └── vertex_deploy.py         # Vertex AI endpoint deployment
├── pipeline/
│   └── pipeline.py              # Vertex AI Pipeline (Kubeflow SDK)
├── tests/
│   ├── test_features.py         # Feature engineering tests
│   └── test_api.py              # API endpoint tests
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.mlflow
├── k8s/
│   └── deployment.yaml          # Kubernetes manifests
├── .github/workflows/
│   └── ci.yml                   # CI/CD pipeline
├── docker-compose.yml
├── pyproject.toml
└── requirements/
    ├── requirements.txt
    └── requirements-dev.txt
```

---

## MLOps Lifecycle

### 1. Feature Engineering
- Joins 3 source tables (`insurance`, `employee`, `vendor`)
- Creates 10 domain-specific features
- Stores processed features in Feature Store (Parquet)

### 2. Training
- XGBoost with `scale_pos_weight` for imbalanced data (95/5)
- MLflow experiment tracking (parameters, metrics, artifacts)
- Model versioning via MLflow Model Registry

### 3. Monitoring

Data drift via **PSI (Population Stability Index)** per feature:

| PSI Range | Status |
|---|---|
| PSI < 0.1 | No drift |
| PSI 0.1 – 0.25 | Warning |
| PSI > 0.25 | Drift detected |

Model Decay: automatic retraining triggered on drift.

### 4. Serving
- FastAPI REST API with input validation (Pydantic)
- Returns prediction + probability + risk level
- Containerized with Docker

### 5. CI/CD
- GitHub Actions runs on every push to `main`
- Trains model + runs all tests automatically

### 6. Vertex AI (Production)
- Pipeline compiled with Kubeflow SDK
- Deployable to Vertex AI Endpoints
- Feature Store integration ready

---

## Foundry Integration

The API exposes the model as a REST endpoint consumable by **Palantir Foundry**:

- **Input:** Raw claim data (JSON)
- **Output:** Prediction + probability + risk level

**Modeling Objectives — configurable thresholds per risk level:**

| Risk Level | Probability | Action |
|---|---|---|
| HIGH | > 0.75 | Auto-block claim |
| MEDIUM | 0.5 – 0.75 | Send to investigator |
| LOW | < 0.5 | Auto-approve |

---

## Dataset

- **Source:** Insurance Claims Fraud Data (Kaggle)
- **Size:** 10,000 claims
- **Tables:** `insurance_data`, `employee_data`, `vendor_data`
- **Target:** `CLAIM_STATUS` (A = Approved 95%, D = Denied 5%)
- **Challenge:** Highly imbalanced dataset

---
## Model Performance Note

The model shows low metrics (F1=0.02, ROC-AUC=0.49) due to the nature of the dataset: 
no feature has correlation above 0.03 with the target variable. The CLAIM_STATUS 
appears to be near-randomly assigned in this public dataset.

In a production scenario, this would be addressed by:
- Working with domain experts to engineer better features
- Incorporating additional data sources (historical fraud patterns, external databases)
- Using more advanced techniques (anomaly detection, graph-based fraud detection)

**The focus of this project is the MLOps infrastructure**, not model optimization. 
The complete pipeline (training, monitoring, serving, CI/CD) works regardless of 
model performance.


---

## Key Design Decisions

| Decision | Reason |
|---|---|
| XGBoost over Neural Networks | Tabular data, small dataset, fast training |
| `scale_pos_weight` over SMOTE | Simpler, no synthetic data, same effect |
| Parquet over CSV for Feature Store | Columnar, compressed, faster reads |
| FastAPI over Flask | Async, auto-docs, Pydantic validation |
| PSI for drift detection | Industry standard, interpretable thresholds |
| Separate containers | Independent scaling, fault isolation |
