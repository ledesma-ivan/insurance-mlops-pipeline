# Insurance Fraud Detection - MLOps Pipeline

End-to-end MLOps pipeline for insurance claim fraud detection. Built with **XGBoost**, **MLflow**, **FastAPI**, **Airflow**, **LangChain + FAISS**, **Kubernetes (Minikube)**, **Docker**, **Kubeflow**, and **Vertex AI**.

---

## Architecture

```
Raw Data (3 CSVs)
        |
        v
┌─────────────────────────────────────────────────────┐
│              Airflow DAG (@weekly)                   │
│                                                      │
│  ingest_data → preprocess → train_model → evaluate  │
│                                              |       │
│                                     BranchPythonOp  │
│                                       ↙         ↘   │
│                               promote_model  reject  │
└─────────────────────────────────────────────────────┘
        |
        v
MLflow Tracking + Model Registry (Production stage)
        |
        v
Serving (FastAPI REST API) --> Foundry / Consumers
        |
        v
Monitoring (PSI Drift Detection) --> triggers DAG
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | XGBoost |
| Experiment Tracking | MLflow |
| Feature Store | Parquet / PyArrow |
| API | FastAPI |
| Retraining Orchestration | Apache Airflow 2.10 |
| Explainability (GenAI) | LangChain + FAISS + Claude Haiku |
| Container Orchestration | Kubernetes (Minikube) + HPA |
| Load Testing | Locust |
| Infrastructure as Code | Terraform (Google provider) |
| Pipeline Orchestration | Kubeflow SDK + Vertex AI |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | Google Cloud Platform (Vertex AI) |
| Monitoring | PSI (Population Stability Index) |
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
# API:     http://localhost:8000/docs
# MLflow:  http://localhost:5000
```

### 5. Run Airflow (retraining pipeline)

```bash
# First time only — initialises DB and creates admin user
docker-compose run --rm airflow-init

# Start all services (MLflow + API + Airflow webserver + scheduler)
docker-compose up --build

# Airflow UI: http://localhost:8080
# Login: admin / admin
```

**Trigger a manual DAG run:**
```bash
docker-compose exec airflow-scheduler \
  airflow dags trigger insurance_fraud_retraining
```

**Change the F1 promotion threshold (default 0.60):**
```bash
docker-compose exec airflow-scheduler \
  airflow variables set f1_threshold 0.65
```

### 6. Run tests

```bash
pip install -r requirements/requirements-dev.txt
python -m pytest tests/ -v
```

### 7. Run drift detection

```bash
python -m monitoring.drift
```

### 8. Run automatic retraining

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

### Explain (RAG — GenAI)

```bash
POST http://localhost:8000/explain
Content-Type: application/json
```

Same request body as `/predict`. Returns prediction + natural language explanation.

```json
{
  "prediction": 1,
  "probability": 0.8731,
  "risk_level": "HIGH",
  "explanation": "Esta claim fue marcada como HIGH RISK por tres factores principales: (1) El ratio claim/prima es de 347x — muy por encima del umbral sospechoso de 10x, lo que indica que el asegurado reclama un monto desproporcionado a su prima. (2) El incidente ocurrió a las 3am sin reporte policial, una combinación clásica de fraude nocturno sin testigos. (3) Se reclama una pérdida mayor (Major Loss) sin lesiones reportadas, lo cual es físicamente inconsistente con un incidente grave."
}
```

> **Sin API key:** el endpoint devuelve una explicación basada en reglas automáticamente.

---

## Project Structure

```
insurance-mlops-pipeline/
├── explainability/
│   ├── domain_docs.py               # Feature descriptions + fraud pattern docs
│   ├── knowledge_base.py            # Builds/loads FAISS index from MLflow + docs
│   └── rag_explainer.py             # LangChain LCEL chain (Claude Haiku)
├── dags/
│   └── insurance_retraining_dag.py  # Airflow DAG (retraining pipeline)
├── feature_store/
│   ├── feature_engineering.py       # Feature creation + joins
│   └── store.py                     # Feature Store (Parquet)
├── training/
│   └── train.py                     # XGBoost + MLflow tracking + registry
├── monitoring/
│   ├── drift.py                     # PSI drift detection
│   └── alerts.py                    # Alerts + automatic retraining
├── serving/
│   ├── app.py                       # FastAPI REST endpoint
│   └── vertex_deploy.py             # Vertex AI endpoint deployment
├── pipeline/
│   └── pipeline.py                  # Vertex AI Pipeline (Kubeflow SDK)
├── tests/
│   ├── test_features.py             # Feature engineering tests
│   └── test_api.py                  # API endpoint tests
├── terraform/
│   ├── main.tf                      # GCS + Artifact Registry + Vertex AI + IAM
│   ├── variables.tf                 # project_id, region, environment
│   ├── outputs.tf                   # registry URL, bucket name, endpoint ID
│   ├── versions.tf                  # provider ~> 5.0, optional GCS backend
│   └── terraform.tfvars.example     # copy → terraform.tfvars (gitignored)
├── load_testing/
│   └── locustfile.py                # Locust load test (validates HPA)
├── scripts/
│   └── minikube_deploy.sh           # One-command Minikube setup
├── docker/
│   ├── Dockerfile.airflow           # Airflow image + ML deps
│   ├── Dockerfile.api
│   └── Dockerfile.mlflow
├── k8s/
│   ├── deployment.yaml              # Deployment + Service + PV/PVC
│   ├── configmap.yaml               # Env vars
│   ├── secret.yaml                  # Secret template (no real values)
│   └── hpa.yaml                     # HorizontalPodAutoscaler (1→5 pods)
├── .github/workflows/
│   └── ci.yml                       # CI/CD pipeline
├── docker-compose.yml               # MLflow + API + Airflow + Postgres
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

### 5. Kubernetes — Autoscaling with HPA

Deploy the FastAPI to Minikube and validate autoscaling with Locust load tests.

**Manifests (`k8s/`):**

| File | Resource | Purpose |
|---|---|---|
| `deployment.yaml` | Deployment + Service + PV/PVC | API pods, NodePort 30080, hostPath model volume |
| `configmap.yaml` | ConfigMap | Env vars (MODEL_PATH, MLFLOW_TRACKING_URI) |
| `secret.yaml` | Secret template | ANTHROPIC_API_KEY (never commit real values) |
| `hpa.yaml` | HorizontalPodAutoscaler | Scale 1→5 pods at 70% CPU / 80% memory |

**HPA behaviour:**
- Scale-up: +2 pods per 60s, stabilisation window 30s (reacts fast to spikes)
- Scale-down: -1 pod per 120s, stabilisation window 120s (avoids flapping)

**One-command deploy:**
```bash
bash scripts/minikube_deploy.sh
# Starts Minikube, enables metrics-server, builds image, applies all manifests
# API: http://$(minikube ip):30080/docs
```

**Watch autoscaling in action:**
```bash
# Terminal 1 — watch HPA
kubectl get hpa fraud-detection-hpa --watch

# Terminal 2 — watch pods
kubectl get pods -l app=fraud-detection-api --watch

# Terminal 3 — load test (50 users, 3 min)
locust -f load_testing/locustfile.py \
  --headless -u 50 -r 30 --run-time 3m \
  --host http://$(minikube ip):30080

# Spike test — drives scale-up fast
locust -f load_testing/locustfile.py \
  --headless -u 100 -r 100 --run-time 2m \
  --host http://$(minikube ip):30080 \
  --tags spike
```

### 6. RAG Explainability Layer (GenAI)

`POST /explain` answers **"¿por qué se marcó esta claim?"** using a RAG pipeline:

```
Claim features
      │
      ▼
FAISS retriever ──────────────────────────────────────────────────┐
  ├─ domain_docs.py: feature descriptions + fraud patterns         │
  └─ MLflow importances: feature weights from Production model     │
                                                                   ▼
                                               LangChain LCEL chain
                                               (Claude Haiku)
                                                   │
                                                   ▼
                                       Natural language explanation
```

**Knowledge base sources (indexed at startup):**
- Static domain docs: 12 documents describing each feature and fraud pattern context
- Dynamic MLflow importances: feature weights extracted from the registered Production model

**Without `ANTHROPIC_API_KEY`:** falls back to a rule-based explanation (no LLM required).

**Rebuild the FAISS index manually:**
```bash
python -m explainability.knowledge_base http://localhost:5000
```

### 6. Airflow Retraining DAG

Weekly automated retraining with promotion gate:

| Task | Operator | Description |
|---|---|---|
| `ingest_data` | PythonOperator | Validates 3 source CSVs, pushes stats to XCom |
| `preprocess` | PythonOperator | Runs feature engineering, saves Parquet to `/tmp` |
| `train_model` | PythonOperator | Trains XGBoost, logs run to MLflow, pushes `run_id` |
| `evaluate` | PythonOperator | Reads F1 from MLflow run via `run_id` |
| `check_f1_threshold` | BranchPythonOperator | Compares F1 vs `f1_threshold` Airflow Variable |
| `promote_model` | PythonOperator | Transitions model version → **Production** in MLflow Registry |
| `reject_model` | PythonOperator | Tags run as rejected, Production model unchanged |

**Default F1 threshold: `0.60`** — configurable at runtime via Airflow Variables (no redeploy needed).

### 7. Terraform — Infrastructure as Code

Provisions all GCP resources needed to run the pipeline in production. A single `terraform apply` creates everything.

**Resources (`terraform/`):**

| Resource | Type | Purpose |
|---|---|---|
| `google_storage_bucket` | GCS | MLflow artifact storage + model binaries. Versioning on, lifecycle moves old versions to Nearline after 90 days |
| `google_artifact_registry_repository` | Docker registry | Stores the fraud-detection-api Docker image |
| `google_vertex_ai_endpoint` | Vertex AI | Serving endpoint — models are deployed here via `vertex_deploy.py` |
| `google_service_account` | IAM | Pipeline SA with `storage.admin`, `artifactregistry.writer`, `aiplatform.user` |
| `google_project_service` | API enablement | Enables Storage, Artifact Registry, Vertex AI, IAM APIs |

**Workflow:**
```bash
cd terraform

# 1. Copy and fill in your vars
cp terraform.tfvars.example terraform.tfvars

# 2. Authenticate
gcloud auth application-default login

# 3. Init + plan + apply
terraform init
terraform plan
terraform apply

# 4. Use the outputs
terraform output artifact_registry_url   # → tag and push Docker image
terraform output mlflow_artifact_root    # → set as MLFLOW_ARTIFACT_ROOT
terraform output vertex_endpoint_id      # → pass to vertex_deploy.py
terraform output docker_push_commands    # → copy-paste ready push commands
```

**Remote state (teams):** uncomment the `backend "gcs"` block in `versions.tf` and create a separate bucket for tfstate before `terraform init`.

**Destroy:**
```bash
terraform destroy   # safe in dev; set mlflow_bucket_force_destroy=false in prod
```

### 8. CI/CD
- GitHub Actions runs on every push to `main`
- Trains model + runs all tests automatically

### 8. Vertex AI (Production)
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
| LangChain LCEL over legacy chains | Composable, type-safe, easier to extend |
| FAISS over Chroma/Pinecone | Zero infra, runs in-process, sufficient for this KB size |
| sentence-transformers for embeddings | Free, no API key, good quality for Spanish/English |
| Claude Haiku as explainability LLM | Fast, cheap, sufficient for structured explanation tasks |
| Fallback to rule-based explanation | Makes `/explain` usable without API key in demos |
| Airflow LocalExecutor over Celery | Single-node dev setup, no Redis overhead |
| F1 threshold via Airflow Variable | Configurable at runtime without redeploy |
| BranchPythonOperator for promote/reject | Native Airflow pattern, visible in DAG graph |
| NodePort over LoadBalancer in Minikube | LoadBalancer requires cloud provider; NodePort works locally |
| hostPath PV over cloud PVC | Zero cloud deps for local Minikube demo |
| HPA on CPU 70% threshold | Industry default; memory added as secondary guard |
| Locust over k6/JMeter | Pure Python, easy to extend with custom fraud payloads |
| Terraform over gcloud scripts | Declarative, idempotent, reviewable diffs, state tracking |
| Single flat Terraform module | Simpler than nested modules for this scope; easy to read |
| `force_destroy = false` in prod | Prevents accidental data loss on `terraform destroy` |
| GCS backend (optional) | Enables team collaboration and remote state locking |