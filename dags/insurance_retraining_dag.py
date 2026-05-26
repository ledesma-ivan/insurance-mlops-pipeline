"""
Airflow DAG: Weekly retraining pipeline for insurance fraud detection.

Flow:
    ingest_data → preprocess → train_model → evaluate → check_f1_threshold
                                                              ↙           ↘
                                                     promote_model    reject_model
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "insurance-fraud-model"
F1_THRESHOLD_DEFAULT = "0.60"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ── Task 1: Ingest ────────────────────────────────────────────────────────────

def ingest_data(**context):
    import pandas as pd

    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    files = {
        "insurance": os.path.join(data_dir, "insurance_data.csv"),
        "employee": os.path.join(data_dir, "employee_data.csv"),
        "vendor": os.path.join(data_dir, "vendor_data.csv"),
    }

    stats = {}
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file missing: {path}")
        df = pd.read_csv(path)
        stats[name] = {"rows": len(df), "cols": len(df.columns)}
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")

    context["ti"].xcom_push(key="data_stats", value=stats)
    context["ti"].xcom_push(key="data_dir", value=data_dir)
    total_rows = sum(v["rows"] for v in stats.values())
    print(f"Ingest complete: {total_rows} total rows across {len(files)} files")


# ── Task 2: Preprocess ────────────────────────────────────────────────────────

def preprocess(**context):
    import sys
    sys.path.insert(0, PROJECT_ROOT)

    from feature_store.feature_engineering import (
        create_features,
        load_raw_data,
        select_model_features,
    )

    data_dir = context["ti"].xcom_pull(key="data_dir", task_ids="ingest_data")
    df = load_raw_data(
        insurance_path=os.path.join(data_dir, "insurance_data.csv"),
        employee_path=os.path.join(data_dir, "employee_data.csv"),
        vendor_path=os.path.join(data_dir, "vendor_data.csv"),
    )
    df = create_features(df)

    feature_cols = select_model_features(df) + ["target"]
    features_path = f"/tmp/features_{context['ds_nodash']}.parquet"
    df[feature_cols].to_parquet(features_path, index=False)

    context["ti"].xcom_push(key="features_path", value=features_path)
    print(f"Features saved: {features_path} ({df.shape[0]} rows, {len(feature_cols)} cols)")


# ── Task 3: Train ─────────────────────────────────────────────────────────────

def train_model(**context):
    import sys
    sys.path.insert(0, PROJECT_ROOT)

    import mlflow
    import mlflow.xgboost
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from feature_store.feature_engineering import select_model_features

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    features_path = context["ti"].xcom_pull(key="features_path", task_ids="preprocess")
    df = pd.read_parquet(features_path)

    feature_cols = select_model_features(df)
    X = df[feature_cols].fillna(0)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": float(scale_pos_weight),
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    mlflow.set_experiment("insurance-fraud-detection")
    run_name = f"airflow-retrain-{context['ds_nodash']}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("dag_run_id", context["run_id"])

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    context["ti"].xcom_push(key="run_id", value=run_id)
    print(f"Training complete | run_id={run_id} | F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")


# ── Task 4: Evaluate ──────────────────────────────────────────────────────────

def evaluate(**context):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    run = mlflow.get_run(run_id)
    m = run.data.metrics

    f1 = m.get("f1", 0.0)
    context["ti"].xcom_push(key="f1_score", value=f1)

    print(
        f"Evaluation results | "
        f"F1={f1:.4f} | "
        f"Precision={m.get('precision', 0):.4f} | "
        f"Recall={m.get('recall', 0):.4f} | "
        f"ROC-AUC={m.get('roc_auc', 0):.4f}"
    )


# ── Task 5: Branch on F1 threshold ───────────────────────────────────────────

def check_f1_threshold(**context):
    threshold = float(Variable.get("f1_threshold", default_var=F1_THRESHOLD_DEFAULT))
    f1 = float(context["ti"].xcom_pull(key="f1_score", task_ids="evaluate"))

    print(f"F1={f1:.4f} | threshold={threshold:.4f}")

    if f1 >= threshold:
        print(f"Decision: PROMOTE (F1 {f1:.4f} >= {threshold:.4f})")
        return "promote_model"
    print(f"Decision: REJECT (F1 {f1:.4f} < {threshold:.4f})")
    return "reject_model"


# ── Task 6a: Promote ──────────────────────────────────────────────────────────

def promote_model(**context):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    f1 = float(context["ti"].xcom_pull(key="f1_score", task_ids="evaluate"))

    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    version = result.version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(MODEL_NAME, version, "promoted_by", "airflow")
    client.set_model_version_tag(MODEL_NAME, version, "f1_score", f"{f1:.4f}")

    context["ti"].xcom_push(key="promoted_version", value=version)
    print(f"Promoted {MODEL_NAME} v{version} → Production (F1={f1:.4f})")


# ── Task 6b: Reject ───────────────────────────────────────────────────────────

def reject_model(**context):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    f1 = float(context["ti"].xcom_pull(key="f1_score", task_ids="evaluate"))
    threshold = float(Variable.get("f1_threshold", default_var=F1_THRESHOLD_DEFAULT))

    client.set_tag(run_id, "promotion_status", "rejected")
    client.set_tag(run_id, "rejection_reason", f"F1={f1:.4f} below threshold={threshold:.4f}")

    print(
        f"Model rejected: F1={f1:.4f} < threshold={threshold:.4f}. "
        "Current Production model is unchanged."
    )


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="insurance_fraud_retraining",
    default_args=default_args,
    description="Weekly retraining pipeline for insurance fraud detection",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "insurance", "retraining"],
) as dag:

    t_ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_data)
    t_preprocess = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_branch = BranchPythonOperator(task_id="check_f1_threshold", python_callable=check_f1_threshold)
    t_promote = PythonOperator(task_id="promote_model", python_callable=promote_model)
    t_reject = PythonOperator(task_id="reject_model", python_callable=reject_model)

    t_ingest >> t_preprocess >> t_train >> t_evaluate >> t_branch >> [t_promote, t_reject]
