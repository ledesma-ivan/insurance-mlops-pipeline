"""
Vertex AI Pipeline usando Kubeflow SDK.
MLOps pipeline: features, training, evaluation, deploy
"""

import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from kfp import dsl  # noqa: E402
from kfp.dsl import Dataset, Input, Metrics, Model, Output  # noqa: E402


# === STEP 1: Feature Engineering ===
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "pyarrow", "scikit-learn"],
)
def feature_engineering_step(
    raw_data_path: str,
    features_output: Output[Dataset],
):
    """Carga datos crudos, crea features y guarda en Feature Store"""
    import pandas as pd

    # Cargar datos
    insurance = pd.read_csv(f"{raw_data_path}/insurance_data.csv")
    employee = pd.read_csv(f"{raw_data_path}/employee_data.csv")
    vendor = pd.read_csv(f"{raw_data_path}/vendor_data.csv")

    df = insurance.merge(employee, on="AGENT_ID", how="left", suffixes=("", "_agent"))
    df = df.merge(vendor, on="VENDOR_ID", how="left", suffixes=("", "_vendor"))

    # Crear features
    df["TXN_DATE_TIME"] = pd.to_datetime(df["TXN_DATE_TIME"])
    df["LOSS_DT"] = pd.to_datetime(df["LOSS_DT"])
    df["REPORT_DT"] = pd.to_datetime(df["REPORT_DT"])
    df["POLICY_EFF_DT"] = pd.to_datetime(df["POLICY_EFF_DT"])

    df["claim_to_premium_ratio"] = df["CLAIM_AMOUNT"] / df["PREMIUM_AMOUNT"].replace(0, 1)
    df["is_high_claim"] = (df["CLAIM_AMOUNT"] > df["CLAIM_AMOUNT"].quantile(0.75)).astype(int)
    df["days_loss_to_report"] = (df["REPORT_DT"] - df["LOSS_DT"]).dt.days
    df["days_report_to_txn"] = (df["TXN_DATE_TIME"] - df["REPORT_DT"]).dt.days
    df["policy_age_days"] = (df["LOSS_DT"] - df["POLICY_EFF_DT"]).dt.days
    df["is_night_incident"] = df["INCIDENT_HOUR_OF_THE_DAY"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_major_loss"] = (df["INCIDENT_SEVERITY"] == "Major Loss").astype(int)
    df["no_police_report"] = (df["POLICE_REPORT_AVAILABLE"] == 0).astype(int)
    df["no_injury_high_claim"] = ((df["ANY_INJURY"] == 0) & (df["is_high_claim"] == 1)).astype(int)
    risk_map = {"L": 0, "M": 1, "H": 2}
    df["risk_encoded"] = df["RISK_SEGMENTATION"].map(risk_map).fillna(0)
    df["target"] = (df["CLAIM_STATUS"] == "D").astype(int)

    df.to_parquet(features_output.path, index=False)
    print(f"Features created: {df.shape}")


# === STEP 2: Training ===
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "xgboost", "scikit-learn", "mlflow"],
)
def training_step(
    features_input: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
):
    """Entrena XGBoost y loguea en MLflow"""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    # Cargar features
    df = pd.read_parquet(features_input.path)

    feature_cols = [
        "PREMIUM_AMOUNT",
        "CLAIM_AMOUNT",
        "AGE",
        "TENURE",
        "NO_OF_FAMILY_MEMBERS",
        "INCIDENT_HOUR_OF_THE_DAY",
        "claim_to_premium_ratio",
        "is_high_claim",
        "days_loss_to_report",
        "days_report_to_txn",
        "policy_age_days",
        "is_night_incident",
        "is_major_loss",
        "no_police_report",
        "no_injury_high_claim",
        "risk_encoded",
    ]

    X = df[feature_cols].fillna(0)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Entrenar
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluar
    y_pred = model.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
    }

    # Guardar modelo
    model.save_model(model_output.path + ".json")

    # Loguear metricas
    for name, value in metrics.items():
        metrics_output.log_metric(name, value)

    print(f"Model trained. F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")


# === STEP 3: Evaluation ===
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "xgboost"],
)
def evaluation_step(
    metrics_input: Input[Metrics],
    model_input: Input[Model],
    deploy_decision: Output[Dataset],
):
    """Decide si el modelo es suficientemente bueno para deploy"""
    import json

    # Umbral minimo para deploy
    MIN_F1 = 0.01  # Bajo porque el dataset es dificil

    f1 = metrics_input.metadata.get("f1", 0)

    decision = {
        "deploy": f1 >= MIN_F1,
        "f1": f1,
        "reason": "F1 meets threshold" if f1 >= MIN_F1 else "F1 below threshold",
    }

    with open(deploy_decision.path, "w") as f:
        json.dump(decision, f)

    print(f"Evaluation: deploy={decision['deploy']}, F1={f1:.4f}")


# === PIPELINE ===
@dsl.pipeline(
    name="insurance-fraud-pipeline",
    description="MLOps pipeline: features, training, evaluation, deploy",
)
def insurance_fraud_pipeline(
    raw_data_path: str = "gs://insurance-mlops-data/raw",
    project_id: str = "your-gcp-project-id",
    region: str = "us-central1",
):
    # Step 1: Feature Engineering
    features_task = feature_engineering_step(raw_data_path=raw_data_path)

    # Step 2: Training (depende de Step 1)
    training_task = training_step(
        features_input=features_task.outputs["features_output"],
    )

    # Step 3: Evaluation (depende de Step 2)
    evaluation_step(
        metrics_input=training_task.outputs["metrics_output"],
        model_input=training_task.outputs["model_output"],
    )


# === Compilar y ejecutar ===
if __name__ == "__main__":
    from kfp import compiler

    # Compilar pipeline a YAML (esto si se puede hacer localmente)
    compiler.Compiler().compile(
        pipeline_func=insurance_fraud_pipeline,
        package_path="pipeline/insurance_fraud_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/insurance_fraud_pipeline.yaml")
    print("To run on Vertex AI:")
    print("   from google.cloud import aiplatform")
    print("   aiplatform.PipelineJob(...).run()")
