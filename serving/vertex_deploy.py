"""
Deploy model to Vertex AI Endpoint.
Requires: google-cloud-aiplatform
pip install google-cloud-aiplatform
"""

from google.cloud import aiplatform


def deploy_to_vertex(
    project_id: str,
    region: str = "us-central1",
    model_path: str = "models/latest/model.json",
    display_name: str = "insurance-fraud-model",
    endpoint_display_name: str = "insurance-fraud-endpoint",
    machine_type: str = "n1-standard-2",
):
    """Deploy XGBoost model to Vertex AI Endpoint"""

    # 1. Inicializar Vertex AI
    aiplatform.init(project=project_id, location=region)
    print(f"Vertex AI initialized: project={project_id}, region={region}")

    # 2. Subir modelo a Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    )
    print(f"Model uploaded: {model.display_name} (ID: {model.resource_name})")

    # 3. Crear endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
    )
    print(f"Endpoint created: {endpoint.display_name}")

    # 4. Deploy modelo al endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=3,
        traffic_split={"0": 100},
    )
    print(f"Model deployed to endpoint: {endpoint.resource_name}")

    return endpoint


def predict_from_vertex(
    project_id: str,
    endpoint_id: str,
    region: str = "us-central1",
):
    """Hacer prediccion desde Vertex AI Endpoint"""

    aiplatform.init(project=project_id, location=region)

    endpoint = aiplatform.Endpoint(endpoint_id)

    # Ejemplo de prediccion
    instance = {
        "PREMIUM_AMOUNT": 150.0,
        "CLAIM_AMOUNT": 50000,
        "AGE": 35,
        "TENURE": 24,
        "NO_OF_FAMILY_MEMBERS": 3,
        "INCIDENT_HOUR_OF_THE_DAY": 2,
        "claim_to_premium_ratio": 333.33,
        "is_high_claim": 1,
        "days_loss_to_report": 10,
        "days_report_to_txn": 3,
        "policy_age_days": 1324,
        "is_night_incident": 1,
        "is_major_loss": 1,
        "no_police_report": 1,
        "no_injury_high_claim": 1,
        "risk_encoded": 2,
    }

    prediction = endpoint.predict(instances=[instance])
    print(f"Prediction: {prediction.predictions}")

    return prediction


if __name__ == "__main__":
    print("=" * 60)
    print("VERTEX AI DEPLOYMENT SCRIPT")
    print("=" * 60)
    print("")
    print("This script deploys the model to Vertex AI.")
    print("Requires GCP credentials and project setup.")
    print("")
    print("Usage:")
    print("  deploy_to_vertex(project_id='your-project-id')")
    print("")
    print("  predict_from_vertex(")
    print("      project_id='your-project-id',")
    print("      endpoint_id='your-endpoint-id'")
    print("  )")
