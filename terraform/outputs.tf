output "mlflow_bucket_name" {
  description = "GCS bucket name — set as MLFLOW_ARTIFACT_ROOT in MLflow config"
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "mlflow_artifact_root" {
  description = "gs:// URI to pass as MLFLOW_ARTIFACT_ROOT"
  value       = "gs://${google_storage_bucket.mlflow_artifacts.name}"
}

output "artifact_registry_url" {
  description = "Docker registry URL — use for docker push and Kubernetes imagePullPolicy"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "vertex_endpoint_id" {
  description = "Vertex AI endpoint ID — pass to vertex_deploy.py"
  value       = google_vertex_ai_endpoint.fraud_detection.id
}

output "vertex_endpoint_name" {
  description = "Full Vertex AI endpoint resource name"
  value       = google_vertex_ai_endpoint.fraud_detection.name
}

output "service_account_email" {
  description = "MLOps service account email — use for Workload Identity / key generation"
  value       = google_service_account.mlops_sa.email
}

output "docker_push_commands" {
  description = "Commands to tag and push the API image to Artifact Registry"
  value = <<-EOT
    gcloud auth configure-docker ${var.region}-docker.pkg.dev
    docker tag fraud-detection-api:latest \
      ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/fraud-detection-api:latest
    docker push \
      ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/fraud-detection-api:latest
  EOT
}
