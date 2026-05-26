provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  name_suffix = "${var.environment}"

  required_apis = [
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "iam.googleapis.com",
  ]
}

# ── Enable required GCP APIs ──────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each           = toset(local.required_apis)
  service            = each.value
  disable_on_destroy = false
}

# ── Service Account ───────────────────────────────────────────────────────────

resource "google_service_account" "mlops_sa" {
  account_id   = "insurance-mlops-${local.name_suffix}"
  display_name = "Insurance MLOps SA (${var.environment})"
  description  = "Used by Airflow, training jobs, and Vertex AI pipeline"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "mlops_sa_roles" {
  for_each = toset([
    "roles/storage.admin",
    "roles/artifactregistry.writer",
    "roles/aiplatform.user",
    "roles/iam.serviceAccountUser",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.mlops_sa.email}"
}

# ── GCS Bucket — MLflow artifacts + model binaries ───────────────────────────

resource "google_storage_bucket" "mlflow_artifacts" {
  name          = "${var.project_id}-mlflow-artifacts-${local.name_suffix}"
  location      = var.region
  force_destroy = var.mlflow_bucket_force_destroy

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  # Move old artifact versions to cheaper storage after 90 days
  lifecycle_rule {
    condition {
      age        = 90
      with_state = "ARCHIVED"
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Delete archived versions older than 1 year
  lifecycle_rule {
    condition {
      age        = 365
      with_state = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# Grant the SA read/write on the artifacts bucket
resource "google_storage_bucket_iam_member" "mlops_sa_bucket_access" {
  bucket = google_storage_bucket.mlflow_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlops_sa.email}"
}

# ── Artifact Registry — Docker images ────────────────────────────────────────

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "insurance-mlops-${local.name_suffix}"
  format        = "DOCKER"
  description   = "Docker images for insurance fraud detection (${var.environment})"

  depends_on = [google_project_service.apis]
}

# Grant the SA push access to the Docker repo
resource "google_artifact_registry_repository_iam_member" "mlops_sa_ar_access" {
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.mlops_sa.email}"
}

# ── Vertex AI Endpoint ────────────────────────────────────────────────────────
# The endpoint is the serving target. Models are deployed to it via the
# Vertex AI SDK (serving/vertex_deploy.py) after training and registration.

resource "google_vertex_ai_endpoint" "fraud_detection" {
  name         = "insurance-fraud-${local.name_suffix}"
  display_name = "Insurance Fraud Detection (${var.environment})"
  location     = var.region
  description  = "Serves the registered XGBoost fraud detection model"

  depends_on = [google_project_service.apis]
}
