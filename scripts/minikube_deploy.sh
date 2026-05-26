#!/usr/bin/env bash
# minikube_deploy.sh — full setup for local Kubernetes demo
#
# Prerequisites: minikube, kubectl, docker
# Run from the repo root: bash scripts/minikube_deploy.sh
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[k8s]${NC} $*"; }
warn() { echo -e "${YELLOW}[k8s]${NC} $*"; }

# ── 1. Start Minikube ─────────────────────────────────────────────────────────
log "Starting Minikube (4 CPUs, 4 GB RAM)..."
minikube start --cpus=4 --memory=4096 --driver=docker

log "Enabling metrics-server (required for HPA)..."
minikube addons enable metrics-server

# ── 2. Build image inside Minikube's Docker daemon ────────────────────────────
log "Pointing Docker CLI to Minikube daemon..."
eval "$(minikube docker-env)"

log "Building fraud-detection-api image..."
docker build -f docker/Dockerfile.api -t fraud-detection-api:latest .

# ── 3. Mount model files into Minikube ────────────────────────────────────────
log "Mounting ./models into Minikube at /mnt/models (background)..."
minikube mount "$(pwd)/models:/mnt/models" &
MOUNT_PID=$!
echo "$MOUNT_PID" > /tmp/minikube_mount.pid
warn "Mount PID $MOUNT_PID saved to /tmp/minikube_mount.pid — kill it when done."
sleep 3  # let mount stabilise

# ── 4. Apply manifests ────────────────────────────────────────────────────────
log "Applying Kubernetes manifests..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml

# ── 5. Wait for rollout ───────────────────────────────────────────────────────
log "Waiting for deployment rollout..."
kubectl rollout status deployment/fraud-detection-api --timeout=120s

# ── 6. Print access info ──────────────────────────────────────────────────────
NODE_IP=$(minikube ip)
log "Deployment ready!"
echo ""
echo "  API:       http://${NODE_IP}:30080"
echo "  API docs:  http://${NODE_IP}:30080/docs"
echo "  Health:    http://${NODE_IP}:30080/health"
echo ""
echo "  Watch pods:   kubectl get pods -l app=fraud-detection-api --watch"
echo "  Watch HPA:    kubectl get hpa fraud-detection-hpa --watch"
echo ""
echo "  Run load test (ramp to 50 users over 30s, 3 minutes):"
echo "    locust -f load_testing/locustfile.py \\"
echo "      --headless -u 50 -r 30 --run-time 3m \\"
echo "      --host http://${NODE_IP}:30080"
echo ""
echo "  Tear down:    minikube delete"
