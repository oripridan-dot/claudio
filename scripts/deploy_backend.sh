#!/bin/bash
# deploy_backend.sh
# Deploys Claudio backend to Google Cloud Run

set -e

PROJECT_ID="too-loo-zi8g7e"
REGION="me-west1"
SERVICE_NAME="claudio-backend"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "📦 Building Docker image..."
# Fix: Ensure we are in the claudio root for the docker build context
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
# Use CLOUDSDK_PYTHON to ensure compatibility
CLOUDSDK_PYTHON=/usr/bin/python3 /Users/oripridan/google-cloud-sdk/bin/gcloud builds submit --tag $IMAGE --project=$PROJECT_ID

echo "🚀 Deploying to Cloud Run..."
CLOUDSDK_PYTHON=/usr/bin/python3 /Users/oripridan/google-cloud-sdk/bin/gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --min-instances 0 \
  --max-instances 4 \
  --memory 2Gi \
  --cpu 2 \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --project=$PROJECT_ID

echo "✅ Backend deployed!"
