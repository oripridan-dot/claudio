#!/bin/bash
# deploy_backend.sh
# Deploys Claudio backend to Google Cloud Run

set -e

PROJECT_ID="your-gcp-project-id"  # To be replaced
REGION="me-west1"
SERVICE_NAME="claudio-backend"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "📦 Building Docker image..."
gcloud builds submit --tag $IMAGE

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --min-instances 0 \
  --max-instances 4

echo "✅ Backend deployed!"
