#!/bin/bash
# =============================================================
#  DEPLOY TO GOOGLE CLOUD RUN (FREE TIER)
# =============================================================
#
# WHAT THIS SCRIPT DOES (step by step):
#   1. Checks that gcloud CLI is installed
#   2. Sets your Google Cloud project
#   3. Enables the required Google Cloud APIs
#   4. Builds the Docker image using Cloud Build (free tier)
#   5. Pushes the image to Google Container Registry
#   6. Deploys to Cloud Run with free-tier friendly settings
#
# BEFORE RUNNING:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Run: gcloud auth login
#   3. Edit the variables below (PROJECT_ID, SERVICE_NAME, REGION)
#
# RUN WITH:
#   chmod +x deploy.sh
#   ./deploy.sh
# =============================================================

set -e  # Exit immediately if any command fails

# ── ✏️  EDIT THESE VALUES ─────────────────────────────────────────────────────
PROJECT_ID="your-gcp-project-id"      # Your Google Cloud project ID
SERVICE_NAME="gpt2-chat"               # Name for your Cloud Run service
REGION="us-central1"                   # Cloud Run region (free tier available here)
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
# ─────────────────────────────────────────────────────────────────────────────

echo "=================================================="
echo "  GPT-2 Chat — Google Cloud Run Deployment"
echo "=================================================="
echo "  Project  : ${PROJECT_ID}"
echo "  Service  : ${SERVICE_NAME}"
echo "  Region   : ${REGION}"
echo "  Image    : ${IMAGE_NAME}"
echo "=================================================="
echo ""

# ── Step 1: Verify gcloud is installed ────────────────────────────────────────
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "   Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "✅ gcloud CLI found."

# ── Step 2: Set the active project ────────────────────────────────────────────
echo ""
echo "📋 Setting project to: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

# ── Step 3: Enable required Google Cloud APIs ─────────────────────────────────
echo ""
echo "🔧 Enabling required APIs (this may take a moment)..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    --quiet

echo "✅ APIs enabled."

# ── Step 4: Build the Docker image using Cloud Build ──────────────────────────
# Cloud Build builds the image in Google's cloud — you don't need Docker
# installed locally! The free tier gives you 120 build-minutes/day.
echo ""
echo "🏗️  Building Docker image with Cloud Build..."
echo "    (This takes 5-10 minutes on first build — grab a coffee! ☕)"
gcloud builds submit \
    --tag "${IMAGE_NAME}" \
    --timeout=20m \
    .

echo "✅ Docker image built and pushed: ${IMAGE_NAME}"

# ── Step 5: Deploy to Cloud Run ───────────────────────────────────────────────
echo ""
echo "🚀 Deploying to Cloud Run..."

# Generate a random secret key for production
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")

gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    \
    # ── Free tier friendly settings ──
    --memory "2Gi" \
    --cpu "1" \
    --min-instances "0" \        # Scale to zero when idle (saves money!)
    --max-instances "3" \        # Limit instances to control cost
    --concurrency "1" \          # 1 request per instance (GPT model is big)
    --timeout "120s" \           # Allow 2 minutes for model generation
    \
    # ── Allow unauthenticated access (public website) ──
    --allow-unauthenticated \
    \
    # ── Environment variables ──
    --set-env-vars "DEBUG=False" \
    --set-env-vars "DJANGO_SECRET_KEY=${SECRET_KEY}" \
    --set-env-vars "ALLOWED_HOSTS=.run.app,localhost" \
    \
    --quiet

echo ""
echo "=================================================="
echo "  ✅ DEPLOYMENT COMPLETE!"
echo "=================================================="

# Get the deployed URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo ""
echo "  🌐 Your app is live at:"
echo "     ${SERVICE_URL}"
echo ""
echo "  📊 View logs:"
echo "     gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "  💰 Cost estimate:"
echo "     Cloud Run free tier includes:"
echo "     • 2 million requests/month"
echo "     • 360,000 GB-seconds of memory/month"
echo "     • 180,000 vCPU-seconds/month"
echo "     Scaling to zero means you pay nothing when idle!"
echo ""
echo "=================================================="