#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-focused-sentry-442212-q8}
PROJECT_NUMBER=${PROJECT_NUMBER:-437410826517}
REGION=${REGION:-us}
AR_REPO=${AR_REPO:-gptoss-120-server}
IMAGE_NAME=${IMAGE_NAME:-gptoss-mxfp4}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD)}
IMAGE_URI=${IMAGE_URI:-"${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}"}
CLOUDBUILD_CONFIG=${CLOUDBUILD_CONFIG:-cloudbuild.yaml}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found. Install Google Cloud SDK first." >&2
  exit 1
fi

echo "Using project: ${PROJECT_ID}"

gcloud config set project "${PROJECT_ID}" >/dev/null

if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repository ${AR_REPO} in ${REGION}"
  gcloud artifacts repositories create "${AR_REPO}" \
    --location="${REGION}" \
    --repository-format=docker \
    --description="GPT-OSS container images"
fi

echo "Submitting Cloud Build: ${IMAGE_URI}:${IMAGE_TAG}"

gcloud builds submit \
  --config "${CLOUDBUILD_CONFIG}" \
  --substitutions _IMAGE_URI="${IMAGE_URI}",_IMAGE_TAG="${IMAGE_TAG}" \
  .

echo "Build submitted. Track progress in Cloud Build history or via gcloud builds log." 
