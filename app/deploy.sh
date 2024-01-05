#!/bin/bash
# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found."
    exit 1
fi

# Check if the repository exists
if ! gcloud artifacts repositories describe $REPO_NAME \
    --project=$PROJECT_ID \
    --location=$CLOUD_RUN_LOCATION &>/dev/null; then

    # Repository does not exist, create it
    gcloud artifacts repositories create $REPO_NAME \
        --project=$PROJECT_ID \
        --location=$CLOUD_RUN_LOCATION \
        --repository-format=docker

    echo "Repository $REPO_NAME created."
else
    echo "Repository $REPO_NAME already exists."
fi

# Build Docker image
docker build -t wikiart-app .

# Tag local image
DOCKER_IMAGE="$CLOUD_RUN_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/wikiart-app-img"
docker tag wikiart-app $DOCKER_IMAGE

# Configure Docker with the correct region
gcloud auth configure-docker "$CLOUD_RUN_LOCATION-docker.pkg.dev"

# Push the image to Artifact Registry repo
docker push $DOCKER_IMAGE 

# Deploy to Cloud Run
gcloud run deploy wikiart-app-run \
    --image $DOCKER_IMAGE \
    --memory 2Gi \
    --port 7860 \
    --allow-unauthenticated \
    --project $PROJECT_ID