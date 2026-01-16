#!/bin/bash

# Set the version here
VERSION="v6.00"

echo ""
echo ""
echo "> Building Docker image with tag: ${VERSION}"
echo "> Abort if you forget to update the version"

echo ""
echo ""
echo "> Logging in to dockerdex.umcn.nl:5005"
echo "> For dockerdex instructions please checkout: https://diagnijmegen.github.io/deepops-sol-config/gitlab-docker-registry-instructions/#4-tag-your-docker-image" 
docker logout dockerdex.umcn.nl:5005 # reset previously stored credentials
docker login dockerdex.umcn.nl:5005

# Default values
IMAGE_TAG="nnunet-for-pathology"
REGISTRY_URL="dockerdex.umcn.nl:5005"
REPO_NAME="nnunet-for-pathology"
DOCKER_USER=$(grep '"auth"' ~/.docker/config.json | cut -d'"' -f4 | base64 --decode | cut -d':' -f1 | tr '[:upper:]' '[:lower:]')

# Tag the existing Docker image with the version
VERSION_TAG=${REGISTRY_URL}/${DOCKER_USER}/${REPO_NAME}:${VERSION}
docker tag ${IMAGE_TAG} ${VERSION_TAG}
# Push the versioned image
docker push ${VERSION_TAG}

# Tag the image as 'latest'
LATEST_TAG=${REGISTRY_URL}/${DOCKER_USER}/${REPO_NAME}:latest
docker tag ${VERSION_TAG} ${LATEST_TAG}
# Push the 'latest' tag
docker push ${LATEST_TAG}

docker logout dockerdex.umcn.nl:5005 # reset stored credentials