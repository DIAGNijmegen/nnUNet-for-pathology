#!/bin/bash

VERSION="v6.00"
DOCKER_USERNAME="joeyspronck67"

docker build -t nnunet-for-pathology . && \
docker tag nnunet-for-pathology $DOCKER_USERNAME/nnunet-for-pathology:$VERSION && \
docker push $DOCKER_USERNAME/nnunet-for-pathology:$VERSION && \

docker tag $DOCKER_USERNAME/nnunet-for-pathology:$VERSION $DOCKER_USERNAME/nnunet-for-pathology:latest
docker push $DOCKER_USERNAME/nnunet-for-pathology:latest
