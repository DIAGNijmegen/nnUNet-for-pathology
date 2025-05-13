#!/bin/bash

VERSION="v1.00"

docker build -t joeyspronck67/nnunet_for_pathology/nnunet_for_pathology:$VERSION . && \
docker push joeyspronck67/nnunet_for_pathology/nnunet_for_pathology:$VERSION && \

docker tag joeyspronck67/nnunet_for_pathology/nnunet_for_pathology:$VERSION joeyspronck67/nnunet_for_pathology/nnunet_for_pathology:latest
docker push joeyspronck67/nnunet_for_pathology/nnunet_for_pathology:latest
