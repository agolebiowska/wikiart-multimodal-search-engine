#!/bin/bash -e

image_name=us-central1-docker.pkg.dev/test-vertex-400014/repo-wikiart/generate-embeddings-component
image_tag=latest
full_image_name=${image_name}:${image_tag}

cd "$(dirname "$0")"

docker build -t "$full_image_name" .
docker push "$full_image_name"

docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"