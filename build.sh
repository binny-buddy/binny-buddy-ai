#!/bin/sh

docker buildx build \
  --platform linux/arm64,linux/amd64 \
  -t cocopam/binny-buddy-ai:latest \
  --push .