#!/usr/bin/env bash
set -e

IMAGE_NAME=smoking-pipeline:gpu

docker build -t $IMAGE_NAME -f docker/Dockerfile.gpu .

# kiểm tra GPU trong container
docker run --rm --gpus all \
  -v "$PWD/models:/workspace/models" \
  -v "$PWD/input:/workspace/input" \
  -v "$PWD/output:/workspace/output" \
  $IMAGE_NAME \
  python src/env_check.py

# chạy pipeline trên GPU 0
docker run --rm --gpus all \
  -v "$PWD/models:/workspace/models" \
  -v "$PWD/input:/workspace/input" \
  -v "$PWD/output:/workspace/output" \
  $IMAGE_NAME \
  python src/pipeline.py \
    --det /workspace/models/det_best.pt \
    --cls /workspace/models/cls_best.pt \
    --img /workspace/input/img1.jpg \
    --out /workspace/output \
    --device 0 \
    --smoking_action_class smoking \
    --cigarette_class cigarette \
    --cls_positive_labels "smoking,hút_thuốc" \
    --cls_threshold 0.6
