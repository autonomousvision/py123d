#!/usr/bin/env bash

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/
# s3://argoverse/datasets/av2/lidar/
# s3://argoverse/datasets/av2/motion-forecasting/
# s3://argoverse/datasets/av2/tbv/

DATASET_NAMES=("sensor" "lidar" "motion_forecasting" "tbv")
TARGET_DIR="/path/to/argoverse"

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    mkdir -p "$TARGET_DIR/$DATASET_NAME"
    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" "$TARGET_DIR/$DATASET_NAME"
done
