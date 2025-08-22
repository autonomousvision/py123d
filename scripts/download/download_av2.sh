#!/usr/bin/env bash

# Dataset URIs
# s3://argoverse/datasets/av2/sensor/
# s3://argoverse/datasets/av2/lidar/
# s3://argoverse/datasets/av2/motion-forecasting/
# s3://argoverse/datasets/av2/tbv/

DATASET_NAMES=("sensor" "lidar" "motion-forecasting" "tbv")
TARGET_DIR="/path/to/argoverse"

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    mkdir -p "$TARGET_DIR/$DATASET_NAME"
    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" "$TARGET_DIR/$DATASET_NAME"
done


# wget -r s3://argoverse/datasets/av2/sensor/test/0f0cdd79-bc6c-35cd-9d99-7ae2fc7e165c/sensors/cameras/ring_front_center/315965893599927217.jpg
# wget http://argoverse.s3.amazonaws.com/datasets/av2/sensor/test/0f0cdd79-bc6c-35cd-9d99-7ae2fc7e165c/sensors/cameras/ring_front_center/315965893599927217.jpg
