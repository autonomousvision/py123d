# NOTE: Please check the LICENSE file when downloading the nuPlan dataset
# wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE

# train: nuplan_train

# val: nuplan_val

# test: nuplan_test

# mini: nuplan_mini_train, nuplan_mini_val, nuplan_mini_test
for split in {0..8}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_camera_${split}.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_lidar_${split}.zip
done
