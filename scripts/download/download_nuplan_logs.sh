# NOTE: Please check the LICENSE file when downloading the nuPlan dataset
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE

# 1. Maps (required for nuplan and nuplan-mini)
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip

# 2. Logs
# 2.1 nuplan_train
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_boston.zip
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_pittsburgh.zip
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_singapore.zip
for split in {1..6}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_train_vegas_${split}.zip
done

# 2.2 nuplan_val
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip

# 2.3 nuplan_test
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_val.zip

# 2.4 nuplan-mini_train, nuplan-mini_val, nuplan-mini_test
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_mini.zip


# 3. Sensor blobs
# 3.1 train: nuplan_train
for split in {0..42}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_camera_${split}.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_lidar_${split}.zip
done

# 3.2 val: nuplan_val
for split in {0..11}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set/nuplan-v1.1_val_camera_${split}.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set/nuplan-v1.1_val_lidar_${split}.zip
done

# 3.3 test: nuplan_test
for split in {0..11}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/test_set/nuplan-v1.1_test_camera_${split}.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/test_set/nuplan-v1.1_test_lidar_${split}.zip
done


# 3.4 mini: nuplan_mini_train, nuplan_mini_val, nuplan_mini_test
for split in {0..8}; do
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_camera_${split}.zip
    wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_lidar_${split}.zip
done
