

CACHE_PATH=/home/daniel/cache_test


python $D123_DEVKIT_ROOT/d123/script/run_preprocessing.py \
experiment_name="smart_preprocessing" \
scene_filter="nuplan_mini_train" \
scene_filter.max_num_scenes=1000 \
cache_path="${CACHE_PATH}/training"


python $D123_DEVKIT_ROOT/d123/script/run_preprocessing.py \
experiment_name="smart_preprocessing" \
scene_filter="nuplan_mini_val" \
scene_filter.max_num_scenes=1000 \
cache_path="${CACHE_PATH}/validation"
