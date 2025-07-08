

CACHE_PATH=/home/daniel/cache_test


python $ASIM_DEVKIT_ROOT/asim/script/run_preprocessing.py \
experiment_name="smart_preprocessing" \
scene_filter="nuplan_mini_train" \
scene_filter.max_num_scenes=1000 \
cache_path="${CACHE_PATH}/training"


python $ASIM_DEVKIT_ROOT/asim/script/run_preprocessing.py \
experiment_name="smart_preprocessing" \
scene_filter="nuplan_mini_val" \
scene_filter.max_num_scenes=1000 \
cache_path="${CACHE_PATH}/validation"
