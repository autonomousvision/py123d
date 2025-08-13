s3_uri = "/data/jbwang/d123/data/nuplan_mini_train/2021.10.11.07.12.18_veh-50_00211_00304.arrow"
# s3_uri = "/data/jbwang/d123/data/nuplan_private_test/2021.09.22.13.20.34_veh-28_01446_01583.arrow"
# s3_uri = "/data/jbwang/d123/data/carla/_Rep0_routes_validation1_route0_07_23_14_33_15.arrow"
# s3_uri = "/data/jbwang/d123/data/nuplan_mini_val/2021.06.07.12.54.00_veh-35_01843_02314.arrow"

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.dataset as ds

import os

s3_fs = fs.S3FileSystem()
from d123.common.utils.timer import Timer


timer = Timer()
timer.start()

dataset = ds.dataset(f"{s3_uri}", format="ipc")
timer.log("1. Dataset loaded")

# Get all column names and remove the ones you want to drop
all_columns = dataset.schema.names
# print("all_columns", all_columns)
# print("Schema:")
# print(dataset.schema)
# columns_to_keep = [col for col in all_columns if col not in ["front_cam_demo", "front_cam_transform"]]
timer.log("2. Columns filtered")

table = dataset.to_table(columns=all_columns)
# print("table",table)
# print(table["token"])
for col in table.column_names:
    if col == "lidar":
        continue
    print(f"Column: {col}, Type: {table.schema.field(col).type}")
    tokens = table[col]    # 或 table.column("token")
    # print(len(tokens))
    print(tokens.slice(0, 4).to_pylist())
# print(table["traffic_light_ids"])
timer.log("3. Table created")
# Save locally
# with pa.ipc.new_file("filtered_file.arrow", table.schema) as writer:
#     writer.write_table(table)
timer.log("4. Table saved locally")

timer.end()
timer.stats(verbose=False)

# 查看nuplan数据库的表结构和内容

# from pathlib import Path
# from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
# from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
# from sqlalchemy import inspect, select
# from sqlalchemy.orm import Session
# from sqlalchemy import func
# from nuplan.database.nuplan_db_orm.ego_pose import EgoPose

# NUPLAN_DATA_ROOT = Path("/nas/datasets/nuplan/")  # 按你实际路径
# log_path = "/nas/datasets/nuplan/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"

# db = NuPlanDB(NUPLAN_DATA_ROOT, log_path, None)
# # print(db.log)
# print(db.log.map_version)
# # print("log.cameras",db.log.cameras)
# # print("Log name:", db.log_name)
# # print("lidar",db.lidar_pc)
# # print("scenario_tags", db.scenario_tag)
# # print(db.log._session.query(EgoPose).order_by(func.abs(EgoPose.timestamp)).first())

# # persp = Path("/nas/datasets/KITTI-360/calibration/perspective.txt")
# # with open(persp, "r") as f:
# #     lines = [ln.strip() for ln in f if ln.strip()]
# #     print(lines)

# from d123.dataset.dataset_specific.kitti_360.kitti_360_data_converter import get_kitti360_camera_metadata

# print(get_kitti360_camera_metadata())



# from d123.dataset.dataset_specific.kitti_360.kitti_360_data_converter import _read_timestamps
# result = _read_timestamps("2013_05_28_drive_0000_sync")
# print(len(result))
# print([result[0].time_us])