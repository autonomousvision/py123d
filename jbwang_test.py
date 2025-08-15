# from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB

# # # 打开数据库文件
# # db = NuPlanDB(db_path="/nas/datasets/nuplan/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db")
# NUPLAN_DATA_ROOT = "/nas/datasets/nuplan/nuplan-v1.1/splits/mini"
# log_path 
# log_db = NuPlanDB(NUPLAN_DATA_ROOT, str(log_path), None)

# # 获取第1050帧数据
# frame = db.get_frame(1050)
# img_front = frame.camera_front  # 前视图像
# point_cloud = frame.lidar       # 点云

# # 获取本片段所有车辆状态
# status_data = db.get_vehicle_status()  # 返回DataFrame
# print(status_data)



# from d123.dataset.dataset_specific.nuplan.nuplan_data_converter import NuplanDataConverter, DataConverterConfig
# spits = ["nuplan_mini_train"]
# log_path = "/nas/datasets/nuplan/nuplan-v1.1/splits/mini/"
# converter = NuplanDataConverter(
#     log_path=log_path,
#     splits=spits,
#     data_converter_config=DataConverterConfig(output_path="data/jbwang/d123"),
# )
# # converter.convert_logs()
from pathlib import Path
log_paths_per_split = {
    "nuplan_mini_train": [
        "2021","2022"]
        }
log_args = [
        {
            "log_path": log_path,
            "split": split,
        }
        for split, log_paths in log_paths_per_split.items()
        for log_path in log_paths
    ]
PATH_2D_RAW_ROOT = Path("/nas/datasets/KITTI-360/data_3d_raw/")
candidates = sorted(p for p in PATH_2D_RAW_ROOT.iterdir() if p.is_dir() and p.name.endswith("_sync"))
# print(log_args)
# print(candidates)
# print(candidates[0].name)
# print(candidates[0].stem)
# print(type(candidates[0].name))
# print(type(candidates[0].stem))
# PATH_2D_RAW_ROOT_new = PATH_2D_RAW_ROOT/"123"/candidates[0].name
# print(PATH_2D_RAW_ROOT_new)



# import hashlib
# def create_token(input_data: str) -> str:
#     # TODO: Refactor this function.
#     # TODO: Add a general function to create tokens from arbitrary data.
#     if isinstance(input_data, str):
#         input_data = input_data.encode("utf-8")

#     hash_obj = hashlib.sha256(input_data)
#     return hash_obj.hexdigest()[:16]

# log_name = "1230_asd_"
# for i in range(20):
#     a = create_token(f"{log_name}_{i}")
#     print(a)ee


import numpy as np
from pathlib import Path
a =  np.loadtxt("/data/jbwang/d123/data_poses/2013_05_28_drive_0000_sync/oxts/data/0000000000.txt")
b = np.loadtxt("/nas/datasets/KITTI-360/data_poses/2013_05_28_drive_0018_sync/poses.txt")
data = b
ts = data[:, 0].astype(np.int32)
poses = np.reshape(data[:, 1:], (-1, 3, 4))
poses = np.concatenate((poses, np.tile(np.array([0, 0, 0, 1]).reshape(1,1,4),(poses.shape[0],1,1))), 1)
print(a)    
print(b.shape)
print(ts.shape)
print(poses.shape)

ccc = Path("/data/jbwang/d123/data_poses/2013_05_28_drive_0000_sync/oxts/data/")
print(len(list(ccc.glob("*.txt"))))