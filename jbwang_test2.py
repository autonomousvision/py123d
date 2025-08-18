# # import numpy as np
# # import pickle

# # # path = "/nas/datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000000.bin"
# # # a = np.fromfile(path, dtype=np.float32)

# # # print(a.shape)
# # # print(a[:10])

# # # path2 = "/nas/datasets/KITTI-360/calibration/calib_cam_to_pose.txt"
# # # c = np.loadtxt(path2)
# # # print(c)

# # import open3d as o3d
# # import numpy as np

# # def read_ply_file(file_path):
# #     # 读取 PLY 文件
# #     pcd = o3d.io.read_point_cloud(file_path)
# #     print(len(pcd.points), len(pcd.colors))
# #     # 提取顶点信息
# #     points = np.asarray(pcd.points)  # x, y, z
# #     colors = np.asarray(pcd.colors)  # red, green, blue
# #     # semantics = np.asarray(pcd.semantic)  # semanticID, instanceID, isVisible, confidence

# #     # 将所有信息合并到一个数组中
# #     vertices = np.hstack((points, colors))

# #     return vertices

# # # 示例用法
# # file_path = '/nas/datasets/KITTI-360/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000002_0000000385.ply'  # 替换为你的 PLY 文件路径
# # vertices = read_ply_file(file_path)

# # # 打印前几个顶点信息
# # print("顶点信息 (前5个顶点):")
# # print(vertices[:5])

# import numpy as np
# from scipy.linalg import polar
# from scipy.spatial.transform import Rotation as R

# def polar_decompose_rotation_scale(A: np.ndarray):
#     """
#     A: 3x3 (含旋转+缩放+剪切)
#     返回:
#       Rm: 纯旋转
#       Sm: 对称正定 (缩放+剪切)
#       scale: 近似轴缩放（从 Sm 特征值开方或对角提取；若存在剪切需谨慎）
#       yaw,pitch,roll: 使用 ZYX 序列 (常对应 yaw(Z), pitch(Y), roll(X))
#     """
#     Rm, Sm = polar(A)              # A = Rm @ Sm
#     # 近似各向缩放（若无剪切）:
#     scale = np.diag(Sm)
#     # 欧拉角
#     yaw, pitch, roll = R.from_matrix(Rm).as_euler('zyx', degrees=False)
#     return {
#         "R": Rm,
#         "S": Sm,
#         "scale_diag": scale,
#         "yaw_pitch_roll": (yaw, pitch, roll),
#     }

# M = np.array([
#     [-3.97771668e+00, -1.05715942e+00,-2.18206085e-02],
#     [2.43555284e+00, -1.72707462e+00, -1.03932284e-02],
#     [-4.41359095e-02, -2.94448305e-02, 1.39303744e+00],
# ])
# out = polar_decompose_rotation_scale(M)
# print(out)

# import numpy as np
# path = "/nas/datasets/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000000.bin"
# a = np.fromfile(path, dtype=np.float32)
# a = a.reshape((-1,4))
# print(a[10000:10010,:3]) 


import gc
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np
from collections import defaultdict
import datetime
import hashlib
import xml.etree.ElementTree as ET
import pyarrow as pa
from PIL import Image
import logging

from d123.common.datatypes.detection.detection_types import DetectionType
from d123.dataset.dataset_specific.kitti_360.kitti_360_helper import KITTI360Bbox3D


bbox_3d_path = Path("/nas/datasets/KITTI-360/data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml")

tree = ET.parse(bbox_3d_path)
root = tree.getroot()

KIITI360_DETECTION_NAME_DICT = {
    "truck": DetectionType.VEHICLE,
    "bus": DetectionType.VEHICLE,
    "car": DetectionType.VEHICLE,
    "motorcycle": DetectionType.BICYCLE,
    "bicycle": DetectionType.BICYCLE,
    "pedestrian": DetectionType.PEDESTRIAN,
}

for child in root:
    label = child.find('label').text
    if child.find('transform') is None or label not in KIITI360_DETECTION_NAME_DICT.keys():
        continue
    obj = KITTI360Bbox3D()
    obj.parseBbox(child)
    # print(obj.Rm)
    # print(Sigma)