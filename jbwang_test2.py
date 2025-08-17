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


import glob
import os
import cv2

def to_video(folder_path, fps=15, downsample=2):
    imgs_path = glob.glob(os.path.join(folder_path, '*png*'))
    # imgs_path = sorted(imgs_path)[:19]
    imgs_path = sorted(imgs_path)[:700:1]
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width // downsample, height //
                               downsample), interpolation=cv2.INTER_AREA)
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)

    # media.write_video(os.path.join(folder_path, 'video.mp4'), img_array, fps=10)
    mp4_path = os.path.join("/data/jbwang/d123/video/", 'video_one_episode.mp4')
    if os.path.exists(mp4_path):
        os.remove(mp4_path)
    out = cv2.VideoWriter(
        mp4_path,
        cv2.VideoWriter_fourcc(*'DIVX'), fps, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

to_video("/nas/datasets/KITTI-360/2013_05_28_drive_0000_sync/image_00/data_rect/")

