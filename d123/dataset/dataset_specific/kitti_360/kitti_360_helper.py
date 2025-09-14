import numpy as np

from collections import defaultdict
from typing import Dict, Optional, Any, List
from scipy.linalg import polar
from scipy.spatial.transform import Rotation as R

from d123.geometry import BoundingBoxSE3, StateSE3
from d123.dataset.dataset_specific.kitti_360.labels import kittiId2label

import os
from pathlib import Path

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])
DIR_CALIB = "calibration"
PATH_CALIB_ROOT: Path = KITTI360_DATA_ROOT / DIR_CALIB

DEFAULT_ROLL = 0.0
DEFAULT_PITCH = 0.0

kitti3602nuplan_imu_calibration_ideal = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

KITTI3602NUPLAN_IMU_CALIBRATION = kitti3602nuplan_imu_calibration_ideal

MAX_N = 1000
def local2global(semanticId, instanceId):
    globalId = semanticId*MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int32)
    else:
        return int(globalId)
    
def global2local(globalId):
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int32), instanceId.astype(np.int32)
    else:
        return int(semanticId), int(instanceId)

class KITTI360Bbox3D():
    # Constructor
    def __init__(self):

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1
        self.globalID = -1

        # the window that contains the bbox
        self.start_frame = -1
        self.end_frame = -1

        # timestamp of the bbox (-1 if statis)
        self.timestamp = -1

        # name
        self.name = '' 

        #label
        self.label = ''
           
    def parseOpencvMatrix(self, node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')
    
        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d)<1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseBbox(self, child):
        semanticIdKITTI = int(child.find('semanticId').text)
        self.semanticId = kittiId2label[semanticIdKITTI].id
        self.instanceId = int(child.find('instanceId').text)
        self.name = kittiId2label[semanticIdKITTI].name
 
        self.start_frame = int(child.find('start_frame').text) 
        self.end_frame = int(child.find('end_frame').text)

        self.timestamp = int(child.find('timestamp').text)

        self.annotationId = int(child.find('index').text) + 1

        self.label = child.find('label').text

        self.globalID = local2global(self.semanticId, self.instanceId)

        self.valid_frames = {"global_id": self.globalID, "records": []}

        self.parseVertices(child)
        self.parse_scale_rotation()

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3,:3]
        T = transform[:3,3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        
        self.R = R
        self.T = T
    
    def parse_scale_rotation(self):
        Rm, Sm = polar(self.R) 
        if np.linalg.det(Rm) < 0:
            Rm[0] = -Rm[0]
        scale = np.diag(Sm)
        yaw, pitch, roll = R.from_matrix(Rm).as_euler('zyx', degrees=False)

        self.Rm = np.array(Rm)
        self.scale = scale
        self.yaw = yaw
        self.pitch = pitch  
        self.roll = roll
        
    def get_state_array(self):
        center = StateSE3(
            x=self.T[0],
            y=self.T[1],
            z=self.T[2],
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
        )
        scale = self.scale
        bounding_box_se3 = BoundingBoxSE3(center, scale[0], scale[1], scale[2])

        return bounding_box_se3.array

    def filter_by_radius(self,ego_state_xyz,radius=50.0):
        ''' first stage of detection, used to filter out detections by radius '''
        d = np.linalg.norm(ego_state_xyz - self.T[None, :], axis=1)
        idxs = np.where(d <= radius)[0]
        for idx in idxs:
            self.valid_frames["records"].append({
                "timestamp": idx,
                "points_in_box": None,
                })

    def box_visible_in_point_cloud(self, points):
        ''' points: (N,3) , box: (8,3) '''
        box = self.vertices
        O, A, B, C = box[0], box[1], box[2], box[5]
        OA = A - O
        OB = B - O
        OC = C - O
        POA, POB, POC = (points @ OA[..., None])[:, 0], (points @ OB[..., None])[:, 0], (points @ OC[..., None])[:, 0]
        mask = (np.dot(O, OA) < POA) & (POA < np.dot(A, OA)) & \
            (np.dot(O, OB) < POB) & (POB < np.dot(B, OB)) & \
            (np.dot(O, OC) < POC) & (POC < np.dot(C, OC))
        
        points_in_box = np.sum(mask)
        visible = True if points_in_box > 50 else False
        return visible, points_in_box
    
    def load_detection_preprocess(self, records_dict: Dict[int, Any]):
        if self.globalID in records_dict:
            self.valid_frames["records"] = records_dict[self.globalID]["records"]


def get_lidar_extrinsic() -> np.ndarray:
    cam2pose_txt = PATH_CALIB_ROOT / "calib_cam_to_pose.txt"
    if not cam2pose_txt.exists():
        raise FileNotFoundError(f"calib_cam_to_pose.txt file not found: {cam2pose_txt}")
    
    cam2velo_txt = PATH_CALIB_ROOT / "calib_cam_to_velo.txt"
    if not cam2velo_txt.exists():
        raise FileNotFoundError(f"calib_cam_to_velo.txt file not found: {cam2velo_txt}")
    
    lastrow = np.array([0,0,0,1]).reshape(1,4)

    with open(cam2pose_txt, 'r') as f:
        image_00 = next(f)
        values = list(map(float, image_00.strip().split()[1:]))
        matrix = np.array(values).reshape(3, 4)
        cam2pose = np.concatenate((matrix, lastrow))
        cam2pose = KITTI3602NUPLAN_IMU_CALIBRATION @ cam2pose
    
    cam2velo = np.concatenate((np.loadtxt(cam2velo_txt).reshape(3,4), lastrow))
    extrinsic =  cam2pose @ np.linalg.inv(cam2velo)
    return extrinsic