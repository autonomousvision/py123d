import numpy as np

from collections import defaultdict
from typing import Dict, Any, List, Tuple
import copy
from scipy.linalg import polar

from py123d.geometry import BoundingBoxSE3, StateSE3
from py123d.geometry.polyline import Polyline3D
from py123d.geometry.rotation import EulerAngles
from py123d.conversion.datasets.kitti_360.kitti_360_labels import kittiId2label,BBOX_LABLES_TO_DETECTION_NAME_DICT

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
def local2global(semanticId: int, instanceId: int) -> int: 
    globalId = semanticId*MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int32)
    else:
        return int(globalId)
    
def global2local(globalId: int) -> Tuple[int, int]:
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int32), instanceId.astype(np.int32)
    else:
        return int(semanticId), int(instanceId)

class KITTI360Bbox3D():

    # global id(only used for sequence 0004)
    dynamic_global_id = 2000000
    static_global_id = 1000000

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
           
    def parseBbox(self, child):
        self.timestamp = int(child.find('timestamp').text)

        self.annotationId = int(child.find('index').text) + 1

        self.label = child.find('label').text

        if child.find('semanticId') is None:
            self.name = BBOX_LABLES_TO_DETECTION_NAME_DICT.get(self.label, 'unknown')
            self.is_dynamic = int(child.find('dynamic').text)
            if self.is_dynamic != 0:
                dynamicSeq = int(child.find('dynamicSeq').text)
                self.globalID = KITTI360Bbox3D.dynamic_global_id + dynamicSeq
            else:
                self.globalID = KITTI360Bbox3D.static_global_id
                KITTI360Bbox3D.static_global_id += 1
        else:
            self.start_frame = int(child.find('start_frame').text) 
            self.end_frame = int(child.find('end_frame').text)      
            
            semanticIdKITTI = int(child.find('semanticId').text)
            self.semanticId = kittiId2label[semanticIdKITTI].id
            self.instanceId = int(child.find('instanceId').text)
            self.name = kittiId2label[semanticIdKITTI].name

            self.globalID = local2global(self.semanticId, self.instanceId)

        self.valid_frames = {"global_id": self.globalID, "records": []}

        self.parseVertices(child)
        self.parse_scale_rotation()

    def parseVertices(self, child):
        transform = parseOpencvMatrix(child.find('transform'))
        R = transform[:3,:3]
        T = transform[:3,3]
        vertices = parseOpencvMatrix(child.find('vertices'))
        self.vertices_template = copy.deepcopy(vertices)
        
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        
        self.R = R
        self.T = T
    
    def parse_scale_rotation(self):
        Rm, Sm = polar(self.R) 
        if np.linalg.det(Rm) < 0:
            Rm[0] = -Rm[0]
        scale = np.diag(Sm)
        # yaw, pitch, roll = R.from_matrix(Rm).as_euler('zyx', degrees=False)
        euler_angles = EulerAngles.from_rotation_matrix(Rm)
        yaw,pitch,roll = euler_angles.yaw, euler_angles.pitch, euler_angles.roll
        obj_quaternion = euler_angles.quaternion
        # obj_quaternion = EulerAngles(roll=roll, pitch=pitch, yaw=yaw).quaternion

        self.Rm = np.array(Rm)
        self.Sm = np.array(Sm)
        self.scale = scale
        self.yaw = yaw
        self.pitch = pitch  
        self.roll = roll
        self.qw = obj_quaternion.qw
        self.qx = obj_quaternion.qx
        self.qy = obj_quaternion.qy
        self.qz = obj_quaternion.qz
        
    def get_state_array(self) -> np.ndarray:
        center = StateSE3(
            x=self.T[0],
            y=self.T[1],
            z=self.T[2],
            qw=self.qw,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
        )
        scale = self.scale
        bounding_box_se3 = BoundingBoxSE3(center, scale[0], scale[1], scale[2])

        return bounding_box_se3.array

    def filter_by_radius(self, ego_state_xyz: np.ndarray, valid_timestamp: List[int], radius: float = 50.0) -> None:
        ''' first stage of detection, used to filter out detections by radius '''
        d = np.linalg.norm(ego_state_xyz - self.T[None, :], axis=1)
        idxs = np.where(d <= radius)[0]
        for idx in idxs:
            self.valid_frames["records"].append({
                "timestamp": valid_timestamp[idx],
                "points_in_box": None,
                })

    def box_visible_in_point_cloud(self, points: np.ndarray) -> Tuple[bool, int]:
        ''' points: (N,3) , box: (8,3) '''
        box = self.vertices.copy()
        # avoid calculating ground point cloud
        z_offset = 0.1
        box[:,2] += z_offset
        O, A, B, C = box[0], box[1], box[2], box[5]
        OA = A - O
        OB = B - O
        OC = C - O
        POA, POB, POC = (points @ OA[..., None])[:, 0], (points @ OB[..., None])[:, 0], (points @ OC[..., None])[:, 0]
        mask = (np.dot(O, OA) < POA) & (POA < np.dot(A, OA)) & \
            (np.dot(O, OB) < POB) & (POB < np.dot(B, OB)) & \
            (np.dot(O, OC) < POC) & (POC < np.dot(C, OC))
        
        points_in_box = np.sum(mask)
        visible = True if points_in_box > 40 else False
        return visible, points_in_box
    
    def load_detection_preprocess(self, records_dict: Dict[int, Any]):
        if self.globalID in records_dict:
            self.valid_frames["records"] = records_dict[self.globalID]["records"]

class KITTI360_MAP_Bbox3D():
    def __init__(self):
        self.id = -1
        self.label = ' '

        self.vertices: Polyline3D = None
        self.R = None
        self.T = None
    
    def parseVertices_plane(self, child):
        transform = parseOpencvMatrix(child.find('transform'))
        R = transform[:3,:3]
        T = transform[:3,3]
        if child.find("transform_plane").find('rows').text == '0':
            vertices = parseOpencvMatrix(child.find('vertices'))
        else:
            vertices = parseOpencvMatrix(child.find('vertices_plane'))
        
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = Polyline3D.from_array(vertices)
        
        self.R = R
        self.T = T

    def parseBbox(self, child):
        self.id = int(child.find('index').text)
        self.label = child.find('label').text
        self.parseVertices_plane(child)
        

def parseOpencvMatrix(node):
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