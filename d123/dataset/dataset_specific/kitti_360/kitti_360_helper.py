import numpy as np

from collections import defaultdict

from scipy.linalg import polar
from scipy.spatial.transform import Rotation as R

from d123.common.geometry.base import StateSE3
from d123.common.geometry.bounding_box.bounding_box import BoundingBoxSE3
from d123.dataset.dataset_specific.kitti_360.labels import kittiId2label

DEFAULT_ROLL = 0.0
DEFAULT_PITCH = 0.0

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

        self.globalID = local2global(self.semanticId, self.instanceId)
        transform = self.parseOpencvMatrix(child.find('transform'))
        self.R = transform[:3,:3]
        self.T = transform[:3,3]
    
    def polar_decompose_rotation_scale(self):
        Rm, Sm = polar(self.R) 
        scale = np.diag(Sm)
        yaw, pitch, roll = R.from_matrix(Rm).as_euler('zyx', degrees=False)

        return scale, (yaw, pitch, roll)

    def get_state_array(self):
        scale, (yaw, pitch, roll) = self.polar_decompose_rotation_scale()
        center = StateSE3(
            x=self.T[0],
            y=self.T[1],
            z=self.T[2],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        bounding_box_se3 = BoundingBoxSE3(center, scale[0], scale[1], scale[2])

        return bounding_box_se3.array