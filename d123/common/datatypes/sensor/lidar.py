from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class LiDAR:

    point_cloud: npt.NDArray[np.float32]

    @property
    def xyz(self) -> npt.NDArray[np.float32]:
        """
        Returns the point cloud as an Nx3 array of x, y, z coordinates.
        """
        return self.point_cloud[:3].T
