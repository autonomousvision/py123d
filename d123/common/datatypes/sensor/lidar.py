from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class LiDAR:

    point_cloud: npt.NDArray[np.float32]
