from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:

    pass

    def get_view_matrix(self) -> np.ndarray:
        # Compute the view matrix based on the camera's position and orientation
        pass
