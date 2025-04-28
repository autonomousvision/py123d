from dataclasses import dataclass
from typing import Optional

import shapely

from asim.dataset.dataset_specific.carla.opendrive.elements.objects import Object
from asim.dataset.dataset_specific.carla.opendrive.elements.reference import Border


@dataclass
class OpenDriveObjectHelper:

    object_id: int

    @property
    def shapely_polygon(self) -> shapely.Polygon:
        raise NotImplementedError


def get_object_helper(object: Object, reference_border: Border) -> OpenDriveObjectHelper:

    object_helper: Optional[OpenDriveObjectHelper] = None

    if len(object.outline) < 1:
        raise NotImplementedError

    else:
        pass

    return object_helper
