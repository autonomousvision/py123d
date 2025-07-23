import abc

from asim.common.datatypes.time.time_point import TimePoint
from asim.common.datatypes.vehicle_state.ego_state import DynamicStateSE2, EgoStateSE2


class AbstractMotionModel(abc.ABC):
    """
    Interface for generic ego motion model.
    """

    @abc.abstractmethod
    def step(
        self,
        ego_state: EgoStateSE2,
        ideal_dynamic_state: DynamicStateSE2,
        next_timepoint: TimePoint,
    ) -> EgoStateSE2:
        """
        Propagate the ego state using the ideal dynamic state and next timepoint.
        :param ego_state: The current ego state.
        :param ideal_dynamic_state: The ideal dynamic state to propagate.
        :param next_timepoint: The next timepoint for propagation.
        :return: The updated ego state after propagation.
        """
