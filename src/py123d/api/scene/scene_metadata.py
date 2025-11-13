from dataclasses import dataclass


@dataclass(frozen=True)
class SceneMetadata:
    """Metadata for a scene extracted from a log."""

    initial_uuid: str
    """UUID of the scene, i.e., the UUID of the starting frame of the scene."""

    initial_idx: int
    """Index of the starting frame of the scene in the log."""

    duration_s: float
    """Duration of the scene in seconds."""

    history_s: float
    """History duration of the scene in seconds."""

    iteration_duration_s: float
    """Duration of each iteration in seconds."""

    @property
    def number_of_iterations(self) -> int:
        """Number of iterations in the scene."""
        return round(self.duration_s / self.iteration_duration_s)

    @property
    def number_of_history_iterations(self) -> int:
        """Number of history iterations in the scene."""
        return round(self.history_s / self.iteration_duration_s)

    @property
    def end_idx(self) -> int:
        """Index of the end frame of the scene."""
        return self.initial_idx + self.number_of_iterations
