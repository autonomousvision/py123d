from dataclasses import dataclass


@dataclass(frozen=True)
class SceneMetadata:

    initial_uuid: str
    initial_idx: int
    duration_s: float
    history_s: float
    iteration_duration_s: float

    @property
    def number_of_iterations(self) -> int:
        return round(self.duration_s / self.iteration_duration_s)

    @property
    def number_of_history_iterations(self) -> int:
        return round(self.history_s / self.iteration_duration_s)

    @property
    def end_idx(self) -> int:
        return self.initial_idx + self.number_of_iterations
