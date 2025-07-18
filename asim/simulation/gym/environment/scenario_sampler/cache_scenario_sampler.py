from logging import getLogger
from typing import List, Optional

import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from asim.simulation.gym.cache.gym_scenario_cache import GymScenarioCache
from asim.simulation.gym.environment.scenario_sampler.abstract_scenario_sampler import AbstractScenarioSampler

logger = getLogger(__name__)


class CacheScenarioSampler(AbstractScenarioSampler):
    """
    Scenario sampler that loads scenarios from the Gym cache structure.
    NOTE: It is possible to implement a scenario sampler from nuPlan SQL database, but not included in the code.
        We tried that. It was too slow.
    """

    def __init__(
        self, log_names: List[str], cache_path: str, format: str = "gz", ignore_log_names: bool = False
    ) -> None:
        """
        Initializes the CacheScenarioSampler.
        :param log_names: Log names to include during training.
        :param cache_path: Path to the cache directory where scenarios are saved.
        :param format: Format of the scenario cache (i.e. gzip), defaults to "gz"
        """

        self._log_names = log_names
        self._scenario_cache = GymScenarioCache(cache_path, format)

        # NOTE: Additional conditions (e.g. depending on scenario type) could be added heres
        if ignore_log_names:
            logger.warning(
                "Ignoring provided log names: all scenarios from the cache will be loaded. "
                "This may lead to training on validation/test splits if the cache was not properly filtered. "
                "Ensure your cache only contains appropriate training scenarios."
            )
            self._file_paths = self._scenario_cache.file_paths
        else:
            self._file_paths = [
                file_path
                for file_path, log_name in zip(self._scenario_cache.file_paths, self._scenario_cache.log_names)
                if log_name in self._log_names
            ]

    def sample(self, seed: Optional[int] = None) -> AbstractScenario:
        """Inherited, see super class."""
        return self.sample_batch(1, seed=seed)[0]

    def sample_batch(self, batch_size: int, seed: Optional[int] = None) -> List[AbstractScenario]:
        """Inherited, see super class."""
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(self._file_paths), size=batch_size)

        scenarios: List[AbstractScenario] = []
        for idx in indices:
            file_path = self._file_paths[idx]
            scenarios.append(self._scenario_cache.load_scenario(file_path))
        return scenarios
