import dataclasses
import gzip
import json
import pickle
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional

import msgpack
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from asim.simulation.gym.cache.gym_scenario import GymScenario
from asim.simulation.gym.cache.gym_scenario_data import GymScenarioData
from asim.simulation.gym.cache.helper.convert_gym_scenario import (
    extract_to_gym_scenario_data,
)


class GymScenarioCache:
    """Helper class to save and load GymScenarioData to/from disk."""

    def __init__(
        self,
        cache_path: str,
        format: Literal["gz", "msgpack", "json"],
        compression_level: Optional[int] = None,
    ) -> None:
        """
        Initializes the GymScenarioCache object.
        :param cache_path: Path to the cache directory where scenarios will be saved/loaded.
        :param format: Format to save the scenarios, can be 'gz', 'msgpack', or 'json'.
        :param compression_level: compression level used in gzip, defaults to None
        """
        assert format in ["gz", "msgpack", "json"], f"Invalid format {format}"

        self.cache_path = Path(cache_path)
        self.format = format
        self.compression_level = compression_level

    def save(self, data: GymScenarioData, file_path: Path) -> None:
        """
        Saves the GymScenarioData to a file in the specified format.
        :param data: GymScenarioData to save.
        :param file_path: Path to the file where the data will be saved.
        """
        assert str(file_path).endswith(
            f".{self.format}"
        ), f"File {file_path} does not have the expected format {self.format}"

        data_dict = dataclasses.asdict(data)

        if self.format == "json":
            with open(file_path, "w") as f:
                json.dump(data_dict, f)
        elif self.format == "gz":
            with gzip.open(file_path, "wb", compresslevel=self.compression_level) as f:
                pickle.dump(data_dict, f)
        elif self.format == "msgpack":
            with open(file_path, "wb") as f:
                packed = msgpack.packb(data_dict, use_bin_type=True)
                f.write(packed)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def load(self, file_path: Path) -> GymScenarioData:
        """
        Loads the GymScenarioData from a file in the specified format.
        :param file_path: Path to the file from which the data will be loaded.
        :raises ValueError: If the file format is not supported.
        :return: GymScenarioData loaded from the file.
        """
        assert file_path.exists(), f"File {file_path} does not exist"
        assert str(file_path).endswith(
            f".{self.format}"
        ), f"File {file_path} does not have the expected format {self.format}"

        if self.format == "json":
            with open(file_path, "r") as f:
                data_dict = json.load(f)
        elif self.format == "gz":
            with gzip.open(file_path, "rb") as f:
                data_dict = pickle.load(f)
        elif self.format == "msgpack":
            with open(file_path, "rb") as f:
                data_dict = msgpack.unpack(f)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        return GymScenarioData.deserialize(data_dict)

    def save_scenario(self, scenario: AbstractScenario) -> None:
        """
        Saves a GymScenario to the cache.
        :param scenario: AbstractScenario to save.
        """
        data = extract_to_gym_scenario_data(scenario)
        file_path = self.get_file_path(scenario)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.save(data, file_path)

    def load_scenario(self, file_path: Path) -> AbstractScenario:
        """
        Loads a GymScenario from the cache.
        :param file_path: path of file to load.
        :return: Abstract scenario interface of nuPlan, based on pre-cached GymScenarioData.
        """
        return GymScenario(
            self.get_token(file_path),
            self.get_scenario_type(file_path),
            self.get_log_name(file_path),
            self.load(file_path),
        )

    def get_file_path(self, scenario: AbstractScenario) -> Path:
        """
        Helper function for the file path structure for a given scenario.
        :param scenario: Abstract scenario interface of nuPlan
        :return: file path, depending on format of cache object.
        """
        return self.cache_path / scenario.log_name / scenario.scenario_type / f"{scenario.token}.{self.format}"

    def get_token(self, file_path: Path) -> str:
        """
        Helper function for the token identifier of a given file path.
        :param file_path: Path of the file storing the scenario data.
        :return: token as string.
        """
        return file_path.stem

    def get_scenario_type(self, file_path: Path) -> str:
        return file_path.parent.name

    def get_log_name(self, file_path: Path) -> str:
        return file_path.parent.parent.name

    @cached_property
    def file_paths(self) -> List[Path]:
        """
        Returns all file paths in the cache directory that match the specified format.
        :return: List of file paths that match the specified format.
        """
        file_paths: List[Path] = []
        for log_path in self.cache_path.iterdir():
            for scenario_type_path in log_path.iterdir():
                for token_path in scenario_type_path.iterdir():
                    if token_path.name.endswith(f".{self.format}"):
                        file_paths.append(token_path)
        return file_paths

    @cached_property
    def tokens(self) -> List[str]:
        """
        Returns a list of tokens for all scenarios in the cache.
        :return: List of tokens
        """
        file_paths = self.file_paths
        return [self.get_token(file_path) for file_path in file_paths]

    @cached_property
    def scenario_types(self) -> List[str]:
        """
        Returns a list of scenario types for all scenarios in the cache.
        :return: List of scenario types
        """
        file_paths = self.file_paths
        return [file_path.parent.name for file_path in file_paths]

    @cached_property
    def log_names(self) -> List[str]:
        """
        Returns a list of log names for all scenarios in the cache.
        :return: List of log names
        """
        file_paths = self.file_paths
        return [self.get_log_name(file_path) for file_path in file_paths]

    def __len__(self) -> int:
        """
        Returns the number of scenarios in the cache.
        :return: Number of scenarios.
        """
        return len(self.file_paths)
