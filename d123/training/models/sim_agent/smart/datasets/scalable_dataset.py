import logging
import pickle
from pathlib import Path
from typing import Callable, List, Optional

from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        raw_dir = Path(raw_dir)
        self._raw_paths = [p.as_posix() for p in sorted(raw_dir.glob("*"))]
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None

        logger.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")).as_posix()
        return data
