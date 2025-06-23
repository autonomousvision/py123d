import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


def _numpy_dict_to_torch(data: dict) -> dict:
    """
    Convert numpy arrays in a dictionary to torch tensors.
    :param data: Dictionary with numpy arrays.
    :return: Dictionary with torch tensors.
    """
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = torch.tensor(value)
            if data[key].dtype == torch.float64:
                data[key] = data[key].to(torch.float32)
        elif isinstance(value, dict):
            _numpy_dict_to_torch(value)


class WaymoTargetBuilderTrain(BaseTransform):
    def __init__(self, max_num: int) -> None:
        super(WaymoTargetBuilderTrain, self).__init__()
        self.step_current = 10
        self.max_num = max_num

    def __call__(self, data) -> HeteroData:
        _numpy_dict_to_torch(data)

        pos = data["agent"]["position"]
        av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        distance = torch.norm(pos - pos[av_index], dim=-1)

        # we do not believe the perception out of range of 150 meters
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

        # we do not predict vehicle too far away from ego car
        role_train_mask = data["agent"]["role"].any(-1)
        extra_train_mask = (distance[:, self.step_current] < 100) & (
            data["agent"]["valid_mask"][:, self.step_current + 1 :].sum(-1) >= 5
        )

        train_mask = extra_train_mask | role_train_mask
        if train_mask.sum() > self.max_num:  # too many vehicle
            _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
            selected_indices = _indices[torch.randperm(_indices.size(0))[: self.max_num - role_train_mask.sum()]]
            data["agent"]["train_mask"] = role_train_mask
            data["agent"]["train_mask"][selected_indices] = True
        else:
            data["agent"]["train_mask"] = train_mask  # [n_agent]

        return HeteroData(data)


class WaymoTargetBuilderVal(BaseTransform):
    def __init__(self) -> None:
        super(WaymoTargetBuilderVal, self).__init__()

    def __call__(self, data) -> HeteroData:
        _numpy_dict_to_torch(data)
        return HeteroData(data)
