import torch
from torchmetrics import Metric


class AverageMeter(Metric):
    def __init__(self, **kwargs) -> None:
        super(AverageMeter, self).__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, val: torch.Tensor) -> None:
        self.sum += val.sum()
        self.count += val.numel()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
