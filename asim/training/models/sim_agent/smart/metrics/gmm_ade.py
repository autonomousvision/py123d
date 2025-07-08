import torch
from torch import Tensor, tensor
from torchmetrics import Metric


class GMMADE(Metric):
    def __init__(self) -> None:
        super(GMMADE, self).__init__()
        self.add_state("sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        logits: Tensor,  # [n_agent, n_step, n_k]
        pred: Tensor,  # [n_agent, n_step, n_k, 2]
        target: Tensor,  # [n_agent, n_step, 2]
        valid: Tensor,  # [n_agent, n_step]
    ) -> None:
        n_agent, n_step, _ = logits.shape
        idx_max = logits.argmax(-1)  # [n_agent, n_step]
        pred_max = pred[
            torch.arange(n_agent).unsqueeze(1),
            torch.arange(n_step).unsqueeze(0),
            idx_max,
        ]  # [n_agent, n_step, 2]

        dist = torch.norm(pred_max - target, p=2, dim=-1)  # [n_agent, n_step]
        dist = ((dist * valid).sum(-1)) / (valid.sum(-1) + 1e-6)  # [n_agent]
        self.sum += dist.sum()
        self.count += valid.any(-1).sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
