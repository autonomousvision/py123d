import torch
from torchmetrics import Metric


class TokenCls(Metric):
    def __init__(self, max_guesses: int = 6, **kwargs) -> None:
        super(TokenCls, self).__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,  # next_token_logits: [n_agent, 16, n_token]
        pred_valid: torch.Tensor,  # next_token_idx_gt: [n_agent, 16]
        target: torch.Tensor,  # next_token_idx_gt: [n_agent, 16]
        target_valid: torch.Tensor,  # [n_agent, 16]
    ) -> None:
        target = target[..., None]
        acc = (torch.topk(pred, k=self.max_guesses, dim=-1)[1] == target).any(dim=-1)
        valid_mask = pred_valid & target_valid
        acc = acc * valid_mask
        self.sum += acc.sum()
        self.count += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
