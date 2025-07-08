from typing import Optional

import torch
from torch import Tensor, tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from torchmetrics.metric import Metric

from .utils import get_euclidean_targets


class EgoNLL(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        use_gt_raw: bool,
        gt_thresh_scale_length: float,  # {"veh": 4.8, "cyc": 2.0, "ped": 1.0}
        hard_assignment: bool,
        rollout_as_gt: bool,
    ) -> None:
        super().__init__()
        self.use_gt_raw = use_gt_raw
        self.gt_thresh_scale_length = gt_thresh_scale_length
        self.hard_assignment = hard_assignment
        self.rollout_as_gt = rollout_as_gt
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        # ! action that goes from [(10->15), ..., (85->90)]
        ego_next_logits: Tensor,  # [n_batch, 16, n_k_ego_gmm]
        ego_next_poses: Tensor,  # [n_batch, 16, n_k_ego_gmm, 3]
        ego_next_valid: Tensor,  # [n_batch, 16]
        ego_next_cov: Tensor,  # [2], one for pos, one for heading.
        # ! for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
        pred_pos: Tensor,  # [n_batch, 18, 2]
        pred_head: Tensor,  # [n_batch, 18]
        pred_valid: Tensor,  # [n_batch, 18]
        # ! for step {5, 10, ..., 90}
        gt_pos_raw: Tensor,  # [n_batch, 18, 2]
        gt_head_raw: Tensor,  # [n_batch, 18]
        gt_valid_raw: Tensor,  # [n_batch, 18]
        # or use the tokenized gt
        gt_pos: Tensor,  # [n_batch, 18, 2]
        gt_head: Tensor,  # [n_batch, 18]
        gt_valid: Tensor,  # [n_batch, 18]
        token_agent_shape: Tensor,  # [n_agent, 2]
        # ! for rollout_as_gt
        next_token_action: Optional[Tensor] = None,  # [n_batch, 16, 3]
        **kwargs,
    ) -> None:
        # ! use raw or tokenized GT
        if self.use_gt_raw:
            gt_pos = gt_pos_raw
            gt_head = gt_head_raw
            gt_valid = gt_valid_raw

        # ! GT is valid if it's close to the rollout.
        if self.gt_thresh_scale_length > 0:
            dist = torch.norm(pred_pos - gt_pos, dim=-1)  # [n_agent, n_step]
            _thresh = token_agent_shape[:, 1] * self.gt_thresh_scale_length  # [n_agent]
            gt_valid = gt_valid & (dist < _thresh.unsqueeze(1))  # [n_agent, n_step]

        # ! get prob_targets
        target, target_valid = get_euclidean_targets(
            pred_pos=pred_pos,
            pred_head=pred_head,
            pred_valid=pred_valid,
            gt_pos=gt_pos,
            gt_head=gt_head,
            gt_valid=gt_valid,
        )
        if self.rollout_as_gt and (next_token_action is not None):
            target = next_token_action

        # ! transform yaw angle to unit vector
        ego_next_poses = torch.cat(
            [
                ego_next_poses[..., :2],
                ego_next_poses[..., [-1]].cos(),
                ego_next_poses[..., [-1]].sin(),
            ],
            dim=-1,
        )
        ego_next_poses = ego_next_poses.flatten(0, 1)  # [n_batch*n_step, K, 4]
        cov = ego_next_cov.repeat_interleave(2)[None, None, :].expand(*ego_next_poses.shape)  # [n_batch*n_step, K, 4]

        n_batch, n_step = target_valid.shape
        target = torch.cat(
            [target[..., :2], target[..., [-1]].cos(), target[..., [-1]].sin()], dim=-1
        )  # [n_batch, n_step, 4]
        target = target.flatten(0, 1)  # [n_batch*n_step, 4]

        ego_next_logits = ego_next_logits.flatten(0, 1)  # [n_batch*n_step, K]
        if self.hard_assignment:
            idx_hard_assign = (ego_next_poses - target.unsqueeze(1))[..., :2].norm(dim=-1).argmin(-1)
            n_batch_step = idx_hard_assign.shape[0]
            ego_next_poses = ego_next_poses[torch.arange(n_batch_step), idx_hard_assign].unsqueeze(1)
            cov = cov[torch.arange(n_batch_step), idx_hard_assign].unsqueeze(1)
            ego_next_logits = ego_next_logits[torch.arange(n_batch_step), idx_hard_assign].unsqueeze(1)

        gmm = MixtureSameFamily(
            Categorical(logits=ego_next_logits),
            Independent(Normal(ego_next_poses, cov), 1),
        )

        loss = -gmm.log_prob(target)  # [n_batch*n_step]
        loss = loss.view(n_batch, n_step)  # [n_batch, n_step]

        loss_weighting_mask = target_valid & ego_next_valid  # [n_batch, n_step]

        self.loss_sum += (loss * loss_weighting_mask).sum()
        self.count += (loss_weighting_mask > 0).sum()

    def compute(self) -> Tensor:
        return self.loss_sum / self.count
