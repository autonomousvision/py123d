from typing import Optional

import torch
from torch import Tensor, tensor
from torch.nn.functional import cross_entropy
from torchmetrics.metric import Metric

from .utils import get_euclidean_targets, get_prob_targets


class CrossEntropy(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        use_gt_raw: bool,
        gt_thresh_scale_length: float,  # {"veh": 4.8, "cyc": 2.0, "ped": 1.0}
        label_smoothing: float,
        rollout_as_gt: bool,
    ) -> None:
        super().__init__()
        self.use_gt_raw = use_gt_raw
        self.gt_thresh_scale_length = gt_thresh_scale_length
        self.label_smoothing = label_smoothing
        self.rollout_as_gt = rollout_as_gt
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        # ! action that goes from [(10->15), ..., (85->90)]
        next_token_logits: Tensor,  # [n_agent, 16, n_token]
        next_token_valid: Tensor,  # [n_agent, 16]
        # ! for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
        pred_pos: Tensor,  # [n_agent, 18, 2]
        pred_head: Tensor,  # [n_agent, 18]
        pred_valid: Tensor,  # [n_agent, 18]
        # ! for step {5, 10, ..., 90}
        gt_pos_raw: Tensor,  # [n_agent, 18, 2]
        gt_head_raw: Tensor,  # [n_agent, 18]
        gt_valid_raw: Tensor,  # [n_agent, 18]
        # or use the tokenized gt
        gt_pos: Tensor,  # [n_agent, 18, 2]
        gt_head: Tensor,  # [n_agent, 18]
        gt_valid: Tensor,  # [n_agent, 18]
        # ! for tokenization
        token_agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
        # ! for filtering intersting agent for training
        train_mask: Optional[Tensor] = None,  # [n_agent]
        # ! for rollout_as_gt
        next_token_action: Optional[Tensor] = None,  # [n_agent, 16, 3]
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
        euclidean_target, euclidean_target_valid = get_euclidean_targets(
            pred_pos=pred_pos,
            pred_head=pred_head,
            pred_valid=pred_valid,
            gt_pos=gt_pos,
            gt_head=gt_head,
            gt_valid=gt_valid,
        )
        if self.rollout_as_gt and (next_token_action is not None):
            euclidean_target = next_token_action

        prob_target = get_prob_targets(
            target=euclidean_target,  # [n_agent, n_step, 3] x,y,yaw in local
            token_agent_shape=token_agent_shape,  # [n_agent, 2]
            token_traj=token_traj,  # [n_agent, n_token, 4, 2]
        )  # [n_agent, n_step, n_token] prob, last dim sum up to 1

        loss = cross_entropy(
            next_token_logits.transpose(1, 2),  # [n_agent, n_token, n_step], logits
            prob_target.transpose(1, 2),  # [n_agent, n_token, n_step], prob
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # [n_agent, n_step=16]

        # ! weighting final loss [n_agent, n_step]
        loss_weighting_mask = next_token_valid & euclidean_target_valid
        if self.training:
            loss_weighting_mask &= train_mask.unsqueeze(1)  # [n_agent, n_step]

        self.loss_sum += (loss * loss_weighting_mask).sum()
        self.count += (loss_weighting_mask > 0).sum()

    def compute(self) -> Tensor:
        return self.loss_sum / self.count
