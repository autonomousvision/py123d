from typing import Tuple

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from d123.training.models.sim_agent.smart.utils.geometry import wrap_angle
from d123.training.models.sim_agent.smart.utils.rollout import cal_polygon_contour, transform_to_local


@torch.no_grad()
def get_prob_targets(
    target: Tensor,  # [n_agent, n_step, 3] x,y,yaw in local coord
    token_agent_shape: Tensor,  # [n_agent, 2]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
) -> Tensor:  # [n_agent, n_step, n_token] prob, last dim sum up to 1
    # ! tokenize to index, then compute prob
    contour = cal_polygon_contour(
        target[..., :2],  # [n_agent, n_step, 2]
        target[..., 2],  # [n_agent, n_step]
        token_agent_shape[:, None, :],  # [n_agent, 1, 1, 2]
    )  # [n_agent, n_step, 4, 2] in local coord

    # [n_agent, n_step, 1, 4, 2] - [n_agent, 1, n_token, 4, 2]
    target_token_index = (
        torch.norm(contour.unsqueeze(2) - token_traj[:, None, :, :, :], dim=-1).sum(-1).argmin(-1)
    )  # [n_agent, n_step]

    # [n_agent, n_step, n_token] bool
    prob_target = one_hot(target_token_index, num_classes=token_traj.shape[1])
    prob_target = prob_target.to(target.dtype)
    return prob_target


@torch.no_grad()
def get_euclidean_targets(
    pred_pos: Tensor,  # [n_agent, 18, 2]
    pred_head: Tensor,  # [n_agent, 18]
    pred_valid: Tensor,  # [n_agent, 18]
    gt_pos: Tensor,  # [n_agent, 18, 2]
    gt_head: Tensor,  # [n_agent, 18]
    gt_valid: Tensor,  # [n_agent, 18]
) -> Tuple[Tensor, Tensor]:
    """
    Return: action that goes from [(10->15), ..., (85->90)]
        target: [n_agent, 16, 3], x,y,yaw
        target_valid: [n_agent, 16]
    """
    gt_last_pos = gt_pos.roll(shifts=-1, dims=1).flatten(0, 1)
    gt_last_head = gt_head.roll(shifts=-1, dims=1).flatten(0, 1)
    gt_last_valid = gt_valid.roll(shifts=-1, dims=1)  # [n_agent, 18]
    gt_last_valid[:, -1:] = False  # [n_agent, 18]

    target_pos, target_head = transform_to_local(
        pos_global=gt_last_pos.unsqueeze(1),  # [n_agent*18, 1, 2]
        head_global=gt_last_head.unsqueeze(1),  # [n_agent*18, 1]
        pos_now=pred_pos.flatten(0, 1),  # [n_agent*18, 2]
        head_now=pred_head.flatten(0, 1),  # [n_agent*18]
    )
    target_valid = pred_valid & gt_last_valid  # [n_agent, 18]

    target_pos = target_pos.squeeze(1).view(gt_pos.shape)  # n_agent, 18, 2]
    target_head = wrap_angle(target_head)  # [n_agent, 18]
    target_head = target_head.squeeze(1).view(gt_head.shape)
    target = torch.cat((target_pos, target_head.unsqueeze(-1)), dim=-1)

    # truncate [(5->10), ..., (90->5)] to [(10->15), ..., (85->90)]
    target = target[:, 1:-1]  # [n_agent, 16, 3], x,y,yaw
    target_valid = target_valid[:, 1:-1]  # [n_agent, 16]
    return target, target_valid
