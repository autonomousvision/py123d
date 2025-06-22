from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SMARTRolloutSampling:
    num_k: int = 1
    temp: float = 1.0
    criteria: Optional[str] = "topk_prob"  # {topk_dist_sampled_with_prob, topk_prob, topk_prob_sampled_with_dist}


@dataclass
class SMARTConfig:

    lr: float = 0.0005
    lr_warmup_steps: int = 0
    lr_total_steps: int = 100000
    lr_min_ratio: float = 0.05

    val_open_loop: bool = True
    val_closed_loop: bool = True

    # Tokenizer
    map_token_file: str = "map_traj_token5.pkl"
    agent_token_file: str = "agent_vocab_555_s2.pkl"

    map_token_sampling: SMARTRolloutSampling = field(
        default_factory=lambda: SMARTRolloutSampling(num_k=1, temp=1.0, criteria=None)
    )
    agent_token_sampling: SMARTRolloutSampling = field(
        default_factory=lambda: SMARTRolloutSampling(num_k=1, temp=1.0, criteria=None)
    )

    # Rollout Sampling
    validation_rollout_sampling: SMARTRolloutSampling = field(
        default_factory=lambda: SMARTRolloutSampling(num_k=5, temp=1.0, criteria="topk_prob")
    )
    training_rollout_sampling: SMARTRolloutSampling = field(
        default_factory=lambda: SMARTRolloutSampling(num_k=-1, temp=1.0, criteria="topk_prob")
    )

    # Decoder
    hidden_dim: int = 128
    num_freq_bands: int = 64
    num_heads: int = 8
    head_dim: int = 16
    dropout: float = 0.1
    hist_drop_prob: float = 0.1
    num_map_layers: int = 3
    num_agent_layers: int = 6
    pl2pl_radius: float = 10
    pl2a_radius: float = 30
    a2a_radius: float = 60
    time_span: Optional[int] = 30
    num_historical_steps: int = 11
    num_future_steps: int = 80

    # train loss
    use_gt_raw: bool = True
    gt_thresh_scale_length: float = -1.0  # {"veh": 4.8, "cyc": 2.0, "ped": 1.0}
    label_smoothing: float = 0.1
    rollout_as_gt: bool = False

    # else:
    n_rollout_closed_val: int = 10
    n_vis_batch: int = 2
    n_vis_scenario: int = 2
    n_vis_rollout: int = 5
