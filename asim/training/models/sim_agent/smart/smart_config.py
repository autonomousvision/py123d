from dataclasses import dataclass


@dataclass
class SmartConfig:
    """
    Configuration for the SMART agent.
    """

    num_historical_steps: int = 11
    num_future_steps: int = 80
    use_intention: bool = True
    token_size: int = 2048

    mode: str = "train"
    predictor: str = "smart"
    dataset: str = "waymo"

    input_dim: int = 2
    hidden_dim: int = 128
    output_dim: int = 2
    output_head: bool = False
    num_heads: int = 8
    head_dim: int = 16
    dropout: float = 0.1
    num_freq_bands: int = 64
    lr: float = 0.0005
    warmup_steps: int = 0
    total_steps: int = 32

    decoder_num_map_layers: int = 3
    decoder_num_agent_layers: int = 6
    decoder_a2a_radius: float = 60
    decoder_pl2pl_radius: float = 10
    decoder_pl2a_radius: float = 30
    decoder_time_span: float = 30
