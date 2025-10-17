from dataclasses import dataclass, field
from typing import Optional

from py123d.common.visualization.color.color import BLACK, Color


@dataclass
class PlotConfig:
    fill_color: Color = field(default_factory=lambda: BLACK)
    fill_color_alpha: float = 1.0
    line_color: Color = BLACK
    line_color_alpha: float = 1.0
    line_width: float = 1.0
    line_style: str = "-"
    marker_style: str = "o"
    marker_size: float = 1.0
    marker_edge_color: Color = field(default_factory=lambda: BLACK)
    zorder: int = 0

    smoothing_radius: Optional[float] = None
