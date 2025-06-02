from dataclasses import dataclass
from typing import Optional

from asim.common.visualization.color.color import BLACK, Color


@dataclass
class PlotConfig:
    fill_color: Color
    fill_color_alpha: float = 1.0
    line_color: Color = BLACK
    line_color_alpha: float = 0.0
    line_width: float = 1.0
    line_style: str = "-"
    marker_style: Optional[str] = None
    marker_size: Optional[float] = 5.0
    marker_edge_color: Optional[Color] = BLACK
    zorder: int = 0
