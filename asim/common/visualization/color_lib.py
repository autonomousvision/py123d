from dataclasses import dataclass
from typing import Tuple

from PIL import ImageColor


@dataclass
class Color:

    hex: str

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return ImageColor.getcolor(self.hex, "RGB")

    @property
    def rgba(self) -> Tuple[int, int, int]:
        return ImageColor.getcolor(self.hex, "RGBA")

    @property
    def rgb_norm(self) -> Tuple[float, float, float]:
        return tuple([c / 255 for c in self.rgb])
