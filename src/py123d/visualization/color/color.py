from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import ImageColor


@dataclass(frozen=True)
class Color:

    hex: str

    @classmethod
    def from_rgb(cls, rgb: Tuple[int, int, int]) -> Color:
        r, g, b = rgb
        return cls(f"#{r:02x}{g:02x}{b:02x}")

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return ImageColor.getcolor(self.hex, "RGB")

    @property
    def rgba(self) -> Tuple[int, int, int]:
        return ImageColor.getcolor(self.hex, "RGBA")

    @property
    def rgb_norm(self) -> Tuple[float, float, float]:
        return tuple([c / 255 for c in self.rgb])

    @property
    def rgba_norm(self) -> Tuple[float, float, float]:
        return tuple([c / 255 for c in self.rgba])

    def set_brightness(self, factor: float) -> Color:
        r, g, b = self.rgb
        return Color.from_rgb(
            (
                max(min(int(r * factor), 255), 0),
                max(min(int(g * factor), 255), 0),
                max(min(int(b * factor), 255), 0),
            )
        )

    def __str__(self) -> str:
        return self.hex


BLACK: Color = Color("#000000")
WHITE: Color = Color("#FFFFFF")
LIGHT_GREY: Color = Color("#D3D3D3")
DARK_GREY: Color = Color("#9e9d9d")
DARKER_GREY: Color = Color("#787878")

ELLIS_5: Dict[int, Color] = {
    0: Color("#DE7061"),  # red
    1: Color("#B0E685"),  # green
    2: Color("#4AC4BD"),  # cyan
    3: Color("#E38C47"),  # orange
    4: Color("#699CDB"),  # blue
}

TAB_10: Dict[int, Color] = {
    0: Color("#1f77b4"),  # blue
    1: Color("#ff7f0e"),  # orange
    2: Color("#2ca02c"),  # green
    3: Color("#d62728"),  # red
    4: Color("#9467bd"),  # violet
    5: Color("#8c564b"),  # brown
    6: Color("#e377c2"),  # pink
    7: Color("#7f7f7f"),  # grey
    8: Color("#bcbd22"),  # yellow
    9: Color("#17becf"),  # cyan
}

NEW_TAB_10: Dict[int, Color] = {
    0: Color("#4e79a7"),  # blue
    1: Color("#f28e2b"),  # orange
    2: Color("#e15759"),  # red
    3: Color("#76b7b2"),  # cyan
    4: Color("#59a14f"),  # green
    5: Color("#edc948"),  # yellow
    6: Color("#b07aa1"),  # violet
    7: Color("#ff9da7"),  # pink-ish
    8: Color("#9c755f"),  # brown
    9: Color("#bab0ac"),  # grey
}
