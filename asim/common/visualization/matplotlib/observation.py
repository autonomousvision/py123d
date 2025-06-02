from typing import Union

import matplotlib.pyplot as plt

from asim.common.visualization.color.default import BOX_DETECTION_CONFIG
from asim.common.visualization.matplotlib.utils import add_shapely_polygon_to_ax
from asim.dataset.observation.detection.detection import BoxDetectionSE2, BoxDetectionSE3, BoxDetectionWrapper


def add_box_detections_to_ax(ax: plt.Axes, box_detections: BoxDetectionWrapper) -> None:
    for box in box_detections:
        # TODO: Optionally, continue on boxes outside of plot.
        add_box_detection_to_ax(ax, box)


def add_box_detection_to_ax(
    ax: plt.Axes,
    box_detection: Union[BoxDetectionSE2, BoxDetectionSE3],
) -> None:
    plot_config = BOX_DETECTION_CONFIG[box_detection.metadata.detection_type]
    add_shapely_polygon_to_ax(ax, box_detection.shapely_polygon, plot_config)

    if plot_config.marker_style is not None:
        assert plot_config.marker_style in ["-", "^"]

        if plot_config.marker_style == "-":
            pass
        elif plot_config.marker_style == "^":
            pass
