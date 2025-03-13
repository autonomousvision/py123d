import matplotlib.pyplot as plt

from asim.common.visualization.color.lib.opendrive import OPENDRIVE_LANE_TYPES
from asim.dataset.dataset_specific.carla.opendrive.convert_opendrive_map import convert_opendrive_map
from asim.dataset.dataset_specific.carla.opendrive.elements.opendrive import OpenDRIVE


def plot_opendrive_map(opendrive: OpenDRIVE) -> None:

    fig, ax = plt.subplots(figsize=(10, 10))

    lane_groups = convert_opendrive_map(opendrive)
    for road_id, road_lane_groups in lane_groups.items():
        for lane_group in road_lane_groups:
            for lane in lane_group.lane_helper:
                polygon = lane.shapely_polygon
                exterior_x, exterior_y = polygon.exterior.xy
                ax.fill(
                    exterior_x,
                    exterior_y,
                    color=OPENDRIVE_LANE_TYPES[lane.type].hex,
                    linewidth=0.0,
                    alpha=0.9,
                )
