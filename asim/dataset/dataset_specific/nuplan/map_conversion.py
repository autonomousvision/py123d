import os
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import geopandas as gpd
import pandas as pd
import pyogrio

from asim.dataset.maps.gpkg.utils import get_all_rows_with_value, get_row_with_value
from asim.dataset.maps.map_datatypes import MapSurfaceType


# LANE = 0
# LANE_GROUP = 1
# INTERSECTION = 2
# CROSSWALK = 3
# WALKWAY = 4
# CARPARK = 5
# GENERIC_DRIVABLE = 6

MapSurfaceType.LANE

NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
MAP_LOCATIONS = {"sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood"}
GPKG_LAYERS: List[str] = [
    "baseline_paths",
    "carpark_areas",
    "generic_drivable_areas",
    "dubins_nodes",
    "lane_connectors",
    "intersections",
    "boundaries",
    "crosswalks",
    "lanes_polygons",
    "lane_group_connectors",
    "lane_groups_polygons",
    "road_segments",
    "stop_polygons",
    "traffic_lights",
    "walkways",
    "gen_lane_connectors_scaled_width_polygons",
]


class NuPlanMapConverter:
    # TODO: create general map interface similar to carla

    def __init__(self):

        self._gdf: Optional[Dict[str, gpd.GeoDataFrame]] = None
        pass

    def convert(self, map_name: str = "us-pa-pittsburgh-hazelwood") -> None:
        assert map_name in MAP_LOCATIONS, f"Map name {map_name} is not supported."

        # find map.pkg path
        # TODO: make this more general
        map_file_path = Path(NUPLAN_MAPS_ROOT) / "us-pa-pittsburgh-hazelwood" / "9.17.1937" / "map.gpkg"

        self._load_dataframes(map_file_path)

        lane_df = self._extract_lane_dataframe()
        map_file_name = f"{map_name}.gpkg"
        lane_df.to_file(map_file_name, layer=MapSurfaceType.LANE.serialize(), driver="GPKG")
        # walkways_df.to_file(map_file_name, layer=MapSurfaceType.WALKWAY.serialize(), driver="GPKG", mode="a")
        # carpark_df.to_file(map_file_name, layer=MapSurfaceType.CARPARK.serialize(), driver="GPKG", mode="a")

        # Extract lane dataframe

        # extract lane group dataframe

        # extract

    def _load_dataframes(self, map_file_path: Path) -> None:

        # The projected coordinate system depends on which UTM zone the mapped location is in.
        map_meta = gpd.read_file(map_file_path, layer="meta", engine="pyogrio")
        projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]

        self._gdf = {}
        for layer_name in GPKG_LAYERS:
            with warnings.catch_warnings():
                # Suppress the warnings from the GPKG operations below so that they don't spam the training logs.
                warnings.filterwarnings("ignore")

                gdf_in_pixel_coords = pyogrio.read_dataframe(map_file_path, layer=layer_name, fid_as_index=True)
                # gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
                gdf_in_utm_coords = gdf_in_pixel_coords

                # For backwards compatibility, cast the index to string datatype.
                #   and mirror it to the "fid" column.
                gdf_in_utm_coords.index = gdf_in_utm_coords.index.map(str)
                gdf_in_utm_coords["fid"] = gdf_in_utm_coords.index

            self._gdf[layer_name] = gdf_in_utm_coords

    def _extract_lane_dataframe(self) -> gpd.GeoDataFrame:
        assert self._gdf is not None, "Call `.initialize()` before retrieving data!"
        return self._extract_nuplan_lane_dataframe()

    def _extract_nuplan_lane_dataframe(self) -> gpd.GeoDataFrame:
        assert self._gdf is not None, "Call `.initialize()` before retrieving data!"

        lane_ids = self._gdf["lanes_polygons"].lane_fid.to_list()
        lane_group_ids = self._gdf["lanes_polygons"].lane_group_fid.to_list()
        speed_limits_mps = self._gdf["lanes_polygons"].speed_limit_mps.to_list()
        predecessor_ids = []
        successor_ids = []
        left_boundaries = []
        right_boundaries = []
        baseline_paths = []
        geometries = self._gdf["lanes_polygons"].geometry.to_list()

        for lane_id in lane_ids:

            # 1. predecessor_ids, successor_ids
            _predecessor_ids = get_all_rows_with_value(
                self._gdf["lane_connectors"],
                "entry_lane_fid",
                lane_id,
            )["fid"].tolist()
            _successor_ids = get_all_rows_with_value(
                self._gdf["lane_connectors"],
                "exit_lane_fid",
                lane_id,
            )["fid"].tolist()
            predecessor_ids.append(_predecessor_ids)
            successor_ids.append(_successor_ids)

            # 2. left_boundaries, right_boundaries
            lane_series = get_row_with_value(self._gdf["lanes_polygons"], "fid", str(lane_id))
            left_boundary_fid = lane_series["left_boundary_fid"]
            left_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(left_boundary_fid))["geometry"]
            left_boundaries.append(left_boundary)

            right_boundary_fid = lane_series["right_boundary_fid"]
            right_boundary = get_row_with_value(self._gdf["boundaries"], "fid", str(right_boundary_fid))["geometry"]
            right_boundaries.append(right_boundary)

            # 3. baseline_paths
            baseline_path = get_row_with_value(self._gdf["baseline_paths"], "lane_fid", float(lane_id))["geometry"]
            baseline_paths.append(baseline_path)

        data = pd.DataFrame(
            {
                "id": lane_ids,
                "lane_group_id": lane_group_ids,
                "predecessor_ids": predecessor_ids,
                "successor_ids": successor_ids,
                "left_boundary": left_boundaries,
                "right_boundary": right_boundaries,
                "baseline_path": baseline_paths,
            }
        )

        gdf = gpd.GeoDataFrame(data, geometry=geometries)
        return gdf

    def _extract_nuplan_lane_connector_dataframe(self) -> None:
        pass
