import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pyogrio
from shapely.geometry import LineString
import shapely.geometry as geom

from d123.dataset.conversion.map.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from d123.dataset.maps.gpkg.utils import get_all_rows_with_value, get_row_with_value
from d123.dataset.maps.map_datatypes import MapLayer, RoadEdgeType, RoadLineType
from d123.geometry.polyline import Polyline3D
from d123.dataset.dataset_specific.kitti_360.kitti_360_helper import KITTI360_MAP_Bbox3D

MAX_ROAD_EDGE_LENGTH = 100.0  # meters, used to filter out very long road edges

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])

DIR_3D_BBOX = "data_3d_bboxes"

PATH_3D_BBOX_ROOT: Path = KITTI360_DATA_ROOT / DIR_3D_BBOX

KIITI360_MAP_BBOX = [
    "road",
    "sidewalk",
    # "railtrack",
    # "ground",
    # "driveway",
]

def convert_kitti360_map(log_name: str, map_path: Path) -> None:

    xml_path = PATH_3D_BBOX_ROOT / "train_full" / f"{log_name}.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs: List[KITTI360_MAP_Bbox3D] = []
    for child in root:
        label = child.find('label').text
        if child.find("transform") is None or label not in KIITI360_MAP_BBOX:                
            continue
        obj = KITTI360_MAP_Bbox3D()
        obj.parseBbox(child)
        objs.append(obj)

    dataframes: Dict[MapLayer, gpd.GeoDataFrame] = {}
    dataframes[MapLayer.LANE] = _get_none_data()
    dataframes[MapLayer.LANE_GROUP] = _get_none_data()
    dataframes[MapLayer.INTERSECTION] = _get_none_data()
    dataframes[MapLayer.CROSSWALK] = _get_none_data()
    dataframes[MapLayer.WALKWAY] = _extract_walkway_df(objs)
    dataframes[MapLayer.CARPARK] = _get_none_data()
    dataframes[MapLayer.GENERIC_DRIVABLE] = _extract_generic_drivable_df(objs)
    dataframes[MapLayer.ROAD_EDGE] = _extract_road_edge_df(objs)
    dataframes[MapLayer.ROAD_LINE] = _get_none_data()

    map_file_name = map_path
    for layer, gdf in dataframes.items():
        gdf.to_file(map_file_name, layer=layer.serialize(), driver="GPKG", mode="a")

def _get_none_data() -> gpd.GeoDataFrame:
    ids = []
    geometries = []
    data = pd.DataFrame({"id": ids})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf

def _extract_generic_drivable_df(objs: list[KITTI360_MAP_Bbox3D]) -> gpd.GeoDataFrame:
    ids: List[int] = []
    outlines: List[geom.LineString] = []
    geometries: List[geom.Polygon] = []
    for obj in objs:
        if obj.label != "road":
            continue
        ids.append(obj.id)
        outlines.append(obj.vertices.linestring)
        geometries.append(geom.Polygon(obj.vertices.array[:, :2]))
    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf

def _extract_walkway_df(objs: list[KITTI360_MAP_Bbox3D]) -> gpd.GeoDataFrame:
    ids: List[int] = []
    outlines: List[geom.LineString] = []
    geometries: List[geom.Polygon] = []
    for obj in objs:
        if obj.label != "sidewalk":
            continue
        ids.append(obj.id)
        outlines.append(obj.vertices.linestring)
        geometries.append(geom.Polygon(obj.vertices.array[:, :2]))

    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf

def _extract_road_edge_df(objs: list[KITTI360_MAP_Bbox3D]) -> gpd.GeoDataFrame:
    geometries: List[geom.Polygon] = []
    for obj in objs:
        if obj.label != "road":
            continue
        geometries.append(geom.Polygon(obj.vertices.array[:, :2]))
    road_edge_linear_rings = get_road_edge_linear_rings(geometries)
    road_edges = split_line_geometry_by_max_length(road_edge_linear_rings, MAX_ROAD_EDGE_LENGTH)

    ids = []
    road_edge_types = []
    for idx in range(len(road_edges)):
        ids.append(idx)
        # TODO @DanielDauner: Figure out if other types should/could be assigned here.
        road_edge_types.append(int(RoadEdgeType.ROAD_EDGE_BOUNDARY))

    data = pd.DataFrame({"id": ids, "road_edge_type": road_edge_types})
    return gpd.GeoDataFrame(data, geometry=road_edges)