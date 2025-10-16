import os
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import shapely.geometry as geom

from d123.conversion.utils.map_utils.road_edge.road_edge_2d_utils import (
    get_road_edge_linear_rings,
    split_line_geometry_by_max_length,
)
from d123.datatypes.maps.map_datatypes import RoadEdgeType
from d123.geometry.polyline import Polyline3D
from d123.conversion.datasets.kitti_360.kitti_360_helper import KITTI360_MAP_Bbox3D
from d123.conversion.map_writer.abstract_map_writer import AbstractMapWriter
from d123.datatypes.maps.cache.cache_map_objects import (
    CacheGenericDrivable,
    CacheWalkway,
    CacheRoadEdge,
)

MAX_ROAD_EDGE_LENGTH = 100.0  # meters, used to filter out very long road edges

KITTI360_DATA_ROOT = Path(os.environ["KITTI360_DATA_ROOT"])

DIR_3D_BBOX = "data_3d_bboxes"

PATH_3D_BBOX_ROOT: Path = KITTI360_DATA_ROOT / DIR_3D_BBOX

KITTI360_MAP_BBOX = [
    "road",
    "sidewalk",
    # "railtrack",
    # "ground",
    # "driveway",
]

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
        geometries.append(geom.Polygon(obj.vertices.array[:, :3]))
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
        geometries.append(geom.Polygon(obj.vertices.array[:, :3]))

    data = pd.DataFrame({"id": ids, "outline": outlines})
    gdf = gpd.GeoDataFrame(data, geometry=geometries)
    return gdf

def _extract_road_edge_df(objs: list[KITTI360_MAP_Bbox3D]) -> gpd.GeoDataFrame:
    geometries: List[geom.Polygon] = []
    for obj in objs:
        if obj.label != "road":
            continue
        geometries.append(geom.Polygon(obj.vertices.array[:, :3]))
    road_edge_linear_rings = get_road_edge_linear_rings(geometries)
    road_edges = split_line_geometry_by_max_length(road_edge_linear_rings, MAX_ROAD_EDGE_LENGTH)

    ids = []
    road_edge_types = []
    for idx in range(len(road_edges)):
        ids.append(idx)
        road_edge_types.append(int(RoadEdgeType.ROAD_EDGE_BOUNDARY))

    data = pd.DataFrame({"id": ids, "road_edge_type": road_edge_types})
    return gpd.GeoDataFrame(data, geometry=road_edges)


def convert_kitti360_map_with_writer(log_name: str, map_writer: AbstractMapWriter) -> None:
    """
    Convert KITTI-360 map data using the provided map writer.
    This function extracts map data from KITTI-360 XML files and writes them using the map writer interface.
    
    :param log_name: The name of the log to convert
    :param map_writer: The map writer to use for writing the converted map
    """
    xml_path = PATH_3D_BBOX_ROOT / "train_full" / f"{log_name}.xml"
    if not xml_path.exists():
        xml_path = PATH_3D_BBOX_ROOT / "train" / f"{log_name}.xml"
    
    if not xml_path.exists():
        raise FileNotFoundError(f"BBox 3D file not found: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs: List[KITTI360_MAP_Bbox3D] = []
    
    for child in root:
        label = child.find('label').text
        if child.find("transform") is None or label not in KITTI360_MAP_BBOX:
            continue
        obj = KITTI360_MAP_Bbox3D()
        obj.parseBbox(child)
        objs.append(obj)
    

    generic_drivable_gdf = _extract_generic_drivable_df(objs)
    walkway_gdf = _extract_walkway_df(objs)
    road_edge_gdf = _extract_road_edge_df(objs)
    
    for idx, row in generic_drivable_gdf.iterrows():
        if not row.geometry.is_empty:
            map_writer.write_generic_drivable(
                CacheGenericDrivable(
                    object_id=idx,
                    geometry=row.geometry
                )
            )
    
    for idx, row in walkway_gdf.iterrows():
        if not row.geometry.is_empty:
            map_writer.write_walkway(
                CacheWalkway(
                    object_id=idx,
                    geometry=row.geometry
                )
            )
    
    for idx, row in road_edge_gdf.iterrows():
        if not row.geometry.is_empty:
            if hasattr(row.geometry, 'exterior'):
                road_edge_line = row.geometry.exterior
            else:
                road_edge_line = row.geometry
            
            map_writer.write_road_edge(
                CacheRoadEdge(
                    object_id=idx,
                    road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY,
                    polyline=Polyline3D.from_linestring(road_edge_line)
                )
            )