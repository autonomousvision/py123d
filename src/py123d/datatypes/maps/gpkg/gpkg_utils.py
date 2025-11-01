from typing import List

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import trimesh
from shapely import wkt

from py123d.geometry.polyline import Polyline3D


def load_gdf_with_geometry_columns(gdf: gpd.GeoDataFrame, geometry_column_names: List[str] = []):
    # TODO: refactor
    # Convert string geometry columns back to shapely objects
    for col in geometry_column_names:
        if col in gdf.columns and len(gdf) > 0 and isinstance(gdf[col].iloc[0], str):
            try:
                gdf[col] = gdf[col].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to geometry: {str(e)}")


def get_all_rows_with_value(
    elements: gpd.geodataframe.GeoDataFrame, column_label: str, desired_value: str
) -> gpd.geodataframe.GeoDataFrame:
    """
    Extract all matching elements. Note, if no matching desired_key is found and empty list is returned.
    :param elements: data frame from MapsDb.
    :param column_label: key to extract from a column.
    :param desired_value: key which is compared with the values of column_label entry.
    :return: a subset of the original GeoDataFrame containing the matching key.
    """
    # Note: in nurec one referenced column can not be converted to int
    return elements.iloc[np.where(elements[column_label].to_numpy() == desired_value)]


def get_row_with_value(elements: gpd.geodataframe.GeoDataFrame, column_label: str, desired_value: str) -> gpd.GeoSeries:
    """
    Extract a matching element.
    :param elements: data frame from MapsDb.
    :param column_label: key to extract from a column.
    :param desired_value: key which is compared with the values of column_label entry.
    :return row from GeoDataFrame.
    """
    if column_label == "fid":
        return elements.loc[desired_value]

    matching_rows = get_all_rows_with_value(elements, column_label, desired_value)
    assert len(matching_rows) > 0, f"Could not find the desired key = {desired_value}"
    assert len(matching_rows) == 1, (
        f"{len(matching_rows)} matching keys found. Expected to only find one." "Try using get_all_rows_with_value"
    )
    return matching_rows.iloc[0]


def get_trimesh_from_boundaries(
    left_boundary: Polyline3D, right_boundary: Polyline3D, resolution: float = 0.25
) -> trimesh.Trimesh:

    def _interpolate_polyline(polyline_3d: Polyline3D, num_samples: int) -> npt.NDArray[np.float64]:
        if num_samples < 2:
            num_samples = 2
        distances = np.linspace(0, polyline_3d.length, num=num_samples, endpoint=True, dtype=np.float64)
        return polyline_3d.interpolate(distances)

    average_length = (left_boundary.length + right_boundary.length) / 2
    num_samples = int(average_length // resolution) + 1
    left_boundary_array = _interpolate_polyline(left_boundary, num_samples)
    right_boundary_array = _interpolate_polyline(right_boundary, num_samples)
    return _create_lane_mesh_from_boundary_arrays(left_boundary_array, right_boundary_array)


def _create_lane_mesh_from_boundary_arrays(
    left_boundary_array: npt.NDArray[np.float64], right_boundary_array: npt.NDArray[np.float64]
) -> trimesh.Trimesh:

    # Ensure both polylines have the same number of points
    if left_boundary_array.shape[0] != right_boundary_array.shape[0]:
        raise ValueError("Both polylines must have the same number of points")

    n_points = left_boundary_array.shape[0]
    vertices = np.vstack([left_boundary_array, right_boundary_array])

    faces = []
    for i in range(n_points - 1):
        faces.append([i, i + n_points, i + 1])
        faces.append([i + 1, i + n_points, i + n_points + 1])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh
