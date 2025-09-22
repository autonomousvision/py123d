from typing import List

import geopandas as gpd
import numpy as np
from shapely import wkt


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
    return elements.iloc[np.where(elements[column_label].to_numpy().astype(int) == int(desired_value))]


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
