import geopandas as gpd
import numpy as np


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
    # if column_label == "id":
    #     return elements.loc[desired_value]

    # matching_rows = get_all_rows_with_value(elements, column_label, desired_value)
    # assert len(matching_rows) > 0, f"Could not find the desired key = {desired_value}"
    # assert len(matching_rows) == 1, (
    #     f"{len(matching_rows)} matching keys found. Expected to only find one." "Try using get_all_rows_with_value"
    # )

    # return matching_rows.iloc[0]
    return elements.loc[elements[column_label] == desired_value].iloc[0]
