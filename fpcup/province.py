"""
(Try to) load map backgrounds from file so they can be plotted.
"""

import geopandas as gpd
import numpy as np
from pandas import Series
from tqdm import tqdm

from ._typing import Iterable
from .constants import CRS_AMERSFOORT
from .settings import DEFAULT_DATA

# Load the outline of the Netherlands
nl = gpd.read_file(DEFAULT_DATA/"NL_borders.geojson")
nl_boundary = nl.boundary

# Load the provinces
provinces = gpd.read_file(DEFAULT_DATA/"NL_provinces.geojson")
provinces_coarse = gpd.read_file(DEFAULT_DATA/"NL_provinces_coarse.geojson")
province_names = list(provinces["naamOfficieel"]) + ["Friesland"]

# Access individual provinces using a dictionary, e.g. province_boundary["Zuid-Holland"]
province_area = {name: poly for name, poly in zip(provinces["naamOfficieel"], provinces["geometry"])}
province_boundary = {name: gpd.GeoSeries(outline) for name, outline in zip(provinces["naamOfficieel"], provinces.boundary)}
province_coarse = {name: poly for name, poly in zip(provinces_coarse["naamOfficieel"], provinces_coarse["geometry"])}

# Add an alias for Friesland/Fryslân
province_area["Friesland"] = province_area["Fryslân"]
province_boundary["Friesland"] = province_boundary["Fryslân"]

def is_in_province(data: gpd.GeoDataFrame, province: str, province_data: dict=province_coarse, use_centroid=True) -> Iterable[bool]:
    """
    For a series of geometries (e.g. BRP plots), determine if they are in the given province.
    Enable `use_centroid` to see if the centre of each plot falls within the province rather than the entire plot - this is useful for plots that are split between provinces.
    """
    area = province_data[province]
    if use_centroid:
        data_here = data.centroid
    else:
        data_here = data

    # Step 1: use the convex hull for a coarse selection
    selection_coarse = data_here.within(area.convex_hull)

    # Step 2: use the real shape of the province
    selection_fine = data_here.loc[selection_coarse].within(area)

    # Update the coarse selection with the new information
    selection_coarse.loc[selection_coarse] = selection_fine

    return selection_coarse

def add_provinces(data: gpd.GeoDataFrame, new_column: str="province", province_data: dict=province_coarse, **kwargs) -> None:
    """
    Add a column with province names.
    Note: can get very slow for long dataframes.
    """
    # Generate an empty Series which will be populated with time
    province_list = Series(data=np.tile("", len(data)), name="province", dtype=str, index=data.index)

    # Loop over the provinces, find the relevant entries, and fill in the list
    for province_name in tqdm(province_data.keys(), desc="Assigning labels", unit="province"):
        # Find the plots that have not been assigned a province yet to prevent duplicates
        where_empty = (province_list == "")
        data_empty = data.loc[where_empty]

        # Find the items that are in this province
        selection = is_in_province(data_empty, province_name, province_data=province_data, **kwargs)

        # Set elements of where_empty to True if they are within this province
        where_empty.loc[where_empty] = selection

        # Assign the values
        province_list.loc[where_empty] = province_name

    # Add the series to the dataframe
    data[new_column] = province_list
