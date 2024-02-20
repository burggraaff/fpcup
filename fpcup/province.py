"""
(Try to) load map backgrounds from file so they can be plotted.
"""

import geopandas as gpd
import numpy as np
from pandas import DataFrame, Series
from tqdm import tqdm

from ._typing import Iterable, Optional
from .constants import CRS_AMERSFOORT
from .settings import DEFAULT_DATA

# Load the outline of the Netherlands
nl = gpd.read_file(DEFAULT_DATA/"NL_borders.geojson")
nl_boundary = nl.boundary

# Load the provinces
_provinces = gpd.read_file(DEFAULT_DATA/"NL_provinces.geojson")
_provinces_coarse = gpd.read_file(DEFAULT_DATA/"NL_provinces_coarse.geojson")
province_names = list(_provinces["naamOfficieel"]) + ["Friesland"]

# Access individual provinces using a dictionary, e.g. province_boundary["Zuid-Holland"]
province_area = {name: poly for name, poly in zip(_provinces["naamOfficieel"], _provinces["geometry"])}
province_boundary = {name: gpd.GeoSeries(outline) for name, outline in zip(_provinces["naamOfficieel"], _provinces.boundary)}
province_coarse = {name: poly for name, poly in zip(_provinces_coarse["naamOfficieel"], _provinces_coarse["geometry"])}

# Add an alias for Friesland/Fryslân
province_area["Friesland"] = province_area["Fryslân"]
province_boundary["Friesland"] = province_boundary["Fryslân"]

def process_input_province(province: str) -> str:
    """
    Take an input province name and turn it into the standard format.
    """
    # Convert to title case, e.g. zuid-holland -> Zuid-Holland
    province = province.title()

    # Apply common alternatives
    if province in ("Friesland", "Fryslan"):
        province = "Fryslân"

    return province

def is_in_province(data: gpd.GeoDataFrame, province: str, *, province_data: dict=province_coarse, use_centroid=True) -> Iterable[bool]:
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

def add_provinces(data: gpd.GeoDataFrame, *, new_column: str="province", province_data: dict=province_coarse, progressbar=True, leave_progressbar=True, **kwargs) -> None:
    """
    Add a column with province names.
    Note: can get very slow for long dataframes.
    """
    # Convert to the right CRS first
    data_crs = data.to_crs(CRS_AMERSFOORT)

    # Generate an empty Series which will be populated with time
    province_list = Series(data=np.tile("", len(data_crs)), name="province", dtype=str, index=data_crs.index)

    # Loop over the provinces, find the relevant entries, and fill in the list
    for province_name in tqdm(province_data.keys(), desc="Assigning labels", unit="province", disable=not progressbar, leave=leave_progressbar):
        # Find the plots that have not been assigned a province yet to prevent duplicates
        where_empty = (province_list == "")
        data_empty = data_crs.loc[where_empty]

        # Find the items that are in this province
        selection = is_in_province(data_empty, province_name, province_data=province_data, **kwargs)

        # Set elements of where_empty to True if they are within this province
        where_empty.loc[where_empty] = selection

        # Assign the values
        province_list.loc[where_empty] = province_name

    # Add the series to the dataframe
    data[new_column] = province_list

def add_province_geometry(data: DataFrame, which: str="area", *, column_name: Optional[str]=None, crs: str=CRS_AMERSFOORT) -> gpd.GeoDataFrame:
    """
    Add a column with province geometry to a DataFrame with a province name column/index.
    """
    # Remove entries that are not in the province list
    pass

    # Use the index if no column was provided
    if column_name is None:
        column = data.index.to_series()
    else:
        column = data[column_name]

    # Apply the dictionary mapping
    geometry = column.map(province_area)
    data_new = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)

    # Change to outline if desired
    # (Doing this at the start gives an error)
    if which.lower() == "area":
        pass
    elif which.lower() == "outline":
        data_new["geometry"] = data_new.boundary
    else:
        raise ValueError(f"Cannot add geometries of type `{which}`")

    return data_new
