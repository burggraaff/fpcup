"""
Geography-related constants and methods.
Includes polygons/outlines of the Netherlands and its provinces.
"""
import geopandas as gpd
gpd.options.io_engine = "pyogrio"
import h3pandas
import numpy as np
from pandas import DataFrame, Series
from tqdm import tqdm

from ._typing import AreaDict, BoundaryDict, Callable, Iterable, Optional
from .constants import CRS_AMERSFOORT, WGS84
from .settings import DEFAULT_DATA

# Constants
PROVINCE_NAMES = ("Fryslân", "Gelderland", "Noord-Brabant", "Noord-Holland", "Overijssel", "Zuid-Holland",  "Groningen", "Zeeland", "Drenthe", "Flevoland", "Limburg", "Utrecht")  # Sorted by area
NETHERLANDS = "Netherlands"
NAMES = PROVINCE_NAMES + (NETHERLANDS, )

# For convenience: iterate over NAMES with tqdm
iterate_netherlands = lambda disable=False, leave=True: tqdm(NAMES, desc="Looping over provinces", unit="province", disable=disable, leave=leave)

# Load the Netherlands shapefile
_netherlands = gpd.read_file(DEFAULT_DATA/"NL_borders.geojson")
_area_netherlands = {NETHERLANDS: _netherlands.iloc[0].geometry}  # Polygon - to be used in comparisons
_boundary_netherlands = {NETHERLANDS: _netherlands.boundary}  # GeoSeries - to be used with .plot()

# Load the province shapefiles
_provinces = gpd.read_file(DEFAULT_DATA/"NL_provinces.geojson")
_provinces_coarse = gpd.read_file(DEFAULT_DATA/"NL_provinces_coarse.geojson")

# Access individual provinces using a dictionary, e.g. area["Zuid-Holland"]
# Note: these contain bare Polygon/MultiPolygon objects, with no CRS.
area = {name: poly for name, poly in zip(_provinces["naamOfficieel"], _provinces["geometry"])}
area = {**_area_netherlands, **area}

area_coarse = {name: poly for name, poly in zip(_provinces_coarse["naamOfficieel"], _provinces_coarse["geometry"])}

# Access individual provinces using a dictionary, e.g. boundary["Zuid-Holland"]
# Note: these contain GeoSeries objects with 1 entry, with a CRS set.
boundary = {name: gpd.GeoSeries(outline, crs=CRS_AMERSFOORT) for name, outline in zip(_provinces["naamOfficieel"], _provinces.boundary)}
boundary = {**_boundary_netherlands, **boundary}

boundary_coarse = {name: gpd.GeoSeries(poly.boundary, crs=CRS_AMERSFOORT) for name, poly in area_coarse.items()}


def process_input_province(province: str) -> str:
    """
    Take an input province name and turn it into the standard format.
    """
    # Convert to title case, e.g. zuid-holland -> Zuid-Holland
    province = province.title()

    # Apply common alternatives
    if province in ("Friesland", "Fryslan"):
        province = "Fryslân"
    elif province in ("the Netherlands", "NL", "All"):
        province = "Netherlands"

    return province


def is_in_province(_data: gpd.GeoDataFrame, province: str, *,
                   province_data: AreaDict=area_coarse, use_centroid=True) -> Iterable[bool]:
    """
    For a series of geometries (e.g. BRP plots), determine if they are in the given province.
    Enable `use_centroid` to see if the centre of each plot falls within the province rather than the entire plot - this is useful for plots that are split between provinces.
    """
    assert province in NAMES, f"Unknown province '{province}'."

    area = province_data[province]
    if use_centroid:
        data = _data.centroid
    else:
        data = _data

    # Step 1: use the convex hull for a coarse selection
    selection = data.within(area.convex_hull)

    # Step 2: use the real shape of the province
    selection_fine = data.loc[selection].within(area)

    # Update the selection with the new information
    selection.loc[selection] = selection_fine

    return selection


def add_provinces(data: gpd.GeoDataFrame, *,
                  new_column: str="province", province_data: AreaDict=area_coarse, progressbar=True, leave_progressbar=True, **kwargs) -> None:
    """
    Add a column with province names.
    Note: can get very slow for long dataframes.
    """
    # Convert to the right CRS first
    data_crs = data.to_crs(CRS_AMERSFOORT)

    # Generate an empty Series which will be populated with time
    province_list = Series(data=np.tile("", len(data_crs)),
                           name="province", dtype=str, index=data_crs.index)

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


def add_province_geometry(data: DataFrame, which: str="area", *,
                          column_name: Optional[str]=None, crs: str=CRS_AMERSFOORT) -> gpd.GeoDataFrame:
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
    geometry = column.map(area)
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


def entries_in_province(data: gpd.GeoDataFrame, province: str) -> gpd.GeoDataFrame:
    """
    Return only those entries from `data` that are in the given `province`.
    Shorthand function that tries different approaches:
        1. Check if `data` has a "province" column and use that.
        2. Filter based on the geometry in `data` using is_in_province.
    """
    assert province in PROVINCE_NAMES, f"Unknown province '{province}'."

    # First: try the "province" column
    if "province" in data.columns:
        entries = (data["province"] == province)

    # Second: Check the data manually
    elif "geometry" in data.columns:
        # Convert the data to the same CRS as the province data
        data_crs = data.to_crs(CRS_AMERSFOORT)
        entries = is_in_province(data_crs, province)

    # No other cases currently
    else:
        raise ValueError(f"Input does not have a 'province' column nor any geometry information.")

    return data.loc[entries]


def aggregate_h3(_data: gpd.GeoDataFrame, functions: dict[str, Callable] | Callable | str="mean", *,
                 level: int=6, clipto: Optional[str]="Netherlands") -> gpd.GeoDataFrame:
    """
    Aggregate data to the H3 hexagonal grid.
    `functions` is passed to DataFrame.agg.
    `clipto` is used to get a geometry, e.g. the Netherlands or one province, to clip the results to. Set it to `None` to preserve the full grid.

    TO DO: See if using `clipto` to filter data to a given province before aggregation is faster.
    """
    # Convert the input to WGS84 for the aggregation
    crs_original = _data.crs
    data = _data.copy()
    data["geometry"] = data.centroid.to_crs(WGS84)

    # Aggregate the data and convert back to the original CRS
    data_h3 = data.h3.geo_to_h3_aggregate(level, functions)
    data_h3.to_crs(crs_original, inplace=True)

    # Clip the data if desired
    if clipto is not None:
        assert clipto in area.keys(), f"Cannot clip the H3 grid to '{clipto}' - unknown name. Please provide a province name, 'Netherlands', or None."
        clipto_geometry = area[clipto]

        data_h3["geometry"] = data_h3.intersection(clipto_geometry)
        data_h3 = data_h3.loc[~data_h3.is_empty]

    return data_h3
