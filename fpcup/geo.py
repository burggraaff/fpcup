"""
Geography-related constants and methods.
Includes polygons/outlines of the Netherlands and its provinces.
"""
import random
from functools import wraps

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
import h3pandas
import numpy as np
from pandas import DataFrame, Series
from shapely import Geometry, Point, Polygon
from tqdm import tqdm

from ._typing import Aggregator, AreaDict, BoundaryDict, Callable, Iterable, Optional, PathOrStr
from .aggregate import default_aggregator, rename_after_aggregation
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


def transform_geometry(geometry: Geometry, crs_old: str, crs_new: str) -> Geometry:
    """
    Transform a bare Geometry object from one CRS to another.
    """
    geometry_gpd = gpd.GeoSeries(geometry, crs=crs_old)
    transformed_gpd = geometry_gpd.to_crs(crs_new)
    geometry_new = transformed_gpd.iloc[0]
    return geometry_new


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
                  new_column: str="province", province_data: AreaDict=area_coarse, remove_empty=True,
                  progressbar=True, leave_progressbar=True, **kwargs) -> None:
    """
    Add a column with province names.
    Note: can get very slow for long dataframes.
    If `remove_empty`, entries that do not fall into any province are removed.
    **kwargs are passed to is_in_province.
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

    # Remove entries not in any province (e.g. from generating random coordinates)
    if remove_empty:
        index_remove = data.loc[data[new_column] == ""].index
        data.drop(index=index_remove, inplace=True)


def add_province_geometry(data: DataFrame, which: str="area", *,
                          column_name: Optional[str]=None) -> gpd.GeoDataFrame:
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
    data_new = gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_AMERSFOORT)

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


def maintain_crs(func: Callable) -> Callable:
    """
    Decorator that ensures the output of an aggregation function is in the same CRS as the input, regardless of transformations that happen along the way.
    """
    @wraps(func)
    def newfunc(data, *args, **kwargs):
        crs_original = data.crs
        data_new = func(data, *args, **kwargs)
        data_new.to_crs(crs_original, inplace=True)
        return data_new

    return newfunc


@maintain_crs
def aggregate_province(_data: gpd.GeoDataFrame, *,
                       aggregator: Optional[Aggregator]=None) -> gpd.GeoDataFrame:
    """
    Aggregate data to the provinces.
    `aggregator` is passed to DataFrame.agg; if none is specified, then means or weighted means (depending on availability of weights) are used.
    """
    # Convert the input to CRS_AMERSFOORT for the aggregation and use the centroids
    data = _data.copy()
    data["geometry"] = data.to_crs(CRS_AMERSFOORT).centroid

    # Add province information if not yet available
    if "province" not in data.columns:
        add_provinces(data, leave_progressbar=False)

    # Aggregate the data
    if aggregator is None:
        aggregator = default_aggregator(data)
    data_province = data.groupby("province").agg(aggregator).rename(columns=rename_after_aggregation)

    # Add the province geometries
    data_province = add_province_geometry(data_province)

    return data_province


def save_aggregate_province(data: gpd.GeoDataFrame, saveto: PathOrStr, **kwargs) -> None:
    """
    Save a provincial aggregate without the geometry information.
    """
    data.drop("geometry", axis=1).to_csv(saveto)


def _default_h3_level(clipto: str) -> int:
    """
    Determine the default level for H3 aggregation.
    Currently a simple choice between 7 (provinces) or 6 (Netherlands or unspecified).
    Defined as a function rather than a dictionary to allow for future expansions, e.g. determining it by area coverage instead.
    """
    if clipto in PROVINCE_NAMES:
        level = 7
    else:
        level = 6

    return level


@maintain_crs
def aggregate_h3(_data: gpd.GeoDataFrame, *,
                 aggregator: Optional[Aggregator]=None, level: Optional[int]=None, clipto: Optional[str]="Netherlands", weightby: str="area") -> gpd.GeoDataFrame:
    """
    Aggregate data to the H3 hexagonal grid.
    `aggregator` is passed to DataFrame.agg; if none is specified, then means or weighted means (depending on availability of weights) are used.
    `clipto` is used to get a geometry, e.g. the Netherlands or one province, to clip the results to. Set it to `None` to preserve the full grid.
    """
    # Filter the data to the desired province
    if clipto != "Netherlands":
        _data = entries_in_province(_data, clipto)

    # Find the centroids in a projected CRS, then convert to WGS84 for aggregation
    data = _data.copy()
    if not data.crs.is_projected:
        data.to_crs(CRS_AMERSFOORT, inplace=True)
    data["geometry"] = data.centroid.to_crs(WGS84)

    # Aggregate the data
    if aggregator is None:
        aggregator = default_aggregator(data, weightby=weightby)
    if level is None:
        level = _default_h3_level(clipto)
    data_h3 = data.h3.geo_to_h3_aggregate(level, aggregator).rename(columns=rename_after_aggregation)

    # Clip the data if desired
    if clipto is not None:
        assert clipto in area.keys(), f"Cannot clip the H3 grid to '{clipto}' - unknown name. Please provide a province name, 'Netherlands', or None."
        clipto_geometry = area[clipto]

        data_h3.to_crs(CRS_AMERSFOORT, inplace=True)
        data_h3["geometry"] = data_h3.intersection(clipto_geometry)
        data_h3 = data_h3.loc[~data_h3.is_empty]

    return data_h3


def _generate_random_point_within_bounds(min_lon, min_lat, max_lon, max_lat) -> Point:
    """
    Generate one random shapely Point within the given bounds.
    The inputs are in the same order as the outputs of Polygon.bounds (in WGS84), so this function can be called as _generate_random_point_within_bounds(*bounds).
    """
    latitude = random.uniform(min_lat, max_lat)
    longitude = random.uniform(min_lon, max_lon)
    return Point(longitude, latitude)


def _generate_random_point_in_geometry(geometry: Geometry, *, crs=CRS_AMERSFOORT) -> Point:
    """
    Generate a single random point that lies within a geometry.
    Unfortunately quite slow due to the many "contains" checks.
    """
    polygon = transform_geometry(geometry, crs, WGS84)  # Convert to WGS84 coordinates
    while True:  # Infinite generator
        # Generate points randomly until one falls within the given geometry
        p = _generate_random_point_within_bounds(*polygon.bounds)

        # If a point was successfully generated, yield it
        if polygon.contains(p):
            # Doing a first check with the convex hull does not help here
            yield p


def _generate_random_point_in_geometry_batch(geometry: Geometry, n: int, *,  crs=CRS_AMERSFOORT) -> gpd.GeoSeries:
    """
    Generate a batch of random points and return the ones that fall within the given geometry.
    """
    polygon = transform_geometry(geometry, crs, WGS84)  # Convert to WGS84 coordinates

    points = gpd.GeoSeries((_generate_random_point_within_bounds(*polygon.bounds) for i in range(n)), crs=WGS84)
    points_in_polygon = points.loc[points.within(polygon)]

    return points_in_polygon


def coverage_of_bounding_box(geometry: Geometry) -> float:
    """
    Calculate the fraction of its bounding box covered by a given geometry.
    """
    min_x, min_y, max_x, max_y = geometry.bounds
    area_box = (max_x - min_x) * (max_y - min_y)
    return geometry.area / area_box
