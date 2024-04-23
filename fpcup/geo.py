"""
Geography-related constants and methods.
Includes polygons/outlines of the Netherlands and its provinces.
"""
import random
from functools import partial, wraps

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
import h3pandas
import numpy as np
from pandas import DataFrame, Series
from shapely import Geometry, Point, Polygon
from tqdm import tqdm

from ._netherlands import ABBREVIATION2NAME, NAME2ABBREVIATION, ALIASES, NAMES, PROVINCE_NAMES, _basemaps, apply_aliases
from ._netherlands import NETHERLANDS as NETHERLANDS_LABEL
from .constants import CRS_AMERSFOORT, WGS84
from .multiprocessing import multiprocess_site_generation
from .typing import AreaDict, BoundaryDict, Callable, Coordinates, Iterable, Optional, PathOrStr, RealNumber


### PROVINCE CONVENIENCE CLASS
class Province:
    """
    Handles everything related to areas such as provinces, countries, etc.
    """
    ### CREATION AND INITIALISATION
    def __init__(self, area: Geometry, *,
                 area_coarse: Optional[Geometry]=None, crs: str=CRS_AMERSFOORT,
                 name: Optional[str]=None, abbreviation: Optional[str]=None, level: Optional[str]="province"):
        # Initialise geospatial parameters
        self.area = area
        self.area_coarse = area_coarse
        self._crs = crs

        # Initialise identifiers etc
        self.name = name
        if abbreviation is None and name is not None:
            try:
                abbreviation = NAME2ABBREVIATION[name]
            except KeyError:
                abbreviation = None
        self.abbreviation = abbreviation
        self.level = level

    @classmethod
    def from_dataframe(cls, name: str, data: gpd.GeoDataFrame=_basemaps, *,
                       namecol: str="name", levelcol: str="level"):
        """
        For a given name, get the relevant data from a GeoDataFrame and create a Province object.
        """
        # Main data
        mainrow = data.loc[data[namecol] == name].iloc[0]
        area = mainrow.geometry
        level = mainrow["level"]

        # Coarse geometry (if available)
        try:
            coarserow = data.loc[data[namecol] == name+"_coarse"].iloc[0]
        except IndexError:
            area_coarse = None
        else:
            area_coarse = coarserow.geometry

        return cls(area, name=name, area_coarse=area_coarse, crs=data.crs, level=level)


    ### GEOSPATIAL PROPERTIES AND MANIPULATIONS
    @property
    def crs(self) -> str:
        return self._crs

    def __repr__(self) -> str:
        return f"{self.name} (crs={self.crs})"

    def __str__(self) -> str:
        return self.name

    @property
    def boundary(self) -> Geometry:
        """
        Get the boundary of self.area; implemented as a property for ease in CRS conversions.
        """
        return self.area.boundary

    @property
    def boundary_coarse(self) -> Geometry | None:
        if self.area_coarse is not None:
            return self.area_coarse.boundary
        else:
            return None

    @property
    def centroid(self) -> Point:
        return self.area.centroid

    @property
    def convex_hull(self) -> Geometry:
        return self.area.convex_hull

    def to_crs(self, crs_new: str):
        """
        Generate a new object in a given new CRS.
        """
        cls = self.__class__
        transform = partial(transform_geometry, crs_old=self.crs, crs_new=crs_new)

        area = transform(self.area)
        if self.area_coarse is not None:
            area_coarse = transform(self.area_coarse)
        else:
            area_coarse = None

        return cls(area, area_coarse=area_coarse, crs=crs_new, name=self.name, abbreviation=self.abbreviation)


    ### COMPLEX GEOSPATIAL FUNCTIONS
    def contains(self, other: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *,
                 use_coarse=False, use_centroid=True) -> bool | Iterable[bool]:
        """
        Check if a given geometry or series thereof are within the current province.
        If `other` is a bare Geometry, it is assumed to be in the same CRS as the Province.
        If `other` is a GeoSeries/GeoDataFrame, it is converted first.
        """
        # Set up data that will be checked
        SINGLE_INPUT = isinstance(other, Geometry)
        if SINGLE_INPUT:
            data = gpd.GeoSeries(other, crs=self.crs)
        else:
            data = other.to_crs(self.crs)

        # Set up self
        if use_coarse:
            assert self.area_coarse is not None, f"Cannot use coarse area for {self} because it is not defined."
            target = self.area_coarse
        else:
            target = self.area

        # Perform the actual check
        selection = is_in_geometry(data, target, use_centroid=use_centroid)
        if SINGLE_INPUT:
            selection = selection.iloc[0]

        return selection

    def select_entries_in_province(self, data: DataFrame) -> gpd.GeoDataFrame:
        """
        Return only those entries from `data` that fall within this Province.
        Shorthand function that tries different approaches:
            1. Check if `data` has a "province" column and use that (DataFrame or GeoDataFrame).
            2. Filter based on the geometry in `data` using Province.contains (GeoDataFrame only).
        """
        # First: try the "province" column
        if "province" in data.columns:
            column = data["province"].replace(ABBREVIATION2NAME)
            entries = (column == self.name)

        # Second: Check the data manually
        elif "geometry" in data.columns:
            entries = self.contains(data)

        # No other cases currently
        else:
            raise ValueError(f"Input does not have a 'province' column nor any geometry information.")

        return data.loc[entries]

    def clip_data(self, _data: gpd.GeoDataFrame, *, remove_empty=True) -> gpd.GeoDataFrame:
        """
        Given a GeoDataframe with geometries, return the intersection with this province.
        Example use case: clipping H3 hexagons.
        """
        data = _data.to_crs(self.crs)
        data["geometry"] = data.intersection(self.area)

        if remove_empty:
            data = data.loc[~data.is_empty]

        return data

    def _generate_random_point(self, *args) -> Point:
        # *args do nothing but are necessary for multiprocessing to pass dummy arguments
        return _generate_random_point_in_geometry(self.area)

    def generate_random_points(self, n: int, *, as_coordinates=True,
                               generate_crs: Optional[str]=None, output_crs: Optional[str]=WGS84,
                               progressbar=True, leave_progressbar=True) -> list[Coordinates] | gpd.GeoSeries:
        """
        Generate n pairs of latitude/longitude coordinates that are (roughly) uniformly distributed over the given province.
        `generate_crs` determines the CRS in which points are generated; this is the native CRS unless otherwise specified.
        `output_crs` is used to transform the outputs; by default this is WGS84.
        Points are generated in the native CRS unless otherwise specified; they may not be uniformly distributed in other CRSes.
        The output is given as a list so it can be iterated over multiple times.
        if `as_geometry` is True, the results are returned as Points; if False (default), as pairs of coordinates.
        """
        # Adjust to the generation CRS
        if generate_crs is None:
            generate_crs = self.crs
            func = self._generate_random_point
        else:
            new_geo = self.to_crs(generate_crs)
            func = new_geo._generate_random_point

        # Generate points
        points = multiprocess_site_generation(func, range(n), progressbar=progressbar, leave_progressbar=leave_progressbar)

        # Convert output to the correct CRS
        points = transform_geometry(points, generate_crs, output_crs)

        # Return as points if desired
        if as_coordinates:
            points = points_to_coordinates(points)

        return points


    ### PLOTTING
    def _plot_geo(self, geo_property: Geometry, *args,
                  crs: Optional[str]=None, **kwargs) -> None:
        """
        General method for plotting some geo property, e.g. area or boundary, by converting it to a GeoSeries first.
        """
        as_series = gpd.GeoSeries(geo_property, crs=self.crs)
        if crs is not None:
            as_series = as_series.to_crs(crs)

        as_series.plot(*args, **kwargs)

    def plot_area(self, *args, use_coarse=False, **kwargs) -> None:
        """
        Plot the area (or coarse area) of this province.
        """
        geo = self.area_coarse if use_coarse else self.area
        self._plot_geo(geo, *args, **kwargs)

    def plot_boundary(self, *args, use_coarse=False, **kwargs) -> None:
        """
        Plot the boundary (or coarse boundary) of this province.
        """
        geo = self.boundary_coarse if use_coarse else self.boundary
        self._plot_geo(geo, *args, **kwargs)


### PRE-MADE GEOMETRIES
provinces = {name: Province.from_dataframe(name) for name in PROVINCE_NAMES}
NETHERLANDS = Province.from_dataframe(NETHERLANDS_LABEL)
locations_full = {NETHERLANDS_LABEL: NETHERLANDS, **provinces}
_province_areas = {name: p.area for name, p in locations_full.items()}
_province_areas_coarse = {name: p.area_coarse for name, p in locations_full.items()}

# Template for easier looping with progressbar
def tqdm_template(data: Iterable, *, label="province"):
    return lambda disable=False, leave=True: tqdm(data, desc=f"Looping over {label}s", unit=label)

iterate_over_provinces = tqdm_template(provinces.values())
iterate_over_locations = tqdm_template(locations_full.values(), label="place")
iterate_over_province_names = tqdm_template(provinces.keys())



### GENERAL GEOSPATIAL FUNCTIONS
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


def format_coordinates(latitude: RealNumber, longitude: RealNumber) -> str:
    """
    Format (lat, lon) coordinates clearly.
    """
    NS = "N" if latitude >= 0 else "S"
    EW = "E" if longitude >= 0 else "W"
    return f"({latitude:.4f}° {NS}, {longitude:.4f}° {EW})"


def points_to_coordinates(points: Iterable[Point]) -> list[Coordinates]:
    """
    Split shapely Points into (latitude, longitude) pairs.
    """
    return [(p.y, p.x) for p in points]


def coverage_of_bounding_box(geometry: Geometry) -> float:
    """
    Calculate the fraction of its bounding box covered by a given geometry.
    """
    min_x, min_y, max_x, max_y = geometry.bounds
    area_box = (max_x - min_x) * (max_y - min_y)
    return geometry.area / area_box


def transform_geometry(geometry: Geometry | Iterable[Geometry], crs_old: str, crs_new: str) -> Geometry | Iterable[Geometry]:
    """
    Transform a bare Geometry object, or iterable thereof, from one CRS to another.
    """
    SINGLE_GEOMETRY = isinstance(geometry, Geometry)

    geometry_gpd = gpd.GeoSeries(geometry, crs=crs_old)
    transformed_gpd = geometry_gpd.to_crs(crs_new)

    geometry_new = transformed_gpd.iloc[0] if SINGLE_GEOMETRY else transformed_gpd
    return geometry_new


def is_in_geometry(_data: gpd.GeoDataFrame, target: Geometry, *,
                   use_centroid=True) -> Iterable[bool]:
    """
    For a series of geometries (e.g. BRP plots), determine if they are in the given target geometry.
    Enable `use_centroid` to see if the centre of each plot falls within the geometry rather than the entire plot - this is useful for plots that are split between provinces.
    """
    if use_centroid:
        data = _data.centroid
    else:
        data = _data

    # Step 1: use the convex hull for a coarse selection
    selection = data.within(target.convex_hull)

    # Step 2: use the real shape of the province
    selection_fine = data.loc[selection].within(target)

    # Update the selection with the new information
    selection.loc[selection] = selection_fine

    return selection


### PROVINCE-RELATED FUNCTIONS
def process_input_province(name: str) -> Province:
    """
    Take an input province name, apply aliases, and return the Province object.
    """
    name = apply_aliases(name)
    return locations_full[name]


def is_single_province(prov: Province) -> bool:
    """
    Helper function for parsing inputs; are we doing a single province or the whole country?
    """
    return (prov.level == "province")


def add_provinces(data: gpd.GeoDataFrame, *,
                  new_column: str="province", use_abbreviation=True, use_coarse=True, remove_empty=True,
                  progressbar=True, leave_progressbar=True, **kwargs) -> None:
    """
    Add a column with province names/abbreviations.
    Note: can get very slow for long dataframes.
    If `remove_empty`, entries that do not fall into any province are removed.
    **kwargs are passed to province.contains.
    """
    # Convert to the right CRS first
    data_crs = data.to_crs(CRS_AMERSFOORT)

    # Generate an empty Series which will be populated with time
    province_list = Series(data=np.tile("", len(data_crs)),
                           name="province", dtype=str, index=data_crs.index)

    # Loop over the provinces, find the relevant entries, and fill in the list
    for province in tqdm(provinces.values(), desc="Assigning labels", unit="province", disable=not progressbar, leave=leave_progressbar):
        province_label = province.abbreviation if use_abbreviation else province.name

        # Find the plots that have not been assigned a province yet to prevent duplicates
        where_empty = (province_list == "")
        data_empty = data_crs.loc[where_empty]

        # Find the items that are in this province
        selection = province.contains(data_empty, use_coarse=use_coarse, **kwargs)

        # Set elements of where_empty to True if they are within this province
        where_empty.loc[where_empty] = selection

        # Assign the values
        province_list.loc[where_empty] = province_label

    # Add the series to the dataframe
    data[new_column] = province_list

    # Remove entries not in any province (e.g. from generating random coordinates)
    if remove_empty:
        index_remove = data.loc[data[new_column] == ""].index
        data.drop(index=index_remove, inplace=True)


def add_province_geometry(data: DataFrame, *,
                          column_name: Optional[str]=None, use_coarse=False) -> gpd.GeoDataFrame:
    """
    Add a column with province geometry to a DataFrame with a province name column/index.
    """
    # Use the index if no column was provided
    if column_name is None:
        column_with_names = data.index.to_series()
    else:
        column_with_names = data[column_name]

    # Check for aliases
    names = column_with_names.apply(apply_aliases)

    # Remove entries that are not in the province list
    pass

    # Select the correct mapping
    mapping = _province_areas_coarse if use_coarse else _province_areas

    # Apply the dictionary mapping
    geometry = names.map(mapping)
    data_new = gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_AMERSFOORT)

    return data_new


### SITE GENERATION
def _generate_random_point_within_bounds(min_lon, min_lat, max_lon, max_lat) -> Point:
    """
    Generate one random shapely Point within the given bounds.
    The inputs are in the same order as the outputs of Polygon.bounds (in WGS84), so this function can be called as _generate_random_point_within_bounds(*bounds).
    """
    latitude = random.uniform(min_lat, max_lat)
    longitude = random.uniform(min_lon, max_lon)
    return Point(longitude, latitude)


def _generate_random_point_in_geometry(polygon: Geometry) -> Point:
    """
    Generate a single random point that lies within a geometry.
    """
    while True:  # Keep going until a point is returned/yielded
        # Generate points randomly until one falls within the given geometry
        p = _generate_random_point_within_bounds(*polygon.bounds)

        # If a point was successfully generated, yield it
        if polygon.contains(p):
            # Doing a first check with the convex hull does not help here
            return p
