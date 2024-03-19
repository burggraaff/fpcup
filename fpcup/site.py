"""
Site-related stuff: load data etc
"""
import random
from itertools import product

import numpy as np
from pandas import concat
from shapely import Point
from tqdm import tqdm, trange

from pcse.util import WOFOST72SiteDataProvider, WOFOST80SiteDataProvider
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from ._typing import Callable, Coordinates, Iterable, RealNumber
from .constants import CRS_AMERSFOORT, WGS84
from .geo import area, _generate_random_point_in_geometry, _generate_random_point_in_geometry_batch, coverage_of_bounding_box, transform_geometry
from .multiprocessing import multiprocess_site_generation


def example(*args, **kwargs) -> PCSESiteDataProvider:
    """
    Just use the default value for now.
    """
    sitedata = WOFOST72SiteDataProvider(WAV=10)
    return sitedata


def combine_and_shuffle_coordinates(latitudes: Iterable[RealNumber], longitudes: Iterable[RealNumber], *,
                                    shuffle: bool=True) -> list[Coordinates]:
    """
    Post-process lists of latitudes and longitudes into a list of combined coordinate pairs.
    """
    # Combine the coordinates into pairs
    coordinates = product(latitudes, longitudes)
    coordinates = list(coordinates)

    # Randomise order
    if shuffle:
        random.shuffle(coordinates)  # In-place

    return coordinates


def generate_sites_range(latitude: tuple[RealNumber], longitude: tuple[RealNumber], *,
                         shuffle: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all pairs of latitude/longitude coordinates in given latitude/longitude ranges.
    Inputs should be tuples that can be passed to np.arange as parameters.
    """
    latitudes = np.arange(*latitude)
    longitudes = np.arange(*longitude)
    return combine_and_shuffle_coordinates(latitudes, longitudes, shuffle=shuffle)


def generate_sites_space(latitude: tuple[RealNumber], longitude: tuple[RealNumber], n: int, *,
                         shuffle: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n pairs of latitude/longitude coordinates in a given area bound by latitude/longitude ranges.
    Inputs should be tuples of (min, max) latitude/longitude.
    Note: the true size of the output may be smaller than n due to rounding.
    """
    n_each = int(np.sqrt(n))
    latitudes = np.linspace(*latitude, n_each)
    longitudes = np.linspace(*longitude, n_each)
    return combine_and_shuffle_coordinates(latitudes, longitudes, shuffle=shuffle)


def points_to_coordinates(points: Iterable[Point]) -> list[Coordinates]:
    """
    Split shapely Points into (latitude, longitude) pairs.
    """
    return [(p.y, p.x) for p in points]


def generate_sites_in_province_frombatch(province: str, n: int, **kwargs) -> list[Coordinates]:
    """
    Generate n pairs of latitude/longitude coordinates that are (roughly) uniformly distributed over the given province.
    Points are generated in WGS84 so they may not be uniformly distributed in other CRSes.
    The output is given as a list so it can be iterated over multiple times.
    """
    geometry = area[province]  # geometry in CRS_AMERSFOORT

    # Estimate roughly by how much to overshoot for each iteration
    coverage = coverage_of_bounding_box(geometry)
    n_safe = int(n / coverage)

    # Generate the first iteration of points
    points = _generate_random_point_in_geometry_batch(geometry, n_safe)

    # Iterate until there are enough points
    while len(points) < n:
        new_points = _generate_random_point_in_geometry_batch(geometry, n_safe//10)
        points = concat([points, new_points])

    # Cut off any excess
    points = points.iloc[:n]

    # Extract and return the latitudes and longitudes
    coordinates = points_to_coordinates(points)
    return coordinates


def generate_sites_in_province(province: str, n: int, *,
                               progressbar=True, leave_progressbar=True) -> list[Coordinates]:
    """
    Generate n pairs of latitude/longitude coordinates that are (roughly) uniformly distributed over the given province.
    Points are generated in WGS84 so they may not be uniformly distributed in other CRSes.
    The output is given as a list so it can be iterated over multiple times.
    """
    geometry = area[province]  # geometry in CRS_AMERSFOORT
    geometry = transform_geometry(geometry, CRS_AMERSFOORT, WGS84)  # Convert to WGS84 coordinates
    geometry_iterable = [geometry] * n

    # Generate points
    points = multiprocess_site_generation(_generate_random_point_in_geometry, geometry_iterable, progressbar=progressbar, leave_progressbar=leave_progressbar)

    coordinates = points_to_coordinates(points)

    return coordinates
