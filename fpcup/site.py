"""
Site-related stuff: load data etc
"""
from itertools import product
from numbers import Number

import numpy as np

from pcse.util import WOFOST72SiteDataProvider, WOFOST80SiteDataProvider

def grid_coordinate_range(latitude: tuple[Number], longitude: tuple[Number]) -> list[tuple[float]]:
    """
    Generate all pairs of latitude/longitude coordinates in given latitude/longitude ranges.
    Inputs should be tuples that can be passed to np.arange.
    The output is given as a list so it can be iterated over multiple times.

    Example:
        coords = grid_coordinate_range(latitudes=(50, 52, 0.1), longitudes=(5, 10, 1))
    """
    latitude_range = np.arange(*latitude)
    longitude_range = np.arange(*longitude)
    coordinates = product(latitude_range, longitude_range)
    coordinates = list(coordinates)
    return coordinates

def grid_coordinate_linspace(latitude: tuple[Number], longitude: tuple[Number], n: int) -> list[tuple[float]]:
    """
    Generate n pairs of latitude/longitude coordinates in a given area bound by latitude/longitude ranges.
    Inputs should be tuples of (min, max) latitude/longitude.
    The output is given as a list so it can be iterated over multiple times.
    Note: the true size of the output may be smaller than n due to rounding.

    Example:
        coords = grid_coordinate_range(latitudes=(50, 52), longitudes=(5, 10), n=1000)
    """
    n_each = int(np.sqrt(n))
    latitude_range = np.linspace(*latitude, n_each)
    longitude_range = np.linspace(*longitude, n_each)
    coordinates = product(latitude_range, longitude_range)
    coordinates = list(coordinates)
    return coordinates
