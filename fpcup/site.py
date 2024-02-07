"""
Site-related stuff: load data etc
"""
from itertools import product

import numpy as np

from pcse.util import WOFOST72SiteDataProvider, WOFOST80SiteDataProvider
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from ._brp_dictionary import brp_categories_NL2EN
from ._typing import RealNumber

def example(*args, **kwargs) -> PCSESiteDataProvider:
    """
    Just use the default value for now.
    """
    sitedata = WOFOST72SiteDataProvider(WAV=10)
    return sitedata

def grid_coordinate_range(latitude: tuple[RealNumber], longitude: tuple[RealNumber]) -> list[tuple[float]]:
    """
    Generate all pairs of latitude/longitude coordinates in given latitude/longitude ranges.
    Inputs should be tuples that can be passed to np.arange as parameters.
    The output is given as a list so it can be iterated over multiple times.

    Example:
        coords = grid_coordinate_range(latitudes=(50, 52, 0.1), longitudes=(5, 10, 1))
    """
    latitude_range = np.arange(*latitude)
    longitude_range = np.arange(*longitude)
    coordinates = product(latitude_range, longitude_range)
    coordinates = list(coordinates)
    return coordinates

def grid_coordinate_linspace(latitude: tuple[RealNumber], longitude: tuple[RealNumber], n: int) -> list[tuple[float]]:
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
