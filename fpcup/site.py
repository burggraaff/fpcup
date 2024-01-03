"""
Site-related stuff: load data etc
"""
from itertools import product
from numbers import Number

import numpy as np

from pcse.util import WOFOST72SiteDataProvider, WOFOST80SiteDataProvider

def grid_coordinate_range(latitude: tuple[Number], longitude: tuple[Number]) -> list[tuple[float]]:
    """
    Generate all pairs of latitude/longitude coordinates.
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
