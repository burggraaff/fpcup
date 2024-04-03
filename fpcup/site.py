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

from ._brp_dictionary import brp_categories_NL2EN
from ._typing import Callable, Coordinates, Iterable, RealNumber
from .constants import CRS_AMERSFOORT, WGS84
from .parameters import site_parameters


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
