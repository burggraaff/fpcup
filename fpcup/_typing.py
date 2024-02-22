"""
Combines abstract base classes from various places and generates some useful aliases.
"""
from numbers import Number, Real as RealNumber
from os import PathLike
from typing import Callable, Iterable, Optional, Type

from geopandas import GeoSeries
from shapely import Point, Polygon

Coordinates = tuple[RealNumber, RealNumber]
PathOrStr = PathLike | str
StringDict = dict[str, str]

AreaDict = dict[str, Polygon]
BoundaryDict = dict[str, GeoSeries]
