"""
Combines abstract base classes from various places and generates some useful aliases.
"""
from numbers import Number, Real as RealNumber
from os import PathLike
from typing import Callable, Iterable, Optional, Type

from geopandas import GeoSeries
from shapely import Point, Polygon

# Combinations of built-in types
Coordinates = tuple[RealNumber, RealNumber]
PathOrStr = PathLike | str

# Mappings and other callables
StringDict = dict[str, str]
FuncDict = dict[str, Callable]
Aggregator = FuncDict | Callable | str

# Geographic data
AreaDict = dict[str, Polygon]
BoundaryDict = dict[str, GeoSeries]
