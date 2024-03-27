"""
Combines abstract base classes from various places and generates some useful aliases.
"""
from dataclasses import dataclass
from numbers import Number, Real as RealNumber
from os import PathLike
from typing import Callable, Iterable, Optional, Type

from geopandas import GeoSeries
from pandas import Series
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

# PCSE parameters
@dataclass
class _PCSEParameterBase:
    name: str
    description: str

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

@dataclass
class _PCSEParameterPlottable:
    plotname: Optional[str] = None

    def __post_init__(self):
        # Uses the description as a default plotname; assumes a description exists
        if self.plotname is None:
            self.plotname = self.description

@dataclass
class PCSEFlag(_PCSEParameterBase):
    def __str__(self) -> str:
        return f"[FLAG] {self.name}: {self.description}"

@dataclass
class PCSELabel(_PCSEParameterPlottable, _PCSEParameterBase):
    def __str__(self) -> str:
        return str(self.plotname)

@dataclass
class PCSENumericParameter(_PCSEParameterPlottable, _PCSEParameterBase):
    unit: Optional[str] = None
    bounds: Iterable[Number] = None
    dtype: type = float

    def __str__(self) -> str:
        base = f"{self.name}: {self.plotname}"
        if self.unit is not None:
            base += f" [{self.unit}]"
        return base

@dataclass
class PCSEDateParameter(_PCSEParameterPlottable, _PCSEParameterBase):
    def __str__(self) -> str:
        return f"{self.name}: {self.plotname}"

@dataclass
class PCSETabularParameter(_PCSEParameterBase):
    x: str
    x_unit: Optional[str] = None
    unit: Optional[str] = None

    def __str__(self) -> str:
        base = f"[TABLE] {self.name}: {self.description}"
        if self.unit is not None:
            base += f" [{self.unit}]"
        base += f" as function of {self.x}"
        if self.x_unit is not None:
            base += f" [{self.x_unit}]"
        return base


PCSEParameter = PCSEFlag | PCSELabel | PCSENumericParameter | PCSETabularParameter
