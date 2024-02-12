"""
Combines abstract base classes from various places and generates some useful aliases.
"""
from numbers import Number, Real as RealNumber
from os import PathLike
from typing import Callable, Iterable, Optional, Type

Coordinates = tuple[RealNumber, RealNumber]
PathOrStr = PathLike | str
StringDict = dict[str, str]
