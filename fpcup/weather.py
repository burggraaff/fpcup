"""
Weather-related stuff: load data etc
"""
from functools import cache
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from pcse.base import WeatherDataProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import CABOWeatherDataProvider, CSVWeatherDataProvider, ExcelWeatherDataProvider

from .settings import DEFAULT_DATA

def load_example_Excel(filename=DEFAULT_DATA/"meteo"/"nl1.xlsx") -> ExcelWeatherDataProvider:
    """
    Load the example Excel weather file provided in the PCSE notebook repository.
    For testing purposes.
    """
    return ExcelWeatherDataProvider(filename)

# @cache
def _load_weather_data_NASAPower_cache(latitude: float, longitude: float, **kwargs) -> NASAPowerWeatherDataProvider:
    """
    Load weather data from the NASA Power database using PCSE's NASAPowerWeatherDataProvider method.
    Cached to speed up duplicate calls (particularly useful when running/debugging in interactive mode).

    Returns a single NASAPowerWeatherDataProvider object.
    """
    weather_data = NASAPowerWeatherDataProvider(latitude=latitude, longitude=longitude, **kwargs)
    return weather_data

def load_weather_data_NASAPower(coordinates: tuple[float] | Iterable[tuple[float]], return_single=True, progressbar=True, leave_progressbar=False, **kwargs) -> NASAPowerWeatherDataProvider | list[NASAPowerWeatherDataProvider]:
    """
    Load weather data from the NASA Power database using PCSE's NASAPowerWeatherDataProvider method.

    Coordinates must be (latitude, longitude) in that order.
    If a single (latitude, longitude) pair is provided and return_single=True, returns a single NASAPowerWeatherDataProvider object.
    If a single (latitude, longitude) pair is provided and return_single=False, returns a list containing a single NASAPowerWeatherDataProvider object.

    If multiple latitudes and/or longitudes are provided, return a list of NASAPowerWeatherDataProvider objects.
    """
    # Check if multiple coordinate pairs were provided - if not, make them into a single-element iterable
    if isinstance(coordinates, tuple) and not isinstance(coordinates[0], tuple):
        coordinates = [coordinates]
        progressbar = False

    # Try to determine the number of inputs for the progressbar - does not work if coordinates is a generator
    try:
        n = len(coordinates)
    except TypeError:
        n = None

    # Do the actual loading
    weather_data = [_load_weather_data_NASAPower_cache(latitude=lat, longitude=long, **kwargs) for lat, long in tqdm(coordinates, total=n, desc="Fetching weather data", unit="sites", disable=not progressbar, leave=leave_progressbar)]

    # If there was only a single output, provide a single output
    if return_single and len(weather_data) == 1:
        weather_data = weather_data[0]

    return weather_data
