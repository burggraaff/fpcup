"""
Weather-related stuff: load data etc
"""
from functools import cache
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pcse.base import WeatherDataProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import CABOWeatherDataProvider, CSVWeatherDataProvider, ExcelWeatherDataProvider
from pcse import settings as pcse_settings
PCSE_METEO_CACHE_DIR = Path(pcse_settings.METEO_CACHE_DIR)

from .settings import DEFAULT_DATA
from ._typing import Coordinates, Iterable

def load_example_Excel(filename=DEFAULT_DATA/"meteo"/"nl1.xlsx") -> ExcelWeatherDataProvider:
    """
    Load the example Excel weather file provided in the PCSE notebook repository.
    For testing purposes.
    """
    return ExcelWeatherDataProvider(filename)

def load_example_csv(filename=DEFAULT_DATA/"meteo"/"nl1.csv") -> CSVWeatherDataProvider:
    """
    Load the example CSV weather file provided in the PCSE repository documentation.
    For testing purposes.
    """
    return CSVWeatherDataProvider(filename)

def load_weather_data_NASAPower(coordinates: Coordinates | Iterable[Coordinates], return_single=True, progressbar=True, leave_progressbar=False, **kwargs) -> NASAPowerWeatherDataProvider | list[NASAPowerWeatherDataProvider]:
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
    coordinates_tqdm = tqdm(coordinates, total=n, desc="Fetching weather data", unit="sites", disable=not progressbar, leave=leave_progressbar)
    weather_data = [NASAPowerWeatherDataProvider(latitude=lat, longitude=long, **kwargs) for lat, long in coordinates_tqdm]

    # If there was only a single output, provide a single output
    if return_single and len(weather_data) == 1:
        weather_data = weather_data[0]

    return weather_data
