"""
Weather-related stuff: load data etc
"""
from functools import cache
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from pcse.db import NASAPowerWeatherDataProvider

@cache
def _load_weather_data_NASAPower_cache(latitude: float, longitude: float, **kwargs) -> NASAPowerWeatherDataProvider:
    """
    Load weather data from the NASA Power database using PCSE's NASAPowerWeatherDataProvider method.
    Cached to speed up duplicate calls (particularly useful when running/debugging in interactive mode).

    Returns a single NASAPowerWeatherDataProvider object.
    """
    weather_data = NASAPowerWeatherDataProvider(latitude=latitude, longitude=longitude, **kwargs)
    return weather_data

def load_weather_data_NASAPower(latitude: float | Iterable[float], longitude: float | Iterable[float], return_single=True, **kwargs) -> NASAPowerWeatherDataProvider | list[NASAPowerWeatherDataProvider]:
    """
    Load weather data from the NASA Power database using PCSE's NASAPowerWeatherDataProvider method.

    If a single (latitude, longitude) pair is provided and return_single=True, returns a single NASAPowerWeatherDataProvider object.
    If a single (latitude, longitude) pair is provided and return_single=False, returns a list containing a single NASAPowerWeatherDataProvider object.

    If multiple latitudes and/or longitudes are provided, return a list of NASAPowerWeatherDataProvider objects.
    """
    # Check if the inputs are iterables - if not, make them into single-element iterables
    if not isinstance(latitude, Iterable):
        latitude = [latitude]
    if not isinstance(longitude, Iterable):
        longitude = [longitude]

    # Try to determine the number of inputs for the progressbar - does not work if they are generators
    try:
        n = len(latitude) * len(longitude)
    except TypeError:
        n = None

    # Generate coordinate pairs
    coordinates = product(latitude, longitude)

    # Do the actual loading
    weather_data = [_load_weather_data_NASAPower_cache(latitude=lat, longitude=long, **kwargs) for lat, long in tqdm(coordinates, total=n, desc="Fetching weather data", unit="sites")]

    # If there was only a single output, provide a single output
    if return_single and len(weather_data) == 1:
        weather_data = weather_data[0]

    return weather_data
