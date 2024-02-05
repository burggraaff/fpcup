"""
Process JRC MARS agro-meteorological data.

The following conversions need to take place to become PCSE-compatible:
    Precipitation: from mm/day to cm/day
    Wind speed: from wind speed at 10m to wind speed at 2m altitude
    Radiation: from kJ/m²/day to J/m²/day
"""
from datetime import date
from pathlib import Path

import pandas as pd

from pcse.util import wind10to2

filename = Path("data/meteo/NL-weather_ver2024_01_28946_162168065.csv")

# Define converters
convert_day = date.fromisoformat
convert_windspeed = lambda wind10: wind10to2(float(wind10))  # from 10 m to 2 m altitude
convert_precipitation = lambda precip: float(precip)/10  # mm/day -> cm/day
convert_radiation = lambda rad: float(rad)*1000  # kJ/m²/day -> J/m²/day

convert_all = {"DAY": convert_day, "WINDSPEED": convert_windspeed, "PRECIPITATION": convert_precipitation, "RADIATION": convert_radiation}

# Read the data
data = pd.read_csv(filename, sep=";", converters=convert_all)
print(f"Read data from {filename.absolute()}")
