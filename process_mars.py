"""
Process JRC MARS agro-meteorological data.

The following conversions need to take place to become PCSE-compatible:
    Precipitation: from mm/day to cm/day
    Wind speed: from wind speed at 10m to wind speed at 2m altitude
    Radiation: from kJ/m²/day to J/m²/day
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.interpolate import griddata

from matplotlib import pyplot as plt

from pcse.util import wind10to2

from fpcup.province import nl, CRS_AMERSFOORT

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

# Sort by coordinates
data.sort_values(["LONGITUDE", "LATITUDE"], inplace=True)

# Convert to GeoPandas
data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["LONGITUDE"], data["LATITUDE"], crs="WGS84"))

# Regrid: definitions
data.to_crs(CRS_AMERSFOORT, inplace=True)
xmin, xmax, ymin, ymax = data.total_bounds
cellsize = 250  # m
newx = np.arange(xmin, xmax+cellsize, cellsize)
newy = np.arange(ymin, ymax+cellsize, cellsize)
newx, newy = np.meshgrid(newx, newy)

# Regrid: example
key = "RADIATION"
day = data["DAY"].iloc[1]
data_day = data.loc[data["DAY"] == day]

gridded = griddata((data_day["geometry"].x, data_day["geometry"].y), data_day[key], (newx, newy), method="cubic")

# gridded_df = gpd.GeoDataFrame(gridded, geometry=gpd.points_from_xy(newx, newy, crs=CRS_AMERSFOORT))

# Regrid: plot
vmin, vmax = np.nanmin(gridded), np.nanmax(gridded)
fig, ax = plt.subplots(1, 1, figsize=(5,5))
im = ax.imshow(gridded, origin="lower", extent=(newx.min(), newx.max(), newy.min(), newy.max()), vmin=vmin, vmax=vmax)
data_day.plot(key, ax=ax, edgecolor="black", vmin=vmin, vmax=vmax)
nl.boundary.plot(ax=ax, color="black", lw=1)

fig.colorbar(im, ax=ax, label=key)

plt.show()
plt.close()
