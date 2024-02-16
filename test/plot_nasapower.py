"""
Plot the NASAPowerWeatherDataProvider cache.
"""
from pathlib import Path

import geopandas as gpd
from matplotlib import pyplot as plt
from shapely import Point

import fpcup

# Load the data
cachedir = fpcup.weather.PCSE_METEO_CACHE_DIR
filenames = sorted(cachedir.glob("NASAPowerWeatherDataProvider*.cache"))

# Basic statistics
print(f"Number of cache files: {len(filenames)}")

# Convert to geometry
def latlon_from_path(filename: Path) -> tuple[float, float]:
    latlon_str = filename.stem.split("_")[1:]
    lat, lon = [float(s[3:])/10 for s in latlon_str]
    return lat, lon

latitudes, longitudes = zip(*[latlon_from_path(f) for f in filenames])
geometry = [Point(lon, lat) for lat, lon in zip(latitudes, longitudes)]

df = gpd.GeoDataFrame({"filename": filenames}, geometry=geometry, crs=fpcup.constants.WGS84)

# Plot the data with an outline of the Netherlands
# Do it once with no limits (i.e. showing all data) and once with only areas near NL
for usebounds, crs in zip((False, True), [fpcup.constants.WGS84, fpcup.constants.CRS_AMERSFOORT]):
    fig, ax = plt.subplots(figsize=(10, 10))

    df.to_crs(crs).plot(ax=ax)
    fpcup.plotting.nl_boundary.to_crs(crs).plot(ax=ax, color="k")

    ax.set_xlabel(f"Longitude ({crs})")
    ax.set_ylabel(f"Latitude ({crs})")

    if usebounds:
        bounds = fpcup.plotting.nl_boundary.to_crs(crs).bounds.iloc[0]
        ax.set_xlim(bounds.minx, bounds.maxx)
        ax.set_ylim(bounds.miny, bounds.maxy)

    plt.show()
    plt.close()
