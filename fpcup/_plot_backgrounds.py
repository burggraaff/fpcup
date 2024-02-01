"""
(Try to) load map backgrounds from file so they can be plotted.
"""
import geopandas as gpd
from pyogrio.errors import DataSourceError

from .settings import DEFAULT_DATA

CRS_AMERSFOORT = "EPSG:28992"

# Load the outline of the Netherlands
try:
    nl = gpd.read_file(DEFAULT_DATA/"NL_borders.geojson")
except DataSourceError:
    nl = None
    nl_boundary = None
else:
    nl_boundary = nl.boundary

# Load the provinces
try:
    provinces = gpd.read_file(DEFAULT_DATA/"NL_provinces.geojson")
except DataSourceError:
    provinces = None

# Access individual provinces using a dictionary, e.g. province_boundary["Zuid-Holland"]
province_area = {name: poly for name, poly in zip(provinces["naamOfficieel"], provinces["geometry"])}
province_boundary = {name: gpd.GeoSeries(outline) for name, outline in zip(provinces["naamOfficieel"], provinces.boundary)}
