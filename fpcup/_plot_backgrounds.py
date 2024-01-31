"""
(Try to) load map backgrounds from file so they can be plotted.
"""
import geopandas as gpd

from .settings import DEFAULT_DATA

CRS_AMERSFOORT = "EPSG:28992"

# Load the outline of the Netherlands
try:
    nl = gpd.read_file(DEFAULT_DATA/"NL_borders.geojson")
except IOError:
    nl = None
    nl_boundary = None
else:
    nl_boundary = nl.boundary
