"""
(Try to) load map backgrounds from file so they can be plotted.
"""
import geopandas as gpd

from .settings import DEFAULT_DATA

# Load the outline of the Netherlands
try:
    nl = gpd.read_file(DEFAULT_DATA/"BestuurlijkeGebieden_2024.gpkg")
except IOError:
    nl = None
    nl_boundary = None
else:
    nl_boundary = gpd.GeoSeries(nl.unary_union.boundary, crs=nl.crs)

# Load the coastline
try:
    sea = gpd.read_file(DEFAULT_DATA/"searegions.gml").to_crs(nl.crs)
except (IOError, AttributeError):
    sea = None
    coastline = None
else:
    coastline = gpd.GeoSeries(sea.unary_union, crs=sea.crs)
