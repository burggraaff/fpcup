"""
Generate a plottable outer border for the Netherlands and its constituents: coastline + BE border + DE border.

Example:
    python generate_basemaps.py -d data/top10nl-2023
"""
from pathlib import Path

import pandas as pd
import geopandas as gpd
gpd.options.io_engine = "pyogrio"
gpd.pd.options.mode.chained_assignment = None  # Prevents unneeded warnings


### SETUP
# Constants (cannot be imported from fpcup because it depends on the results from this script)
CRS_AMERSFOORT = "EPSG:28992"

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Generate a plottable outer border for the Netherlands and its constituents: coastline + BE border + DE border.")
parser.add_argument("-d", "--data_dir", help="folder containing basemap data", default="data/top10nl-2023", type=Path)
parser.add_argument("-o", "--output_filename", help="filename to save basemap dataframe to", default="basemaps.geojson", type=Path)
args = parser.parse_args()

# Remove GFS files if they exist - for some reason, these mess with MultiPolygons
gfs_files = args.data_dir.glob("*.gfs")
for gfs in gfs_files:
    gfs.unlink()  # Deletes the file
    print(f"Deleted GFS file: {gfs}")


### READ DATA
# Load the raw data
territories = gpd.read_file(args.data_dir/"top10nl_registratiefgebied.gml", columns=["typeRegistratiefGebied", "naamOfficieel", "geometry"])
print("Loaded land data")

water = gpd.read_file(args.data_dir/"top10nl_waterdeel.gml", columns=["typeWater", "naamOfficieel", "geometry"])
print("Loaded water data")

# Translate to simpler names in English
for data in (territories, water):
    data.set_crs(CRS_AMERSFOORT, inplace=True)
    data.rename(columns={"naamOfficieel": "name", "typeRegistratiefGebied": "level"}, inplace=True)

territories["level"].replace({"land": "country", "provincie": "province"}, inplace=True)

### SELECT AND EDIT WATER GEOMETRIES
# Select water bodies with a surface area only (no small streams)
water = water.loc[water.area > 0]
water["surface_area"] = water.area / 1e6  # km²
water.sort_values("surface_area", inplace=True, ascending=False)

# Sort waters by type and size and select the biggest ones
sea = water.loc[water["typeWater"].isin(["zee", "droogvallend", "droogvallend (LAT)"])].unary_union

inlandwaters = water.loc[water["typeWater"] == "meer, plas"]
inlandwaters_big = inlandwaters.loc[inlandwaters["surface_area"] >= 5].unary_union       # >5   km²
inlandwaters_medium = inlandwaters.loc[inlandwaters["surface_area"] >= 0.5].unary_union  # >0.5 km²

rivers = water.loc[water["typeWater"] == "waterloop"]
rivers_medium = rivers.loc[rivers["surface_area"] >= 0.5].unary_union  # >0.5 km²
rivers_big = rivers.iloc[:4].unary_union  # arbitrary selection: Haringvliet and surrounding waters

# Merge the waters into one object for easy subtraction
waters_big = sea.union(inlandwaters_big).union(rivers_big)
waters_detail = sea.union(inlandwaters_medium).union(rivers_medium)

print("Filtered water geometries")


### SELECT AND EDIT LAND GEOMETRIES
# Select useful land elements
provinces = territories.loc[territories["level"] == "province"]
country = territories.loc[territories["level"] == "country"]
country["name"].iloc[0] = "Netherlands"

print("Filtered land geometries")

# Save original areas as "coarse" versions, useful for faster selections
make_coarse = lambda s: s + "_coarse"
provinces_coarse = provinces.copy()
country_coarse = country.copy()
provinces_coarse["name"] = provinces["name"].apply(make_coarse)
country_coarse["name"] = country["name"].apply(make_coarse)

# Subtract water areas
country["geometry"] = country.difference(waters_big)
provinces["geometry"] = provinces.difference(waters_detail)

# Combine into one GeoDataFrame
basemaps = pd.concat([country, country_coarse, provinces, provinces_coarse])


### SAVE TO FILE
basemaps.to_file(args.output_filename, driver="GeoJSON")
print(f"Saved base maps to file: {args.output_filename.absolute()}")
