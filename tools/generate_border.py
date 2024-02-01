"""
Generate a plottable outer border for the Netherlands: coastline + BE border + DE border.
"""
from pathlib import Path
import geopandas as gpd
gpd.options.io_engine = "pyogrio"
gpd.pd.options.mode.chained_assignment = None  # Prevents unneeded warnings

CRS_AMERSFOORT = "EPSG:28992"

data_dir = Path("data/top10nl-2023")

# Remove GFS files if they exist - for some reason, these mess with MultiPolygons
gfs_files = data_dir.glob("*.gfs")
for gfs in gfs_files:
    gfs.unlink()  # Deletes the file
    print(f"Deleted GFS file: {gfs}")

# Load the raw data
territories = gpd.read_file(data_dir/"top10nl_registratiefgebied.gml", columns=["gml_id", "lokaalID", "typeRegistratiefGebied", "naamOfficieel", "geometry"])
print("Loaded land data")

water = gpd.read_file(data_dir/"top10nl_waterdeel.gml", columns=["gml_id", "lokaalID", "typeWater", "naamOfficieel", "geometry"])
water.set_crs(CRS_AMERSFOORT, inplace=True)
print("Loaded water data")

# Select water bodies with a surface area only (no small streams)
water = water.loc[water.area > 0]
water["surface_area"] = water.area / 1e6  # km²

# Select useful elements
provinces = territories.loc[territories["typeRegistratiefGebied"] == "provincie"]
country = territories.loc[territories["typeRegistratiefGebied"] == "land"].iloc[0].geometry

sea = water.loc[water["typeWater"].isin(["zee", "droogvallend", "droogvallend (LAT)"])].unary_union

# Sort inland waters by size and select the biggest ones
inlandwaters = water.loc[water["typeWater"] == "meer, plas"]
inlandwaters.sort_values("surface_area", inplace=True, ascending=False)

inlandwaters_big = inlandwaters.loc[inlandwaters["surface_area"] >= 5].unary_union  # >5 km²
inlandwaters_medium = inlandwaters.loc[inlandwaters["surface_area"] >= 0.5].unary_union  # >0.5 km²

rivers = water.loc[water["typeWater"] == "waterloop"]
rivers.sort_values("surface_area", inplace=True, ascending=False)
rivers_big = rivers.loc[rivers["surface_area"] >= 0.5].unary_union  # >0.5 km²

# Merge the waters into one object for easy subtraction
waters_big = sea.union(inlandwaters_big)
waters_detail = sea.union(inlandwaters_medium).union(rivers_big)

# Subtract water areas
land = country.difference(waters_big)
provinces["geometry"] = provinces["geometry"].difference(waters_detail)

# Convert to a GeoSeries and save to file
land_series = gpd.GeoSeries(land, crs=CRS_AMERSFOORT)
land_series.to_file("NL_borders.geojson", driver="GeoJSON")
print("Saved land area")

provinces.set_crs(CRS_AMERSFOORT, inplace=True)
provinces.to_file("NL_provinces.geojson", driver="GeoJSON")
print("Saved provinces")
