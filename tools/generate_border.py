"""
Generate a plottable outer border for the Netherlands: coastline + BE border + DE border.
"""
from pathlib import Path
import geopandas as gpd
gpd.options.io_engine = "pyogrio"

CRS_AMERSFOORT = "EPSG:28992"

data_dir = Path("data/top10nl-2023")

# Remove GFS files if they exist - for some reason, these mess with MultiPolygons
gfs_files = data_dir.glob("*.gfs")
for gfs in gfs_files:
    gfs.unlink()  # Deletes the file
    print(f"Deleted GFS file: {gfs}")

# Load the raw data
country_row = slice(2, 3)  # Equivalent to [2:3]
country = gpd.read_file(data_dir/"top10nl_registratiefgebied.gml", rows=country_row)
print("Loaded land data")

water = gpd.read_file(data_dir/"top10nl_waterdeel.gml", columns=["gml_id", "lokaalID", "typeWater", "naamOfficieel", "geometry"])
print("Loaded water data")

# Select useful elements
country = country.iloc[0].geometry
sea = water.loc[water["typeWater"].isin(["zee", "droogvallend", "droogvallend (LAT)"])].unary_union

# Sort inland waters by size and select the biggest ones
inlandwaters = water.loc[water["typeWater"] == "meer, plas"]
inlandwaters["surface_area"] = inlandwaters.area
inlandwaters.sort_values("surface_area", inplace=True, ascending=False)

inlandwaters_big = inlandwaters.loc[inlandwaters["surface_area"] >= 5e6].unary_union  # >5 km²

# Subtract water areas
land = country - sea - inlandwaters_big

# Convert to a GeoSeries and save to file
land_series = gpd.GeoSeries(land, crs=CRS_AMERSFOORT)
land_series.to_file("NL_borders.geojson", driver="GeoJSON")
