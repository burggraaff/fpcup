"""
Read the BRP file in .gpkg format and process it to something more light-weight.
BRP files can be obtained from https://www.pdok.nl/atom-downloadservices/-/article/basisregistratie-gewaspercelen-brp-

Example usage:
    %run process_brp.py data/brp/brpgewaspercelen_definitief_2022.gpkg
"""
import geopandas as gpd
gpd.pd.options.mode.chained_assignment = None  # Prevents unneeded warnings

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Read the BRP file in .gpkg format and process it to something more light-weight.")
parser.add_argument("filename", help="file to be processed", type=fpcup.io.Path)
parser.add_argument("-r", "--results_dir", help="folder to save results to", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_RESULTS / "brp")
parser.add_argument("-p", "--plots", help="generate plots along the way", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Load the file
brp = gpd.read_file(args.filename)
filestem = args.filename.stem
if args.verbose:
    print(f"Loaded file {args.filename.absolute()} -- {len(brp)} entries")

# Remove unnecessary columns and translate to English
brp.drop(columns=["jaar", "status"], inplace=True)
brp.rename({"gewas": "crop", "gewascode": "crop_code"})
brp["category"].replace(fpcup.site.brp_categories_NL2EN, inplace=True)

# Add area column (in ha)
brp["area"] = brp.area * fpcup.constants.m2ha

# Show the distribution of plot types
if args.plots:
    fpcup.plotting.brp_histogram(brp, column="category", xlabel="Category", title=filestem, top5=False, saveto=args.results_dir/"brp-hist_categories.pdf")
    fpcup.plotting.brp_map(brp, column="category", title=f"Land usage\n{filestem}", colour_dict=fpcup.plotting.brp_categories_colours, saveto=args.results_dir/"brp-map_categories.pdf")

# Select cropland
brp_agro = brp.loc[brp["category"] == "Arable land"].drop(columns=["category"])
if args.verbose:
    print(f"Reduced file to agricultural land only -- {len(brp_agro)} entries")

# Show the distribution of crop types
if args.plots:
    fpcup.plotting.brp_histogram(brp_agro, column="crop", figsize=(10, 5), title=filestem, log=True, usexticks=False, saveto=args.results_dir/"brp-hist_crops.pdf")

# Select fpcup crops
brp_fpcup = brp_agro.loc[brp_agro["crop"].isin(fpcup.crop.brp_crops_NL2EN)]
brp_fpcup["crop"].replace(fpcup.crop.brp_crops_NL2EN, inplace=True)
if args.verbose:
    print(f"Reduced file to crops listed in the FPCUP/BRP dictionary -- {len(brp_fpcup)} entries")

# Add a column with the main categories
brp_fpcup["crop_species"] = brp_fpcup["crop"].apply(fpcup.crop.main_croptype)

# Add province information (this takes a while)
fpcup._plot_backgrounds.add_provinces(brp_fpcup)

# Show the distribution of crop types and species
if args.plots:
    fpcup.plotting.brp_histogram(brp_fpcup, column="crop", figsize=(7, 4), title=filestem, saveto=args.results_dir/"brp-hist_crops-filtered.pdf")
    fpcup.plotting.brp_histogram(brp_fpcup, column="crop_species", figsize=(3, 4), title=filestem, saveto=args.results_dir/"brp-hist_crops-filtered-combined.pdf")

# Show the distribution across the country
if args.plots:
    fpcup.plotting.brp_map(brp_fpcup, column="crop_species", title=f"Selected crop types\n{filestem}", colour_dict=fpcup.plotting.brp_crops_colours, saveto=args.results_dir/"brp-map_crops-filtered.pdf")

    fpcup.plotting.brp_map(brp_fpcup, column="crop_species", province="Zuid-Holland", title=f"Selected crop types\n{filestem}", colour_dict=fpcup.plotting.brp_crops_colours, saveto=args.results_dir/"brp-map_crops-filtered-zh.pdf")

    fpcup.plotting.brp_crop_map_split(brp_fpcup, column="crop_species", title=f"Selected crop types\n{filestem}", saveto=args.results_dir/"brp-map_crops-individual.pdf")

# Add centroid coordinates in WGS84 for WOFOST
coordinates = brp_fpcup.centroid.to_crs("WGS84")
brp_fpcup["latitude"] = coordinates.y
brp_fpcup["longitude"] = coordinates.x

# Save the processed dataframe to file
brp_fpcup.to_file(f"{filestem}-processed.gpkg", driver="GPKG")
