"""
Read the BRP file in .gpkg format and process it to something more light-weight.
BRP files can be obtained from https://www.pdok.nl/atom-downloadservices/-/article/basisregistratie-gewaspercelen-brp-

Example usage:
    %run process_brp.py data/brp/brpgewaspercelen_definitief_2022.gpkg -vp
"""
import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Read the BRP file in .gpkg format and process it to something more light-weight.")
parser.add_argument("filename", help="file to be processed", type=fpcup.io.Path)
parser.add_argument("-r", "--results_dir", help="folder to save results to", type=fpcup.io.Path, default=fpcup.DEFAULT_RESULTS / "brp")
parser.add_argument("-p", "--plots", help="generate plots along the way", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Load the file
brp = fpcup.io.read_geodataframe(args.filename)
filestem = args.filename.stem
year = filestem.split("_")[-1]
if args.verbose:
    print(f"Loaded file {args.filename.absolute()} -- {len(brp)} entries")

# Remove unnecessary columns and translate to English
brp.drop(columns=["jaar", "status"], inplace=True)
brp.rename({"gewas": "crop", "gewascode": "crop_code"}, axis=1, inplace=True)
brp.replace({"category": fpcup.site.brp_categories_NL2EN}, inplace=True)

# Add province information
fpcup.geo.add_provinces(brp)
if args.verbose:
    print("Added province column")

# Add area column (in ha)
brp["area"] = brp.area * fpcup.constants.m2ha

# Show the distribution of plot types
if args.plots:
    fpcup.plotting.brp_histogram(brp, column="category", title=filestem, top5=False, saveto=args.results_dir/f"brp{year}-hist_categories.pdf")

    for province in fpcup.geo.iterate_over_locations():
        fpcup.plotting.brp_map_category(brp, province=province, saveto=args.results_dir/f"brp{year}-map_categories-{province.abbreviation}.pdf")

# Select cropland
brp_agro = brp.loc[brp["category"] == "cropland"].drop(columns=["category"])
if args.verbose:
    print(f"Reduced file to cropland only -- {len(brp_agro)} entries")

# Show the distribution of crop types
if args.plots:
    fpcup.plotting.brp_histogram(brp_agro, column="crop", figsize=(10, 5), title=filestem, log=True, usexticks=False, saveto=args.results_dir/f"brp{year}-hist_crops.pdf")

# Select fpcup crops
brp_fpcup = brp_agro.loc[brp_agro["crop"].isin(fpcup.crop.brp_crops_NL2EN)]
brp_fpcup.replace({"crop": fpcup.crop.brp_crops_NL2EN}, inplace=True)
if args.verbose:
    print(f"Reduced file to crops listed in the FPCUP/BRP dictionary -- {len(brp_fpcup)} entries")

# Add a column with the main categories
brp_fpcup["crop_species"] = brp_fpcup["crop"].apply(fpcup.crop.main_croptype)
if args.verbose:
    print("Added crop_species column")

# Show the distribution of crop types and species
if args.plots:
    fpcup.plotting.brp_histogram(brp_fpcup, column="crop", figsize=(7, 4), title=filestem, saveto=args.results_dir/f"brp{year}-hist_crops-filtered.pdf")
    fpcup.plotting.brp_histogram(brp_fpcup, column="crop_species", figsize=(3, 4), title=filestem, saveto=args.results_dir/f"brp{year}-hist_crops-filtered-combined.pdf")

# Show the distribution across the country
if args.plots:
    for province in fpcup.geo.iterate_over_locations():
        fpcup.plotting.brp_map_crop(brp_fpcup, province=province, saveto=args.results_dir/f"brp{year}-map_crops-filtered-{province.abbreviation}.pdf")

        fpcup.plotting.brp_crop_map_split(brp_fpcup, province=province, title=f"Selected crop types in {province} from {filestem}", saveto=args.results_dir/f"brp{year}-map_crops-individual-{province.abbreviation}.pdf")

# Add centroid coordinates in WGS84 for WOFOST
coordinates = brp_fpcup.centroid.to_crs("WGS84")
brp_fpcup["latitude"] = coordinates.y
brp_fpcup["longitude"] = coordinates.x
if args.verbose:
    print("Added WGS84 coordinates")

# Save the processed dataframe to file
brp_fpcup.to_file(f"brp{year}.gpkg", driver="GPKG")
if args.verbose:
    print("Saved processed file")
