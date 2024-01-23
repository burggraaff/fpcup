"""
Read the BRP file in .gpkg format and process it to something more light-weight.
BRP files can be obtained from https://www.pdok.nl/atom-downloadservices/-/article/basisregistratie-gewaspercelen-brp-

Example usage:
    %run process_brp.py data/brp/brpgewaspercelen_definitief_2022.gpkg
"""
import geopandas as gpd
gpd.pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Read the BRP file in .gpkg format and process it to something more light-weight.")
parser.add_argument("filename", help="file to be processed", type=fpcup.io.Path)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-p", "--plots", help="generate plots along the way", action="store_true")
args = parser.parse_args()

# Load the file
brp = gpd.read_file(args.filename)
filestem = args.filename.stem
if args.verbose:
    print(f"Loaded file {args.filename.absolute()} - {len(brp)} entries")

# Remove unnecessary columns
brp.drop(columns=["jaar", "status"], inplace=True)

def plot_crop_distribution(data: gpd.GeoDataFrame, figsize=(3, 2), usexticks=True, xlabel="Crop", ylabel="Number of plots", title=None, top5=True, **kwargs):
    """
    Make a bar plot showing the distribution of plots/crops in BRP data.
    """
    counts = data.value_counts()

    plt.figure(figsize=figsize)
    counts.plot.bar(color='w', edgecolor='k', hatch="//", **kwargs)

    if usexticks:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.tick_params(axis="x", bottom=False, labelbottom=False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if top5:
        top5_text = f"Top 5:\n{counts.head().to_string(header=False)}"
        plt.text(0.99, 0.98, top5_text, transform=plt.gca().transAxes, horizontalalignment="right", verticalalignment="top")

    plt.show()
    plt.close()

# Show the distribution of plot types
if args.plots:
    plot_crop_distribution(brp["category"], xlabel="Category", title=filestem, top5=False)

# Select cropland
brp_agro = brp.loc[brp["category"] == "Bouwland"].drop(columns=["category"])
if args.verbose:
    print(f"Reduced file to agricultural land only - {len(brp_agro)} entries")

# Show the distribution of crop types
if args.plots:
    plot_crop_distribution(brp_agro["gewas"], figsize=(10, 2), title=filestem, log=True, usexticks=False)

# Select fpcup crops
brp_fpcup = brp_agro.loc[brp_agro["gewas"].isin(fpcup.crop.brp_dictionary)]
brp_fpcup["gewas"].replace(fpcup.crop.brp_dictionary, inplace=True)
if args.verbose:
    print(f"Reduced file to crops listed in the FPCUP/BRP dictionary - {len(brp_fpcup)} entries")

# Show the distribution of crop types
if args.plots:
    plot_crop_distribution(brp_fpcup["gewas"], figsize=(7, 2), title=filestem)

# Process polygons
p = brp_fpcup["geometry"].iloc[0]
