"""
Analyse a PCSE ensemble with one varying parameter, as generated by wofost_ensemble_parameters.py.

Example:
    %run plot_ensemble_parameter_single.py outputs/RDMSOL -v
"""
from matplotlib import pyplot as plt
from tqdm import tqdm

import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Analyse a PCSE ensemble with one varying parameter, as generated by wofost_ensemble_parameters.py.")
parser.add_argument("output_dir", help="folder to load PCSE outputs from", type=fpcup.io.Path)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

args.PARAMETER_NAME = args.output_dir.stem
args.PARAMETER = fpcup.parameters.all_parameters[args.PARAMETER_NAME]


### CONSTANTS
OUTPUT_LABELS = ("RD", "DOM", "LAIMAX", "TAGP", "TWSO")  # Parameters to plot
OUTPUT_PARAMETERS = [fpcup.parameters.pcse_summary_outputs[key] for key in OUTPUT_LABELS]


### HELPER FUNCTIONS
def number_of_replicates(data: fpcup.geo.gpd.GeoDataFrame, column: str) -> int:
    """
    Determine the number of replicates in a given column, e.g. number_of_replicates(data, column='geometry') is the number of sites.
    """
    return len(data[column].unique())


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()

    ### SETUP
    # Load the ensemble summary
    inputsummary, summary = fpcup.io.load_ensemble_summary_from_folder(args.output_dir)
    if args.verbose:
        print(f"Loaded input summary, summary from {args.output_dir.absolute()}")

    # Join
    inputsummary.drop(columns=["geometry"], inplace=True)
    summary = summary.join(inputsummary)
    summary.sort_values(args.PARAMETER_NAME, inplace=True)
    if args.verbose:
        print("Joined input/output summary tables")

    # Determine number of runs
    n_crops = number_of_replicates(summary, "crop")


    ### PLOTTING
    # Loop over the crops and generate a figure for each
    for (crop_name, summary_by_crop) in tqdm(summary.groupby("crop"), desc="Plotting figures", unit="crop", leave=args.verbose):
        crop_short = fpcup.crop.CROP2ABBREVIATION[crop_name]

        # Determine number of runs
        n_sites = number_of_replicates(summary_by_crop, "geometry")
        n_soiltypes = number_of_replicates(summary_by_crop, "soiltype")
        n_parameter_values = number_of_replicates(summary_by_crop, args.PARAMETER_NAME)

        # Setup
        fig, axs = plt.subplots(nrows=len(OUTPUT_PARAMETERS), ncols=n_soiltypes, sharex=True, sharey="row", figsize=(10, 10), squeeze=False)

        # Loop over the columns (soil types) first
        for ax_col, (soiltype, summary_by_soiltype) in zip(axs.T, summary_by_crop.groupby("soiltype")):
            ax_col[0].set_title(f"Soil type: {soiltype}")

            # Loop over the rows (summary outputs) next
            for ax, output in zip(ax_col, OUTPUT_PARAMETERS):
                # Plot a line for each site
                summary_by_soiltype.groupby("geometry").plot.line(args.PARAMETER_NAME, output.name, ax=ax, alpha=0.5, legend=False)

        # Add reference line for default value, if available
        try:
            for ax in axs.ravel():
                ax.axvline(args.PARAMETER.default, color="black", linestyle="--", alpha=0.5)
        except (TypeError, AttributeError):
            pass

        # Titles / labels
        try:
            xlim = args.PARAMETER.bounds
        except AttributeError:
            xlim = summary_by_crop[args.PARAMETER_NAME].min(), summary_by_crop[args.PARAMETER_NAME].max()
        axs[0, 0].set_xlim(*xlim)

        for ax, output in zip(axs[:, 0], OUTPUT_PARAMETERS):
            ax.set_ylabel(output.name)

        fig.suptitle(f"WOFOST sensitivity to {args.PARAMETER_NAME}: {crop_name} ({n_sites} sites, {n_parameter_values} values)\n{args.PARAMETER}")
        fig.align_xlabels()
        fig.align_ylabels()

        # Save figure
        plt.savefig(fpcup.DEFAULT_RESULTS/f"{args.PARAMETER_NAME}-{crop_short}.pdf", bbox_inches="tight")
        plt.close()
