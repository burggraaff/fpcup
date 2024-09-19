"""
Functions for plotting data/results relating to sensitivity studies.
"""
import pandas as pd

from matplotlib import pyplot as plt

from ..model import GeoSummary
from ..parameters import PCSEParameter, all_parameters, pcse_summary_outputs
from ..typing import Optional, PathOrStr

### CONSTANTS
OUTPUT_LABELS = ("DOM", "RD", "LAIMAX", "TAGP", "TWSO")  # Parameters to plot
OUTPUT_PARAMETERS = [pcse_summary_outputs[key] for key in OUTPUT_LABELS]


### HELPER FUNCTIONS
def number_of_replicates(data: pd.DataFrame, column: str) -> int:
    """
    Determine the number of replicates in a given column, e.g. number_of_replicates(data, column='geometry') is the number of sites.
    """
    return len(data[column].unique())


def sensitivity_one_crop(summary_by_crop: GeoSummary, crop_name: str, parameter: str | PCSEParameter, *,
                         saveto: Optional[PathOrStr]=None) -> None:
    """
    For a summary table for one crop, plot the sensitivity of certain output parameters to the given input parameter.
    """
    if not isinstance(parameter, PCSEParameter):
        parameter = all_parameters[parameter]

    # Determine number of runs
    n_sites = number_of_replicates(summary_by_crop, "geometry")
    n_soiltypes = number_of_replicates(summary_by_crop, "soiltype")
    n_parameter_values = number_of_replicates(summary_by_crop, parameter.name)

    # Setup
    fig, axs = plt.subplots(nrows=len(OUTPUT_PARAMETERS), ncols=n_soiltypes, sharex=True, sharey="row", figsize=(3*n_soiltypes, 10), squeeze=False)

    # Loop over the columns (soil types) first
    for ax_col, (soiltype, summary_by_soiltype) in zip(axs.T, summary_by_crop.groupby("soiltype")):
        ax_col[0].set_title(f"Soil type: {soiltype}")

        # Loop over the rows (summary outputs) next
        for ax, output in zip(ax_col, OUTPUT_PARAMETERS):
            # Plot a line for each site
            summary_by_soiltype.groupby("geometry").plot.line(parameter.name, output.name, ax=ax, alpha=0.5, legend=False)

    # Add reference line for default value, if available
    try:
        for ax in axs.ravel():
            ax.axvline(parameter.default, color="black", linestyle="--", alpha=0.5)
    except (TypeError, AttributeError):
        pass

    # Titles / labels
    try:
        xlim = parameter.bounds
    except AttributeError:
        xlim = summary_by_crop[parameter.name].min(), summary_by_crop[parameter.name].max()
    axs[0, 0].set_xlim(*xlim)

    for ax in axs[1:, 0]:
        ax.set_ylim(ymin=0)

    for ax, output in zip(axs[:, 0], OUTPUT_PARAMETERS):
        ax.set_ylabel(output.name)

    fig.suptitle(f"WOFOST sensitivity to {parameter.name}: {crop_name} ({n_sites} sites, {n_parameter_values} values)\n{parameter}")
    fig.align_xlabels()
    fig.align_ylabels()

    # Save figure
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
