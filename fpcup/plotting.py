"""
Functions for plotting data and results
"""
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt, patches as mpatches, ticker as mticker
from tqdm import tqdm

from ._brp_dictionary import brp_categories_colours, brp_crops_colours
from .model import parameter_names
from .province import nl_boundary, province_area, province_boundary

# @mticker.FuncFormatter
# def _capitalise_ticks(x, pos):
#     """
#     Helper function to capitalise str ticks in plots.
#     """
#     try:
#         return x.capitalize()
#     except AttributeError:
#         return "abc"
# _capitalise_ticks = mticker.StrMethodFormatter("abc{x:}")

def column_to_title(column: str) -> str:
    """
    Clean up a column name (e.g. "crop_species") so it can be used as a title (e.g. "Crop species").
    """
    return column.capitalize().replace("_", " ")

def brp_histogram(data: gpd.GeoDataFrame, column: str, figsize=(3, 5), usexticks=True, xlabel="Crop", title=None, top5=True, saveto=None, **kwargs) -> None:
    """
    Make a bar plot showing the distribution of plots/crops in BRP data.
    """
    counts = data[column].value_counts()
    areas = data.groupby([column])["area"].sum().reindex_like(counts)  # Area per group, unit [ha]

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=figsize, gridspec_kw={"hspace": 0.1})
    counts.plot.bar(ax=axs[0], color='w', edgecolor='k', hatch="//", **kwargs)
    areas.plot.bar(ax=axs[1], color='w', edgecolor='k', hatch="//", **kwargs)

    axs[0].tick_params(axis="x", bottom=False, labelbottom=False)
    if usexticks:
        # There is no cleaner way to do this because tick_params does not support horizontalalignment
        # Capitalisation: _capitalise_ticks method did not seem to work, its arguments are ints instead of str
        xticklabels = [label.get_text().capitalize() for label in axs[1].get_xticklabels()]
        axs[1].set_xticklabels(xticklabels, rotation=45, horizontalalignment="right")
    else:
        axs[1].tick_params(axis="x", bottom=False, labelbottom=False)

    if not "log" in kwargs:
        for ax in axs:
            # Prevent scientific notation
            ax.ticklabel_format(axis="y", style="plain")

            # Set ymin explicitly
            ax.set_ylim(ymin=0)

    axs[1].set_xlabel(xlabel)
    axs[0].set_ylabel("Number of plots")
    axs[1].set_ylabel("Total area [ha]")
    axs[0].set_title(title)
    fig.align_ylabels()

    if top5:
        float2str = lambda x: f"{x:.0f}"
        top5_text = [f"Top 5:\n{df.head().to_string(header=False, float_format=float2str)}" for df in (counts, areas)]
        for ax, text in zip(axs, top5_text):
            ax.text(0.99, 0.98, text, transform=ax.transAxes, horizontalalignment="right", verticalalignment="top")

    if saveto:
        plt.savefig(saveto, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

def brp_map(data: gpd.GeoDataFrame, column: str, province: str | None=None, figsize=(10, 10), title=None, rasterized=True, colour_dict=None, saveto=None, **kwargs) -> None:
    """
    Create a map of BRP polygons in the given column.
    If `province` is provided, only data within that province will be plotted, with the corresponding outline.
    """
    # Select province data if desired
    if province:
        assert "province" in data.columns, f"Cannot plot data by province - data do not have a 'province' column\n(columns: {data.columns}"
        province = province.title()
        data = data.loc[data["province"] == province]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # If colours are specified, plot those instead of the raw data, and add a legend
    if colour_dict:
        colours = data[column].map(colour_dict)
        data.plot(ax=ax, color=colours, rasterized=rasterized, **kwargs)

        # Generate dummy patches with the same colour mapping and add those to the legend
        colour_patches = [mpatches.Patch(color=colour, label=label.capitalize()) for label, colour in colour_dict.items() if label in data[column].unique()]
        ax.legend(handles=colour_patches, loc="lower right", fontsize=12, edgecolor="black", framealpha=1, title=column_to_title(column))

    # If colours are not specified, simply plot the data and let geopandas handle the colours
    else:
        data.plot(ax=ax, column=column, rasterized=rasterized, **kwargs)

    # Add a country/province outline
    if province:
        boundary = province_boundary[province]
    else:
        boundary = nl_boundary
    if boundary is not None:
        boundary.plot(ax=ax, color="black", lw=1)

    ax.set_title(title)
    ax.set_axis_off()
    ax.axis("equal")

    if saveto:
        plt.savefig(saveto, bbox_inches="tight", dpi=600)
    plt.show()
    plt.close()

def brp_crop_map_split(data: gpd.GeoDataFrame, column: str="crop_species", crops=brp_crops_colours.keys(), figsize=(10, 7), shape=(2, 4), title=None, rasterized=True, saveto=None, **kwargs) -> None:
    """
    Create a map of BRP polygons, with one panel per crop species.
    """
    # Create figure
    fig, axs = plt.subplots(*shape, figsize=figsize)

    for crop, ax in zip(crops, axs.ravel()):
        colour = brp_crops_colours[crop]
        data_here = data.loc[data[column] == crop]
        data_here.plot(ax=ax, color="black", rasterized=rasterized, **kwargs)

        if nl_boundary is not None:
            nl_boundary.plot(ax=ax, color="black", lw=0.2)

        ax.set_title(crop.capitalize(), color=colour)
        ax.set_axis_off()
        ax.axis("equal")

    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto, bbox_inches="tight", dpi=600)
    plt.show()
    plt.close()

def replace_year_in_datetime(date, newyear=2000):
    """
    For a datetime object yyyy-mm-dd, replace yyyy with newyear.
    Note that you may get errors if your data contain leap days (mm-dd = 02-29) but your chosen `newyear` was not a leap year.
    """
    return date.replace(year=newyear)

def plot_wofost_ensemble_results(outputs, keys=None, saveto=None, replace_years=True, show=True):
    """
    Plot WOFOST ensemble results.
    """
    # If no keys were specified, get all of them
    if keys is None:
        keys = outputs[0].keys()

    # Plot curves for outputs
    fig, axs = plt.subplots(nrows=len(keys), sharex=True, figsize=(8,10))

    for df in tqdm(outputs, total=len(outputs), desc="Plotting results", unit="runs"):
        # Remove the year information if desired, e.g. to compare year-by-year results
        if replace_years:
            time_axis = pd.to_datetime(df.index.to_series()).apply(replace_year_in_datetime)
        else:
            time_axis = pd.to_datetime(df.index.to_series())

        # Plot every key in the corresponding panel
        for ax, key in zip(axs, keys):
            ax.plot(time_axis, df[key], alpha=0.25)

    axs[-1].set_xlabel("Time")
    for ax, key in zip(axs, keys):
        ax.set_ylabel(key)
        ax.set_ylim(ymin=0)
        ax.text(1.00, 1.00, parameter_names[key], transform=ax.transAxes, horizontalalignment="right", verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "white"})
        ax.grid()
    fig.align_ylabels()
    axs[0].set_title(f"Results from {len(outputs)} WOFOST runs")

    if saveto is not None:
        fig.savefig(saveto, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
