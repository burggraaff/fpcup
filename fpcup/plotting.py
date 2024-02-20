"""
Functions for plotting data and results
"""
import datetime as dt

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tqdm import tqdm

from matplotlib import pyplot as plt, patches as mpatches, ticker as mticker, dates as mdates
from matplotlib import rcParams
rcParams.update({"axes.grid": True, "figure.dpi": 600, "grid.linestyle": "--", "hist.bins": 15, "legend.edgecolor": "black", "legend.framealpha": 1, "savefig.dpi": 600})
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._brp_dictionary import brp_categories_colours, brp_crops_colours
from ._typing import Iterable, Optional, PathOrStr, StringDict
from .model import Summary, parameter_names
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

def plot_outline(ax: plt.Axes, province: str="All", **kwargs) -> None:
    """
    Plot an outline of the Netherlands ("All") or a specific province (e.g. "Zuid-Holland").
    """
    if province == "All":
        boundary = nl_boundary
    else:
        boundary = province_boundary[province]

    line_kw = {"color": "black", "lw": 1, **kwargs}
    boundary.plot(ax=ax, **line_kw)

def column_to_title(column: str) -> str:
    """
    Clean up a column name (e.g. "crop_species") so it can be used as a title (e.g. "Crop species").
    """
    return column.capitalize().replace("_", " ")

def brp_histogram(data: gpd.GeoDataFrame, column: str, *,
                  figsize=(3, 5), usexticks=True, xlabel: Optional[str]="Crop", title: Optional[str]=None, top5=True, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
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
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

def brp_map(data: gpd.GeoDataFrame, column: str, *,
            province: Optional[str]="All", figsize=(10, 10), title: Optional[str]=None, rasterized=True, colour_dict: Optional[StringDict]=None, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a map of BRP polygons in the given column.
    If `province` is provided, only data within that province will be plotted, with the corresponding outline.
    """
    # Select province data if desired
    if province != "All":
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
        ax.legend(handles=colour_patches, loc="lower right", fontsize=12, title=column_to_title(column))

    # If colours are not specified, simply plot the data and let geopandas handle the colours
    else:
        data.plot(ax=ax, column=column, rasterized=rasterized, **kwargs)

    # Add a country/province outline
    plot_outline(ax, province)

    ax.set_title(title)
    ax.set_axis_off()
    ax.axis("equal")

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

def brp_crop_map_split(data: gpd.GeoDataFrame, column: str="crop_species", *,
                       crops: Iterable[str]=brp_crops_colours.keys(), figsize=(14, 3.5), shape=(1, 5), title: Optional[str]=None, rasterized=True, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a map of BRP polygons, with one panel per crop species.
    Shape is (nrows, ncols).
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
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

def replace_year_in_datetime(date: dt.date, newyear: int=2000) -> dt.date:
    """
    For a datetime object yyyy-mm-dd, replace yyyy with newyear.
    Note that you may get errors if your data contain leap days (mm-dd = 02-29) but your chosen `newyear` was not a leap year.
    """
    return date.replace(year=newyear)

def plot_wofost_ensemble_results(outputs: Iterable[pd.DataFrame], keys: Iterable[str]=None, *,
                                 title: Optional[str]=None, saveto: Optional[PathOrStr]=None, replace_years=True, progressbar=True, leave_progressbar=False) -> None:
    """
    Plot WOFOST ensemble results.
    """
    # If no keys were specified, get all of them
    if keys is None:
        keys = outputs[0].keys()

    # Plot curves for outputs
    fig, axs = plt.subplots(nrows=len(keys), sharex=True, figsize=(8,10))

    for df in tqdm(outputs, total=len(outputs), desc="Plotting outputs", unit="runs", disable=not progressbar, leave=leave_progressbar):
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

    fig.align_ylabels()

    if title is None:
        f"Results from {len(outputs)} WOFOST runs"
    axs[0].set_title(title)

    if saveto is not None:
        fig.savefig(saveto, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

def _numerical_or_date_bins(column: pd.Series) -> int | pd.DatetimeIndex:
    """
    Generate bins for a column based on its data type.
    """
    if is_datetime(column):
        return pd.date_range(column.min() - pd.Timedelta(hours=12), column.max() + pd.Timedelta(hours=12))
    else:
        return rcParams["hist.bins"]

def plot_wofost_ensemble_summary(summary: Summary, keys: Iterable[str]=None, *,
                                 title: Optional[str]=None, province: Optional[str]="All", saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot WOFOST ensemble results.
    """
    # If no keys were specified, get all of them
    if keys is None:
        keys = summary.keys()

    # Create figure and panels
    fig, axs = plt.subplots(nrows=2, ncols=len(keys), sharey="row", figsize=(15, 5), gridspec_kw={"hspace": 0.25, "height_ratios": [1, 1.5]})

    # Determine some parameters before the loop
    summary_cols = summary[keys]
    vmin, vmax = summary_cols.min(), summary_cols.max()
    bins = summary_cols.apply(_numerical_or_date_bins)

    # Loop over keys
    for ax_col, key in zip(axs.T, keys):
        column = summary[key]

        # First row: histograms
        if is_datetime(column):
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax_col[0].xaxis.set_major_locator(locator)
            ax_col[0].xaxis.set_major_formatter(formatter)

        column.plot.hist(ax=ax_col[0], bins=bins[key])
        ax_col[0].set_title(key)
        ax_col[0].set_xlim(vmin[key], vmax[key])

        # Second row: maps
        if is_datetime(column):
            column = column.apply(mdates.date2num)
            vmin_here, vmax_here = column.min(), column.max()
        else:
            vmin_here, vmax_here = vmin[key], vmax[key]

        divider = make_axes_locatable(ax_col[1])
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        im = summary.plot(column, ax=ax_col[1], rasterized=True, vmin=vmin_here, vmax=vmax_here, legend=True, cax=cax, cmap="cividis", legend_kwds={"location": "bottom"})

    # Settings for map panels
    for ax in axs[1]:
        # Add a country/province outline
        plot_outline(ax, province)

        ax.set_axis_off()
        ax.axis("equal")

    axs[0, 0].set_ylabel("Distribution")
    fig.align_xlabels()

    if title is None:
        title = f"Results from {len(summary)} WOFOST runs"
    fig.suptitle(title)

    if saveto is not None:
        fig.savefig(saveto, bbox_inches="tight", dpi=150)
    else:
        plt.show()

    plt.close()
