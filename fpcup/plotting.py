"""
Functions for plotting data and results
"""
import datetime as dt
from functools import partial

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tqdm import tqdm

from matplotlib import pyplot as plt, dates as mdates, patches as mpatches, ticker as mticker
from matplotlib import colormaps, rcParams
rcParams.update({"axes.grid": True,
                 "figure.dpi": 600, "savefig.dpi": 600,
                 "grid.linestyle": "--",
                 "hist.bins": 15,
                 "image.cmap": "cividis",
                 "legend.edgecolor": "black", "legend.framealpha": 1,
                 })
from mpl_toolkits.axes_grid1 import make_axes_locatable
cividis_discrete = colormaps["cividis"].resampled(10)

from ._brp_dictionary import brp_categories_colours, brp_crops_colours
from ._typing import Aggregator, Callable, Iterable, Optional, PathOrStr, RealNumber, StringDict
from .analysis import KEYS_AGGREGATE
from .constants import CRS_AMERSFOORT, WGS84
from .geo import PROVINCE_NAMES, area, area_coarse, boundary, boundary_coarse, aggregate_h3, entries_in_province
from .model import Summary, parameter_names
from .tools import make_iterable

_RASTERIZE_LIMIT_GEO = 250  # Plot geo data in raster format if there are more than this number
_RASTERIZE_GEO = lambda data: (len(data) > _RASTERIZE_LIMIT_GEO)


def plot_outline(ax: plt.Axes, province: str="Netherlands", *,
                 coarse: bool=False, crs: str=CRS_AMERSFOORT, **kwargs) -> None:
    """
    Plot an outline of the Netherlands or a specific province (e.g. "Zuid-Holland").
    """
    if coarse:
        line = boundary_coarse[province].to_crs(crs)
    else:
        line = boundary[province].to_crs(crs)

    line_kw = {"color": "black", "lw": 0.5, **kwargs}
    line.plot(ax=ax, **line_kw)


def _configure_map_panels(axs: plt.Axes | Iterable[plt.Axes], province: str | Iterable[str]="Netherlands", **kwargs) -> None:
    """
    Apply default settings to map panels.
    **kwargs are passed to `plot_outline` - e.g. coarse, crs, ...
    """
    axs = make_iterable(axs)
    provinces = make_iterable(province)
    for ax in axs:
        # Country/Province outline(s)
        for p in provinces:
            plot_outline(ax, p, **kwargs)

        # Axis settings
        ax.set_axis_off()
        ax.axis("equal")


def _column_to_title(column: str) -> str:
    """
    Clean up a column name (e.g. "crop_species") so it can be used as a title (e.g. "Crop species").
    """
    return column.capitalize().replace("_", " ")


def brp_histogram(data: gpd.GeoDataFrame, column: str, *,
                  figsize=(3, 5), usexticks=True, title: Optional[str]=None, top5=True, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Make a bar plot showing the distribution of plots/crops in BRP data.
    """
    # Determine the number of plots and total area per group
    counts = data[column].value_counts()  # Number of plots per group
    areas = data.groupby(column)["area"].sum().reindex_like(counts)  # Area per group, unit [ha]

    # Plot the data
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=figsize, gridspec_kw={"hspace": 0.1})
    for data, ax in zip([counts, areas], axs):
        data.plot.bar(ax=ax, color='w', edgecolor='k', hatch="//", **kwargs)

    # Adjust ticks on x axis
    axs[0].tick_params(axis="x", bottom=False, labelbottom=False)
    if usexticks:
        # There is no cleaner way to do this because tick_params does not support horizontalalignment
        xticklabels = [label.get_text().capitalize() for label in axs[1].get_xticklabels()]
        axs[1].set_xticklabels(xticklabels, rotation=45, horizontalalignment="right")
    else:
        axs[1].tick_params(axis="x", bottom=False, labelbottom=False)

    # Panel settings
    for ax in axs:
        ax.grid(False)

        if not "log" in kwargs:
            # Prevent scientific notation
            ax.ticklabel_format(axis="y", style="plain")

            # Set ymin explicitly
            ax.set_ylim(ymin=0)

    # Adjust labels
    axs[0].set_title(title)
    axs[0].set_ylabel("Number of plots")
    axs[1].set_xlabel(_column_to_title(column))
    axs[1].set_ylabel("Total area [ha]")
    fig.align_ylabels()

    # Add the top 5 list in the corner
    if top5:
        float2str = lambda x: f"{x:.0f}"
        top5_text = [f"Top 5:\n{df.head().to_string(header=False, float_format=float2str)}" for df in (counts, areas)]
        for ax, text in zip(axs, top5_text):
            ax.text(0.99, 0.98, text, transform=ax.transAxes, horizontalalignment="right", verticalalignment="top")

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def brp_map(data: gpd.GeoDataFrame, column: str, *,
            province: str="Netherlands", figsize=(10, 10), title: Optional[str]=None, colour_dict: Optional[StringDict]=None, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a map of BRP polygons in the given column.
    If `province` is provided, only data within that province will be plotted, with the corresponding outline.
    """
    # Select province data if desired
    if province != "Netherlands":
        data = entries_in_province(data, province)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rasterized = _RASTERIZE_GEO(data)

    # If colours are specified, plot those instead of the raw data, and add a legend
    if colour_dict:
        colours = data[column].map(colour_dict)
        data.plot.geo(ax=ax, color=colours, rasterized=rasterized, **kwargs)

        # Generate dummy patches with the same colour mapping and add those to the legend
        colour_patches = [mpatches.Patch(color=colour, label=label.capitalize()) for label, colour in colour_dict.items() if label in data[column].unique()]
        ax.legend(handles=colour_patches, loc="lower right", fontsize=12, title=_column_to_title(column))

    # If colours are not specified, simply plot the data and let geopandas handle the colours
    else:
        data.plot.geo(ax=ax, column=column, rasterized=rasterized, **kwargs)

    # Panel settings
    _configure_map_panels(ax, province)
    ax.set_title(title)

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def brp_crop_map_split(data: gpd.GeoDataFrame, column: str="crop_species", *,
                       province: str="Netherlands", crops: Iterable[str]=brp_crops_colours.keys(), figsize=(14, 3.5), shape=(1, 5), title: Optional[str]=None, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a map of BRP polygons, with one panel per crop species.
    Shape is (nrows, ncols).
    """
    # Select province data if desired
    if province != "Netherlands":
        data = entries_in_province(data, province)

    # Create figure
    fig, axs = plt.subplots(*shape, figsize=figsize)

    # Plot each crop into its own panel
    for crop, ax in zip(crops, axs.ravel()):
        # Plot the plots
        data_here = data.loc[data[column] == crop]
        number_here = len(data_here)
        if number_here > 0:  # Only plot if there are actually plots for this crop
            rasterized = _RASTERIZE_GEO(data_here)
            data_here.plot(ax=ax, color="black", rasterized=rasterized, **kwargs)

        # Panel parameters
        colour = brp_crops_colours[crop]
        ax.set_title(f"{crop.title()} ({number_here})", color=colour)

    # Panel settings
    _configure_map_panels(axs.ravel(), province, lw=0.2)
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    else:
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


def _configure_histogram_datetime(ax: plt.Axes) -> None:
    """
    Adjust the axes on a datetime histogram so they are formatted properly.
    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)

    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)


def wofost_summary_histogram(summary: Summary, keys: Iterable[str]=KEYS_AGGREGATE, *,
                             axs: Optional[Iterable[plt.Axes]]=None, weights: Optional[str | Iterable[RealNumber]]=None, title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot histograms showing WOFOST run summaries.
    If `axs` is specified, use existing Axes and do not apply figure settings or save anything.
    If `axs` is None, create a new figure.
    """
    NEW_FIGURE = (axs is None)

    # If `weights` was given as a column name, get the associated data
    if isinstance(weights, str):
        weights = summary[weights]

    # Determine some parameters before the loop
    summary_cols = summary[keys]
    vmin, vmax = summary_cols.min(), summary_cols.max()
    bins = summary_cols.apply(_numerical_or_date_bins)

    # Configure figure and axes if none were provided
    if NEW_FIGURE:
        fig, axs = plt.subplots(nrows=1, ncols=len(keys), sharey=True, figsize=(3*len(keys), 3))
    else:
        fig = axs[0].figure  # Assume all axes are in the same figure

    # Plot the histograms
    for ax, key in zip(axs, keys):
        column = summary[key]

        if is_datetime(column):
            _configure_histogram_datetime(ax)

        column.plot.hist(ax=ax, bins=bins[key], weights=weights, facecolor="black")

        # Panel settings
        ax.set_title(key)
        ax.set_xlim(vmin[key], vmax[key])

    # Labels
    axs[0].set_ylabel("Distribution")
    fig.align_xlabels()
    if NEW_FIGURE:
        if title is None:
            title = f"Results from {len(summary)} WOFOST runs"
        fig.suptitle(title)

    # Save/show the figure
    if NEW_FIGURE:
        if saveto is not None:
            fig.savefig(saveto, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


def wofost_summary_geo(data_geo: gpd.GeoDataFrame, keys: Iterable[str]=KEYS_AGGREGATE, *,
                       axs: Optional[Iterable[plt.Axes]]=None, title: Optional[str]=None, rasterized: bool=True,
                       province: Optional[str]="Netherlands", coarse: bool=False,
                       saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Plot geographical maps showing WOFOST run summaries.
    If `axs` is specified, use existing Axes and do not apply figure settings or save anything.
    If `axs` is None, create a new figure.
    Note that `province` only determines the outline(s) to be drawn - this function does not aggregate or filter data.
    """
    NEW_FIGURE = (axs is None)

    # Configure figure and axes if none were provided
    if NEW_FIGURE:
        fig, axs = plt.subplots(nrows=1, ncols=len(keys), sharex=True, sharey=True, figsize=(3*len(keys), 3))
    else:
        fig = axs[0].figure  # Assume all axes are in the same figure

    # Plot the maps
    for ax, key in zip(axs, keys):
        # Second row: maps
        column = data_geo[key]
        if is_datetime(column):
            column = column.apply(mdates.date2num)
        vmin, vmax = column.min(), column.max()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        im = data_geo.plot.geo(column, ax=ax, vmin=vmin, vmax=vmax, rasterized=rasterized, legend=True, cax=cax, cmap=cividis_discrete, legend_kwds={"location": "bottom", "label": key}, **kwargs)

    # Settings for map panels
    _configure_map_panels(axs, province, coarse=coarse, crs=data_geo.crs)

    # Labels
    if NEW_FIGURE:
        fig.suptitle(title)

    # Save/show the figure
    if NEW_FIGURE:
        if saveto is not None:
            fig.savefig(saveto, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


def plot_wofost_summary(summary: Summary, keys: Iterable[str]=KEYS_AGGREGATE, *,
                                 aggregate: bool=True, aggregate_kwds={}, weights: Optional[str | Iterable[RealNumber]]=None, title: Optional[str]=None, province: Optional[str]="Netherlands", saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot histograms and (aggregate) maps showing WOFOST run summaries.
    """
    # Create figure and panels
    fig, axs = plt.subplots(nrows=2, ncols=len(keys), sharey="row", figsize=(15, 5), gridspec_kw={"hspace": 0.25, "height_ratios": [1, 1.5]})

    ### First row: histograms
    wofost_summary_histogram(summary, keys, axs=axs[0], weights=weights)

    ### Second row: maps
    # Aggregate the data if desired
    if aggregate:
        data_geo = aggregate_h3(summary, clipto=province, weightby=weights, **aggregate_kwds)
        rasterized = False
        dpi = rcParams["savefig.dpi"]
    else:
        data_geo = summary
        rasterized = _RASTERIZE_GEO(summary)
        dpi = 150

    wofost_summary_geo(data_geo, keys, axs=axs[1], rasterized=rasterized, province=province)

    ### Figure settings
    if title is None:
        title = f"Results from {len(summary)} WOFOST runs"
    fig.suptitle(title)

    ### Save/show results
    if saveto is not None:
        fig.savefig(saveto, bbox_inches="tight", dpi=dpi)
    else:
        plt.show()

    plt.close()


plot_wofost_summary_byprovince = partial(wofost_summary_geo, rasterized=True, province=PROVINCE_NAMES, coarse=True)
