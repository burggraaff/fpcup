"""
Functions for plotting data and results
"""
import datetime as dt
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tqdm import tqdm

from matplotlib import pyplot as plt, dates as mdates, patches as mpatches, patheffects as mpe, ticker as mticker
from matplotlib import colormaps, rcParams

rcParams.update({"axes.grid": True,
                 "figure.dpi": 600, "savefig.dpi": 600,
                 "grid.linestyle": "--",
                 "hist.bins": 15,
                 "image.cmap": "cividis",
                 "legend.edgecolor": "black", "legend.framealpha": 1,
                 })

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._brp_dictionary import brp_categories_colours, brp_crops_colours
from .aggregate import KEYS_AGGREGATE, aggregate_h3
from .constants import CRS_AMERSFOORT, WGS84
from .geo import Province, NETHERLANDS, is_single_province, provinces
from .model import InputSummary, Summary, TimeSeries
from .parameters import all_parameters, pcse_inputs, pcse_outputs, pcse_summary_outputs
from .tools import make_iterable
from .typing import Aggregator, Callable, Iterable, Optional, PathOrStr, RealNumber, StringDict

### CONSTANTS
# Raster/Vector switches
_RASTERIZE_LIMIT_LINES = 1000
_RASTERIZE_LIMIT_GEO = 250  # Plot geo data in raster format if there are more than this number
_RASTERIZE_GEO = lambda data: (len(data) > _RASTERIZE_LIMIT_GEO)

KEYS_AGGREGATE_PLOT = ("n", "area", *KEYS_AGGREGATE)

# Graphical defaults
cividis_discrete = colormaps["cividis"].resampled(10)
default_outline = {"color": "black", "linewidth": 0.5}


### GEOSPATIAL PLOTS
def _configure_map_panels(axs: plt.Axes | Iterable[plt.Axes],
                          province: Province | Iterable[Province]=NETHERLANDS, **kwargs) -> None:
    """
    Apply default settings to map panels.
    **kwargs are passed to `Province.plot_boundary` - e.g. coarse, crs, ...
    """
    axs = make_iterable(axs)
    provinces = make_iterable(province)
    outline_kw = {**default_outline, **kwargs}
    for ax in axs:
        # Country/Province outline(s)
        for p in provinces:
            p.plot_boundary(ax=ax, **outline_kw)

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
            province: Province=NETHERLANDS, colour_dict: Optional[StringDict]=None,
            figsize=(10, 10), title: Optional[str]=None, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a map of BRP polygons in the given column.
    If `province` is provided, only data within that province will be plotted, with the corresponding outline.
    """
    # Select province data if desired
    SINGLE_PROVINCE = is_single_province(province)
    if SINGLE_PROVINCE:
        data = province.select_entries_in_province(data)

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
                       province: Province=NETHERLANDS, crops: Iterable[str]=brp_crops_colours.keys(),
                       figsize=(14, 3.5), shape=(1, 5), title: Optional[str]=None, saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a map of BRP polygons, with one panel per crop species.
    Shape is (nrows, ncols).
    """
    # Select province data if desired
    SINGLE_PROVINCE = is_single_province(province)
    if SINGLE_PROVINCE:
        data = province.select_entries_in_province(data)

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
    _configure_map_panels(axs.ravel(), province, linewidth=0.2)
    fig.suptitle(title)

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_wofost_input_summary(summary: InputSummary | Summary, *,
                              title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot WOFOST input statistics for an ensemble summary.
    """
    # Setup
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), squeeze=False, layout="constrained")

    # Site / soil properties
    summary.plot.hexbin("RDMSOL", "WAV", mincnt=1, gridsize=10, cmap=cividis_discrete, ax=axs[0, 0])
    axs[0, 0].set_xlabel("RDMSOL")
    axs[0, 0].set_ylabel("WAV")

    # Labels
    fig.suptitle(title)

    # Save / close
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
    # Determine rasterization based on number of lines
    rasterized = (len(outputs) > _RASTERIZE_LIMIT_LINES)
    saveto = saveto.with_suffix(".png") if rasterized else saveto.with_suffix(".pdf")

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
            ax.plot(time_axis, df[key], alpha=0.25, rasterized=rasterized)

    axs[-1].set_xlabel("Time")
    for ax, key in zip(axs, keys):
        ax.set_ylabel(key)
        ax.set_ylim(ymin=0)
        ax.text(1.00, 1.00, pcse_outputs[key], transform=ax.transAxes, horizontalalignment="right", verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "white"})

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
                             weights: Optional[str | Iterable[RealNumber]]=None,
                             axs: Optional[Iterable[plt.Axes]]=None, title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot histograms showing WOFOST run summaries.
    If `axs` is specified, use existing Axes and do not apply figure settings or save anything.
    If `axs` is None, create a new figure.
    """
    NEW_FIGURE = (axs is None)

    # If `weights` was given as a column name, get the associated data
    if isinstance(weights, str):
        weights = summary[weights]

    # Configure figure and axes if none were provided
    if NEW_FIGURE:
        fig, axs = plt.subplots(nrows=1, ncols=len(keys), sharey=True, figsize=(3*len(keys), 3))
    else:
        fig = axs[0].figure  # Assume all axes are in the same figure

    # Plot the histograms
    for ax, key in zip(axs, keys):
        # Leave an empty panel for keys which may be passed from plot_wofost_summary but which only apply to the geo plot
        if key in ("n", "area"):
            ax.set_axis_off()
            continue

        column = summary[key]

        if is_datetime(column):
            _configure_histogram_datetime(ax)

        bins = _numerical_or_date_bins(column)
        column.plot.hist(ax=ax, bins=bins, weights=weights, facecolor="black")

        # Panel settings
        ax.set_title(key)
        ax.set_xlim(column.min(), column.max())

    # Labels
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


def _remove_key_from_keys(keys: Iterable[str], key_to_remove: str) -> tuple[str]:
    """
    Remove the key_to_remove from a list of keys.
    """
    return tuple(k for k in keys if k != key_to_remove)


def wofost_summary_geo(data_geo: gpd.GeoDataFrame, keys: Iterable[str]=KEYS_AGGREGATE_PLOT, *,
                       axs: Optional[Iterable[plt.Axes]]=None, title: Optional[str]=None, rasterized: bool=True,
                       province: Province=NETHERLANDS, use_coarse: bool=False,
                       saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Plot geographical maps showing WOFOST run summaries.
    If `axs` is specified, use existing Axes and do not apply figure settings or save anything.
    If `axs` is None, create a new figure.
    Note that `province` only determines the outline(s) to be drawn - this function does not aggregate or filter data.
    """
    NEW_FIGURE = (axs is None)

    # Don't plot n, area if these are not available
    if NEW_FIGURE:
        for key_to_remove in ("n", "area"):
            if key_to_remove in keys and key_to_remove not in data_geo.columns:
                keys = _remove_key_from_keys(keys, key_to_remove)

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
    _configure_map_panels(axs, province, use_coarse=use_coarse, crs=data_geo.crs)

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


def plot_wofost_summary(summary: Summary, keys: Iterable[str]=KEYS_AGGREGATE_PLOT, *,
                                 aggregate: bool=True, aggregate_kwds: dict={}, weight_by_area: bool=True,
                                 province: Province=NETHERLANDS,
                                 title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot histograms and (aggregate) maps showing WOFOST run summaries.
    """
    # Check for area (weight) availability
    if weight_by_area:
        if "area" in summary.columns:
            weights = "area"
        else:
            raise ValueError("Cannot weight by area, area not available.\n"
                             f"Available columns: {summary.columns}")
    else:
        if "area" in keys:
            keys = _remove_key_from_keys(keys, "area")
        weights = None

    # Create figure and panels
    fig, axs = plt.subplots(nrows=2, ncols=len(keys), sharey="row", figsize=(15, 5), gridspec_kw={"hspace": 0.25, "height_ratios": [1, 1.5]})

    ### First row: histograms
    wofost_summary_histogram(summary, keys, axs=axs[0], weights=weights)

    ### Second row: maps
    # Aggregate the data if desired
    if aggregate:
        data_geo = aggregate_h3(summary, province=province, weightby=weights, **aggregate_kwds)
        rasterized = True
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


plot_wofost_summary_byprovince = partial(wofost_summary_geo, rasterized=True, province=provinces.values(), use_coarse=True)


def weighted_mean_loss(loss_per_batch: np.ndarray) -> np.ndarray:
    """
    Return the weighted mean loss per epoch, weighted with a sawtooth.
    """
    # Check dimensionality
    INPUT_IS_1D = (loss_per_batch.ndim == 1)
    if INPUT_IS_1D:
        loss_per_batch = loss_per_batch[np.newaxis, :]

    # Generate sawtooth
    n_batches = loss_per_batch.shape[1]
    sawtooth = np.arange(n_batches) + 1

    # Weighted mean
    loss_per_epoch = np.average(loss_per_batch, weights=sawtooth, axis=1)

    # Return 1D if the input was 1D
    if INPUT_IS_1D:
        loss_per_epoch = loss_per_epoch[0]

    return loss_per_epoch


c_train = "#4477AA"
c_test = "#CCBB44"
pe_epoch = [mpe.Stroke(linewidth=4, foreground="black"),
            mpe.Normal()]
def plot_loss_curve(losses_train: np.ndarray, *, losses_test: Optional[np.ndarray]=None,
                    title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot the loss curve per batch and per epoch.
    """
    # Constants
    n_epochs, n_batches = losses_train.shape
    epochs = np.arange(n_epochs + 1)
    batches = np.arange(losses_train.size) + 1

    # Training data: get loss per batch and per epoch
    loss_initial = [losses_train[0, 0]]
    losses_train_epoch = weighted_mean_loss(losses_train)
    losses_train_epoch = np.concatenate([loss_initial, losses_train_epoch])

    losses_train_batch = losses_train.ravel()

    # Testing data: dummy loss at epoch 0
    losses_test = np.insert(losses_test, 0, np.nan)

    # Variables for limits etc.
    try:
        maxloss = np.nanmax([np.nanmax(losses_train), np.nanmax(losses_test)])
    except AttributeError:  # is no test losses were provided
        maxloss = losses_train.max()

    # Figure setup
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), layout="constrained")

    # Plot training loss per batch
    ax.plot(batches, losses_train_batch, color=c_train, zorder=0)

    ax.set_xlim(0, len(batches))
    ax.set_xlabel("Batch", color=c_train)
    ax.set_ylabel("Loss")
    ax.grid(True, axis="y", ls="--")
    ax.grid(False, axis="x")

    # Plot training/testing loss per epoch
    ax2 = ax.twiny()
    ax2.plot(epochs, losses_train_epoch, color=c_train, path_effects=pe_epoch, label="Train", zorder=1)
    ax2.plot(epochs, losses_test, color=c_test, path_effects=pe_epoch, label="Test", zorder=1)

    ax2.set_xlim(0, n_epochs)
    ax2.set_ylim(0, maxloss*1.05)
    ax2.set_xlabel("Epoch")
    ax2.grid(True, ls="--")
    ax2.legend(loc="best")

    # Final settings
    fig.suptitle(title)

    # Save and close
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    plt.close()
