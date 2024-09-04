"""
Functions for plotting WOFOST input data and outputs.
"""
import datetime as dt
from functools import partial

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tqdm import tqdm

from matplotlib import pyplot as plt, dates as mdates

from .common import _RASTERIZE_LIMIT_LINES, _RASTERIZE_GEO, KEYS_AGGREGATE_PLOT, cividis_discrete, make_axes_locatable
from .brp import _configure_map_panels
from ..aggregate import KEYS_AGGREGATE, aggregate_h3
from ..geo import Province, NETHERLANDS, provinces
from ..model import InputSummary, Summary
from ..parameters import pcse_outputs
from ..typing import Iterable, Optional, PathOrStr, RealNumber


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
        return plt.rcParams["hist.bins"]


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
        dpi = plt.rcParams["savefig.dpi"]
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
