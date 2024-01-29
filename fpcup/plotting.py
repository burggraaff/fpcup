"""
Functions for plotting data and results
"""
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from .model import parameter_names

def brp_histogram(data: gpd.GeoDataFrame, figsize=(3, 2), usexticks=True, xlabel="Crop", ylabel="Number of plots", title=None, top5=True, saveto=None, **kwargs):
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

    if saveto:
        plt.savefig(saveto, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

def brp_map(data: gpd.GeoDataFrame, column: str, figsize=(10, 10), title=None, saveto=None, rasterized=True, **kwargs):
    """
    Create a map of BRP polygons in the given column.
    """
    plt.figure(figsize=figsize)
    ax = data.plot(column=column, rasterized=rasterized, **kwargs)
    ax.set_axis_off()

    plt.title(title)

    if saveto:
        plt.savefig(saveto, bbox_inches="tight", dpi=1200)
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
        ax.text(0.00, 1.00, parameter_names[key], transform=ax.transAxes, horizontalalignment="left", verticalalignment="top", bbox={"boxstyle": "round", "facecolor": "white"})
        ax.grid()
    fig.align_ylabels()
    axs[0].set_title(f"Results from {len(outputs)} WOFOST runs")

    if saveto is not None:
        fig.savefig(saveto, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
