"""
Functions for plotting BRP data.
"""
import geopandas as gpd

from matplotlib import pyplot as plt, patches as mpatches

from .common import _RASTERIZE_GEO, default_outline
from .._brp_dictionary import brp_categories_colours, brp_crops_colours
from ..geo import Province, NETHERLANDS, is_single_province
from ..tools import make_iterable
from ..typing import Iterable, Optional, PathOrStr, StringDict


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
