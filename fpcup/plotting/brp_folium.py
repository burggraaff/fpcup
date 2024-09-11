"""
Functions for plotting BRP data interactively using Folium.
"""
from functools import partial

import folium
import geopandas as gpd

from .._brp_dictionary import brp_categories_colours, brp_crops_colours
from ..constants import WGS84
from ..geo import Province, NETHERLANDS, is_single_province, points_to_coordinates
from ..tools import make_iterable
from ..typing import Iterable, Optional, PathOrStr, StringDict


def brp_map_interactive(data: gpd.GeoDataFrame, *,
                        province: Province=NETHERLANDS, colour_dict: Optional[StringDict]=None,
                        saveto: Optional[PathOrStr]=None, **kwargs) -> None:
    """
    Create a Folium map of BRP polygons in the given column.
    If `province` is provided, only data within that province will be plotted, with the corresponding outline.
    """
    # Convert CRS
    data, province = data.to_crs(WGS84), province.to_crs(WGS84)
    coords = points_to_coordinates([province.centroid])[0]

    # Select province data if desired
    SINGLE_PROVINCE = is_single_province(province)
    if SINGLE_PROVINCE:
        data = province.select_entries_in_province(data)

    # If colours are specified, plot those instead of the raw data, and add a legend
    style_dict = {"color": "black", "weight": 0.5}
    if colour_dict:
        style_func = lambda row: {**style_dict, "fillColor": colour_dict[row["properties"]["crop_species"]]}
    else:
        style_func = lambda row: {**style_dict, "fillColor": "green"}

    # Create figure
    m = folium.Map(location=coords, zoom_start=9, tiles="CartoDB positron")

    # Add BRP data
    popup = folium.GeoJsonPopup(fields=["crop", "area"], aliases=["Crop", "Area [ha]"])
    m_brp = folium.GeoJson(data=data, style_function=style_func, zoom_on_click=False, popup=popup)
    m_brp.add_to(m)

    # Save to file, if wanted
    if saveto is not None:
        m.save(saveto)

    return m


brp_map_category_interactive = partial(brp_map_interactive, colour_dict=brp_categories_colours)
brp_map_crop_interactive = partial(brp_map_interactive, colour_dict=brp_crops_colours)
