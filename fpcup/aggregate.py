"""
Functions for geospatial aggregation.
"""
from datetime import date, datetime

import numpy as np
import geopandas as gpd

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from ._typing import Aggregator, Callable, FuncDict, Iterable, Optional, PathOrStr
from .constants import CRS_AMERSFOORT, WGS84
from .geo import NETHERLANDS, Province, add_provinces, add_province_geometry, is_single_province, maintain_crs
from .parameters import aggregate_parameters

### CONSTANTS AND SETTINGS
H3_AGGREGATION_LEVELS = {"country": 6, "province": 7}

### AGGREGATORS
# Columns in summary data that should be averaged over for aggregates
# H3pandas does not support the tuple-dict system, e.g. {"n": ("DVS", "count")}, so it has to be done in an ugly way
KEYS_AGGREGATE = ["LAIMAX", "TWSO", "CTRAT", "CEVST"] + ["DOE", "DOM"]
count_dict = {"DVS": "size"}
rename_after_aggregation = {"DVS": "n"}
mean_dict = {key: "mean" for key in KEYS_AGGREGATE}
mean_dict = {**count_dict, **mean_dict}

def weighted_mean_for_DF(data: pd.DataFrame, *, weightby: str="area") -> Callable:
    """
    Generate a weighted mean function for the given dataframe, to be used in .agg.
    Example:
        weight_by_area = weighted_mean_for_DF(summary)
        weighted_average_yield = summary.agg(wm_twso=("TWSO", weight_by_area))
    """
    def weighted_mean(x):
        return np.average(x, weights=data.loc[x.index, weightby])

    return weighted_mean

def weighted_mean_datetime(data: pd.DataFrame, *, weightby: str="area") -> Callable:
    """
    Generate a weighted mean function for datetime objects, to be used in .agg.
    Same as `weighted_mean_for_DF` but converts to and from timestamps first.
    """
    def weighted_mean_DT(x):
        x_timestamp = x.apply(datetime.timestamp)
        average_timestamp = np.average(x_timestamp, weights=data.loc[x.index, weightby])
        average = datetime.fromtimestamp(average_timestamp)
        return average

    return weighted_mean_DT

def weighted_mean_dict(data: pd.DataFrame, *,
                       keys=KEYS_AGGREGATE, weightby: str="area") -> FuncDict:
    """
    Generate a dictionary with the relevant weighted mean function for every key.
    """
    wm_numerical = weighted_mean_for_DF(data, weightby=weightby)
    wm_datetime = weighted_mean_datetime(data, weightby=weightby)

    aggregator_mean = {key: wm_datetime if is_datetime(data[key]) else wm_numerical for key in keys}
    aggregator_area = {"area": "sum"}

    aggregator = {**count_dict, **aggregator_area, **aggregator_mean}

    return aggregator

def default_aggregator(data: pd.DataFrame, *,
                       keys=KEYS_AGGREGATE, weightby: str="area") -> FuncDict:
    """
    Generate an aggregator dictionary.
    If weights are available, return weighted_mean_dict(*args, **kwargs).
    If weights are not available, return mean_dict.
    """
    if weightby in data.columns:
        aggregator = weighted_mean_dict(data, keys=keys, weightby=weightby)
    else:
        aggregator = mean_dict.copy()  # Return a copy in case it is edited elsewhere

    return aggregator


### GEOSPATIAL AGGREGATION
@maintain_crs
def aggregate_province(_data: gpd.GeoDataFrame, *,
                       aggregator: Optional[Aggregator]=None) -> gpd.GeoDataFrame:
    """
    Aggregate data to the provinces.
    `aggregator` is passed to DataFrame.agg; if none is specified, then means or weighted means (depending on availability of weights) are used.
    """
    # Convert the input to CRS_AMERSFOORT for the aggregation and use the centroids
    data = _data.copy()
    data["geometry"] = data.to_crs(CRS_AMERSFOORT).centroid

    # Add province information if not yet available
    if "province" not in data.columns:
        add_provinces(data, leave_progressbar=False)

    # Aggregate the data
    if aggregator is None:
        aggregator = default_aggregator(data)
    data_province = data.groupby("province").agg(aggregator).rename(columns=rename_after_aggregation)

    # Add the province geometries
    data_province = add_province_geometry(data_province)

    return data_province


def save_aggregate_province(data: gpd.GeoDataFrame, saveto: PathOrStr, **kwargs) -> None:
    """
    Save a provincial aggregate without the geometry information.
    """
    data.drop("geometry", axis=1).to_csv(saveto, **kwargs)


@maintain_crs
def aggregate_h3(_data: gpd.GeoDataFrame, *,
                 aggregator: Optional[Aggregator]=None, level: Optional[int]=None, province: Province=NETHERLANDS, weightby: str="area") -> gpd.GeoDataFrame:
    """
    Aggregate data to the H3 hexagonal grid.
    `aggregator` is passed to DataFrame.agg; if none is specified, then means or weighted means (depending on availability of weights) are used.
    `province` is used to get a geometry, e.g. the Netherlands or one province, to clip the results to. Set it to `None` to preserve the full grid.
    """
    # Filter the data to the desired province
    CLIPPING = (province is not None)
    SINGLE_PROVINCE = is_single_province(province)
    if SINGLE_PROVINCE:
        data = province.select_entries_in_province(_data)
    else:
        data = _data.copy()

    # Find the centroids in a projected CRS, then convert to WGS84 for aggregation
    if not data.crs.is_projected:
        data.to_crs(CRS_AMERSFOORT, inplace=True)
    data["geometry"] = data.centroid.to_crs(WGS84)

    # Get default parameters for aggregation/clipping
    if aggregator is None:
        aggregator = default_aggregator(data, weightby=weightby)
    if level is None:
        if CLIPPING:
            level = H3_AGGREGATION_LEVELS[province.level]
        else:
            raise ValueError("No H3 aggregation level provided and no Province to derive it from.")

    # Aggregate the data
    data_h3 = data.h3.geo_to_h3_aggregate(level, aggregator).rename(columns=rename_after_aggregation)

    # Clip the data if desired
    if CLIPPING:
        data_h3 = province.clip_data(data_h3)

    return data_h3
