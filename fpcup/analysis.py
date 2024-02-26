"""
Functions for analysis
"""
from datetime import date, datetime

import numpy as np
import geopandas as gpd

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from ._typing import Callable, FuncDict, Iterable

# Columns in summary data that should be averaged over for aggregates
KEYS_AGGREGATE = ["LAIMAX", "TWSO", "CTRAT", "CEVST"] + ["DOE", "DOM"]
mean_dict = {key: "mean" for key in KEYS_AGGREGATE}

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

    return {key: wm_datetime if is_datetime(data[key]) else wm_numerical for key in keys}

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
