"""
Functions for analysis
"""
import numpy as np
import pandas as pd
import geopandas as gpd

from ._typing import Callable

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
