"""
Statistics functions, e.g. performance metrics.
These are optimised for pandas DataFrames.
"""
import pandas as pd


### INDIVIDUAL METRICS
def r_squared(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ Coefficient of determination (R²) """
    SSres = ((y - y_hat)**2).sum()
    SStot = ((y - y.mean())**2).sum()
    R2 = 1 - SSres/SStot
    return R2

def MD(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ Median Deviation """
    return (y_hat - y).median()

def MAD(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ Median Absolute Deviation """
    return (y_hat - y).abs().median()

def relativeMAD(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.Series:
    """ Relative Median Absolute Deviation """
    return (1 - y_hat/y).abs().median()


### AGGREGATES
prediction_metrics = {"R²": r_squared,
                      "MD": MD,
                      "MAD": MAD,
                      "relativeMAD": relativeMAD,}

def compare_predictions(y: pd.DataFrame, y_hat: pd.DataFrame) -> pd.DataFrame:
    """
    Compare baseline (y) and predicted (y_hat) values using various metrics.
    """
    metrics = {key: func(y, y_hat) for key, func in prediction_metrics.items()}
    metrics = pd.DataFrame(metrics).T

    return metrics
