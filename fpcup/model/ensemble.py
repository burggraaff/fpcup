"""
Classes, functions, constants relating to running the WOFOST model in ensembles.
"""
from functools import partial

import pandas as pd
from tqdm import tqdm

from .model import run_pcse_from_raw_data, run_pcse_from_raw_data_with_weather, run_pcse_brp_with_weather
from .rundata import RunData, RunDataBRP
from ..multiprocessing import multiprocess_pcse
from ..weather import load_weather_data_NASAPower
from ..typing import Callable, Coordinates, Iterable, Optional, PathOrStr


### ENSEMBLES WITH FIXED PARAMETERS
def run_pcse_multiple_sites(coordinates: Iterable[Coordinates], run_data_constants: dict, output_dir: PathOrStr, *,
                            progressbar=True, leave_progressbar=True,
                            **kwargs) -> Iterable[bool | RunData]:
    """
    Run a PCSE ensemble with constant parameters for different sites.
    Creates a partial instance of `run_pcse_from_raw_data_with_weather` and runs that for every entry in run_data_variables.
    **kwargs are passed to run_pcse_from_raw_data_with_weather.
    """
    # Initialise partial function
    func = partial(run_pcse_from_raw_data_with_weather, run_data_variables={}, output_dir=output_dir, run_data_constants=run_data_constants, **kwargs)

    # Run model
    statuses = multiprocess_pcse(func, coordinates, progressbar=progressbar, leave_progressbar=leave_progressbar)

    return statuses


def run_pcse_brp_ensemble(brp: pd.DataFrame, year: int, run_data_constants: dict, output_dir: PathOrStr, *,
                          progressbar=True, leave_progressbar=False,
                          **kwargs) -> Iterable[bool | RunDataBRP]:
    """
    Run a PCSE ensemble with constant parameters, for multiple sites using the BRP (or another dataframe).
    Loops over sites, gathering site-specific data (e.g. weather), and running a PCSE ensemble.
    **kwargs are passed to run_pcse_ensemble.
    """
    # Unpack BRP
    brp_rows = list(brp.iterrows())

    # Initialise partial function
    func = partial(run_pcse_brp_with_weather, run_data_variables={}, output_dir=output_dir, run_data_constants=run_data_constants, year=year, **kwargs)

    # Run model
    statuses = multiprocess_pcse(func, brp_rows, progressbar=progressbar, leave_progressbar=leave_progressbar)

    return statuses


### ENSEMBLES WITH VARYING PARAMETERS
def run_pcse_ensemble(run_data_variables: Iterable[dict], output_dir: PathOrStr, *,
                      run_data_constants: Optional[dict]={},
                      progressbar=True, leave_progressbar=False,
                      **kwargs) -> Iterable[bool | RunData]:
    """
    Run a PCSE ensemble with variable, and optionally some constant, parameters.
    Creates a partial instance of `run_pcse_from_raw_data` and runs that for every entry in run_data_variables.
    **kwargs are passed to run_pcse_from_raw_data.
    """
    # Initialise partial function
    func = partial(run_pcse_from_raw_data, output_dir=output_dir, run_data_constants=run_data_constants, **kwargs)

    # Run model
    statuses = multiprocess_pcse(func, run_data_variables, progressbar=progressbar, leave_progressbar=leave_progressbar)

    return statuses


def run_pcse_site_ensemble(coordinates: Iterable[Coordinates], run_data_variables: dict, output_dir: PathOrStr, *,
                           run_data_constants: Optional[dict]={},
                           weather_data_provider: Callable=load_weather_data_NASAPower,
                           progressbar=True, leave_progressbar=False,
                           **kwargs) -> Iterable[bool | RunData]:
    """
    Run a PCSE ensemble with variable, and optionally some constant, parameters, for multiple sites.
    Loops over sites, gathering site-specific data (e.g. weather), and running a PCSE ensemble.
    **kwargs are passed to run_pcse_ensemble.
    """
    statuses_combined = []

    for c in tqdm(coordinates, desc="Sites", unit="site", leave=leave_progressbar):
        # Generate site-specific data
        weatherdata = weather_data_provider(c)
        latitude, longitude = c

        # Bundle parameters
        site_constants = {"latitude": latitude, "longitude": longitude, "weatherdata": weatherdata}
        run_constants = {**run_data_constants, **site_constants}

        ### Run the model
        model_statuses = run_pcse_ensemble(run_data_variables, output_dir, run_data_constants=run_constants, leave_progressbar=False, **kwargs)
        statuses_combined.extend(model_statuses)

    return statuses_combined
