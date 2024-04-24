"""
Classes, functions, constants relating to running the WOFOST model.
"""
from functools import partial

from tqdm import tqdm

from pcse.exceptions import WeatherDataProviderError
from pcse.models import Engine, Wofost72_WLP_FD

from .outputs import Output
from .rundata import RunData
from ..multiprocessing import multiprocess_pcse
from ..weather import load_weather_data_NASAPower
from ..typing import Callable, Coordinates, Iterable, Optional, PathOrStr


### Running PCSE
def run_pcse_single(run_data: RunData, *, model: Engine=Wofost72_WLP_FD) -> Output | None:
    """
    Start a new PCSE model with the given inputs and run it until it terminates.
    """
    # Run the model from start to finish
    try:
        wofost = model(*run_data)
        wofost.run_till_terminate()
    except WeatherDataProviderError as e:
        # This is sometimes caused by missing weather data; currently ignored silently but with a None output
        output = None
    else:
        # Convert outputs to dataframes
        output = Output.from_model(wofost, run_data=run_data)

    return output


def run_pcse_from_raw_data(run_data_variables: dict, output_dir: PathOrStr, *,
                           run_data_type: type=RunData, run_data_constants: Optional[dict]={},
                           model: Engine=Wofost72_WLP_FD) -> bool | RunData:
    """
    Fully run PCSE:
        1. Create a RunData object from the raw data
        2. Write the RunData to file
        3. Run PCSE
        4. Write the PCSE results and summary to file
        5. Check if the run finished successfully
        6. Return the run status to the user

    `run_data_constants` may be used in combination with `functools.partial` to pre-set some variables.
    """
    # Initialise run data
    run_data = run_data_type(**run_data_variables, **run_data_constants)
    run_data.to_file(output_dir)

    # Run PCSE
    output = run_pcse_single(run_data, model=model)

    # Check/Save PCSE outputs
    try:
        output.to_file(output_dir)
    # If the run failed, saving to file will also fail, so we instead note that this run failed
    except AttributeError:
        status = run_data
    else:
        status = True

    return status


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


### Processing PCSE outputs
def process_model_statuses(outputs: Iterable[bool | RunData], *, verbose: bool=True) -> Iterable[RunData]:
    """
    Determine which runs in a PCSE ensemble failed / were skipped.
    Succesful runs will have a True status.
    Skipped runs will have a False status.
    Failed runs will have their RunData as their status.

    The RunData of the failed runs are returned for further analysis.
    """
    n = len(outputs)

    failed_runs = [o for o in outputs if isinstance(o, RunData)]
    if len(failed_runs) > 0:
        print(f"Number of failed runs: {len(failed_runs)}/{n}")
    else:
        if verbose:
            print("No runs failed.")

    skipped_runs = [o for o in outputs if o is False]
    if len(skipped_runs) > 0:
        print(f"Number of skipped runs: {len(skipped_runs)}/{n}")
    else:
        if verbose:
            print("No runs skipped.")

    return failed_runs
