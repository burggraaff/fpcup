"""
Classes, functions, constants relating to running the WOFOST model - basics.
"""
from pcse.exceptions import WeatherDataProviderError
from pcse.models import Engine, Wofost72_WLP_FD

from .outputs import Output
from .rundata import RunData, RunDataBRP
from ..weather import load_weather_data_NASAPower
from ..typing import Callable, Iterable, Optional, PathOrStr, Series


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


def run_pcse_from_raw_data_with_weather(coordinates, *args,
                                        run_data_variables: dict={},
                                        weather_data_provider: Callable=load_weather_data_NASAPower, **kwargs) -> bool | RunData:
    """
    Get weather data, then fully run PCSE.
    """
    weatherdata = weather_data_provider(coordinates)
    run_data_variables_updated = {**run_data_variables, "weatherdata": weatherdata, "latitude": coordinates[0], "longitude": coordinates[1]}
    return run_pcse_from_raw_data(run_data_variables_updated, *args, **kwargs)


def run_pcse_brp_with_weather(i_row: tuple[int, Series], year: int, *args,
                              run_data_variables: dict={},
                              weather_data_provider: Callable=load_weather_data_NASAPower, **kwargs) -> bool | RunDataBRP:
    """
    Pre-process BRP data, get weather data, then fully run PCSE.
    """
    # Unpack BRP data
    i, row = i_row  # Unpack index/data pair
    coordinates = row["latitude"], row["longitude"]

    weatherdata = weather_data_provider(coordinates)
    run_data_variables_updated = {**run_data_variables, "weatherdata": weatherdata, "brpdata": row, "brpyear": year}
    return run_pcse_from_raw_data(run_data_variables_updated, *args, run_data_type=RunDataBRP, **kwargs)


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
