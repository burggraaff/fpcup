"""
Functions that are useful
"""
from multiprocessing import Pool  # Multi-threading

import pandas as pd
from tqdm import tqdm

from pcse.exceptions import WeatherDataProviderError
from pcse.models import Wofost72_WLP_FD

def run_id_from_params(parameters, weatherdata, agromanagement):
    """
    Generate a run ID from PCSE model inputs.
    """
    soil_type = parameters._soildata["SOLNAM"]

    startdate = list(agromanagement[0].keys())[0]
    sowdate = agromanagement[0][startdate]["CropCalendar"]["crop_start_date"]
    crop_type = agromanagement[0][startdate]["CropCalendar"]["crop_name"]

    latitude, longitude = weatherdata.latitude, weatherdata.longitude

    run_id = f"{crop_type}_{soil_type}_sown-{sowdate:%Y-%m-%d}_lat{latitude:.1f}-lon{longitude:.1f}"

    return run_id

def start_and_run_wofost(parameters, weatherdata, agromanagement):
    """
    Start a new PCSE model with the given inputs and run it until it terminates.
    """
    try:
        wofost = Wofost72_WLP_FD(parameters, weatherdata, agromanagement)
        wofost.run_till_terminate()
    except WeatherDataProviderError as e:
        # msg = f"Runid '{run_id}' failed because of missing weather data."
        # print(msg)
        output = None
    else:
        # Convert individual output to Pandas DataFrame
        output = pd.DataFrame(wofost.get_output()).set_index("day")

    # Collect summary results
    try:
        summary = wofost.get_summary_output()[0]
    except IndexError:
        # print(f"IndexError in run '{run_id}'")
        summary = None

    return output, summary

def run_wofost_with_id(run_data):
    """
    Start a new PCSE model with the given inputs and run it until it terminates.
    The results are saved with a unique run ID.
    """
    parameters, weatherdata, agromanagement = run_data
    run_id = run_id_from_params(parameters, weatherdata, agromanagement)
    output, summary = start_and_run_wofost(parameters, weatherdata, agromanagement)

    # Optional: write results to file

    try:
        summary["run_id"] = run_id
    except TypeError:  # This happens if there were no summary results
        pass

    return output, summary

def filter_ensemble_outputs(outputs, summary):
    """
    Filter None and other incorrect entries.
    """
    # Find entries that are None
    valid_entries = [s is not None and o is not None for s, o in zip(summary, outputs)]

    # Apply the filter
    outputs_filtered = [o for o, v in zip(outputs, valid_entries) if v]
    summary_filtered = [s for s, v in zip(summary, valid_entries) if v]

    return outputs_filtered, summary_filtered

def run_pcse_ensemble(all_runs, nr_runs=None):
    """
    Run an entire PCSE ensemble at once.
    all_runs is an iterator that zips the three model inputs (parameters, weatherdata, agromanagement) together, e.g.:
        all_runs = product(parameters_combined, weatherdata, agromanagementdata)
    """
    # Run the models
    outputs, summary = zip(*tqdm(map(run_wofost_with_id, all_runs), total=nr_runs, desc="Running PCSE models", unit="runs"))

    # Clean up the results
    outputs, summary = filter_ensemble_outputs(outputs, summary)

    # Convert the summary to a single dataframe
    summary = pd.DataFrame(summary).set_index("run_id")

    return outputs, summary

def run_pcse_ensemble_parallel(all_runs, nr_runs=None):
    """
    Parallelised version of run_pcse_ensemble.

    Run an entire PCSE ensemble at once.
    all_runs is an iterator that zips the three model inputs (parameters, weatherdata, agromanagement) together, e.g.:
        all_runs = product(parameters_combined, weatherdata, agromanagementdata)
    """
    # Run the models
    with Pool() as p:
        outputs, summary = zip(*tqdm(p.imap(run_wofost_with_id, all_runs, chunksize=10), total=nr_runs, desc="Running PCSE models", unit="runs"))

    # Clean up the results
    outputs, summary = filter_ensemble_outputs(outputs, summary)

    # Convert the summary to a single dataframe
    summary = pd.DataFrame(summary).set_index("run_id")

    return outputs, summary
