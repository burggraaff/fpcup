"""
Functions that are useful
"""
import pandas as pd

from pcse.models import Wofost72_WLP_FD
from pcse.exceptions import WeatherDataProviderError

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

def run_wofost_with_id(parameters, weatherdata, agromanagement):
    """
    Start a new PCSE model with the given inputs and run it until it terminates.
    The results are saved with a unique run ID.
    """
    run_id = run_id_from_params(parameters, weatherdata, agromanagement)
    output, summary = start_and_run_wofost(parameters, weatherdata, agromanagement)

    # Optional: write results to file

    try:
        summary["run_id"] = run_id
    except TypeError:  # This happens if there were no summary results
        pass

    return output, summary

def replace_year_in_datetime(dt, newyear=2000):
    """
    For a datetime object yyyy-mm-dd, replace yyyy with newyear.
    """
    return dt.replace(year=newyear)
