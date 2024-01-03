"""
Functions that are useful
"""
from itertools import product
from multiprocessing import Pool  # Multi-threading
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from pcse.base import MultiCropDataProvider, ParameterProvider, WeatherDataProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.fileinput import CABOFileReader
from pcse.models import Wofost72_WLP_FD
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from .agro import AgromanagementData
from .tools import make_iterable

parameter_names = {"DVS": "Crop development stage",
                   "LAI": "Leaf area index [ha/ha]",
                   "TAGP": "Total above-ground production [kg/ha]",
                   "TWSO": "Total weight - storage organs [kg/ha]",
                   "TWLV": "Total weight - leaves [kg/ha]",
                   "TWST": "Total weight - stems [kg/ha]",
                   "TWRT": "Total weight - roots [kg/ha]",
                   "TRA": "Crop transpiration [cm/day]",
                   "RD": "Crop rooting depth [cm]",
                   "SM": "Soil moisture index",
                   "WWLOW": "Total water [cm]"}

def bundle_agro_parameters(sitedata: PCSESiteDataProvider | Iterable[PCSESiteDataProvider],
                      soildata: CABOFileReader | Iterable[CABOFileReader],
                      cropdata: MultiCropDataProvider | Iterable[MultiCropDataProvider]) -> Iterable[ParameterProvider]:
    """
    Bundle the site, soil, and crop parameters into PCSE ParameterProvider objects.
    """
    # Make sure the data are iterable
    sitedata_iter = make_iterable(sitedata, exclude=[PCSESiteDataProvider])
    soildata_iter = make_iterable(soildata, exclude=[CABOFileReader])
    cropdata_iter = make_iterable(cropdata, exclude=[MultiCropDataProvider])

    # Combine everything
    combined_inputs = product(sitedata_iter, soildata_iter, cropdata_iter)
    parameters_combined = (ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in combined_inputs)

    return parameters_combined

def bundle_parameters(sitedata: PCSESiteDataProvider | Iterable[PCSESiteDataProvider],
                      soildata: CABOFileReader | Iterable[CABOFileReader],
                      cropdata: MultiCropDataProvider | Iterable[MultiCropDataProvider],
                      weatherdata: WeatherDataProvider | Iterable[WeatherDataProvider],
                      agromanagementdata: AgromanagementData | Iterable[AgromanagementData]) -> tuple[Iterable[[ParameterProvider, WeatherDataProvider, AgromanagementData]], int | None]:
    """
    Bundle the site, soil, and crop parameters into PCSE ParameterProvider objects.
    """
    # Make sure the data are iterable
    sitedata_iter = make_iterable(sitedata, exclude=[PCSESiteDataProvider])
    soildata_iter = make_iterable(soildata, exclude=[CABOFileReader])
    cropdata_iter = make_iterable(cropdata, exclude=[MultiCropDataProvider])
    weatherdata_iter = make_iterable(weatherdata, exclude=[WeatherDataProvider])
    agromanagementdata_iter = make_iterable(agromanagementdata, exclude=[AgromanagementData])

    # Determine the total number of parameter combinations, if possible
    try:
        n = len(sitedata_iter) * len(soildata_iter) * len(cropdata_iter) * len(weatherdata_iter) * len(agromanagementdata_iter)
    except TypeError:
        n = None

    # Combine everything
    agro_parameters = bundle_agro_parameters(sitedata_iter, soildata_iter, cropdata_iter)
    combined_parameters = product(agro_parameters, weatherdata_iter, agromanagementdata_iter)

    return combined_parameters, n

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

    # Add the run_id to the output and summary objects for tracking purposes
    try:
        output.run_id = run_id
    except TypeError:  # This happens if the run failed
        pass
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
    n_filtered_out = len(valid_entries) - sum(valid_entries)

    # Apply the filter
    outputs_filtered = [o for o, v in zip(outputs, valid_entries) if v]
    summary_filtered = [s for s, v in zip(summary, valid_entries) if v]

    return outputs_filtered, summary_filtered, n_filtered_out

def run_pcse_ensemble(all_runs, nr_runs=None):
    """
    Run an entire PCSE ensemble at once.
    all_runs is an iterator that zips the three model inputs (parameters, weatherdata, agromanagement) together, e.g.:
        all_runs = product(parameters_combined, weatherdata, agromanagementdata)
    """
    # Run the models
    outputs, summary = zip(*tqdm(map(run_wofost_with_id, all_runs), total=nr_runs, desc="Running PCSE models", unit="runs"))

    # Clean up the results
    outputs, summary, n_filtered_out = filter_ensemble_outputs(outputs, summary)
    if n_filtered_out > 0:
        print(f"{n_filtered_out} runs failed.")

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
    outputs, summary, n_filtered_out = filter_ensemble_outputs(outputs, summary)
    if n_filtered_out > 0:
        print(f"{n_filtered_out} runs failed.")

    # Convert the summary to a single dataframe
    summary = pd.DataFrame(summary).set_index("run_id")

    return outputs, summary
