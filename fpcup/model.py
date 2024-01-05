"""
Functions that are useful
"""
from itertools import product
from multiprocessing import Pool  # Multi-threading
from typing import Callable, Iterable

import pandas as pd
from tqdm import tqdm

from pcse.base import MultiCropDataProvider, ParameterProvider, WeatherDataProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.fileinput import CABOFileReader
from pcse.models import Engine, Wofost72_WLP_FD
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

class Result(pd.DataFrame):
    """
    Stores the results from a single PCSE run.
    Essentially a DataFrame that is initialised from a PCSE model object and contains some useful additional variables.
    """
    _internal_names = pd.DataFrame._internal_names + ["run_id", "summary"]
    _internal_names_set = set(_internal_names)

    def __init__(self, model: Engine, run_id: str=""):
        # Initialise the main DataFrame from the model output
        output = model.get_output()
        super().__init__(output)

        # Sort the results by time
        self.set_index("day", inplace=True)

        # Add the run ID
        # Note: it is not possible to generate these within __init__ because the crop calendar data in model.agromanager are destroyed while the model is run
        self.run_id = run_id

        # Save the summary output
        try:
            self.summary = model.get_summary_output()[0]
        except IndexError:
            self.summary = None
        else:
            self.summary["run_id"] = self.run_id

    def __repr__(self) -> str:
        return ("-----\n"
                f"Run ID: {self.run_id}\n\n"
                f"Summary: {self.summary}\n\n"
                f"Data:\n{super().__repr__()}"
                "\n-----")

class Summary(pd.DataFrame):
    """
    Stores a summary of the results from a PCSE ensemble run.
    """
    def __init__(self, data: Iterable):
        # Initialise the main DataFrame
        super().__init__(data)

        # Sort the results by run ID
        self.set_index("run_id", inplace=True)

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

def run_id_from_params(parameters: ParameterProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData):
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

def run_pcse_single(run_data: tuple[ParameterProvider, WeatherDataProvider, AgromanagementData], *, model: Engine=Wofost72_WLP_FD, run_id_function: Callable=run_id_from_params) -> Result:
    """
    Start a new PCSE model with the given inputs and run it until it terminates.
    """
    parameters, weatherdata, agromanagement = run_data
    run_id = run_id_function(parameters, weatherdata, agromanagement)
    try:
        wofost = model(parameters, weatherdata, agromanagement)
        wofost.run_till_terminate()
    except WeatherDataProviderError as e:
        # msg = f"Runid '{run_id}' failed because of missing weather data."
        # print(msg)
        output = None
    else:
        # Convert individual output to a Result object (modified Pandas DataFrame)
        output = Result(wofost, run_id=run_id)

    return output

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

def run_pcse_ensemble(all_runs: Iterable[tuple[ParameterProvider, WeatherDataProvider, AgromanagementData]], nr_runs:int | None=None, progressbar=True, leave_progressbar=True) -> tuple[Result, Summary]:
    """
    Run an entire PCSE ensemble at once.
    all_runs is an iterator that zips the three model inputs (parameters, weatherdata, agromanagement) together, e.g.:
        all_runs = product(parameters_combined, weatherdata, agromanagementdata)
    """
    # Run the models
    outputs = [run_pcse_single(run_data) for run_data in tqdm(all_runs, total=nr_runs, desc="Running PCSE models", unit="runs", disable=not progressbar, leave=leave_progressbar)]

    # Get the summaries
    summary = [o.summary for o in outputs]

    # Clean up the results
    outputs, summary, n_filtered_out = filter_ensemble_outputs(outputs, summary)
    if n_filtered_out > 0:
        print(f"{n_filtered_out} runs failed.")

    # Convert the summary to a Summary object (modified DataFrame)
    summary = Summary(summary)

    return outputs, summary

def run_pcse_ensemble_parallel(all_runs: Iterable[tuple[ParameterProvider, WeatherDataProvider, AgromanagementData]], nr_runs:int | None=None, progressbar=True, leave_progressbar=True) -> tuple[Result, Summary]:
    """
    Note: Very unstable!
    Parallelised version of run_pcse_ensemble.

    Run an entire PCSE ensemble at once.
    all_runs is an iterator that zips the three model inputs (parameters, weatherdata, agromanagement) together, e.g.:
        all_runs = product(parameters_combined, weatherdata, agromanagementdata)
    """
    # Run the models
    with Pool() as p:
        # outputs = tqdm(p.map(run_pcse_single, all_runs, chunksize=3), total=nr_runs, desc="Running PCSE models", unit="runs", disable=not progressbar, leave=leave_progressbar)
        outputs = p.map(run_pcse_single, all_runs, chunksize=3)

    # Get the summaries
    summary = [o.summary for o in outputs]

    # Clean up the results
    outputs, summary, n_filtered_out = filter_ensemble_outputs(outputs, summary)
    if n_filtered_out > 0:
        print(f"{n_filtered_out} runs failed.")

    # Convert the summary to a Summary object (modified DataFrame)
    summary = Summary(summary)

    return outputs, summary
