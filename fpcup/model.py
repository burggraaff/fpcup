"""
Functions that are useful
"""
from datetime import date, datetime
from itertools import product
from multiprocessing import Pool  # Multi-threading
from pathlib import Path

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
from pyogrio import read_dataframe as read_geodataframe, write_dataframe as write_geodataframe
import pandas as pd
import shapely
from tqdm import tqdm

from pcse.base import MultiCropDataProvider, ParameterProvider, WeatherDataProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.fileinput import CABOFileReader
from pcse.models import Engine, Wofost72_WLP_FD
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from ._typing import Callable, Coordinates, Iterable, Optional, PathOrStr
from .agro import AgromanagementData
from .constants import CRS_AMERSFOORT
from .tools import make_iterable

# Parameter names are from "A gentle introduction to WOFOST", De Wit & Boogaard 2021
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
                   "WWLOW": "Total water [cm]",
                   "LAIMAX": "Maximum leaf area index [ha/ha]",
                   "CTRAT": "Cumulative crop transpiration [cm]",
                   "DOS": "Date of sowing",
                   "DOE": "Date of emergence",
                   "DOA": "Date of anthesis",
                   "DOM": "Date of maturity",
                   "DOH": "Date of harvest",
                   "DOV": "Date of vernalisation",
                   "CEVST": "Cumulative soil evaporation [cm]",
                   "area": "Total area [ha]",
                   "n": "Number of sites",
                   }


class RunData(tuple):
    """
    Stores the data necessary to run a PCSE simulation.
    Primarily a tuple containing the parameters in the order PCSE expects them in, with some options for geometry and run_id generation.
    """
    def __new__(cls, sitedata: PCSESiteDataProvider, soildata: CABOFileReader, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, **kwargs):
        parameters = ParameterProvider(sitedata=sitedata, soildata=soildata, cropdata=cropdata)
        return super().__new__(cls, (parameters, weatherdata, agromanagement))

    def __init__(self, sitedata: PCSESiteDataProvider, soildata: CABOFileReader, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, *, run_id: Optional[str]=None, geometry: Optional[shapely.Geometry | tuple]=None, crs=None):
        # Easier access
        self.sitedata = sitedata
        self.soildata = soildata
        self.cropdata = cropdata
        self.parameters = self[0]
        self.weatherdata = self[1]
        self.agromanagement = self[2]
        self.crop = self.agromanagement.crop_name
        self.soiltype = soildata["SOLNAM"]

        # Set up the shapely geometry object
        if geometry is None:
            # Get the coordinates from the weatherdata; note that this often introduces a loss of precision!
            geometry = shapely.Point(self.weatherdata.longitude, self.weatherdata.latitude)
        elif isinstance(geometry, tuple):
            # Pairs of coordinates - assume these are in (lat, lon) format
            latitude, longitude = geometry
            geometry = shapely.Point(longitude, latitude)
        self.geometry = geometry
        self.crs = crs

        # Assign a run_id, either from user input or from the run parameters
        if run_id is None:
            run_id = self.generate_run_id()
        self.run_id = run_id

    def __repr__(self) -> str:
        text_parameters = type(self.parameters).__name__

        text_weather = f"{type(self.weatherdata).__name__} at ({self.weatherdata.latitude:+.4f} N, {self.weatherdata.longitude:+.4f} E)"

        if isinstance(self.geometry, shapely.Geometry):
            text_geometry = f"{self.geometry.geom_type}, centroid ({self.geometry.centroid.x:.4f}, {self.geometry.centroid.y:.4f}) (CRS: {self.crs})"
        else:
            text_geometry = self.geometry

        return (f"Run '{self.run_id}':\n"
                f"ParameterProvider: {text_parameters}\n"
                f"WeatherDataProvider: {text_weather}\n"
                f"AgromanagementData: {self[2]}\n"
                f"Geometry: {text_geometry}")

    def generate_run_id(self) -> str:
        """
        Generate a run ID from PCSE model inputs.
        """
        sowdate = self.agromanagement.crop_start_date

        run_id = f"{self.crop}_{self.soiltype}_sown{sowdate:%Y%j}_lat{self.geometry.y:.7f}-lon{self.geometry.x:.7f}"

        return run_id


def run_id_BRP(brpyear: int | str, plot_id: int | str, crop: str, sowdate: date) -> str:
    """
    Generate a run ID from BRP data.
    Separate from the RunDataBRP class so it can be called before initialising the weather data (which tends to be slow) when checking for duplicate files.
    """
    return f"brp{brpyear}-plot{plot_id}-{crop}-sown{sowdate:%Y%j}"


class RunDataBRP(RunData):
    """
    Same as RunData but specific to the BRP.
    """
    def __init__(self, sitedata: PCSESiteDataProvider, soildata: CABOFileReader, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, brpdata: pd.Series, brpyear: int, crs=CRS_AMERSFOORT):
        """
        Use a BRP data series to initialise the RunData object.
        `brpyear` is the BRP year, not the weatherdata year, so that e.g. a plot from the 2021 BRP can be simulated in 2022.
        """
        # Extract BRP data
        self.plot_id = brpdata.name
        self.crop_species = brpdata["crop_species"]
        self.crop_code = brpdata["crop_code"]
        self.area = brpdata["area"]
        self.province = brpdata["province"]

        self.brpyear = brpyear

        super().__init__(sitedata, soildata, cropdata, weatherdata, agromanagement, geometry=brpdata["geometry"], crs=crs)

    def generate_run_id(self) -> str:
        sowdate = self.agromanagement.crop_start_date
        return run_id_BRP(self.brpyear, self.plot_id, self.crop, sowdate)


class Summary(gpd.GeoDataFrame):
    """
    Stores a summary of the results from a PCSE ensemble run.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index.set_names("run_id", inplace=True)
        self.sort_index(inplace=True)

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData, *, crs: Optional[str]=None, **kwargs):
        """
        Generate a summary from a finished model, inserting the run_id as an index.
        """
        summary = model.get_summary_output()

        # Get data from run_data
        summary[0]["geometry"] = run_data.geometry
        summary[0]["soiltype"] = run_data.soiltype
        index = [run_data.run_id]

        if crs is None:
            crs = run_data.crs

        # Extract more information for the BRP case
        if isinstance(run_data, RunDataBRP):
            summary[0]["plot_id"] = run_data.plot_id
            summary[0]["crop_species"] = run_data.crop_species
            summary[0]["crop_code"] = run_data.crop_code
            summary[0]["area"] = run_data.area
            summary[0]["province"] = run_data.province
            summary[0]["brpyear"] = run_data.brpyear

        return cls(summary, index=index, crs=crs, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr):
        """
        Load a summary from a GeoJSON (.wsum) file.
        """
        data = read_geodataframe(filename)
        data.set_index("run_id", inplace=True)
        return cls(data)

    @classmethod
    def from_ensemble(cls, summaries_individual: Iterable):
        """
        Combine many Summary objects into one through concatenation.
        """
        # Uses the regular Pandas concat function; when the outputs are GeoDataFrames (or Summaries), the result is a GeoDataFrame
        data = pd.concat(summaries_individual)
        return cls(data)

    @classmethod
    def from_folder(cls, folder: PathOrStr, extension: Optional[str]="*.wsum", progressbar=True, leave_progressbar=True):
        """
        Load an ensemble of Summary objects from a folder and combine them.
        """
        # Find all summary files in the folder, except a previously existing ensemble one (if it exists)
        folder = Path(folder)
        filenames = list(folder.glob(extension))
        assert len(filenames) > 0, f"No files with extension '{extension}' were found in folder {folder.absolute()}"
        filename_ensemble = filenames[0].with_stem("ensemble")
        if filename_ensemble in filenames:
            filenames.remove(filename_ensemble)

        # Load the files (with a tqdm progressbar if desired)
        filenames = tqdm(filenames, desc="Loading summaries", unit="file", disable=not progressbar, leave=leave_progressbar)
        summaries_individual = [cls.from_file(filename) for filename in filenames]
        return cls.from_ensemble(summaries_individual)

    def to_file(self, filename: PathOrStr, **kwargs) -> None:
        """
        Save to as a GeoJSON (.wsum) file.
        The index has to be reset first to ensure it is written to file (annoying 'feature' of the GeoJSON driver).
        """
        self.reset_index(inplace=True)
        write_geodataframe(self, filename, driver="GeoJSON", **kwargs)
        self.set_index("run_id", inplace=True)

class Result(pd.DataFrame):
    """
    Stores the results from a single PCSE run.
    Essentially a DataFrame that is initialised from a PCSE model object and contains some useful additional variables.
    """
    _internal_names = pd.DataFrame._internal_names + ["run_id", "summary"]
    _internal_names_set = set(_internal_names)

    def __init__(self, data: Iterable, *, run_id: str="", summary: Optional[Summary]=None, **kwargs):
        # Initialise the main DataFrame from the model output
        super().__init__(data, **kwargs)

        # Sort the results by time
        self.set_index("day", inplace=True)

        # Add the run ID and summary
        self.run_id = run_id
        self.summary = summary

    def __repr__(self) -> str:
        return ("-----\n"
                f"Run ID: {self.run_id}\n\n"
                f"Summary: {self.summary}\n\n"
                f"Data:\n{super().__repr__()}"
                "\n-----")

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData, **kwargs):
        """
        Initialise the main DataFrame from a model output.
        """
        output = model.get_output()

        # Get data from run_data if possible
        run_id = run_data.run_id

        # Save the summary output
        try:
            summary = Summary.from_model(model, run_data=run_data)
        except IndexError:
            summary = None

        return cls(output, run_id=run_id, summary=summary, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, *, run_id: Optional[str]=None, include_summary=True, **kwargs):
        """
        Load an output file.
        If a run_id is not provided, use the filename stem.
        """
        filename = Path(filename)
        if run_id is None:
            run_id = filename.stem

        # Load the main data file
        data = pd.read_csv(filename)

        # Try to load the associated summary file
        if include_summary:
            summary_filename = filename.with_suffix(".wsum")
            try:
                summary = Summary.from_file(summary_filename)
            except FileNotFoundError:
                summary = None
        else:
            summary = None

        return cls(data, run_id=run_id, summary=summary, **kwargs)

    def to_file(self, output_directory: PathOrStr, *, filename: Optional[PathOrStr]=None, **kwargs) -> None:
        """
        Save the results and summary to output files:
            output_directory / filename.wout - full results, CSV
            output_directory / filename.wsum - summary results, GeoJSON

        If no filename is provided, default to using the run ID.
        """
        output_directory = Path(output_directory)

        # Generate the output filenames from the user input (`filename`) or from the run id (default)
        if filename is not None:
            filename_base = output_directory / filename
        else:
            filename_base = output_directory / self.run_id
        filename_results = filename_base.with_suffix(filename_base.suffix+".wout")
        filename_summary = filename_base.with_suffix(filename_base.suffix+".wsum")

        # Save the outputs
        self.to_csv(filename_results, **kwargs)
        self.summary.to_file(filename_summary, **kwargs)


def bundle_parameters(sitedata: PCSESiteDataProvider | Iterable[PCSESiteDataProvider],
                      soildata: CABOFileReader | Iterable[CABOFileReader],
                      cropdata: MultiCropDataProvider | Iterable[MultiCropDataProvider],
                      weatherdata: WeatherDataProvider | Iterable[WeatherDataProvider],
                      agromanagementdata: AgromanagementData | Iterable[AgromanagementData]) -> tuple[Iterable[RunData], int | None]:
    """
    Bundle the site, soil, and crop parameters into PCSE ParameterProvider objects.
    For the main parameters, a Cartesian product is used to get all their combinations.
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
    combined_parameters = product(sitedata_iter, soildata_iter, cropdata_iter, weatherdata_iter, agromanagementdata_iter)
    rundata = (RunData(*params) for params in combined_parameters)

    return rundata, n

def run_pcse_single(run_data: RunData, *, model: Engine=Wofost72_WLP_FD) -> Result | None:
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
        # Convert individual output to a Result object (modified Pandas DataFrame)
        output = Result.from_model(wofost, run_data=run_data)

    return output

def filter_ensemble_outputs(outputs: Iterable[Result | None], summary: Iterable[dict | None]) -> tuple[list[Result], list[dict], int]:
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

def run_pcse_ensemble(all_runs: Iterable[RunData], nr_runs: Optional[int]=None, progressbar=True, leave_progressbar=True) -> tuple[list[Result], Summary]:
    """
    Run an entire PCSE ensemble.
    all_runs is an iterator that zips the three model inputs (parameters, weatherdata, agromanagement) together, e.g.:
        all_runs = product(parameters_combined, weatherdata, agromanagementdata)
    """
    # Run the models
    outputs = [run_pcse_single(run_data) for run_data in tqdm(all_runs, total=nr_runs, desc="Running PCSE models", unit="runs", disable=not progressbar, leave=leave_progressbar)]

    # Get the summaries
    summaries_individual = [o.summary for o in outputs]

    # Clean up the results
    outputs, summaries_individual, n_filtered_out = filter_ensemble_outputs(outputs, summaries_individual)
    if n_filtered_out > 0:
        print(f"{n_filtered_out} runs failed.")

    # Convert the summary to a Summary object (modified DataFrame)
    summary = Summary.from_ensemble(summaries_individual)

    return outputs, summary

def run_pcse_ensemble_parallel(all_runs: Iterable[RunData], nr_runs: Optional[int]=None, progressbar=True, leave_progressbar=True) -> tuple[list[Result], Summary]:
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
    summaries_individual = [o.summary for o in outputs]

    # Clean up the results
    outputs, summaries_individual, n_filtered_out = filter_ensemble_outputs(outputs, summaries_individual)
    if n_filtered_out > 0:
        print(f"{n_filtered_out} runs failed.")

    # Convert the summary to a Summary object (modified DataFrame)
    summary = Summary.from_ensemble(summaries_individual)

    return outputs, summary
