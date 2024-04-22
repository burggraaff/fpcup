"""
Classes, functions, constants relating to running the WOFOST model and processing its inputs/outputs.
"""
from datetime import date, datetime
from functools import partial
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

from ._run_id import append_overrides, generate_run_id_base, generate_run_id_BRP
from ._typing import Callable, Coordinates, Iterable, Optional, PathOrStr, RealNumber
from .agro import AgromanagementData
from .constants import CRS_AMERSFOORT, WGS84
from .crop import default as default_cropdata
from .geo import format_coordinates
from .multiprocessing import multiprocess_file_io, multiprocess_pcse
from .site import default as default_sitedata
from .soil import SoilType
from .tools import copy, indent2
from .weather import load_weather_data_NASAPower

### CONSTANTS
_SUFFIX_RUNDATA = ".wrun"
_SUFFIX_OUTPUTS = ".wout"
_SUFFIX_SUMMARY = ".wsum"


### CLASSES THAT STORE PCSE INPUTS
class RunData(tuple):
    """
    Stores the data necessary to run a PCSE simulation.
    Primarily a tuple containing the parameters in the order PCSE expects them in, with some options for geometry and run_id generation.
    """
    ### OBJECT GENERATION AND INITIALISATION
    def __new__(cls, *, soildata: SoilType, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData,
                sitedata: PCSESiteDataProvider=default_sitedata, cropdata: MultiCropDataProvider=default_cropdata, **kwargs):
        """
        Creates (but does not initialise) a new RunData object from the 5 PCSE inputs.
        The first three inputs (sitedata, soildata, cropdata) are wrapped into a PCSE ParameterProvider.
        Default values can be used for sitedata and cropdata.
        The sitedata, soildata, and cropdata are shallow-copied to prevent edits in place.
        The weatherdata are not copied to conserve memory, and because these are treated as read-only in PCSE.
        """
        parameters = ParameterProvider(sitedata=copy(sitedata), soildata=copy(soildata), cropdata=copy(cropdata))
        return super().__new__(cls, (parameters, weatherdata, agromanagement))

    def __init__(self, *, soildata: SoilType, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData,
                 latitude: RealNumber, longitude: RealNumber,
                 sitedata: PCSESiteDataProvider=default_sitedata, cropdata: MultiCropDataProvider=default_cropdata,
                 override: Optional[dict]={},
                 run_id: Optional[str]=None, prefix: Optional[str]=None, suffix: Optional[str]=None):
        """
        Initialises a newly created RunData object.
        """
        # Easier access
        self.parameters = self[0]
        self.weatherdata = self[1]
        self.agromanagement = self[2]
        self.soiltype = soildata.name

        # Apply override parameters
        self.parameters.update(override)
        self.overrides = copy(self.parameters._override)

        # Assign latitude, longitude
        self.latitude = latitude
        self.longitude = longitude

        # Assign a run_id, either from user input or from the run parameters
        if run_id is None:
            run_id = self.generate_run_id()
            run_id = "_".join(s for s in (prefix, run_id, suffix) if s is not None)
        self.run_id = run_id


    ### SHORTCUTS TO PROPERTIES
    @property
    def crop_name(self) -> str:
        return self[2].crop_name

    @property
    def crop_variety(self) -> str:
        return self[2].crop_variety

    @property
    def sowdate(self) -> date:
        return self[2].crop_start_date


    ### STRING REPRESENTATIONS
    @property
    def site_description(self) -> str:
        return f"Site: {self.parameters._sitedata.__class__.__name__}(WAV={self.parameters['WAV']})"

    @property
    def soil_description(self) -> str:
        return f"Soil: {self.soiltype} ({self.parameters['SOLNAM']})"

    @property
    def crop_description(self) -> str:
        return f"Crop: {self.crop_name} ('{self.crop_variety}' from {self.parameters._cropdata.__class__.__name__})"

    @property
    def overrides_description(self) -> str:
        if len(self.overrides) == 0:
            overridetext = "None"
        else:
            overridetext = repr(self.overrides)
        return f"Overrides: {overridetext}"

    @property
    def parameter_description(self) -> str:
        individual_descriptions = "\n".join([self.site_description,
                                             self.soil_description,
                                             self.crop_description,
                                             self.overrides_description])

        return "\n".join([f"{self.parameters.__class__.__name__}",
                          indent2(individual_descriptions)])

    @property
    def weather_description(self) -> str:
        return f"{type(self.weatherdata).__name__} at {format_coordinates(self.weatherdata.latitude, self.weatherdata.longitude)}"

    @property
    def agro_description(self) -> str:
        return str(self[2])

    @property
    def geometry_description(self) -> str:
        return f"Location: {format_coordinates(self.latitude, self.longitude)}"

    def __repr__(self) -> str:
        return f"Run '{self.run_id}'"

    def __str__(self) -> str:
        individual_descriptions = "\n".join([self.parameter_description,
                                             self.weather_description,
                                             self.agro_description,
                                             self.geometry_description])

        return "\n".join([repr(self),
                          indent2(individual_descriptions)])


    ### RUN ID GENERATION
    def _generate_run_id_base(self) -> str:
        """
        Basic run ID generation; this function should be overridden by inheriting classes.
        """
        return generate_run_id_base(crop_name=self.crop_name, soiltype=self.soiltype, sowdate=self.sowdate, latitude=self.latitude, longitude=self.longitude)

    def generate_run_id(self) -> str:
        """
        Generate a run ID from PCSE model inputs.
        """
        run_id = self._generate_run_id_base()
        run_id = append_overrides(run_id, self.overrides)
        return run_id


    ### FILE INPUT / OUTPUT
    def summary_dict(self) -> dict:
        """
        Return inputs that should be in the Summary; mostly used for inheritance.
        """
        return {}

    def input_dict(self) -> dict:
        """
        Return all inputs as a dictionary.
        """
        return {"latitude": self.latitude,
                "longitude": self.longitude,
                "soiltype": self.soiltype,
                "crop": self.crop_name,
                "variety": self.crop_variety,
                "WAV": self.parameters["WAV"],
                "RDMSOL": self.parameters["RDMSOL"],
                "DOS": self.sowdate,
                **self.overrides}

    def to_file(self, output_directory: PathOrStr, **kwargs) -> None:
        """
        Save to a .wrun file, with a filename based on the current run_id.
        """
        # Set up filename
        output_directory = Path(output_directory)
        filename = output_directory / (self.run_id + _SUFFIX_RUNDATA)

        # Create dataframe and save
        df = pd.DataFrame(self.input_dict(), index=[self.run_id])
        df["DOS"] = pd.to_datetime(df["DOS"])
        df.to_csv(filename)


class RunDataBRP(RunData):
    """
    Same as RunData but specific to the BRP.
    """
    ### OBJECT GENERATION AND INITIALISATION
    def __init__(self, *, soildata: SoilType, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, brpdata: pd.Series, brpyear: int,
                sitedata: PCSESiteDataProvider=default_sitedata, cropdata: MultiCropDataProvider=default_cropdata, **kwargs):
        """
        Use a BRP data series to initialise the RunData object.
        `brpyear` is the BRP year, not the weatherdata year, so that e.g. a plot from the 2021 BRP can be simulated in 2022.
        """
        # Extract BRP data
        self.plot_id = brpdata.name
        self.crop_species = brpdata["crop_species"]
        self.crop_code = brpdata["crop_code"]
        self.area = brpdata["area"]
        self.province_name = brpdata["province"]  # Abbreviation, not Province object

        self.brpyear = brpyear

        super().__init__(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, latitude=brpdata["latitude"], longitude=brpdata["longitude"], **kwargs)


    ### RUN ID GENERATION
    def _generate_run_id_base(self) -> str:
        return generate_run_id_BRP(brpyear=self.brpyear, plot_id=self.plot_id, crop_name=self.crop_name, sowdate=self.sowdate)


    ### FILE INPUT / OUTPUT
    def brp_dict(self) -> dict:
        """
        Return those values that are specific to BRP data.
        """
        return {"brpyear": self.brpyear,
                "plot_id": self.plot_id,
                "province": self.province_name,
                "area": self.area,
                "crop_code": self.crop_code}


    def summary_dict(self) -> dict:
        """
        Return those values required by a Summary object as a dictionary.
        """
        return {**self.brp_dict(),
                **super().summary_dict()}

    def input_dict(self) -> dict:
        """
        Return all inputs as a dictionary.
        """
        return {**self.brp_dict(),
                **super().input_dict()}


### CLASSES THAT STORE PCSE OUTPUTS
class GeneralSummary(gpd.GeoDataFrame):
    """
    General class for Summary-like objects.
    Subclassed for inputs and outputs.
    """
    _suffix_default = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index.set_names("run_id", inplace=True)
        self.sort_index(inplace=True)

    @classmethod
    def from_file(cls, filename: PathOrStr):
        """
        Load a summary from a GeoJSON (.wrun / .wsum) file.
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
    def from_folder(cls, folder: PathOrStr, extension: Optional[str]=None, *,
                    use_existing=True, progressbar=True, leave_progressbar=True):
        """
        Load an ensemble of Summary objects from a folder and combine them.
        """
        if extension is None:
            extension = "*" + cls._suffix_default

        # Find all summary files in the folder, except a previously existing ensemble one (if it exists)
        folder = Path(folder)
        filenames = list(folder.glob(extension))
        assert len(filenames) > 0, f"No files with extension '{extension}' were found in folder {folder.absolute()}"
        filename_ensemble = filenames[0].with_stem("ensemble")
        ENSEMBLE_EXISTS = (filename_ensemble in filenames)

        # If there is an existing summary file, use that and append the new files to it
        if use_existing and ENSEMBLE_EXISTS:
            ensemble = cls.from_file(filename_ensemble)  # Note that the ensemble may get loaded twice
            filenames = [f for f in filenames if f.stem not in ensemble.index]

            # If all files are in the ensemble already, simply return that
            if len(filenames) == 1:
                return ensemble

        # If desired, do not load the ensemble, but use all individual summaries instead
        elif not use_existing and ENSEMBLE_EXISTS:
            filenames.remove(filename_ensemble)

        # Load the files (with a tqdm progressbar if desired)
        summaries_individual = multiprocess_file_io(cls.from_file, filenames, progressbar=progressbar, leave_progressbar=leave_progressbar, desc="Loading summaries")

        return cls.from_ensemble(summaries_individual)

    def to_file(self, filename: PathOrStr, **kwargs) -> None:
        """
        Save to a GeoJSON file.
        The index has to be reset first to ensure it is written to file (annoying 'feature' of the GeoJSON driver).
        """
        self.reset_index(inplace=True)
        write_geodataframe(self, filename, driver="GeoJSON", **kwargs)
        self.set_index("run_id", inplace=True)


class InputSummary(GeneralSummary):
    """
    Store a summary of the inputs for a PCSE ensemble run.
    """
    _suffix_default = _SUFFIX_RUNDATA

    @classmethod
    def from_rundata(cls, run_data: RunData):
        """
        Generate an Inputs objects from a RunData object.
        """
        return cls(index=[run_data.run_id], data=run_data.input_dict(), geometry=[run_data.geometry], crs=run_data.crs)


class Summary(GeneralSummary):
    """
    Stores a summary of the results from a PCSE ensemble run.
    """
    _suffix_default = _SUFFIX_SUMMARY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index.set_names("run_id", inplace=True)
        self.sort_index(inplace=True)

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData, *,
                   crs: Optional[str]=None, **kwargs):
        """
        Generate a summary from a finished model, inserting the run_id as an index.
        """
        data = model.get_summary_output()

        # Get data from run_data
        data[0] = {**run_data.summary_dict(), **data[0]}
        index = [run_data.run_id]
        if crs is None:
            crs = run_data.crs

        return cls(data, index=index, crs=crs, **kwargs)


class Result(pd.DataFrame):
    """
    Stores the results from a single PCSE run.
    Essentially a DataFrame that is initialised from a PCSE model object and contains some useful additional variables.
    """
    _internal_names = pd.DataFrame._internal_names + ["run_id", "summary"]
    _internal_names_set = set(_internal_names)

    def __init__(self, data: Iterable, *,
                 run_id: str="", summary: Optional[Summary]=None, **kwargs):
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
    def from_file(cls, filename: PathOrStr, *,
                  run_id: Optional[str]=None, include_summary=False, **kwargs):
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
            summary_filename = filename.with_suffix(_SUFFIX_SUMMARY)
            try:
                summary = Summary.from_file(summary_filename)
            except FileNotFoundError:
                summary = None
        else:
            summary = None

        return cls(data, run_id=run_id, summary=summary, **kwargs)

    def to_file(self, output_directory: PathOrStr, *,
                filename: Optional[PathOrStr]=None, **kwargs) -> None:
        """
        Save the results and summary to output files:
            output_directory / filename.wout - full results, CSV
            output_directory / filename.wsum - summary results, GeoJSON

        If no filename is provided, default to using the run ID.
        """
        output_directory = Path(output_directory)

        # Generate the output filenames from the user input (`filename`) or from the run id (default)
        # The suffix step cannot be done with just .with_suffix(".wout") in case the filename contains .
        if filename is not None:
            filename_base = output_directory / filename
        else:
            filename_base = output_directory / self.run_id
        filename_results = filename_base.with_suffix(filename_base.suffix + _SUFFIX_OUTPUTS)
        filename_summary = filename_base.with_suffix(filename_base.suffix + _SUFFIX_SUMMARY)

        # Save the outputs
        self.to_csv(filename_results, **kwargs)
        self.summary.to_file(filename_summary, **kwargs)


### Running PCSE
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
        site_constants = {"geometry": c, "weatherdata": weatherdata}
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
