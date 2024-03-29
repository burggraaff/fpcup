"""
Classes, functions, constants relating to running the WOFOST model and processing its inputs/outputs.
"""
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
from pyogrio import read_dataframe as read_geodataframe, write_dataframe as write_geodataframe
import pandas as pd
import shapely

from pcse.base import MultiCropDataProvider, ParameterProvider, WeatherDataProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.fileinput import CABOFileReader
from pcse.models import Engine, Wofost72_WLP_FD
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from ._typing import Callable, Coordinates, Iterable, Optional, PathOrStr
from .agro import AgromanagementData
from .constants import CRS_AMERSFOORT
from .multiprocessing import multiprocess_file_io, multiprocess_pcse
from .soil import SoilType
from .tools import copy, indent2

### Constants
_SUFFIX_RUNDATA = ".wrun"
_SUFFIX_OUTPUTS = ".wout"
_SUFFIX_SUMMARY = ".wsum"
_GEOMETRY_DEFAULT = (0, 0)


### Classes to help store PCSE inputs and outputs
class RunData(tuple):
    """
    Stores the data necessary to run a PCSE simulation.
    Primarily a tuple containing the parameters in the order PCSE expects them in, with some options for geometry and run_id generation.
    """
    def __new__(cls, sitedata: PCSESiteDataProvider, soildata: SoilType, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, **kwargs):
        """
        Creates (but does not initialise) a new RunData object from the 5 PCSE inputs.
        The first three inputs (sitedata, soildata, cropdata) are wrapped into a PCSE ParameterProvider.
        The sitedata, soildata, and cropdata are shallow-copied to prevent edits in place.
        The weatherdata are not copied to conserve memory, and because these are treated as read-only in PCSE.
        """
        parameters = ParameterProvider(sitedata=copy(sitedata), soildata=copy(soildata), cropdata=copy(cropdata))
        return super().__new__(cls, (parameters, weatherdata, agromanagement))

    def __init__(self, sitedata: PCSESiteDataProvider, soildata: SoilType, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, *,
                 override: Optional[dict]={},
                 run_id: Optional[str]=None, prefix: Optional[str]=None, suffix: Optional[str]=None,
                 geometry: Optional[shapely.Geometry | Coordinates]=None, crs=None):
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

        # Assign geometry parameters
        self._initialise_geometry(geometry, crs)

        # Assign a run_id, either from user input or from the run parameters
        if run_id is None:
            run_id = self.generate_run_id()
            run_id = "_".join(s for s in (prefix, run_id, suffix) if s is not None)
        self.run_id = run_id

    def _initialise_geometry(self, geometry: Optional[shapely.Geometry | Coordinates]=None, crs: Optional[str]=None) -> None:
        """
        Add geometry/CRS properties to self.
        """
        # Set up the shapely geometry object
        if geometry is None:
            print(f"Warning: no geometry data provided - defaulting to {_GEOMETRY_DEFAULT}.")
            geometry = _GEOMETRY_DEFAULT

        if isinstance(geometry, tuple):
            # Pairs of coordinates - assume these are in (lat, lon) format
            latitude, longitude = geometry
            geometry = shapely.Point(longitude, latitude)

        self.geometry = geometry
        self.crs = crs

    @property
    def crop_name(self) -> str:
        return self[2].crop_name

    @property
    def crop_variety(self) -> str:
        return self[2].crop_variety

    @property
    def latitude(self) -> float:
        return self.geometry.centroid.y

    @property
    def longitude(self) -> float:
        return self.geometry.centroid.x

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
        return f"{type(self.weatherdata).__name__} at ({self.weatherdata.latitude:+.4f} N, {self.weatherdata.longitude:+.4f} E)"

    @property
    def agro_description(self) -> str:
        return str(self[2])

    @property
    def geometry_description(self) -> str:
        if isinstance(self.geometry, shapely.Geometry):
            return f"{self.geometry.geom_type}, centroid ({self.latitude:.4f}, {self.longitude:.4f}) (CRS: {self.crs})"
        else:
            return str(self.geometry)

    def __repr__(self) -> str:
        return f"Run '{self.run_id}'"

    def __str__(self) -> str:
        individual_descriptions = "\n".join([self.parameter_description,
                                             self.weather_description,
                                             self.agro_description,
                                             self.geometry_description])

        return "\n".join([repr(self),
                          indent2(individual_descriptions)])

    def _generate_run_id_base(self) -> str:
        """
        Basic run ID generation; this function should be overridden by inheriting classes.
        """
        sowdate = self.agromanagement.crop_start_date
        return f"{self.crop_name}_{self.soiltype}_dos{sowdate:%Y%j}_lat{self.latitude:.7f}-lon{self.longitude:.7f}"

    def generate_run_id(self) -> str:
        """
        Generate a run ID from PCSE model inputs.
        """
        run_id = self._generate_run_id_base()

        if len(self.overrides) > 0:
            override_text = "_".join(f"{key}-{value}" for key, value in sorted(self.overrides.items()))
            run_id = "_".join([run_id, override_text])

        return run_id

    def summary_dict(self) -> dict:
        """
        Return inputs that should be in the Summary; mostly used for inheritance.
        """
        return {"geometry": self.geometry}

    def input_dict(self) -> dict:
        """
        Return all inputs as a dictionary.
        """
        return {"WAV": self.parameters["WAV"],
                "soiltype": self.soiltype,
                "crop": self.crop_name,
                "variety": self.crop_variety,
                **self.overrides,
                "geometry": self.geometry}

    def to_file(self, output_directory: PathOrStr, **kwargs) -> None:
        """
        Save to a GeoJSON (.wrun) file.
        Essentially shorthand for creating an InputSummary object and saving that to file, with a filename based on the current run_id.
        """
        # Set up filename
        output_directory = Path(output_directory)
        filename = output_directory / (self.run_id + _SUFFIX_RUNDATA)

        # Create dataframe and save
        input_summary = InputSummary.from_rundata(self)
        input_summary.to_file(filename, **kwargs)


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
    def __init__(self, sitedata: PCSESiteDataProvider, soildata: CABOFileReader, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, brpdata: pd.Series, brpyear: int, *,
                 crs=CRS_AMERSFOORT, **kwargs):
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

        super().__init__(sitedata, soildata, cropdata, weatherdata, agromanagement, geometry=brpdata["geometry"], crs=crs, **kwargs)

    def _generate_run_id_base(self) -> str:
        sowdate = self.agromanagement.crop_start_date
        return run_id_BRP(self.brpyear, self.plot_id, self.crop_name, sowdate)

    def summary_dict(self) -> dict:
        """
        Return those values required by a Summary object as a dictionary.
        """
        return {"province": self.province,
                "crop_species": self.crop_species,
                "area": self.area,
                "brpyear": self.brpyear,
                **super().summary_dict()}

    def input_dict(self) -> dict:
        """
        Return all inputs as a dictionary.
        """
        return {"plot_id": self.plot_id,
                "province": self.province,
                "crop_species": self.crop_species,
                "crop_code": self.crop_code,
                "area": self.area,
                "brpyear": self.brpyear,
                **super().input_dict()}


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
