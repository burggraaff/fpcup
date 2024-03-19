"""
Classes, functions, constants relating to running the WOFOST model and processing its inputs/outputs.
"""
from datetime import date, datetime
from functools import partial
from itertools import product
from multiprocessing import Pool
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
from .soil import SoilType
from .tools import make_iterable

# Constants
_THRESHOLD_PARALLEL_PCSE = 200

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
    def __new__(cls, sitedata: PCSESiteDataProvider, soildata: SoilType, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, **kwargs):
        parameters = ParameterProvider(sitedata=sitedata, soildata=soildata, cropdata=cropdata)
        return super().__new__(cls, (parameters, weatherdata, agromanagement))

    def __init__(self, sitedata: PCSESiteDataProvider, soildata: SoilType, cropdata: MultiCropDataProvider, weatherdata: WeatherDataProvider, agromanagement: AgromanagementData, *,
                 run_id: Optional[str]=None, geometry: Optional[shapely.Geometry | tuple]=None, crs=None):
        # Easier access
        self.sitedata = sitedata
        self.soildata = soildata
        self.cropdata = cropdata
        self.parameters = self[0]
        self.weatherdata = self[1]
        self.agromanagement = self[2]
        self.crop = self.agromanagement.crop_name
        self.soiltype = soildata.name

        # Set up the shapely geometry object
        if geometry is None:
            print("Warning: no geometry data provided - defaulting to (0, 0).")
            geometry = (0, 0)

        if isinstance(geometry, tuple):
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
        text_soil = f"{self.soiltype} ({self.soildata['SOLNAM']})"
        text_crop = f"{self.cropdata.current_crop_name} ('{self.cropdata.current_variety_name}' from {type(self.cropdata).__name__})"

        text_weather = f"{type(self.weatherdata).__name__} at ({self.weatherdata.latitude:+.4f} N, {self.weatherdata.longitude:+.4f} E)"

        if isinstance(self.geometry, shapely.Geometry):
            text_geometry = f"{self.geometry.geom_type}, centroid ({self.geometry.centroid.x:.4f}, {self.geometry.centroid.y:.4f}) (CRS: {self.crs})"
        else:
            text_geometry = self.geometry

        return (f"Run '{self.run_id}':\n"
                f"Soil: {text_soil}\n"
                f"Crop: {text_crop}\n"
                f"WeatherDataProvider: {text_weather}\n"
                f"AgromanagementData: {self[2]}\n"
                f"Geometry: {text_geometry}")

    def generate_run_id(self) -> str:
        """
        Generate a run ID from PCSE model inputs.
        """
        sowdate = self.agromanagement.crop_start_date

        run_id = f"{self.crop}_{self.soiltype}_dos{sowdate:%Y%j}_lat{self.geometry.y:.7f}-lon{self.geometry.x:.7f}"

        return run_id

    def as_summary_dict(self) -> dict:
        """
        Return those values required by a Summary object as a dictionary.
        """
        return {"soiltype": self.soiltype,
                "geometry": self.geometry}


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

    def as_summary_dict(self) -> dict:
        """
        Return those values required by a Summary object as a dictionary.
        """
        summary_general = super().as_summary_dict()

        return {"plot_id": self.plot_id,
                "province": self.province,
                "crop_species": self.crop_species,
                "crop_code": self.crop_code,
                "area": self.area,
                "brpyear": self.brpyear,
                **summary_general}


class Summary(gpd.GeoDataFrame):
    """
    Stores a summary of the results from a PCSE ensemble run.
    """
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
        summary = model.get_summary_output()

        # Get data from run_data
        summary[0] = {**summary[0], **run_data.as_summary_dict()}
        index = [run_data.run_id]
        if crs is None:
            crs = run_data.crs

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
    def from_folder(cls, folder: PathOrStr, extension: Optional[str]="*.wsum", *,
                    use_existing=True, progressbar=True, leave_progressbar=True):
        """
        Load an ensemble of Summary objects from a folder and combine them.
        """
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
        n = len(filenames)
        filenames = tqdm(filenames, desc="Loading summaries", unit="file", disable=not progressbar, leave=leave_progressbar)
        if n < 1000:
            summaries_individual = map(cls.from_file, filenames)
        else:
            with Pool() as p:
                summaries_individual = list(p.imap_unordered(cls.from_file, filenames, chunksize=100))

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
            summary_filename = filename.with_suffix(".wsum")
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
        if filename is not None:
            filename_base = output_directory / filename
        else:
            filename_base = output_directory / self.run_id
        filename_results = filename_base.with_suffix(filename_base.suffix+".wout")
        filename_summary = filename_base.with_suffix(filename_base.suffix+".wsum")

        # Save the outputs
        self.to_csv(filename_results, **kwargs)
        self.summary.to_file(filename_summary, **kwargs)


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


def run_pcse_ensemble(func_pcse: Callable, data_pcse: Iterable, *,
                      n: Optional[int]=None, chunksize: int=10,
                      unit: str="run",
                      verbose: bool=False) -> Iterable[RunData]:
    """
    Run an entire PCSE ensemble.
    `func_pcse` is a function that runs PCSE for a single instance of `data_pcse`.
    For example, data_pcse might be a list of coordinates, for which func_pcse retrieves the corresponding site/weather data and then runs WOFOST.
    Models are run in parallel if there are more than _THRESHOLD_PARALLEL_PCSE to run.
    """
    # Determine the number of inputs and set up a progressbar
    if n is None:
        try:
            n = len(data_pcse)
        except TypeError:
            n = None
    _tqdm_here = partial(tqdm, total=n, desc="Running models", unit=unit, leave=verbose)

    # Determine whether to use multiprocessing
    if n is None:
        USE_MULTIPROCESSING = False
    elif n < _THRESHOLD_PARALLEL_PCSE:
        USE_MULTIPROCESSING = False
    else:
        USE_MULTIPROCESSING = True

    # Actually run the model
    if USE_MULTIPROCESSING:
        with Pool() as p:
            outputs = list(_tqdm_here(p.imap_unordered(func_pcse, data_pcse, chunksize=chunksize)))
    else:
        outputs = list(map(func_pcse, _tqdm_here(data_pcse)))

    # Determine which runs failed / were skipped
    failed_runs = [o for o in outputs if isinstance(o, RunData)]
    if len(failed_runs) > 0:
        print(f"Number of failed runs: {len(failed_runs)}/{n}")
    else:
        if verbose:
            print("No runs failed.")

    skipped_runs = [o for o in outputs if o is False]
    if verbose:
        print(f"Number of skipped runs: {len(skipped_runs)}/{n}")

    return failed_runs
