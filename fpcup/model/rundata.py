"""
Everything related to RunData classes that encapsulate WOFOST/PCSE input data.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from pcse.base import MultiCropDataProvider, ParameterProvider, WeatherDataProvider
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from .run_id import append_overrides, generate_run_id_base, generate_run_id_BRP
from ..agro import AgromanagementData
from ..crop import default as default_cropdata
from ..geo import format_coordinates
from ..site import default as default_sitedata
from ..soil import SoilType
from ..tools import copy, indent2
from ..typing import Optional, PathOrStr, RealNumber


### CONSTANTS
SUFFIX_RUNDATA = ".wrun"


### CLASSES
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
    def sowdate(self) -> dt.date:
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
        filename = output_directory / (self.run_id + SUFFIX_RUNDATA)

        # Create dataframe and save
        df = pd.DataFrame(self.input_dict(), index=[self.run_id])
        df.index.name = "run_id"
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
