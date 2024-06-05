"""
Crop-related stuff: load data etc
"""
from datetime import date, datetime
from itertools import product

from tqdm import tqdm

from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

from ._brp_dictionary import brp_crops_NL2EN
from .agro import AgromanagementDataSingleCrop, template, multiyear_template
from .parameters import crop_parameters
from .tools import invert_dict, make_iterable
from .typing import Iterable

default = YAMLCropDataProvider()

CROP2ABBREVIATION = {"barley": "B",
                     "maize": "M",
                     "sorghum": "S",
                     "soybean": "Y",
                     "wheat": "W",}

ABBREVIATION2CROP = invert_dict(CROP2ABBREVIATION)

def main_croptype(crop: str) -> str:
    """
    Takes in a crop type, e.g. "Wheat (winter)", and returns the main type, e.g. "Wheat".
    """
    return crop.split()[0]


### CROP CONVENIENCE CLASS
class Crop:
    """
    Convenience class for one crop species/variety.
    Primarily used for holding data and generating some variables.
    """
    ### CREATION AND INITIALISATION
    def __init__(self, name: str, variety: str, *,
                 firstdos: int, lastdos: int, max_duration: int) -> None:
        # General properties
        self.name = name
        self.variety = variety
        self.abbreviation = CROP2ABBREVIATION[self.name]

        # Agromanagement-related
        self.sowdoys = range(firstdos, lastdos)
        self.max_duration = max_duration
        self.agro_template = template.format(crop_name=self.name, variety=self.variety, max_duration=self.max_duration)


    ### STRING REPRESENTATIONS
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name} ({self.abbreviation}), {self.variety}, sowing doy {self.sowdoys.start}-{self.sowdoys.stop}, max. duration {self.max_duration} days)"


    ### AGROMANAGEMENT
    def first_sowingdate(self, year: int | Iterable[int]) -> datetime | list[datetime]:
        """
        Generate a datetime object representing the first sowing date in the given year, for this crop.
        """
        MULTIPLE_YEARS = isinstance(year, Iterable)
        dates = generate_sowingdates(year, self.sowdoys.start)

        if not MULTIPLE_YEARS:
            dates = dates[0]

        return dates

    def all_sowingdates(self, year: int | Iterable[int]) -> list[datetime]:
        """
        Generate a list of date objects representing sowing dates for a given year, for this crop.
        """
        return generate_sowingdates(year, self.sowdoys)

    def agromanagement(self, sowdate: date) -> AgromanagementDataSingleCrop:
        """
        Generate a single agromanagement template for a given sowdate, for this crop.
        """
        template = multiyear_template(self.agro_template, sowdate=sowdate, duration=self.max_duration)
        calendar = AgromanagementDataSingleCrop.from_template(template)
        return calendar

    def agromanagement_first_sowingdate(self, year: int | Iterable[int]) -> AgromanagementDataSingleCrop | list[AgromanagementDataSingleCrop]:
        """
        Generate agromanagement templates for the first sowing date in the given years, for this crop.
        """
        MULTIPLE_YEARS = isinstance(year, Iterable)
        year = make_iterable(year)

        dates = self.first_sowingdate(year)
        calendars = [self.agromanagement(d) for d in dates]

        if not MULTIPLE_YEARS:
            calendars = calendars[0]

        return calendars

    def agromanagement_all_sowingdates(self, year: int | Iterable[int], *,
                                       progressbar=True, leave_progressbar=False) -> list[AgromanagementDataSingleCrop]:
        """
        Generate agromanagement semplates for all sowing dates in the given years, for this crop.
        """
        dates = self.all_sowingdates(year)
        dates = tqdm(dates, desc="Loading agromanagement", unit="calendar", disable=not progressbar, leave=leave_progressbar)

        calendars = [self.agromanagement(d) for d in dates]
        return calendars


# Predefined crops
# Sources: see Google Doc
SpringBarley = Crop("barley", "Spring_barley_301", firstdos=18, lastdos=125, max_duration=240)
WinterBarley = Crop("barley", "Spring_barley_301", firstdos=257, lastdos=304, max_duration=240)
GreenMaize = Crop("maize", "Grain_maize_201", firstdos=115, lastdos=122, max_duration=210)  # From the WOFOST crop parameter repository: maize.yaml
SpringWheat = Crop("wheat", "Winter_wheat_102", firstdos=14, lastdos=104, max_duration=365)
WinterWheat = Crop("wheat", "Winter_wheat_102", firstdos=244, lastdos=365, max_duration=365)
Sorghum = Crop("sorghum", "Sorghum_VanHeemst_1988", firstdos=130, lastdos=140, max_duration=250)  # From https://edepot.wur.nl/427964
Soybean = Crop("soybean", "Soybean_901", firstdos=118, lastdos=119, max_duration=190)  # From the WOFOST crop parameter repository: soybean.yaml

crops = {"barley": SpringBarley,
         "barley (spring)": SpringBarley,
         "barley (winter)": WinterBarley,
         "maize": GreenMaize,
         "maize (green)": GreenMaize,
         "maize (grain)": GreenMaize,
         "maize (mix)": GreenMaize,
         "maize (silage)": GreenMaize,
         "maize (sweet)": GreenMaize,
         "maize (energy)": GreenMaize,
         "sorghum": Sorghum,
         "soybean": Soybean,
         "wheat": WinterWheat,
         "wheat (spring)": SpringWheat,
         "wheat (winter)": WinterWheat,}

def select_crop(name: str) -> Crop:
    """
    Given a name, return the corresponding Crop object.
    """
    # First check for abbreviations
    if name.upper() in ABBREVIATION2CROP.keys():
        name = ABBREVIATION2CROP[name.upper()]

    # Return the crop
    return crops[name.lower()]


### GENERAL FUNCTIONS
def generate_sowingdates(year: int | Iterable[int], days_of_year: int | Iterable[int]) -> list[datetime]:
    """
    Generate a list of date objects representing sowing dates for a given year and list of days of the year (DOYs).
    Both inputs can be a single number or an iterable of numbers.
    """
    # Ensure both variables are iterables, then generate all possible pairs
    years = make_iterable(year)
    doys = make_iterable(days_of_year)
    years_and_doys = product(years, doys)
    return [datetime.strptime(f"{year}-{doy}", "%Y-%j") for year, doy in years_and_doys]
