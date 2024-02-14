"""
Agromanagement-related stuff: load data etc
"""
import datetime as dt
from functools import cache
from itertools import product

from tqdm import tqdm

from ._agro_templates import load_agrotemplate_yaml, _template_example_springbarley, template_date
from ._typing import Callable, Iterable, Type
from .tools import make_iterable, dict_product

class AgromanagementData(list):
    """
    This class is essentially a reskinned list with some convenience methods attached.
    It allows us to type check specifically for agromanagement data rather than for a generic list.
    """
    @classmethod
    def from_template(cls, template: Callable, **kwargs):
        return cls(load_agrotemplate_yaml(template, **kwargs))

class AgromanagementDataSingleCrop(AgromanagementData):
    """
    AgromanagementData for agromanagement calendars that only consider a single crop.
    Still essentially a list, but with some parameters more easily available.
    """
    def __init__(self, data):
        super().__init__(data)

        # Extract general calendar data
        self.calendar_start = list(data[0].keys())[0]
        self.calendar_end = list(data[1].keys())[0]
        self.calendar = data[0][self.calendar_start]["CropCalendar"]

        # Extract data from the calendar; done explicitly so ensure completeness
        self.crop_name = self.calendar["crop_name"]
        self.crop_variety = self.calendar["variety_name"]
        self.crop_start_date = self.calendar["crop_start_date"]
        self.crop_start_type = self.calendar["crop_start_type"]
        self.crop_end_date = self.calendar["crop_end_date"]
        self.crop_end_type = self.calendar["crop_end_type"]

    def __repr__(self) -> str:
        if self.crop_end_date is None:
            end = self.crop_end_type
        else:
            end = f"{self.crop_end_date} ({self.crop_end_type})"

        return ("Agromanagement data for a single crop.\n"
                f"Crop: {self.crop_name} (variety: {self.crop_variety})\n"
                f"Start: {self.crop_start_date} ({self.crop_start_type})\n"
                f"End: {end}")

agromanagement_example = AgromanagementDataSingleCrop.from_template(_template_example_springbarley)

def load_agrotemplate(crop: str, **kwargs) -> AgromanagementDataSingleCrop:
    """
    Load an agromanagement template for a given crop, formatted with the provided kwargs.
    Any **kwargs are passed to AgromanagementDataSingleCrop.from_template.
    """
    template = template_date[crop]
    agromanagement = AgromanagementDataSingleCrop.from_template(template, **kwargs)
    return agromanagement

def load_agrotemplate_multi(crop: str, progressbar=True, leave_progressbar=False, **kwargs) -> list[AgromanagementDataSingleCrop]:
    """
    Load an agromanagement template for a given crop, formatted with the provided kwargs.
    This will iterate over every iterable in kwargs; for example, you can provide multiple dates or multiple crops.
    Note that any kwargs not found in the template are simply ignored.
    """
    # Create a Cartesian product of all kwargs, so they can be iterated over
    kwargs_iterable = dict_product(kwargs)

    try:
        n = len(kwargs_iterable)
    except TypeError:
        n = None

    kwargs_iterable = tqdm(kwargs_iterable, total=n, desc="Loading agromanagement", unit="calendars", disable=not progressbar, leave=leave_progressbar)
    agromanagement = [load_agrotemplate(crop, **k) for k in kwargs_iterable]

    return agromanagement

def generate_sowingdates(year: int | Iterable[int], days_of_year: int | Iterable[int]) -> list[dt.datetime]:
    """
    Generate a list of date objects representing sowing dates for a given year and list of days of the year (DOYs).
    Both inputs can be a single number or an iterable of numbers.
    """
    # Ensure both variables are iterables, then generate all possible pairs
    years = make_iterable(year)
    doys = make_iterable(days_of_year)
    years_and_doys = product(years, doys)
    return [dt.datetime.strptime(f"{year}-{doy}", "%Y-%j") for year, doy in years_and_doys]

# Sowing date ranges, in day-of-year (doy) format
sowdoys_springbarley = range(40, 86)  # From the WOFOST crop parameter repository: barley.yaml
sowdoys_greenmaize = range(115, 122)  # From the WOFOST crop parameter repository: maize.yaml
sowdoys_sorghum = range(130, 140)  # From https://www.melkvee.nl/artikel/195909-teelttips-sorghum/
sowdoys_soy = range(118, 119)  # From the WOFOST crop parameter repository: soybean.yaml
sowdoys_winterwheat = range(244, 334)  # From the WOFOST crop parameter repository: wheat.yaml

sowdoys = {"barley": sowdoys_springbarley,
           "barley (spring)": sowdoys_springbarley,
           "barley (winter)": sowdoys_springbarley,
           "maize": sowdoys_greenmaize,
           "maize (green)": sowdoys_greenmaize,
           "maize (grain)": sowdoys_greenmaize,
           "maize (mix)": sowdoys_greenmaize,
           "maize (silage)": sowdoys_greenmaize,
           "maize (sweet)": sowdoys_greenmaize,
           "maize (energy)": sowdoys_greenmaize,
           "sorghum": sowdoys_sorghum,
           "soy": sowdoys_soy,
           "wheat": sowdoys_winterwheat,
           "wheat (spring)": sowdoys_winterwheat,
           "wheat (winter)": sowdoys_winterwheat,}

@cache
def sowdate_range(crop: str, year: int) -> list[dt.datetime]:
    """
    Return the typical range of sowing dates for the given crop in the given year.
    """
    return generate_sowingdates(year, sowdoys[crop])
