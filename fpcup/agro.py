"""
Agromanagement-related stuff: load data etc
"""
import datetime as dt
from itertools import product

import yaml
from tqdm import tqdm

from ._agro_templates import template_crop_date, template_springbarley_date, template_springbarley
from ._typing import Iterable
from .tools import make_iterable, dict_product

class AgromanagementData(list):
    """
    This class is nothing more than a reskinned list.
    It allows us to type check specifically for agromanagement data rather than for a generic list.
    """
    pass

def load_formatted(template: str, **kwargs) -> AgromanagementData:
    """
    Load an agromanagement template (YAML), formatted with the provided kwargs.
    Note that any kwargs not found in the template are simply ignored.

    Example:
        agro = '''
        - {date:%Y}-01-01:
            CropCalendar:
                crop_name: 'barley'
                variety_name: 'Spring_barley_301'
                crop_start_date: {date:%Y-%m-%d}
                crop_start_type: sowing
                crop_end_date:
                crop_end_type: maturity
                max_duration: 300
            TimedEvents: null
            StateEvents: null
        - {date:%Y}-12-01: null
        '''
        agromanagement = load_formatted(agro, date=dt.datetime(2020, 1, 1, 0, 0))
    """
    template_formatted = template.format(**kwargs)
    agromanagement = yaml.safe_load(template_formatted)
    agromanagement = AgromanagementData(agromanagement)
    return agromanagement

def load_formatted_multi(template: str, progressbar=True, leave_progressbar=False, **kwargs) -> list[AgromanagementData]:
    """
    Load an agromanagement template (YAML), formatted with the provided kwargs.
    This will iterate over every iterable in kwargs; for example, you can provide multiple dates or multiple crops.
    Note that any kwargs not found in the template are simply ignored.
    """
    # Create a Cartesina product of all kwargs, so they can be iterated over
    kwargs_iterable = dict_product(kwargs)

    try:
        n = len(kwargs_iterable)
    except TypeError:
        n = None

    agromanagement = [load_formatted(template, **k) for k in tqdm(kwargs_iterable, total=n, desc="Loading agromanagement", unit="calendars", disable=not progressbar, leave=leave_progressbar)]

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
