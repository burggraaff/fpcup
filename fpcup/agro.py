"""
Agromanagement-related stuff: load data etc
"""
import datetime as dt
from typing import Iterable

import yaml
from tqdm import tqdm

from ._agro_templates import template_crop_date, template_springbarley_date, template_springbarley
from .tools import dict_product

def load_formatted(template: str, **kwargs) -> list[dict]:
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
    return agromanagement

def load_formatted_multi(template: str, **kwargs) -> list[list[dict]]:
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

    agromanagement = [load_formatted(template, **k) for k in tqdm(kwargs_iterable, total=n, desc="Loading agromanagement", unit="calendars")]

    return agromanagement

def generate_sowingdates(year: int, days_of_year: Iterable[int]) -> list[dt.datetime]:
    """
    Generate a list of datetime objects representing sowing dates for a given year and list of days of the year (DOYs).

    TO DO: Iterate over years too.
    """
    return [dt.datetime.strptime(f"{year}-{doy}", "%Y-%j") for doy in days_of_year]
