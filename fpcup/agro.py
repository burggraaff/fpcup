"""
Agromanagement-related stuff: load data etc
"""
import datetime as dt
from functools import cache
from itertools import product
import yaml

from tqdm import tqdm

from .tools import indent2
from .typing import Callable


### CONVENIENCE CLASSES FOR AGROMANAGEMENT CALENDARS
class AgromanagementData(list):
    """
    This class is essentially a reskinned list with some convenience methods attached.
    It allows us to type check specifically for agromanagement data rather than for a generic list.
    """
    @classmethod
    def from_template(cls, template: str, **kwargs):
        return cls(_load_agrotemplate_yaml(template, **kwargs))


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

    @property
    def crop(self) -> str:
        return f"{self.crop_name}/{self.crop_variety}"

    @property
    def _start(self) -> str:
        return f"{self.crop_start_date} ({self.crop_start_type})"

    @property
    def _end(self) -> str:
        if self.crop_end_date is None:
            end = self.crop_end_type
        else:
            end = f"{self.crop_end_date} ({self.crop_end_type})"
        return end

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.crop_name}/{self.crop_variety}) ; {self._start} -- {self._end}"

    def __str__(self) -> str:
        classtext = f"{self.__class__.__name__}"
        datatext = (f"Crop: {self.crop}\n"
                    f"Start: {self._start}\n"
                    f"End: {self._end}")

        return "\n".join([classtext,
                          indent2(datatext)])


### TEMPLATES
template = """
- {{sowdate:%Y-%m}}-01:
    CropCalendar:
        crop_name: '{crop_name}'
        variety_name: '{variety}'
        crop_start_date: {{sowdate:%Y-%m-%d}}
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: {max_duration}
    TimedEvents: null
    StateEvents: null
- {{enddate:%Y-%m-%d}}: null
"""


def multiyear_template(template: str, sowdate: dt.date, duration: int) -> str:
    enddate = sowdate + dt.timedelta(days=duration)
    return template.format(sowdate=sowdate, enddate=enddate)


def _load_agrotemplate_yaml(template: str, **kwargs) -> list:
    """
    Load an agromanagement template (YAML), formatted with the provided kwargs.
    Note that any kwargs not found in the template are simply ignored.

    Example:
        agrotemplate = '''
        - {sowdate:%Y}-01-01:
            CropCalendar:
                crop_name: 'barley'
                variety_name: 'Spring_barley_301'
                crop_start_date: {sowdate:%Y-%m-%d}
                crop_start_type: sowing
                crop_end_date:
                crop_end_type: maturity
                max_duration: 365
            TimedEvents: null
            StateEvents: null
        - {sowdate:%Y}-12-31: null
        '''
        agromanagement = load_agrotemplate_yaml(agrotemplate, sowdate=dt.datetime(2020, 1, 1))
    """
    template_formatted = template.format(**kwargs)
    agromanagement = yaml.safe_load(template_formatted)
    return agromanagement
