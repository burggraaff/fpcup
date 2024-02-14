"""
Some agro templates.
These should only be accessed through the .agro library.

The main template uses double braces for the sow date and single braces for the crop type/variety.
This means the crop type and variety have to be set first (together), and the date has to be set second.
A different order of operations could be achieved by using a template in which the brace order is reversed (e.g. {sowdate} and {{croptype}}).
string.Template objects were used previously, which are more powerful, but their functionality is not necessary at this point in time.

Templates are accessible as functions, e.g. template_date["barley"](sowdate=dt.datetime(2022, 3, 3)) - this is slightly more flexible than only using string.format
"""
import datetime as dt
from functools import partial
import yaml

from ._typing import Callable

def load_agrotemplate_yaml(template: Callable, **kwargs) -> list:
    """
    Load an agromanagement template (YAML), formatted with the provided kwargs.
    Note that any kwargs not found in the template are simply ignored.

    Example:
        agro = '''
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
        agrotemplate = agro.format
        agromanagement = load_formatted(agrotemplate, sowdate=dt.datetime(2020, 1, 1))
    """
    template_formatted = template(**kwargs)
    agromanagement = yaml.safe_load(template_formatted)
    return agromanagement

_template = """
- {{sowdate:%Y-%m}}-01:
    CropCalendar:
        crop_name: '{croptype}'
        variety_name: '{variety}'
        crop_start_date: {{sowdate:%Y-%m-%d}}
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: {{duration}}
    TimedEvents: null
    StateEvents: null
- {{enddate:%Y-%m-%d}}: null
"""
def multiyear(template, sowdate, duration):
    enddate = sowdate + dt.timedelta(days=duration)
    return template.format(sowdate=sowdate, enddate=enddate, duration=duration)

# Templates for a single crop with variable sowing dates
_template_date_springbarley = _template.format(croptype="barley", variety="Spring_barley_301")
_template_date_greenmaize = _template.format(croptype="maize", variety="Grain_maize_201")
_template_date_sorghum = _template.format(croptype="sorghum", variety="Sorghum_VanHeemst_1988")
_template_date_soy = _template.format(croptype="soybean", variety="Soybean_901")
_template_date_winterwheat = _template.format(croptype="wheat", variety="Winter_wheat_102")

template_date = {"barley":              partial(multiyear, template=_template_date_springbarley, duration=240),
                 "barley (spring)":     partial(multiyear, template=_template_date_springbarley, duration=240),
                 "barley (winter)":     partial(multiyear, template=_template_date_springbarley, duration=240),
                 "maize":               partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "maize (green)":       partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "maize (grain)":       partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "maize (mix)":         partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "maize (silage)":      partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "maize (sweet)":       partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "maize (energy)":      partial(multiyear, template=_template_date_greenmaize, duration=210),
                 "sorghum":             partial(multiyear, template=_template_date_sorghum, duration=250),
                 "soy":                 partial(multiyear, template=_template_date_soy, duration=190),
                 "wheat":               partial(multiyear, template=_template_date_winterwheat, duration=365),
                 "wheat (spring)":      partial(multiyear, template=_template_date_winterwheat, duration=365),
                 "wheat (winter)":      partial(multiyear, template=_template_date_winterwheat, duration=365),}

# Simplest example: spring barley with a set sowing date
sowdate_example = dt.date(2005, 3, 3)
_template_example_springbarley = template_date["barley (spring)"](sowdate=sowdate_example).format
