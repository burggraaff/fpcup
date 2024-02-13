"""
Some agro templates.
These should only be accessed through the .agro library.

The main template uses double braces for the sow date and single braces for the crop type/variety.
This means the crop type and variety have to be set first (together), and the date has to be set second.
A different order of operations could be achieved by using a template in which the brace order is reversed (e.g. {sowdate} and {{croptype}}).
string.Template objects were used previously, which are more powerful, but their functionality is not necessary at this point in time.
"""
import datetime as dt
import yaml

def load_agrotemplate(template: str, **kwargs) -> list:
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

date_example = dt.date(2005, 3, 3)

_template_crop_date = """
- {{sowdate:%Y}}-01-01:
    CropCalendar:
        crop_name: '{croptype}'
        variety_name: '{variety}'
        crop_start_date: {{sowdate:%Y-%m-%d}}
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {{sowdate:%Y}}-12-01: null
"""

# Templates for a single crop with variable sowing dates
template_date_springbarley = _template_crop_date.format(croptype="barley", variety="Spring_barley_301")

# Simplest example: spring barley with a set sowing date
template_example_springbarley = template_date_springbarley.format(sowdate=date_example)
