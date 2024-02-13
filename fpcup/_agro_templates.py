"""
Some agro templates.
These should only be accessed through the .agro library.

string.Template objects are used to easily generate variants of the same template.
The end user does not see these Templates, only the resulting YAML-format strings.
"""
import datetime as dt
from string import Template

date_example = dt.date(2005, 3, 3)

def parse_template(template: Template, **kwargs) -> str:
    """
    Parse a Template object, applying all kwargs and removing $ signs afterwards.
    """
    template_parsed = template.safe_substitute(**kwargs)
    template_parsed = template_parsed.replace("$", "")
    return template_parsed

_template_crop_date = Template("""
- ${date:%Y}-01-01:
    CropCalendar:
        crop_name: '${croptype}'
        variety_name: '${variety}'
        crop_start_date: ${date:%Y-%m-%d}
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- ${date:%Y}-12-01: null
""")

# Custom crop, variety, sowing dates
template_crop_date = parse_template(_template_crop_date)

# Spring barley with custom sowing dates
template_date_springbarley = parse_template(_template_crop_date, croptype="barley", variety="Spring_barley_301")

# Simplest example: spring barley with a set sowing date
template_example_springbarley = template_date_springbarley.format(date=date_example)
