"""
Handles Netherlands-specific geospatial operations and constants.
"""
import geopandas as gpd
gpd.options.io_engine = "pyogrio"

from .settings import DEFAULT_DATA
from .tools import invert_dict

### NAMES AND LABELS
# Constants
PROVINCE_NAMES = ("Fryslân", "Gelderland", "Noord-Brabant", "Noord-Holland", "Overijssel", "Zuid-Holland",  "Groningen", "Zeeland", "Drenthe", "Flevoland", "Limburg", "Utrecht")  # Sorted by area
NETHERLANDS = "Netherlands"
NAME2ABBREVIATION = {"Drenthe": "DR", "Flevoland": "FL", "Fryslân": "FR", "Gelderland": "GD", "Groningen": "GR", "Limburg": "LB", "Noord-Brabant": "NB", "Noord-Holland": "NH", "Overijssel": "OV", "Utrecht": "UT", "Zuid-Holland": "ZH", "Zeeland": "ZL", "Netherlands": "NL"}

ALIASES = {"Friesland": "Fryslân", "Fryslan": "Fryslân",
           "the Netherlands": "Netherlands", "All": "Netherlands"}

# Derived
ABBREVIATION2NAME = invert_dict(NAME2ABBREVIATION)
NAMES = PROVINCE_NAMES + (NETHERLANDS, )


def apply_aliases(name: str) -> str:
    """
    For a given province name, which may be an alias or abbreviation, find the corresponding true name.
    """
    name_title = name.title()
    name_caps = name.upper()
    if name_title in NAMES:  # Input was the true name already
        return name_title
    elif name_caps in ABBREVIATION2NAME.keys():  # Input is an abbreviation
        return ABBREVIATION2NAME[name_caps]
    elif name_title in ALIASES.keys():  # Input is an alias, e.g. Friesland
        return ALIASES[name_title]
    else:
        raise ValueError(f"Cannot interpret location name {name}")

### GEOSPATIAL
# Load the Netherlands shapefile
_basemaps = gpd.read_file(DEFAULT_DATA/"basemaps.geojson")
