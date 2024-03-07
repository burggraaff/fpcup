"""
Crop-related stuff: load data etc
"""
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

from ._brp_dictionary import brp_crops_NL2EN

# Parameter names are from "A gentle introduction to WOFOST" (De Wit & Boogaard 2021) and from the YAML crop parameter files
parameter_names = {"TSUMEM": "Temperature sum from sowing to emergence [°C day]",
                   "TSUM1": "Temperature sum from emergence to anthesis [°C day]",
                   "TSUM2": "Temperature sum from anthesis to maturity [°C day]",
                   "TBASEM": "Lower threshold temperature for emergence [°C]",
                   "TEFFMX": "Maximum effective temperature for emergence [°C]",
                   "SLATB": "Specific leaf area as a function of DVS [; ha/kg]",
                   "AMAXTB": "Maximum leaf CO2 assimilation rate as function of DVS [;kg/ha/hr]",
                   "RDI": "Initial rooting depth [cm]",
                   "RRI": "Maximum daily increase in rooting depth [cm/day]",
                   "RDMCR": "Maximum rooting depth [cm]",
                   }

default = YAMLCropDataProvider()

def main_croptype(crop: str) -> str:
    """
    Takes in a crop type, e.g. "Wheat (winter)", and returns the main type, e.g. "Wheat".
    """
    return crop.split()[0]
