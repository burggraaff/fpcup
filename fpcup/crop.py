"""
Crop-related stuff: load data etc
"""
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

from ._brp_dictionary import brp_crops_NL2EN
from ._typing import PathOrStr, PCSEFlag, PCSELabel, PCSENumericParameter, PCSETabularParameter
from .tools import parameterdict

# Parameter information from "A gentle introduction to WOFOST" (De Wit & Boogaard 2021) and YAML file descriptions
C = "°C"
Cday = f"{C} day"

TSUMEM = PCSENumericParameter(name="TSUMEM", description="Temperature sum from sowing to emergence", unit=Cday, bounds=(0, 170))
TSUM1 = PCSENumericParameter(name="TSUM1", description="Temperature sum from emergence to anthesis", unit=Cday, bounds=(150, 1050))
TSUM2 = PCSENumericParameter(name="TSUM2", description="Temperature sum from anthesis to maturity", unit=Cday, bounds=(600, 1550))
TBASEM = PCSENumericParameter(name="TBASEM", description="Lower threshold temperature for emergence", unit=C, bounds=(-10, 8))
TEFFMX = PCSENumericParameter(name="TEFFMX", description="Maximum effective temperature for emergence", unit=C, bounds=(18, 32))
RDI = PCSENumericParameter(name="RDI", description="Initial rooting depth", unit="cm", bounds=(10, 50))
RRI = PCSENumericParameter(name="RRI", description="Maximum daily increase in rooting depth", unit="cm / day", bounds=(0, 3))
RDMCR = PCSENumericParameter(name="RDMCR", description="Maximum rooting depth", unit="cm", bounds=(50, 400))

SLATB = PCSETabularParameter(name="SLATB", description="Specific leaf area", x="DVS", x_unit=None, unit="ha / kg")
AMAXTB = PCSETabularParameter(name="AMAXTB", description="Maximum leaf CO2 assimilation rate", x="DVS", x_unit=None, unit="kg / ha / hr")

parameters = parameterdict(TSUMEM, TSUM1, TSUM2, TBASEM, TEFFMX, RDI, RRI, RDMCR, SLATB, AMAXTB)


default = YAMLCropDataProvider()

def main_croptype(crop: str) -> str:
    """
    Takes in a crop type, e.g. "Wheat (winter)", and returns the main type, e.g. "Wheat".
    """
    return crop.split()[0]
