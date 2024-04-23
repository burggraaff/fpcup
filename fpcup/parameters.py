"""
Classes and functions for dealing with PCSE input and output parameters in a standardised way.
Includes a non-comprehensive list of parameters, sorted by subject and input/output status.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from .tools import dict_product
from .typing import Iterable, Optional, RealNumber

### Classes for dealing with parameters
@dataclass
class _PCSEParameterBase:
    name: str
    description: str

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

@dataclass
class _PCSEParameterPlottable:
    plotname: Optional[str] = None

    def __post_init__(self):
        # Uses the description as a default plotname; assumes a description exists
        if self.plotname is None:
            self.plotname = self.description

@dataclass
class PCSEFlag(_PCSEParameterBase):
    def __str__(self) -> str:
        return f"[FLAG] {self.name}: {self.description}"

@dataclass
class PCSELabel(_PCSEParameterPlottable, _PCSEParameterBase):
    def __str__(self) -> str:
        return str(self.plotname)

@dataclass
class PCSENumericParameter(_PCSEParameterPlottable, _PCSEParameterBase):
    unit: Optional[str] = None
    bounds: Optional[Iterable[RealNumber]] = None
    default: Optional[RealNumber] = None
    dtype: type = float

    def __str__(self) -> str:
        base = f"{self.name}: {self.plotname}"
        if self.unit is not None:
            base += f" [{self.unit}]"
        return base

    def generate_range(self, step) -> np.ndarray[dtype]:
        if self.bounds is None:
            raise ValueError(f"No bounds defined for Parameter {self.name}")

        return np.arange(*self.bounds, step=step, dtype=self.dtype)

    def generate_space(self, n) -> np.ndarray[dtype]:
        if self.bounds is None:
            raise ValueError(f"No bounds defined for Parameter {self.name}")

        return np.linspace(*self.bounds, num=n, dtype=self.dtype)

@dataclass
class PCSEDateParameter(_PCSEParameterPlottable, _PCSEParameterBase):
    def __str__(self) -> str:
        return f"{self.name}: {self.plotname}"

@dataclass
class PCSETabularParameter(_PCSEParameterBase):
    x: str
    x_unit: Optional[str] = None
    unit: Optional[str] = None

    def __str__(self) -> str:
        base = f"[TABLE] {self.name}: {self.description}"
        if self.unit is not None:
            base += f" [{self.unit}]"
        base += f" as function of {self.x}"
        if self.x_unit is not None:
            base += f" [{self.x_unit}]"
        return base

PCSEParameter = PCSEFlag | PCSELabel | PCSENumericParameter | PCSETabularParameter

### Parameters
# Parameter information from "A gentle introduction to WOFOST" (De Wit & Boogaard 2021), CABO file descriptions, YAML file descriptions, WOFOSTSiteProvider defaults
C = "°C"
Cday = f"{C} day"
cm3percm3 = "cm^3 / cm^3"
cmperday = "cm / day"
kgperha = "kg / ha"

IFUNRN = PCSEFlag(name="IFUNRN", description="Flag indicating the way the non-infiltrating fraction of rainfall is determined")

SOLNAM = PCSELabel(name="SOLNAM", description="Soil name")

n = PCSENumericParameter(name="n", description="Number of sites")
area = PCSENumericParameter(name="area", description="Total plot area", unit="ha")

WAV = PCSENumericParameter(name="WAV", description="Initial amount of water in rootable zone in excess of wilting point", plotname="Initial excess water", unit="cm", bounds=(0, 50), default=10)
NOTINF = PCSENumericParameter(name="NOTINF", description="Non-infiltrating fraction", bounds=(0, 1), default=0)
SMLIM = PCSENumericParameter(name="SMLIM", description="Maximum initial soil moisture in rooted zone", plotname="Maximum initial soil moisture", unit="cm", bounds=(0, 10), default=0.4)
SSI = PCSENumericParameter(name="SSI", description="Initial surface storage", unit="cm", bounds=(0, 2), default=0)
SSMAX = PCSENumericParameter(name="SSMAX", description="Maximum surface storage capacity", unit="cm", bounds=(0, 2), default=0)

CRAIRC = PCSENumericParameter(name="CRAIRC", description="Critical soil air content for aeration (used when IOX = 1)", plotname="Critical soil air content for aeration", unit=cm3percm3, bounds=(0.04, 0.1))
SM0 = PCSENumericParameter(name="SM0", description="Soil moisture content of saturated soil", plotname="Saturated soil moisture content", unit=cm3percm3, bounds=(0.3, 0.9))
SMFCF = PCSENumericParameter(name="SMFCF", description="Soil moisture content at field capacity", unit=cm3percm3, bounds=(0.05, 0.74))
SMW = PCSENumericParameter(name="SMW", description="Soil moisture content at wilting point", unit=cm3percm3, bounds=(0.01, 0.35))
RDMSOL = PCSENumericParameter(name="RDMSOL", description="Maximum rootable depth of soil", plotname="Maximum rootable depth", unit="cm", bounds=(10, 150), default=120)
K0 = PCSENumericParameter(name="K0", description="Hydraulic conductivity of saturated soil", unit=cmperday, bounds=(0.1, 14))
KSUB = PCSENumericParameter(name="KSUB", description="Maximum percolation rate of water to subsoil", unit=cmperday, bounds=(0.1, 14))
SOPE = PCSENumericParameter(name="SOPE", description="Maximum percolation rate of water through the root zone", unit=cmperday, bounds=(0, 10))
SPADS = PCSENumericParameter(name="SPADS", description="1st topsoil seepage parameter deep seedbed")
SPODS = PCSENumericParameter(name="SPODS", description="2nd topsoil seepage parameter deep seedbed")
SPASS = PCSENumericParameter(name="SPADS", description="1st topsoil seepage parameter shallow seedbed")
SPOSS = PCSENumericParameter(name="SPADS", description="2nd topsoil seepage parameter shallow seedbed")
DEFLIM = PCSENumericParameter(name="DEFLIM", description="Required moisture deficit deep seedbed")

TSUMEM = PCSENumericParameter(name="TSUMEM", description="Temperature sum from sowing to emergence", unit=Cday, bounds=(0, 170))
TSUM1 = PCSENumericParameter(name="TSUM1", description="Temperature sum from emergence to anthesis", unit=Cday, bounds=(150, 1050))
TSUM2 = PCSENumericParameter(name="TSUM2", description="Temperature sum from anthesis to maturity", unit=Cday, bounds=(600, 1550))
TBASEM = PCSENumericParameter(name="TBASEM", description="Lower threshold temperature for emergence", unit=C, bounds=(-10, 8))
TEFFMX = PCSENumericParameter(name="TEFFMX", description="Maximum effective temperature for emergence", unit=C, bounds=(18, 32))
RDI = PCSENumericParameter(name="RDI", description="Initial rooting depth", unit="cm", bounds=(10, 50))
RRI = PCSENumericParameter(name="RRI", description="Maximum daily increase in rooting depth", unit="cm / day", bounds=(0, 3))
RDMCR = PCSENumericParameter(name="RDMCR", description="Maximum rooting depth", unit="cm", bounds=(50, 400))

DVS = PCSENumericParameter(name="DVS", description="Crop development state (-0.1 = sowing; 0 = emergence; 1 = flowering; 2 = maturity)", plotname="Crop development state", bounds=(-0.1, 2))
LAI = PCSENumericParameter(name="LAI", description="Leaf area index", unit="ha / ha", bounds=(0, 12))
TAGP = PCSENumericParameter(name="TAGP", description="Total above-ground production", unit=kgperha, bounds=(0, 150000))
TWSO = PCSENumericParameter(name="TWSO", description="Total weight of storage organs", unit=kgperha, bounds=(0, 100000))
TWLV = PCSENumericParameter(name="TWLV", description="Total weight of leaves", unit=kgperha, bounds=(0, 100000))
TWST = PCSENumericParameter(name="TWST", description="Total weight of stems", unit=kgperha, bounds=(0, 100000))
TWRT = PCSENumericParameter(name="TWSO", description="Total weight of roots", unit=kgperha, bounds=(0, 100000))
TRA = PCSENumericParameter(name="TRA", description="Crop transpiration", unit="cm / day")
RD = PCSENumericParameter(name="RD", description="Rooting depth", unit="cm", bounds=(10, 150))
SM = PCSENumericParameter(name="SM", description="Actual soil moisture content in rooted zone", plotname="Soil moisture index", bounds=(0.01, 0.9))
WWLOW = PCSENumericParameter(name="WWLOW", description="Amount of water in whole rootable zone", plotname="Water in rootable zone", unit="cm", bounds=(0, 150))
LAIMAX = PCSENumericParameter(name="LAIMAX", description="Maximum LAI reached during growth cycle", plotname="Maximum leaf area index", unit="ha / ha")
CTRAT = PCSENumericParameter(name="CTRAT", description="Cumulative crop transpiration", unit="cm", bounds=(0, 100))
CEVST = PCSENumericParameter(name="CEVST", description="Cumulative soil transpiration", unit="cm")

DOS = PCSEDateParameter(name="DOS", description="Date of sowing")
DOE = PCSEDateParameter(name="DOE", description="Date of emergence")
DOA = PCSEDateParameter(name="DOA", description="Date of anthesis")
DOM = PCSEDateParameter(name="DOM", description="Date of maturity")
DOH = PCSEDateParameter(name="DOH", description="Date of harvest")
DOV = PCSEDateParameter(name="DOV", description="Date of vernalisation")

CONTAB = PCSETabularParameter(name="CONTAB", description="10-log hydraulic conductivity", x="pF", x_unit="log(cm)", unit=f"log({cmperday})")
SMTAB = PCSETabularParameter(name="SMTAB", description="Volumetric soil moisture content", x="pF", x_unit="log(cm)", unit=cm3percm3)
SLATB = PCSETabularParameter(name="SLATB", description="Specific leaf area", x="DVS", x_unit=None, unit="ha / kg")
AMAXTB = PCSETabularParameter(name="AMAXTB", description="Maximum leaf CO2 assimilation rate", x="DVS", x_unit=None, unit="kg / ha / hr")


### Collections
def parameterdict(*params: Iterable[PCSEParameter]) -> dict[str, PCSEParameter]:
    """
    Combines an iterable of PCSEParameter objects into a dictionary, with their name as their key.
    """
    return {p.name: p for p in params}

pcse_inputs = parameterdict(RDMSOL, WAV, DOS)
pcse_outputs = parameterdict(DVS, LAI, TAGP, TWSO, TWLV, TWST, TWRT, TRA, RD, SM, WWLOW)
pcse_summary_outputs = parameterdict(LAIMAX, CTRAT, CEVST, TAGP, TWSO, TWLV, TWST, TWRT, RD, DOS, DOE, DOA, DOM, DOH, DOV)

aggregate_parameters = parameterdict(n, area)
crop_parameters = parameterdict(TSUMEM, TSUM1, TSUM2, TBASEM, TEFFMX, RDI, RRI, RDMCR, SLATB, AMAXTB)
soil_parameters = parameterdict(SOLNAM, CRAIRC, SM0, SMFCF, SMW, RDMSOL, K0, KSUB, SOPE, SPADS, SPODS, SPASS, SPOSS, DEFLIM, SMTAB, CONTAB)
site_parameters = parameterdict(WAV, NOTINF, SMLIM, SSI, SSMAX, IFUNRN)

all_parameters = {**pcse_inputs, **pcse_outputs, **pcse_summary_outputs,
                  **aggregate_parameters, **crop_parameters, **soil_parameters, **site_parameters}


### Functions related to generating ensembles
def generate_ensemble_space(*parameter_names, n: int=100) -> dict:
    """
    Generate an iterable spanning the input space for any number of parameter_names.
    """
    parameters = [all_parameters[name] for name in sorted(parameter_names)]
    parameter_ranges = {p.name: p.generate_space(n) for p in parameters}
    combined = dict_product(parameter_ranges)
    return combined
