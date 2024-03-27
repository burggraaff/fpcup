"""
Soil-related stuff: load data etc
"""
from pathlib import Path

from pcse.fileinput import CABOFileReader
from pcse.util import DummySoilDataProvider

from ._typing import PathOrStr, PCSELabel, PCSENumericParameter, PCSETabularParameter
from .settings import DEFAULT_DATA
from .tools import parameterdict

SoilType = CABOFileReader | DummySoilDataProvider

DEFAULT_SOIL_DATA = DEFAULT_DATA / "soil"

# Parameter information from "A gentle introduction to WOFOST" (De Wit & Boogaard 2021) and CABO file descriptions
cm3percm3 = "cm^3 / cm^3"
cmperday = "cm / day"
SOLNAM = PCSELabel(name="SOLNAM", description="Soil name")

CRAIRC = PCSENumericParameter(name="CRAIRC", description="Critical soil air content for aeration (used when IOX = 1)", plotname="Critical soil air content for aeration", unit=cm3percm3, bounds=(0.04, 0.1))
SM0 = PCSENumericParameter(name="SM0", description="Soil moisture content of saturated soil", plotname="Saturated soil moisture content", unit=cm3percm3, bounds=(0.3, 0.9))
SMFCF = PCSENumericParameter(name="SMFCF", description="Soil moisture content at field capacity", unit=cm3percm3, bounds=(0.05, 0.74))
SMW = PCSENumericParameter(name="SMW", description="Soil moisture content at wilting point", unit=cm3percm3, bounds=(0.01, 0.35))
RDMSOL = PCSENumericParameter(name="RDMSOL", description="Maximum rootable depth of soil", plotname="Maximum rootable depth", unit="cm", bounds=(10, 150))
K0 = PCSENumericParameter(name="K0", description="Hydraulic conductivity of saturated soil", unit=cmperday, bounds=(0.1, 14))
KSUB = PCSENumericParameter(name="KSUB", description="Maximum percolation rate of water to subsoil", unit=cmperday, bounds=(0.1, 14))
SOPE = PCSENumericParameter(name="SOPE", description="Maximum percolation rate of water through the root zone", unit=cmperday, bounds=(0, 10))
SPADS = PCSENumericParameter(name="SPADS", description="1st topsoil seepage parameter deep seedbed")
SPODS = PCSENumericParameter(name="SPODS", description="2nd topsoil seepage parameter deep seedbed")
SPASS = PCSENumericParameter(name="SPADS", description="1st topsoil seepage parameter shallow seedbed")
SPOSS = PCSENumericParameter(name="SPADS", description="2nd topsoil seepage parameter shallow seedbed")
DEFLIM = PCSENumericParameter(name="DEFLIM", description="Required moisture deficit deep seedbed")

SMTAB = PCSETabularParameter(name="SMTAB", description="Volumetric soil moisture content", x="pF", x_unit="log(cm)", unit=cm3percm3)
CONTAB = PCSETabularParameter(name="CONTAB", description="10-log hydraulic conductivity", x="pF", x_unit="log(cm)", unit=f"log({cmperday})")

parameters = parameterdict(SOLNAM, CRAIRC, SM0, SMFCF, SMW, RDMSOL, K0, KSUB, SOPE, SPADS, SPODS, SPASS, SPOSS, DEFLIM, SMTAB, CONTAB)

# Dummy with default settings
dummy = DummySoilDataProvider()
dummy.name = "dummy"


def load_soil_file(filename: PathOrStr) -> CABOFileReader:
    """
    Load a single soil file and do some minimal pre-processing.
    """
    filename = Path(filename)
    data = CABOFileReader(filename)
    data.name = filename.stem

    return data


def load_folder(folder: PathOrStr, pattern: str="ec*") -> dict[str, CABOFileReader]:
    """
    Load all soil files from a given folder, matching a given pattern.
    """
    # Ensure the folder is a Path object
    folder = Path(folder)

    # Find and then load the files
    filenames = folder.glob(pattern)
    soil_files = {fn.stem: load_soil_file(fn) for fn in filenames}

    return soil_files

soil_types = load_folder(DEFAULT_SOIL_DATA)
