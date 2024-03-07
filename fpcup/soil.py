"""
Soil-related stuff: load data etc
"""
from pathlib import Path

from pcse.fileinput import CABOFileReader
from pcse.util import DummySoilDataProvider

from ._typing import PathOrStr
from .settings import DEFAULT_DATA

SoilType = CABOFileReader | DummySoilDataProvider

DEFAULT_SOIL_DATA = DEFAULT_DATA / "soil"

# Parameter names are from "A gentle introduction to WOFOST" (De Wit & Boogaard 2021) and from the CABO file descriptions
parameter_names = {"SOLNAM": "Soil name",
                   "CRAIRC": "Critical soil air content for aeration [cm^3 / cm^3]",
                   "SM0": "Soil moisture content of saturated soil [cm^3 / cm^3]",
                   "SMTAB": "Vol. soil moisture content as function of pF [log(cm) ; cm^3 / cm^3]",
                   "SMFCF": "Soil moisture content at field capacity [cm^3 / cm^3]",
                   "SMW": "Soil moisture content at wilting point [cm^3 / cm^3]",
                   "RDMSOL": "Maximum rootable depth of soil [cm]",
                   "K0": "Hydraulic conductivity of saturated soil [cm / day]",
                   "KSUB": "Maximum percolation rate of water to subsoil [cm / day]",
                   "SOPE": "Maximum percolation rate of water through the root zone [cm / day]",
                   "SPADS": "1st topsoil seepage parameter deep seedbed",
                   "SPODS": "2nd topsoil seepage parameter deep seedbed",
                   "SPASS": "1st topsoil seepage parameter shallow seedbed",
                   "SPOSS": "2nd topsoil seepage parameter shallow seedbed",
                   "DEFLIM": "required moisture deficit deep seedbed",
                   "CONTAB": "10-log hydraulic conductivity as function of pF [log(cm) ; log(cm/day)]",
                   }

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
