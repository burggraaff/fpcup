"""
Soil-related stuff: load data etc
"""
from pathlib import Path

from pcse.fileinput import CABOFileReader
from pcse.util import DummySoilDataProvider

from ._typing import PathOrStr
from .parameters import soil_parameters
from .settings import DEFAULT_DATA

SoilType = CABOFileReader | DummySoilDataProvider

DEFAULT_SOIL_DATA = DEFAULT_DATA / "soil"

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
