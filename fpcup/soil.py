"""
Soil-related stuff: load data etc
"""
from pathlib import Path

from pcse.fileinput import CABOFileReader
from pcse.util import DummySoilDataProvider

from ._typing import PathOrStr
from .settings import DEFAULT_DATA

DEFAULT_SOIL_DATA = DEFAULT_DATA / "soil"

# N.B. type hinting does not account for this yet
dummy = DummySoilDataProvider()

def load_folder(folder: PathOrStr, pattern: str="ec*") -> dict[str, CABOFileReader]:
    """
    Load all soil files from a given folder, matching a given pattern.
    """
    # Ensure the folder is a Path object
    folder = Path(folder)

    # Find and then load the files
    filenames = folder.glob(pattern)
    soil_files = {fn.stem: CABOFileReader(fn) for fn in filenames}

    return soil_files

soil_types = load_folder(DEFAULT_SOIL_DATA)
