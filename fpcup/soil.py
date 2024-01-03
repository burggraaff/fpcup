"""
Soil-related stuff: load data etc
"""
from pathlib import Path

from pcse.fileinput import CABOFileReader

def load_folder(folder: Path | str, pattern: str="ec*") -> list[CABOFileReader]:
    """
    Load all soil files from a given folder, matching a given pattern.
    """
    # Ensure the folder is a Path object
    folder = Path(folder)

    # Find and then load the files
    filenames = folder.glob(pattern)
    soil_files = [CABOFileReader(fn) for fn in filenames]

    return soil_files
