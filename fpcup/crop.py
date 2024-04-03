"""
Crop-related stuff: load data etc
"""
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

from ._brp_dictionary import brp_crops_NL2EN
from .parameters import crop_parameters
from .tools import invert_dict

default = YAMLCropDataProvider()

CROP2ABBREVIATION = {"barley": "B",
                     "maize": "M",
                     "sorghum": "S",
                     "soy": "Y",
                     "wheat": "W",}

ABBREVIATION2CROP = invert_dict(CROP2ABBREVIATION)

def main_croptype(crop: str) -> str:
    """
    Takes in a crop type, e.g. "Wheat (winter)", and returns the main type, e.g. "Wheat".
    """
    return crop.split()[0]
