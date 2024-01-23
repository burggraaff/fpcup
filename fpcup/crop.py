"""
Crop-related stuff: load data etc
"""
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

from ._crop_dictionary import brp_dictionary

default = YAMLCropDataProvider()
