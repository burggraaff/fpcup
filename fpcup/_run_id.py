"""
Generation of run IDs - for use in .model, refactored here for simplicity.
"""
from base64 import urlsafe_b64encode, urlsafe_b64decode
from datetime import date
from struct import pack, unpack

from ._typing import Coordinates, RealNumber
from .crop import CROP2ABBREVIATION

### CONSTANTS
_COORDINATE_FORMAT = "ff"  # Note f has 32-bit / up to 7 digit precision
_JOINER = "_"


### HELPER FUNCTIONS
def _process_crop_name(crop_name: str) -> str:
    crop_name = crop_name.lower()
    return CROP2ABBREVIATION[crop_name]


def _process_sowdate(sowdate: date) -> str:
    return f"s{sowdate:%y%j}"


def _process_coordinates(latitude: RealNumber, longitude: RealNumber) -> str:
    """
    Pack lat/lon coordinates into a shorter format.
    """
    as_bytes = pack(_COORDINATE_FORMAT, latitude, longitude)
    as_string = urlsafe_b64encode(as_bytes).decode()
    return as_string


def _unpack_coordinates(packed_string: str) -> str:
    """
    Unpack a lat/lon string.
    """
    as_bytes = urlsafe_b64decode(packed_string)
    latitude, longitude = unpack(_COORDINATE_FORMAT, as_bytes)
    return latitude, longitude


def _process_override_single(key: str, value: RealNumber) -> str:
    """
    Process a single PCSE ParameterProvider override with an arbitrary key and value.
    Currently essentially a dummy function.
    """
    return f"{key}-{value:.5f}"

### RUN ID GENERATION FUNCTIONS
def generate_run_id_base(*, crop_name: str, soiltype: str, sowdate: date, latitude: RealNumber, longitude: RealNumber) -> str:
    """
    Basic run ID.
    """
    sitestr = "c" + _process_coordinates(latitude, longitude)
    cropstr = _process_crop_name(crop_name)
    soilstr = soiltype
    datestr = _process_sowdate(sowdate)

    return _JOINER.join([sitestr, soilstr, cropstr, datestr])


def generate_run_id_BRP(*, brpyear: int, plot_id: int, crop_name: str, sowdate: date) -> str:
    """
    Generate a run ID from BRP data.
    Separate from the RunDataBRP class so it can be called before initialising the weather data (which tends to be slow) when checking for duplicate files.
    """
    sitestr = f"brp{brpyear % 100}-{plot_id}"
    cropstr = _process_crop_name(crop_name)
    datestr = _process_sowdate(sowdate)

    return _JOINER.join([sitestr, cropstr, datestr])


def append_overrides(base: str, overrides: dict) -> str:
    """
    Append additional information from a dictionary, such as PCSE ParameterProvider overrides, to a base run ID.
    """
    # Do nothing if there are no overrides
    if len(overrides) == 0:
        return base

    # Process and combine individual overrides
    individual_strings = [_process_override_single(key, value) for key, value in sorted(overrides.items())]
    overridestr = _JOINER.join(individual_strings)

    # Combine with base and return
    return _JOINER.join([base, overridestr])
