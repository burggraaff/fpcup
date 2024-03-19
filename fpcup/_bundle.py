"""
Legacy code for bundling parameters.
Probably won't be re-used.
"""
from itertools import product

import geopandas as gpd
gpd.options.io_engine = "pyogrio"

from pcse.base import MultiCropDataProvider, ParameterProvider, WeatherDataProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.fileinput import CABOFileReader
from pcse.models import Engine, Wofost72_WLP_FD
from pcse.util import _GenericSiteDataProvider as PCSESiteDataProvider

from ._typing import Callable, Coordinates, Iterable, Optional, PathOrStr
from .agro import AgromanagementData
from .constants import CRS_AMERSFOORT
from .model import RunData, RunDataBRP
from .soil import SoilType
from .tools import make_iterable


def bundle_parameters(sitedata: PCSESiteDataProvider | Iterable[PCSESiteDataProvider],
                      soildata: CABOFileReader | Iterable[CABOFileReader],
                      cropdata: MultiCropDataProvider | Iterable[MultiCropDataProvider],
                      weatherdata: WeatherDataProvider | Iterable[WeatherDataProvider],
                      agromanagementdata: AgromanagementData | Iterable[AgromanagementData]) -> tuple[Iterable[RunData], int | None]:
    """
    Bundle the site, soil, and crop parameters into PCSE ParameterProvider objects.
    For the main parameters, a Cartesian product is used to get all their combinations.
    """
    # Make sure the data are iterable
    sitedata_iter = make_iterable(sitedata, exclude=[PCSESiteDataProvider])
    soildata_iter = make_iterable(soildata, exclude=[CABOFileReader])
    cropdata_iter = make_iterable(cropdata, exclude=[MultiCropDataProvider])
    weatherdata_iter = make_iterable(weatherdata, exclude=[WeatherDataProvider])
    agromanagementdata_iter = make_iterable(agromanagementdata, exclude=[AgromanagementData])

    # Determine the total number of parameter combinations, if possible
    try:
        n = len(sitedata_iter) * len(soildata_iter) * len(cropdata_iter) * len(weatherdata_iter) * len(agromanagementdata_iter)
    except TypeError:
        n = None

    # Combine everything
    combined_parameters = product(sitedata_iter, soildata_iter, cropdata_iter, weatherdata_iter, agromanagementdata_iter)
    rundata = (RunData(*params) for params in combined_parameters)

    return rundata, n
