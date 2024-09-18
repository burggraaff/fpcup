from . import aggregate, constants, geo, io, plotting, settings, stats
from . import agro, crop, site, soil, weather
from .model import run_pcse_single
from .nn import dataset, network
from .settings import DEFAULT_DATA, DEFAULT_OUTPUT, DEFAULT_RESULTS
from .tools import RUNNING_IN_IPYTHON


def test() -> None:
    """
    Simple function to assert the import was successful.
    """
    from importlib.metadata import version
    module_version = version(__package__)
    print(f"Succesfully imported {__package__} version {module_version} from {__file__}.")
