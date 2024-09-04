"""
General functions relating to plotting / visualisation.
"""
import numpy as np

from matplotlib import rcParams
from matplotlib import colormaps
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..aggregate import KEYS_AGGREGATE

# Plot settings
rcParams.update({"axes.grid": True,
                 "figure.dpi": 600, "savefig.dpi": 600,
                 "grid.linestyle": "--",
                 "hist.bins": 15,
                 "image.cmap": "cividis",
                 "legend.edgecolor": "black", "legend.framealpha": 1,
                 })


### CONSTANTS
# Raster/Vector switches
_RASTERIZE_LIMIT_LINES = 1000
_RASTERIZE_LIMIT_GEO = 250  # Plot geo data in raster format if there are more than this number
_RASTERIZE_GEO = lambda data: (len(data) > _RASTERIZE_LIMIT_GEO)

KEYS_AGGREGATE_PLOT = ("n", "area", *KEYS_AGGREGATE)

# Graphical defaults
cividis_discrete = colormaps["cividis"].resampled(10)
default_outline = {"color": "black", "linewidth": 0.5}


def symmetric_lims(lims: tuple) -> tuple:
    """
    Given lims, make them symmetric.
    e.g. (-5, 3) -> (-5, 5)  ;  (-2, 6) -> (-6, 6)
    """
    val = np.abs(lims).max()
    newlims = (-val, val)
    return newlims
