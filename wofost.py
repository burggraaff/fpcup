"""
Playing around with the PCSE implementation of WOFOST.
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import pcse
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.util import WOFOST72SiteDataProvider
from pcse.db import NASAPowerWeatherDataProvider
# from progressbar import printProgressBar

# Initialise model
wofost_object = pcse.start_wofost(grid=31031, crop=1, year=2000, mode='wlp')

# Run model and get output
wofost_object.run_till_terminate()
output = wofost_object.get_output()
df = pd.DataFrame(output)

# Show output
summary_output = wofost_object.get_summary_output()
msg = "Reached maturity at {DOM} with total biomass {TAGP} kg/ha and a yield of {TWSO} kg/ha."
print(msg.format(**summary_output[0]))

# Do the same but in a loop
