"""
Speed test PCSE by running an ensemble of replicates.

Usage example:
    %run speedtest_minimal.py -t excel -n 100
        Does 100 iterations of the same run, using 100 ExcelWeatherDataProvider objects
"""
from itertools import product
from pathlib import Path

from tqdm import tqdm, trange
import yaml

from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.fileinput import CABOFileReader
from pcse.fileinput import CABOWeatherDataProvider, CSVWeatherDataProvider, ExcelWeatherDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.util import WOFOST72SiteDataProvider

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Speed test PCSE by running an ensemble of replicates.")
parser.add_argument("-n", "--number", help="number of replicates; result may be lower due to rounding", type=int, default=400)
args = parser.parse_args()

# Fetch site data
coords = (53, 6)
sitedata = WOFOST72SiteDataProvider(WAV=10)

# Soil data
soil_filename = Path.cwd() / "data" / "soil" / "ec1.soil"
soildata = CABOFileReader(soil_filename)

# Crop data
cropdata = YAMLCropDataProvider()

# Agromanagement calendars
agro_template = """
- 2005-01-01:
    CropCalendar:
        crop_name: 'barley'
        variety_name: 'Spring_barley_301'
        crop_start_date: 2005-03-03
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- 2005-12-01: null
"""
agromanagementdata = yaml.safe_load(agro_template)

parameters = ParameterProvider(sitedata=sitedata, soildata=soildata, cropdata=cropdata)

# Run the simulation ensemble
for i in trange(args.number, desc="Running PCSE models", unit="runs"):
    weatherdata = NASAPowerWeatherDataProvider(*coords)
    try:
        wofost = Wofost72_WLP_FD(parameters, weatherdata, agromanagementdata)
        wofost.run_till_terminate()
    except WeatherDataProviderError:
        output = None
    else:
        output = wofost.get_output()
