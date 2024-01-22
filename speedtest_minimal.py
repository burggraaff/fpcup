"""
Speed test PCSE by running an ensemble of replicates.

Usage example:
    %run speedtest_minimal.py -t excel -n 100
        Does 100 iterations of the same run, using 100 ExcelWeatherDataProvider objects
"""
from itertools import product
from pathlib import Path

from tqdm import tqdm
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
parser.add_argument("-t", "--type", help="which variable to replicate", choices=["site", "soil", "crop", "weather", "agro", "nasa", "excel", "csv", "csvcrop"])
parser.add_argument("-n", "--number", help="number of replicates; result may be lower due to rounding", type=int, default=400)
args = parser.parse_args()

# Fetch site data
coords = (53, 6)
sitedata = [WOFOST72SiteDataProvider(WAV=10)]
if args.type == "site":
    sitedata = [WOFOST72SiteDataProvider(WAV=10) for i in range(args.number)]

# Fetch weather data
if args.type == "excel":
    weatherdata = [ExcelWeatherDataProvider(Path.cwd() / "data" / "meteo" / "nl1.xlsx") for i in range(args.number)]
elif args.type == "csv":
    weatherdata = [CSVWeatherDataProvider(Path.cwd() / "data" / "meteo" / "nl1.csv") for i in range(args.number)]
elif args.type == "csvcrop":
    weatherdata = [CSVWeatherDataProvider(Path.cwd() / "data" / "meteo" / "nl1_cropped.csv") for i in range(args.number)]
elif args.type == "nasa":
    weatherdata = [NASAPowerWeatherDataProvider(*coords) for i in range(args.number)]
elif args.type == "weather":  # Reusing a single WeatherDataProvider object
    weatherdata = [NASAPowerWeatherDataProvider(*coords)] * args.number
else:
    weatherdata = [NASAPowerWeatherDataProvider(*coords)]

# Soil data
soil_filename = Path.cwd() / "data" / "soil" / "ec1.soil"
soildata = [CABOFileReader(soil_filename)]
if args.type == "soil":
    soildata = [CABOFileReader(soil_filename) for i in range(args.number)]

# Crop data
cropdata = [YAMLCropDataProvider()]
if args.type == "crop":
    cropdata = [YAMLCropDataProvider() for i in range(args.number)]

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
agromanagementdata = [yaml.safe_load(agro_template)]
if args.type == "agro":
    agromanagementdata = [yaml.safe_load(agro_template) for i in range(args.number)]

# Create an iterator that loops over all the combinations of input data
combined_inputs = product(sitedata, soildata, cropdata)
parameters_combined = (ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in combined_inputs)
all_runs = product(parameters_combined, weatherdata, agromanagementdata)
all_runs = list(all_runs)
all_runs = tqdm(all_runs, desc="Running PCSE models", unit="runs")

# Run the simulation ensemble
for run_data in all_runs:
    parameters, weatherdata, agromanagement = run_data
    try:
        wofost = Wofost72_WLP_FD(parameters, weatherdata, agromanagement)
        wofost.run_till_terminate()
    except WeatherDataProviderError:
        output = None
    else:
        output = wofost.get_output()
