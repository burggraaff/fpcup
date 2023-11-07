"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook (https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb)
"""
import os, sys
from pathlib import Path
data_dir = Path("data/wofost")
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import yaml

import pcse
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.util import WOFOST72SiteDataProvider
from pcse.db import NASAPowerWeatherDataProvider

# Load crop/soil/weather data
crop_dir = data_dir/"CROPD"
crop_files = [CABOFileReader(filename) for filename in crop_dir.glob("*.CAB")]

soil_dir = data_dir/"SOILD"
soil_files = [CABOFileReader(filename) for filename in soil_dir.glob("EC*")]

sited = WOFOST72SiteDataProvider(WAV=10, CO2=360., SMLIM=0.35)

weatherdata = NASAPowerWeatherDataProvider(longitude=5, latitude=52)
print(weatherdata)
weatherdf = pd.DataFrame(weatherdata.export()).set_index("DAY")

# Define agromanagement
agro_maize = """
- {year}-03-01:
    CropCalendar:
        crop_name: '{crop}'
        variety_name: 'maize'
        crop_start_date: {year}-04-15
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {year}-12-01: null
"""

agro_potato = """
- {year}-03-01:
    CropCalendar:
        crop_name: '{crop}'
        variety_name: 'potato'
        crop_start_date: {year}-05-01
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {year}-12-01: null
"""

agro_sugarbeet = """
- {year}-03-01:
    CropCalendar:
        crop_name: '{crop}'
        variety_name: 'sugar_beet'
        crop_start_date: {year}-05-01
        crop_start_type: sowing
        crop_end_date: {year}-10-15
        crop_end_type: harvest
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {year}-12-01: null
"""
agro_templates = [agro_maize, agro_potato, agro_sugarbeet]

# Main loop
summary_results = []
years = range(2004, 2008)

# Loop over crops, soils and years
crops_calendars = zip(crop_files, agro_templates)
all_runs = product(crops_calendars, soil_files, years)
nruns = len(years) * len(crop_files) * len(soil_files)

for i, inputs in enumerate(all_runs):
    (cropd, agro), soild, year = inputs
    crop_type = cropd['CRPNAM']
    soil_type = soild['SOLNAM']
    # String to identify this run
    run_id = "{crop}_{soil}_{year}".format(crop=crop_type, soil=soil_type, year=year)

    # Set the agromanagement with correct year and crop
    agromanagement = yaml.load(agro.format(year=year, crop=crop_type))

    # Encapsulate parameters
    parameters = ParameterProvider(sitedata=sited, soildata=soild, cropdata=cropd)

    # Start WOFOST, run the simulation
    try:
        wofost = Wofost72_WLP_FD(parameters, weatherdata, agromanagement)
        wofost.run_till_terminate()
    except WeatherDataProviderError as e:
        msg = "Runid '%s' failed because of missing weather data." % run_id
        print(msg)
        continue

    # convert daily output to Pandas DataFrame and store it
    df = pd.DataFrame(wofost.get_output()).set_index("day")
    fname = os.path.join(data_dir, "output", run_id + ".xlsx")
    df.to_excel(fname)

    # Collect summary results
    r = wofost.get_summary_output()[0]
    r['run_id'] = run_id
    summary_results.append(r)
