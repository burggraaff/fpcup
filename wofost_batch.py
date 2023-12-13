"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
import sys
from pathlib import Path
data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs"
from itertools import product

import yaml
import pandas as pd
from matplotlib import pyplot as plt

import pcse
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.util import WOFOST72SiteDataProvider
from pcse.db import NASAPowerWeatherDataProvider
# from progressbar import printProgressBar

print("This notebook was built with:")
print(f"python version: {sys.version}")
print(f"PCSE version: {pcse.__version__}")

cropd = YAMLCropDataProvider()
soil_dir = data_dir / "soil"
soil_files = [CABOFileReader(soil_filename) for soil_filename in soil_dir.glob("ec*")]
sited = WOFOST72SiteDataProvider(WAV=10)

agro_maize = """
- {year}-03-01:
    CropCalendar:
        crop_name: 'maize'
        variety_name: 'Grain_maize_201'
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
        crop_name: 'potato'
        variety_name: 'Fontane'
        crop_start_date: {year}-05-01
        crop_start_type: sowing
        crop_end_date: {year}-09-25
        crop_end_type: harvest
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {year}-12-01: null
"""

agro_sugarbeet = """
- {year}-03-01:
    CropCalendar:
        crop_name: 'sugarbeet'
        variety_name: 'Sugarbeet_601'
        crop_start_date: {year}-05-01
        crop_start_type: sowing
        crop_end_date: {year}-10-15
        crop_end_type: harvest
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {year}-12-01: null
"""
agro_templates = {"maize": agro_maize,
                  "potato": agro_potato,
                  "sugarbeet": agro_sugarbeet
                 }

weatherdata = NASAPowerWeatherDataProvider(longitude=5, latitude=52)
print(weatherdata)

weatherdf = pd.DataFrame(weatherdata.export()).set_index("DAY")

# Placeholder for storing summary results
summary_results = []

# Years to simulate
years = range(2004, 2008)

# Loop over crops, soils and years
all_runs = product(agro_templates.items(), soil_files, years)
nruns = len(agro_templates) * len(soil_files) * len(years)

# printProgressBar(0, nruns, prefix = "Progress:", suffix = "Complete", length = 50)
for i, inputs in enumerate(all_runs):
    (crop_type, agro), soild, year = inputs

    # Set the agromanagement with correct year and crop
    agromanagement = yaml.safe_load(agro.format(year=year))

    # String to identify this run
    soil_type = soild["SOLNAM"]
    run_id = "{crop}_{soil}_{year}".format(crop=crop_type, soil=soil_type, year=year)

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
    # finally:
        # printProgressBar(i+1, nruns, prefix = "Progress:", suffix = "Complete", length = 50)

    # convert daily output to Pandas DataFrame and store it
    df = pd.DataFrame(wofost.get_output()).set_index("day")
    fname = output_dir / (run_id + ".xlsx")
    df.to_excel(fname)

    # Collect summary results
    r = wofost.get_summary_output()[0]
    r["run_id"] = run_id
    summary_results.append(r)

# Write the summary results to an Excel file
df_summary = pd.DataFrame(summary_results).set_index("run_id")
fname = data_dir / "output" / "summary_results.xlsx"
df_summary.to_excel(fname)
