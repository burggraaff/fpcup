"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs"
results_dir = Path.cwd() / "results"

from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import pcse
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.util import WOFOST72SiteDataProvider

import fpcup

print(f"PCSE version: {pcse.__version__}")

cropd = YAMLCropDataProvider()
soil_dir = data_dir / "soil"
soil_files = [CABOFileReader(soil_filename) for soil_filename in soil_dir.glob("ec*")]
sited = WOFOST72SiteDataProvider(WAV=10)

agro = """
- {date:%Y}-01-01:
    CropCalendar:
        crop_name: 'barley'
        variety_name: 'Spring_barley_301'
        crop_start_date: {date:%Y-%m-%d}
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {date:%Y}-12-01: null
"""
crop_type = "barley"

# Fetch weather data for the Netherlands (European part)
longitudes = np.arange(3, 9, 0.5)
latitudes = np.arange(49, 54.1, 0.5)
n_locations = len(longitudes)*len(latitudes)
coords = product(latitudes, longitudes)
weatherdata = [NASAPowerWeatherDataProvider(latitude=lat, longitude=long) for lat, long in tqdm(coords, total=n_locations, desc="Fetching weather data", unit="sites")]

# Set up iterables
sitedata = [sited]
soildata = soil_files
cropdata = [cropd]

parameters_combined = [ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in product(sitedata, soildata, cropdata)]

# Sowing dates to simulate
years = range(2000, 2021, 1)
doys = range(60, 91, 10)
years_doys = product(years, doys)
sowing_dates = [datetime.strptime(f"{year}-{doy}", "%Y-%j") for year, doy in years_doys]
agromanagementdata = [yaml.safe_load(agro.format(date=date)) for date in tqdm(sowing_dates, total=len(sowing_dates), desc="Loading agromanagement data", unit="calendars")]

# Loop over input data
all_runs = product(parameters_combined, weatherdata, agromanagementdata)
nruns = len(parameters_combined) * len(weatherdata) * len(agromanagementdata)
print(f"Number of runs: {nruns}")
# (this does not work when the inputs are all generators)

# Run the simulation ensemble
# raise Exception
outputs, df_summary = fpcup.run_pcse_ensemble_parallel(all_runs, nr_runs=nruns)

# Write the summary results to an Excel file
fname = output_dir / "summary_results.xlsx"
df_summary.to_excel(fname)

# Plot curves for outputs
keys = outputs[0].keys()
fig, axs = plt.subplots(nrows=len(keys), sharex=True, figsize=(8,10))

for df in outputs:
    time_without_year = pd.to_datetime(df.index.to_series()).apply(fpcup.replace_year_in_datetime)
    for ax, key in zip(axs, keys):
        ax.plot(time_without_year, df[key], alpha=0.25)
axs[-1].set_xlabel("Time")
for ax, key in zip(axs, keys):
    ax.set_ylabel(key)
    ax.set_ylim(ymin=0)
    ax.grid()
fig.align_ylabels()
axs[0].set_title(f"Results from {len(outputs)} WOFOST runs")
fig.savefig(results_dir / "WOFOST_batch.pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
