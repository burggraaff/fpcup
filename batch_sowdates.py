"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path
data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs"
results_dir = Path.cwd() / "results"

from itertools import product
from datetime import datetime

import yaml
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import pcse
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.util import WOFOST72SiteDataProvider
from pcse.db import NASAPowerWeatherDataProvider

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

weatherd = NASAPowerWeatherDataProvider(longitude=5, latitude=53)

# Sowing dates to simulate
year = 2020
doys = range(1, 222)
sowing_dates = [datetime.strptime(f"{year}-{doy}", "%Y-%j") for doy in doys]

# Set up iterables
sitedata = [sited]
soildata = soil_files
cropdata = [cropd]

parameters_combined = [ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in product(sitedata, soildata, cropdata)]
weatherdata = [weatherd]
agromanagementdata = [yaml.safe_load(agro.format(date=date)) for date in sowing_dates]

# Loop over input data
all_runs = product(parameters_combined, weatherdata, agromanagementdata)
nruns = len(parameters_combined) * len(weatherdata) * len(agromanagementdata)
print(f"Number of runs: {nruns}")
# (this does not work when the inputs are all generators)

# Placeholder for storing (summary) results
outputs = []
summary_results = []

for parameters, weatherdata, agromanagement in tqdm(all_runs, total=nruns, desc="Running models", unit="runs"):
    # String to identify this run
    soil_type = parameters._soildata["SOLNAM"]
    startdate = list(agromanagement[0].keys())[0]
    sowdate = agromanagement[0][startdate]["CropCalendar"]["crop_start_date"]
    crop_type = agromanagement[0][startdate]["CropCalendar"]["crop_name"]
    run_id = f"{crop_type}_{soil_type}_sown-{sowdate:%Y-%m-%d}"

    # Start WOFOST, run the simulation
    try:
        wofost = Wofost72_WLP_FD(parameters, weatherdata, agromanagement)
        wofost.run_till_terminate()
    except WeatherDataProviderError as e:
        msg = f"Runid '{run_id}' failed because of missing weather data."
        print(msg)
        continue

    # Convert individual output to Pandas DataFrame
    df = pd.DataFrame(wofost.get_output()).set_index("day")
    outputs.append(df)

    # Save individual output to file
    fname = output_dir / (run_id + ".csv")
    df.to_csv(fname)

    # Collect summary results
    try:
        r = wofost.get_summary_output()[0]
    except IndexError:
        # print(f"IndexError in run '{run_id}'")
        continue
    r["run_id"] = run_id
    summary_results.append(r)

# Write the summary results to an Excel file
df_summary = pd.DataFrame(summary_results).set_index("run_id")
fname = output_dir / "summary_results.xlsx"
df_summary.to_excel(fname)

# Plot curves for outputs
keys = df.keys()
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
fig.savefig(results_dir / "WOFOST_batch_sowdates.pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
