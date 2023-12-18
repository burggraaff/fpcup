"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path
data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs"
results_dir = Path.cwd() / "results"

from itertools import product

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

print(f"PCSE version: {pcse.__version__}")

cropd = YAMLCropDataProvider()
soil_dir = data_dir / "soil"
soil_files = [CABOFileReader(soil_filename) for soil_filename in soil_dir.glob("ec*")]
sited = WOFOST72SiteDataProvider(WAV=10)

agro = """
- {year}-03-01:
    CropCalendar:
        crop_name: 'barley'
        variety_name: 'Spring_barley_301'
        crop_start_date: {year}-03-03
        crop_start_type: sowing
        crop_end_date:
        crop_end_type: maturity
        max_duration: 300
    TimedEvents: null
    StateEvents: null
- {year}-12-01: null
"""
crop_type = "barley"

weatherdata = NASAPowerWeatherDataProvider(longitude=4.836232064803372, latitude=53.10069070497469)

# Placeholder for storing summary results
summary_results = []

# Years to simulate
years = range(1984, 2022)

# Loop over crops, soils and years
all_runs = product(soil_files, years)
nruns = len(soil_files) * len(years)
print(f"Number of runs: {nruns}")

outputs = []

for inputs in tqdm(all_runs, total=nruns, desc="Running models", unit="runs"):
    soild, year = inputs

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
        msg = f"Runid '{run_id}' failed because of missing weather data."
        print(msg)
        continue

    # convert daily output to Pandas DataFrame and store it
    df = pd.DataFrame(wofost.get_output()).set_index("day")
    fname = output_dir / (run_id + ".csv")
    df.to_csv(fname)
    outputs.append(df)

    # Collect summary results
    try:
        r = wofost.get_summary_output()[0]
    except IndexError:
        print(f"IndexError in {year}")
        continue
    r["run_id"] = run_id
    summary_results.append(r)

# Write the summary results to an Excel file
df_summary = pd.DataFrame(summary_results).set_index("run_id")
fname = output_dir / "summary_results.xlsx"
df_summary.to_excel(fname)

def replace_year_in_datetime(dt, newyear=2000):
    """
    For a datetime object yyyy-mm-dd, replace yyyy with newyear.
    """
    return dt.replace(year=newyear)

# Plot curves for outputs
keys = df.keys()
fig, axs = plt.subplots(nrows=len(keys), sharex=True, figsize=(8,10))

for df in outputs:
    time_without_year = pd.to_datetime(df.index.to_series()).apply(replace_year_in_datetime)
    for ax, key in zip(axs, keys):
        ax.plot(time_without_year, df[key], alpha=0.25)
axs[-1].set_xlabel("Time")
for ax, key in zip(axs, keys):
    ax.set_ylabel(key)
    ax.set_ylim(ymin=0)
    ax.grid()
fig.align_ylabels()
axs[0].set_title(f"Results from {len(outputs)} WOFOST runs")
fig.savefig(results_dir / "WOFOST_batch_years.pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
