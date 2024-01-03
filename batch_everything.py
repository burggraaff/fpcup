"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs" / "batch"
results_dir = Path.cwd() / "results"

from datetime import datetime
from itertools import product

import numpy as np

from pcse.base import ParameterProvider
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

import fpcup

cropd = YAMLCropDataProvider()
soil_dir = data_dir / "soil"
soil_files = [CABOFileReader(soil_filename) for soil_filename in soil_dir.glob("ec*")]
sited = fpcup.site.WOFOST72SiteDataProvider(WAV=10)

# Fetch weather data for the Netherlands (European part)
coords = fpcup.site.grid_coordinate_range(latitude=(49, 54.1, 0.25), longitude=(3, 9, 0.25))
weatherdata = fpcup.weather.load_weather_data_NASAPower(coords)

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
agromanagementdata = fpcup.agro.load_formatted_multi(fpcup.agro.template_springbarley_date, date=sowing_dates)

# Loop over input data
all_runs = product(parameters_combined, weatherdata, agromanagementdata)
nruns = len(parameters_combined) * len(weatherdata) * len(agromanagementdata)
print(f"Number of runs: {nruns}")
# (this does not work when the inputs are all generators)

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble_parallel(all_runs, nr_runs=nruns)

# Write the summary results to a CSV file
fpcup.io.save_ensemble_summary(summary, output_dir / "summary.csv")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, output_dir)
