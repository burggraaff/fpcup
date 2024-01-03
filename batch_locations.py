"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs" / "locations"
results_dir = Path.cwd() / "results"

from itertools import product

from pcse.base import ParameterProvider

import fpcup

# Fetch site & weather data
coords = fpcup.site.grid_coordinate_range(latitude=(49, 54.1, 0.2), longitude=(3, 9, 0.2))
# coords = fpcup.site.grid_coordinate_linspace(latitude=(49, 54), longitude=(3, 9), n=100)
sitedata = fpcup.site.example(coords)
weatherdata = fpcup.weather.load_weather_data_NASAPower(coords)

# Soil data
soil_dir = data_dir / "soil"
soildata = fpcup.soil.load_folder(soil_dir)

# Crop data
cropdata = [fpcup.crop.default]

# Agromanagement calendars
agromanagementdata = [fpcup.agro.load_formatted(fpcup.agro.template_springbarley)]

# Set up iterables
parameters_combined = [ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in product(sitedata, soildata, cropdata)]

# Loop over input data
all_runs = product(parameters_combined, weatherdata, agromanagementdata)
nruns = len(parameters_combined) * len(weatherdata) * len(agromanagementdata)
print(f"Number of runs: {nruns}")
# (this does not work when the inputs are all generators)

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble(all_runs, nr_runs=nruns)

# Write the summary results to a CSV file
fpcup.io.save_ensemble_summary(summary, output_dir / "summary.csv")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, output_dir)
