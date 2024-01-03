"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs" / "locations"
results_dir = Path.cwd() / "results"

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

# Loop over input data
all_runs, n_runs = fpcup.model.bundle_parameters(sitedata, soildata, cropdata, weatherdata, agromanagementdata)

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble(all_runs, nr_runs=n_runs)

# Write the summary results to a CSV file
fpcup.io.save_ensemble_summary(summary, output_dir / "summary.csv")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, output_dir)
