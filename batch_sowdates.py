"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs" / "sowdates"
results_dir = Path.cwd() / "results"

import fpcup

# Fetch site & weather data
coords = (5, 53)
sitedata = fpcup.site.example(coords)
weatherdata = fpcup.weather.load_weather_data_NASAPower(coordinates=coords, return_single=False)

# Soil data
soil_dir = data_dir / "soil"
soildata = fpcup.soil.load_folder(soil_dir)

# Crop data
cropdata = [fpcup.crop.default]

# Agromanagement calendars
sowing_dates = fpcup.agro.generate_sowingdates(year=2020, days_of_year=range(1, 222))
agromanagementdata = fpcup.agro.load_formatted_multi(fpcup.agro.template_springbarley_date, date=sowing_dates)

# Loop over input data
all_runs, n_runs = fpcup.model.bundle_parameters(sitedata, soildata, cropdata, weatherdata, agromanagementdata)

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble(all_runs, nr_runs=n_runs)

# Write the summary results to a CSV file
fpcup.io.save_ensemble_summary(summary, output_dir / "summary.csv")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, output_dir)
