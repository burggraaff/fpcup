"""
Playing around with the PCSE implementation of WOFOST.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

data_dir = Path("../pcse_notebooks/data")
output_dir = Path.cwd() / "outputs" / "sowdates"
results_dir = Path.cwd() / "results"

from itertools import product

from pcse.base import ParameterProvider
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider

import fpcup

cropd = YAMLCropDataProvider()
soil_dir = data_dir / "soil"
soil_files = [CABOFileReader(soil_filename) for soil_filename in soil_dir.glob("ec*")]
sited = fpcup.site.WOFOST72SiteDataProvider(WAV=10)

weatherdata = fpcup.weather.load_weather_data_NASAPower(coordinates=(5, 53), return_single=False)

# Sowing dates to simulate
sowing_dates = fpcup.agro.generate_sowingdates(year=2020, days_of_year=range(1, 222))

# Set up iterables
sitedata = [sited]
soildata = soil_files
cropdata = [cropd]

parameters_combined = [ParameterProvider(sitedata=site, soildata=soil, cropdata=crop) for site, soil, crop in product(sitedata, soildata, cropdata)]
agromanagementdata = fpcup.agro.load_formatted_multi(fpcup.agro.template_springbarley_date, date=sowing_dates)

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
