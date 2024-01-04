"""
Run a PCSE ensemble for multiple location with one sowing date.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for multiple location with one sowing date.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=Path, default=fpcup.settings.DEFAULT_OUTPUT / "locations")
parser.add_argument("-n", "--number", help="number of locations; result may be lower due to rounding", type=int, default=400)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Fetch site data
# coords = fpcup.site.grid_coordinate_range(latitude=(49, 54.1, 0.2), longitude=(3, 9, 0.2))
coords = fpcup.site.grid_coordinate_linspace(latitude=(49, 54), longitude=(3, 9), n=args.number)
sitedata = fpcup.site.example(coords)
if args.verbose:
    print("Loaded site data")

# Fetch weather data
weatherdata = fpcup.weather.load_weather_data_NASAPower(coords, leave_progressbar=args.verbose)
if args.verbose:
    print("Loaded weather data")

# Soil data
soil_dir = args.data_dir / "soil"
soildata = fpcup.soil.load_folder(soil_dir)
if args.verbose:
    print("Loaded soil data")

# Crop data
cropdata = fpcup.crop.default
if args.verbose:
    print("Loaded crop data")

# Agromanagement calendars
agromanagementdata = fpcup.agro.load_formatted(fpcup.agro.template_springbarley)
if args.verbose:
    print("Loaded agro management data")

# Loop over input data
all_runs, n_runs = fpcup.model.bundle_parameters(sitedata, soildata, cropdata, weatherdata, agromanagementdata)
if args.verbose:
    print(f"Prepared data for {n_runs} runs")

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble(all_runs, nr_runs=n_runs)
if args.verbose:
    print("Finished runs")

# Write the summary results to a CSV file
summary_filename = args.output_dir / "summary.csv"
fpcup.io.save_ensemble_summary(summary, summary_filename)
if args.verbose:
    print(f"Saved ensemble summary to {summary_filename.absolute()}")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, args.output_dir)
if args.verbose:
    print(f"Saved individual ensemble results to {args.output_dir.absolute()}")
