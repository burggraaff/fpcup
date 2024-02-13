"""
Run a PCSE ensemble for one location with multiple sowing dates.
Based on the example notebook: https://github.com/ajwdewit/pcse_notebooks/blob/master/04%20Running%20PCSE%20in%20batch%20mode.ipynb
"""
from pathlib import Path

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for one location with multiple sowing dates.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=Path, default=fpcup.settings.DEFAULT_OUTPUT / "sowdates")
parser.add_argument("-y", "--year", help="year to simulate", type=int, default=2020, choices=range(1984,2022), metavar="[1984-2022]")
parser.add_argument("-f", "--first", help="first sowing date (day-of-year)", type=int, default=1)
parser.add_argument("-l", "--last", help="last sowing date (day-of-year)", type=int, default=222)
parser.add_argument("-s", "--step", help="sowing date step size", type=int, default=1)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Fetch site data
coords = (53, 6)
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
sowing_dates = fpcup.agro.generate_sowingdates(year=args.year, days_of_year=range(args.first, args.last, args.step))
agromanagementdata = fpcup.agro.load_agrotemplate_multi("barley (spring)", sowdate=sowing_dates, leave_progressbar=args.verbose)
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
summary_filename = args.output_dir / "ensemble.wsum"
summary.to_csv(summary_filename)
if args.verbose:
    print(f"Saved ensemble summary to {summary_filename.absolute()}")

# Write the individual outputs to CSV files
fpcup.io.save_ensemble_results(outputs, args.output_dir)
if args.verbose:
    print(f"Saved individual ensemble results to {args.output_dir.absolute()}")
