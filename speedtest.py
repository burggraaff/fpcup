"""
Speed test PCSE by running an ensemble of replicates.
"""
from pathlib import Path

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Speed test PCSE by running an ensemble of replicates.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-t", "--type", help="which variable to replicate", choices=["site", "soil", "crop", "weather", "agro", "nasa", "excel"])
parser.add_argument("-n", "--number", help="number of replicates; result may be lower due to rounding", type=int, default=400)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Fetch site data
coords = (53, 6)
sitedata = fpcup.site.example(coords)
if args.verbose:
    print("Loaded site data")

# Fetch weather data
if args.type == "excel":
    weatherdata = [fpcup.weather.load_example_Excel() for i in range(args.number)]
elif args.type == "nasa":
    print("-- Did you disable caching in fpcup.weather? --")
    coords = [coords] * args.number
    weatherdata = fpcup.weather.load_weather_data_NASAPower(coords, leave_progressbar=args.verbose)
else:
    weatherdata = fpcup.weather.load_weather_data_NASAPower(coords, leave_progressbar=args.verbose)
if args.verbose:
    print("Loaded weather data")

# Soil data
soil_dir = args.data_dir / "soil"
soildata = fpcup.soil.load_folder(soil_dir)[0]
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
if args.type == "site":
    sitedata = [sitedata] * args.number
elif args.type == "soil":
    soildata = [soildata] * args.number
elif args.type == "crop":
    cropdata = [cropdata] * args.number
elif args.type == "weather":
    weatherdata = [weatherdata] * args.number
elif args.type == "agro":
    agromanagementdata = [agromanagementdata] * args.number
all_runs, n_runs = fpcup.model.bundle_parameters(sitedata, soildata, cropdata, weatherdata, agromanagementdata)
all_runs = list(all_runs)
if args.verbose:
    print(f"Prepared data for {n_runs} runs")

# Run the simulation ensemble
outputs, summary = fpcup.run_pcse_ensemble(all_runs, nr_runs=n_runs)
if args.verbose:
    print("Finished runs")
