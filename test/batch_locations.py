"""
Run a PCSE ensemble for multiple (WGS84) coordinates, with all other parameters held constant.

Example:
    %run test/batch_locations.py -v -n 1000 -p Zeeland
"""
from pathlib import Path

from tqdm import tqdm

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for multiple location with one sowing date.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=Path, default=fpcup.settings.DEFAULT_OUTPUT / "locations")
parser.add_argument("-n", "--number", help="number of locations; result may be lower due to rounding", type=int, default=400)
parser.add_argument("-p", "--province", help="province to simulate plots in (or all)", default="Netherlands", choices=fpcup.geo.NAMES, type=fpcup.geo.process_input_province)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Generate the coordinates
crs = fpcup.constants.WGS84
SINGLE_PROVINCE = (args.province != "Netherlands")

if SINGLE_PROVINCE:
    coords = fpcup.site.generate_sites_in_province(args.province, args.number, leave_progressbar=args.verbose)
else:
    coords = fpcup.site.generate_sites_space(latitude=(50.7, 53.6), longitude=(3.3, 7.3), n=args.number)
if args.verbose:
    print(f"Generated {len(coords)} coordinates")

### Generate constants
# Soil data
soildata = fpcup.soil.soil_types["ec1"]
if args.verbose:
    print("Loaded soil data")

# Crop data
cropdata = fpcup.crop.default
if args.verbose:
    print("Loaded crop data")

# Agromanagement calendars
agromanagement = fpcup.agro.agromanagement_example
if args.verbose:
    print("Loaded agro management data:")
    print(agromanagement)

### Loop over sites (coordinates)
failed_runs = []
for c in tqdm(coords, desc="Running models", unit="site", leave=args.verbose):
    # Get site data
    sitedata = fpcup.site.example(c)

    # Get weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower(c)

    # Combine input data
    run = fpcup.model.RunData(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, geometry=c, crs=crs)

    # Run model
    output = fpcup.model.run_pcse_single(run)

    # Save the results to file
    try:
        output.to_file(args.output_dir)
    # If the run failed, saving to file will also fail, so we instead note that this run failed
    except AttributeError:
        failed_runs.append(c)

# Feedback on failed runs: if any failed, let the user know. If none failed, only let the user know in verbose mode.
print()
if len(failed_runs) > 0:
    print(f"Number of failed runs: {len(failed_runs)}/{len(coords)}")
else:
    if args.verbose:
        print("No runs failed.")

# Save an ensemble summary
fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
