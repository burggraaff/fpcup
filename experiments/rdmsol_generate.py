"""
Run a PCSE ensemble for different values of RDMSOL (maximum rooting depth) to determine its effect.
All available soil types are tested.
A single site, roughly central to the Netherlands, is used.

Example:
    %run experiments/rdmsol_generate.py -v -c barley -n 9
"""
from pathlib import Path

from tqdm import tqdm

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for different values of RDMSOL (maximum rooting depth) to determine its effect.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=Path, default=fpcup.settings.DEFAULT_OUTPUT / "rdmsol")
parser.add_argument("-c", "--crop", help="crop to run simulations on", default="barley", choices=("barley", "maize", "sorghum", "soy", "wheat"), type=str.lower)
parser.add_argument("-n", "--number", help="number of locations; result may be lower due to rounding", type=int, default=25)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Constants
YEAR = 2022
RDMSOLs = list(range(10, 150, 1))

# Generate the coordinates
crs = fpcup.constants.WGS84
coords = fpcup.site.generate_sites_space(latitude=(51, 53), longitude=(4, 6), n=args.number)
if args.verbose:
    print(f"Generated {len(coords)} coordinates")

### Generate constants
# Crop data
cropdata = fpcup.crop.default
if args.verbose:
    print("Loaded crop data")

# Agromanagement calendars
sowdate = fpcup.agro.sowdate_range(args.crop, YEAR)[0]
agromanagement = fpcup.agro.load_agrotemplate(args.crop, sowdate=sowdate)
if args.verbose:
    print("Loaded agro management data:")
    print(agromanagement)

### Loop over sites (coordinates)
failed_runs = []
for c in tqdm(coords, desc="Sites", unit="site", leave=args.verbose):
    ### Generate site-specific data
    # Get site data
    sitedata = fpcup.site.example(c)

    # Get weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower(c)

    ### Loop over soil data and depths
    for soildata in tqdm(fpcup.soil.soil_types.values(), desc="Soil types", unit="type", leave=False, disable=fpcup.RUNNING_IN_IPYTHON):
        for depth in tqdm(RDMSOLs, desc="Running models", unit="RDMSOL", leave=False, disable=fpcup.RUNNING_IN_IPYTHON):
            soildata["RDMSOL"] = depth
            run_id = f"{args.crop}_{soildata.name}_RDMSOL-{depth:d}_dos{sowdate:%Y%j}_lat{c[0]:.7f}-lon{c[1]:.7f}"

            # Combine input data
            run = fpcup.model.RunData(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, geometry=c, crs=crs, run_id=run_id)

            # Run model
            output = fpcup.model.run_pcse_single(run)

            # Save the results to file
            try:
                output.to_file(args.output_dir)
            # If the run failed, saving to file will also fail, so we instead note that this run failed
            except AttributeError:
                failed_runs.append(run_id)

# Feedback on failed runs: if any failed, let the user know. If none failed, only let the user know in verbose mode.
print()
if len(failed_runs) > 0:
    print(f"Number of failed runs: {len(failed_runs)}/{len(coords)}")
else:
    if args.verbose:
        print("No runs failed.")

# Save an ensemble summary
fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
