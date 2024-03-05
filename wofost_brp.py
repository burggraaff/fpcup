"""
Run PCSE for plots within the BRP.

Example:
    %run wofost_brp.py data/brp/brpgewaspercelen_definitief_2022-processed.gpkg -v -c barley -p zeeland -f
"""
from tqdm import tqdm
import datetime as dt

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run PCSE for plots within the BRP.")
parser.add_argument("brp_filename", help="file to load the BRP from", type=fpcup.io.Path)
parser.add_argument("-c", "--crop", help="crop to run simulations on (or all)", default="All", choices=("barley", "maize", "sorghum", "soy", "wheat"), type=str.lower)
parser.add_argument("-p", "--province", help="province to select plots from (or all)", default="Netherlands", choices=fpcup.geo.NAMES, type=fpcup.geo.process_input_province)
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=fpcup.io.Path, default=None)
parser.add_argument("-f", "--force", help="run all models even if the associated file already exists", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

SINGLE_PROVINCE = (args.province != "Netherlands")

# Get the year from the BRP filename
year = int(args.brp_filename.stem.split("_")[-1].split("-")[0])

# Determine whether to use a common crop type or row-specific ones
USE_COMMON_CROPTYPE = (args.crop.title() != "All")

# Set up a default output folder if a custom one was not provided
if args.output_dir is None:
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / f"brp{year}-{args.crop}"

    if args.verbose:
        print(f"Default save folder: {args.output_dir.absolute()}")

# Make the output folder if it does not exist yet
fpcup.io.makedirs(args.output_dir, exist_ok=True)

# Load the BRP file
brp = fpcup.io.read_gpd(args.brp_filename)
if args.verbose:
    print(f"Loaded BRP data from {args.brp_filename.absolute()} -- {len(brp)} sites")

# If we are only doing one province, select only the relevant lines from the BRP file
if SINGLE_PROVINCE:
    brp = brp.loc[brp["province"] == args.province]

    if args.verbose:
        print(f"Selected only plots in {args.province} -- {len(brp)} sites")

# Set up crop data: sow dates, selecting relevant plots, setting up agromanagement calendars
if USE_COMMON_CROPTYPE:
    # Select only the relevant lines from the BRP file
    brp = brp.loc[brp["crop_species"] == args.crop]

    if args.verbose:
        print(f"Selected only plots growing {args.crop} -- {len(brp)} sites")

    sowdate = fpcup.agro.sowdate_range(args.crop, year)[0]
    agromanagement = fpcup.agro.load_agrotemplate(args.crop, sowdate=sowdate)

    if args.verbose:
        print("Loaded agro management data:")
        print(agromanagement)

# Pre-load data (to be improved)
soildata = fpcup.soil.soil_types["ec1"]
cropdata = fpcup.crop.default

failed_runs = []
# Run the simulations (minimum working example)
for i, row in tqdm(brp.iterrows(), total=len(brp), desc="Running PCSE", unit="plot"):
    coords = (row["latitude"], row["longitude"])

    # Get agromanagement data if needed
    if not USE_COMMON_CROPTYPE:
        args.crop = row["crop"]
        sowdate = fpcup.agro.sowdate_range(args.crop, year)[0]
        agromanagement = fpcup.agro.load_agrotemplate(args.crop, sowdate=sowdate)

    # If desired, check if this run has been done already, and skip it if so
    if not args.force:
        run_id = fpcup.model.run_id_BRP(year, i, args.crop, sowdate)
        filename = args.output_dir / f"{run_id}.wout"
        if filename.exists():
            continue

    # Fetch site data
    sitedata = fpcup.site.example(coords)

    # Fetch weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower(coords)

    # Bundle parameters
    run = fpcup.model.RunDataBRP(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, brpdata=row, brpyear=year)

    # Run model
    output = fpcup.model.run_pcse_single(run)

    # Save the results to file
    # try:
    output.to_file(args.output_dir)
    # If the run failed, saving to file will also fail, so we instead note that this run failed
    # except AttributeError:
        # failed_runs.append(i)

# Feedback on failed runs: if any failed, let the user know. If none failed, only let the user know in verbose mode.
print()
if len(failed_runs) > 0:
    print(f"Number of failed runs: {len(failed_runs)}/{len(brp)}")
else:
    if args.verbose:
        print("No runs failed.")

# Save an ensemble summary
fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose, use_existing=not args.force)
