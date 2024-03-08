"""
Run a PCSE ensemble for multiple (WGS84) coordinates, with all other parameters held constant.

Example:
    %run test/batch_locations.py -v -n 1000 -p Zeeland
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

import fpcup

### Parse command line arguments
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for multiple location with one sowing date.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=Path, default=fpcup.settings.DEFAULT_OUTPUT / "locations")
parser.add_argument("-n", "--number", help="number of locations; result may be lower due to rounding", type=int, default=400)
parser.add_argument("-p", "--province", help="province to simulate plots in (or all)", default="Netherlands", choices=fpcup.geo.NAMES, type=fpcup.geo.process_input_province)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

args.SINGLE_PROVINCE = (args.province != "Netherlands")

### Load constants
crs = fpcup.constants.WGS84
soildata = fpcup.soil.soil_types["ec3"]
cropdata = fpcup.crop.default
agromanagement = fpcup.agro.agromanagement_example


### Worker function; this runs PCSE once for one site
def run_pcse(coordinates: fpcup._typing.Coordinates) -> bool | str:
    """
    For a single site, get the site-dependent data, wrap it into a RunData object, and run PCSE on it.
    Returns True if the results were succesfully written to file.
    Returns a str (the run_id) if the run failed.
    """
    # Get site data
    sitedata = fpcup.site.example(coordinates)

    # Get weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower(coordinates)

    # Combine input data
    run = fpcup.model.RunData(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, geometry=coordinates, crs=crs)

    # Run model
    output = fpcup.model.run_pcse_single(run)

    # Save the results to file
    try:
        output.to_file(args.output_dir)
    # If the run failed, saving to file will also fail, so we instead note that this run failed
    except AttributeError:
        return run.run_id
    else:
        return True


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    ### Setup
    # Feedback on constants
    if args.verbose:
        print("Loaded soil data")
        print("Loaded crop data")
        print("Loaded agro management data:")
        print(agromanagement)
        print()

    # Generate the coordinates
    if args.SINGLE_PROVINCE:
        coords = fpcup.site.generate_sites_in_province(args.province, args.number, leave_progressbar=args.verbose)
    else:
        coords = fpcup.site.generate_sites_space(latitude=(50.7, 53.6), longitude=(3.3, 7.3), n=args.number)
    if args.verbose:
        print(f"Generated {len(coords)} coordinates")

    ### Run the model
    with Pool() as p:
        runs_successful = list(tqdm(p.imap(run_pcse, coords, chunksize=25), total=len(coords), desc="Running models", unit="site"))

    # Feedback on failed runs: if any failed, let the user know. If none failed, only let the user know in verbose mode.
    failed_runs = [run_id for run_id in runs_successful if isinstance(run_id, str)]
    print()
    if len(failed_runs) > 0:
        print(f"Number of failed runs: {len(failed_runs)}/{len(coords)}")
    else:
        if args.verbose:
            print("No runs failed.")

    # Save an ensemble summary
    fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
