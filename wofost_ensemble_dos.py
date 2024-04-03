"""
Run a PCSE ensemble for different sowing dates (within the suggested range) for one crop.
All possible crop dates for the specified crop are tested.
All available soil types are tested.
A specified number of sites, roughly central to the Netherlands, are used.

Example:
    python wofost_ensemble_dos.py barley -v -s 16
"""
from functools import partial

from tqdm import tqdm

import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for different sowing dates for one crop.")
parser.add_argument("parameter_names", help="parameter(s) to iterate over", type=str.upper, nargs="*")
parser.add_argument("crop", help="crop to run simulations on", type=fpcup.crop.select_crop)
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to (default: generated from parameters)", type=fpcup.io.Path, default=None)
parser.add_argument("-s", "--number_sites", help="number of sites; result may be lower due to rounding", type=int, default=16)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Generate output folder name from parameters
if args.output_dir is None:
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / "DOS"

### Load constants
YEAR = 2022
crs = fpcup.constants.WGS84
cropdata = fpcup.crop.default
soiltypes = list(fpcup.soil.soil_types.values())


### Worker function; this runs PCSE once for one site
def run_pcse(agromanagement: fpcup.agro.AgromanagementDataSingleCrop, *,
             coordinates: fpcup._typing.Coordinates, sitedata: fpcup.site.PCSESiteDataProvider, weatherdata: fpcup.weather.WeatherDataProvider, soildata: fpcup.soil.SoilType) -> bool | fpcup.model.RunDataBRP:
    """
    For a single agromanagement calendar, wrap the data into a RunData object, and run PCSE on it.
    The other kwargs (e.g. sitedata, weatherdata) must be provided beforehand through a `partial` object.
    Returns True if the results were succesfully written to file.
    Returns the corresponding RunData if a run failed.
    """
    # Combine input data
    run = fpcup.model.RunData(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, geometry=coordinates, crs=crs)
    run.to_file(args.output_dir)

    # Run model
    output = fpcup.model.run_pcse_single(run)

    # Save the results to file
    try:
        output.to_file(args.output_dir)
    # If the run failed, saving to file will also fail, so we instead note that this run failed
    except AttributeError:
        return run
    else:
        return True


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()

    ### SETUP
    # Make the output folder if it does not exist yet
    fpcup.io.makedirs(args.output_dir, exist_ok=True)

    # Feedback on constants
    if args.verbose:
        print(f"Save folder: {args.output_dir.absolute()}")
        print()

    # Generate the iterable
    iterable = args.crop.agromanagement_all_sowingdates(YEAR)
    if args.verbose:
        print(f"Length of iterator: {len(iterable)}")

    # Generate the coordinates
    coords = fpcup.site.generate_sites_space(latitude=(51, 53), longitude=(4, 6), n=args.number_sites)
    if args.verbose:
        print(f"Generated {len(coords)} coordinates.")


    ### RUN PCSE
    if args.verbose:
        print(f"Expected total runs: {len(iterable) * len(coords) * len(soiltypes)}")
    model_statuses_combined = []
    for c in tqdm(coords, desc="Sites", unit="site", leave=args.verbose):
        for soildata in tqdm(soiltypes, desc="Soil types", unit="type", leave=False):
            ### Generate site-specific data
            sitedata = fpcup.site.example(c)
            weatherdata = fpcup.weather.load_weather_data_NASAPower(c)
            run_pcse_site = partial(run_pcse, coordinates=c, sitedata=sitedata, weatherdata=weatherdata, soildata=soildata)

            ### Run the model
            model_statuses = fpcup.model.multiprocess_pcse(run_pcse_site, iterable, leave_progressbar=False)
            model_statuses_combined.extend(model_statuses)

    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
