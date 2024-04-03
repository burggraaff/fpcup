"""
Run a PCSE ensemble for different values of input parameters to determine their effect.
All available soil types are tested.
A specified number of sites, roughly central to the Netherlands, are used.

Example:
    python wofost_ensemble_parameters.py rdmsol wav -n 100 -v -c barley -s 16
"""
from functools import partial

from tqdm import tqdm

import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for different values of input parameters to determine their effect.")
parser.add_argument("parameter_names", help="parameter(s) to iterate over", type=str.upper, nargs="*")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to (default: generated from parameters)", type=fpcup.io.Path, default=None)
parser.add_argument("-n", "--number", help="number of values per parameter; note the exponential increase in runs when doing multiple parameters", type=int, default=100)
parser.add_argument("-c", "--crop", help="crop to run simulations on", default="barley", type=fpcup.crop.select_crop)
parser.add_argument("-s", "--number_sites", help="number of sites; result may be lower due to rounding", type=int, default=16)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Generate output folder name from parameters
if args.output_dir is None:
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / "-".join(args.parameter_names)

### Load constants
YEAR = 2022
CRS = fpcup.constants.WGS84
agromanagement = args.crop.agromanagement_first_sowingdate(YEAR)
soiltypes = list(fpcup.soil.soil_types.values())


### Worker function; this runs PCSE once for one site
def run_pcse(overrides: fpcup._typing.Iterable[dict[str, fpcup._typing.Number]], *,
             coordinates: fpcup._typing.Coordinates, weatherdata: fpcup.weather.WeatherDataProvider, soildata: fpcup.soil.SoilType) -> bool | fpcup.model.RunDataBRP:
    """
    For a single dictionary of override values (e.g. RDMSOL and WAV), wrap the data into a RunData object, and run PCSE on it.
    The other kwargs (e.g. sitedata, weatherdata) must be provided beforehand through a `partial` object.
    Returns True if the results were succesfully written to file.
    Returns the corresponding RunData if a run failed.
    """
    # Combine input data
    run = fpcup.model.RunData(override=overrides, soildata=soildata, weatherdata=weatherdata, agromanagement=agromanagement, geometry=coordinates, crs=CRS)
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
        print("Loaded agro management data:")
        print(agromanagement)
        print()

    # Generate the iterable
    iterable = fpcup.parameters.generate_ensemble_space(*args.parameter_names, n=args.number)
    if args.verbose:
        print(f"Iterating over the following parameters: {args.parameter_names}")
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
        ### Generate site-specific data
        weatherdata = fpcup.weather.load_weather_data_NASAPower(c)

        for soildata in tqdm(soiltypes, desc="Soil types", unit="type", leave=False):
            run_pcse_site = partial(run_pcse, coordinates=c, sitedata=sitedata, weatherdata=weatherdata, soildata=soildata)

            ### Run the model
            model_statuses = fpcup.model.multiprocess_pcse(run_pcse_site, iterable, leave_progressbar=False)
            model_statuses_combined.extend(model_statuses)

    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
