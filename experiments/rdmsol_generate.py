"""
Run a PCSE ensemble for different values of RDMSOL (maximum rooting depth) to determine its effect.
All available soil types are tested.
A single site, roughly central to the Netherlands, is used.

Example:
    python experiments/rdmsol_generate.py -v -c barley -n 9
"""
from itertools import product
from functools import partial

from tqdm import tqdm

import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for different values of RDMSOL (maximum rooting depth) to determine its effect.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_OUTPUT / "rdmsol")
parser.add_argument("-c", "--crop", help="crop to run simulations on", default="barley", choices=("barley", "maize", "sorghum", "soy", "wheat"), type=str.lower)
parser.add_argument("-n", "--number", help="number of locations; result may be lower due to rounding", type=int, default=25)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()


### Load constants
parameter_name = "RDMSOL"
parameter = fpcup.parameters.pcse_inputs[parameter_name]
parameterrange = parameter.generate_space(n=100)

YEAR = 2022
soiltypes = fpcup.soil.soil_types.values()
iterable = list(product(soiltypes, parameterrange))
crs = fpcup.constants.WGS84
cropdata = fpcup.crop.default
sowdate = fpcup.agro.sowdate_range(args.crop, YEAR)[0]
agromanagement = fpcup.agro.load_agrotemplate(args.crop, sowdate=sowdate)


### Worker function; this runs PCSE once for one site
def run_pcse(soildata_and_depth: tuple[fpcup.soil.SoilType, int], *,
                                       coordinates: fpcup._typing.Coordinates, sitedata: fpcup.site.PCSESiteDataProvider, weatherdata: fpcup.weather.WeatherDataProvider) -> bool | fpcup.model.RunDataBRP:
    """
    For a single pair of soil type and depth (RDMSOL), wrap the data into a RunData object, and run PCSE on it.
    Uses pre-made sitedata and weatherdata provided through a `partial` object.
    Returns True if the results were succesfully written to file.
    Returns the corresponding RunData if a run failed.
    """
    # Setup
    soildata, depth = soildata_and_depth

    # Combine input data
    run = fpcup.model.RunData(sitedata=sitedata, soildata=soildata, cropdata=cropdata, weatherdata=weatherdata, agromanagement=agromanagement, override={parameter_name: depth}, geometry=coordinates, crs=crs)
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
    ### Setup
    # Make the output folder if it does not exist yet
    fpcup.io.makedirs(args.output_dir, exist_ok=True)

    # Feedback on constants
    if args.verbose:
        print(f"Save folder: {args.output_dir.absolute()}")
        print()
        print("Loaded agro management data:")
        print(agromanagement)
        print()

    # Generate the coordinates
    coords = fpcup.site.generate_sites_space(latitude=(51, 53), longitude=(4, 6), n=args.number)
    if args.verbose:
        print(f"Generated {len(coords)} coordinates.")

    ### Loop over sites (coordinates)
    model_statuses_combined = []
    for c in tqdm(coords, desc="Sites", unit="site", leave=args.verbose):
        ### Generate site-specific data
        sitedata = fpcup.site.example(c)
        weatherdata = fpcup.weather.load_weather_data_NASAPower(c)
        run_pcse_site = partial(run_pcse, coordinates=c, sitedata=sitedata, weatherdata=weatherdata)

        ### Run the model
        model_statuses = fpcup.model.multiprocess_pcse(run_pcse_site, iterable, leave_progressbar=False)
        model_statuses_combined.extend(model_statuses)

    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
