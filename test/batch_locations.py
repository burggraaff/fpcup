"""
Run a PCSE ensemble for multiple (WGS84) coordinates, with all other parameters held constant.

Example:
    python test/batch_locations.py -v -n 1000 -p Zeeland
"""
import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for multiple location with one sowing date.")
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_OUTPUT / "locations")
parser.add_argument("-n", "--number", help="number of locations; result may be lower due to rounding", type=int, default=400)
parser.add_argument("-p", "--province", help="province to simulate plots in (or all)", default=fpcup.geo.NETHERLANDS, type=fpcup.geo.process_input_province)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

args.SINGLE_PROVINCE = fpcup.geo.is_single_province(args.province)


### Load constants
YEAR = 2022
CRS = fpcup.constants.WGS84
soildata = fpcup.soil.soil_types["ec3"]
agromanagement = fpcup.crop.SpringBarley.agromanagement_first_sowingdate(YEAR)


### Worker function; this runs PCSE once for one site
def run_pcse(coordinates: fpcup._typing.Coordinates) -> bool | fpcup.model.RunData:
    """
    For a single site, get the site-dependent data, wrap it into a RunData object, and run PCSE on it.
    Returns True if the results were succesfully written to file.
    Returns the corresponding RunData if a run failed.
    """
    # Get weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower(coordinates)

    # Combine input data
    run = fpcup.model.RunData(soildata=soildata, weatherdata=weatherdata, agromanagement=agromanagement, geometry=coordinates, crs=CRS)
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
        print("Loaded soil data")
        print("Loaded agro management data:")
        print(agromanagement)
        print()

    # Generate the coordinates
    if args.SINGLE_PROVINCE:
        coords = args.province.generate_random_points(args.number, leave_progressbar=args.verbose)
    else:
        coords = fpcup.site.generate_sites_space(latitude=(50.7, 53.6), longitude=(3.3, 7.3), n=args.number)
    if args.verbose:
        print(f"Generated {len(coords)} coordinates")

    ### Run the model
    model_statuses = fpcup.model.multiprocess_pcse(run_pcse, coords, leave_progressbar=args.verbose)
    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
