"""
Run PCSE for plots within the BRP.
Plots for a user-provided crop (e.g. barley) are selected and only these are used.
Currently does not support using different variants/cultivars.

Example:
    python wofost_brp.py data/brp/brp2022.gpkg barley -v -p zeeland -f
"""
import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run PCSE for plots within the BRP.")
parser.add_argument("brp_filename", help="file to load the BRP from", type=fpcup.io.Path)
parser.add_argument("crop", help="crop to run simulations on", type=fpcup.crop.select_crop)
parser.add_argument("-p", "--province", help="province to select plots from (or all)", default=fpcup.geo.NETHERLANDS, type=fpcup.geo.process_input_province)
parser.add_argument("-d", "--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("-o", "--output_dir", help="folder to save PCSE outputs to", type=fpcup.io.Path, default=None)
parser.add_argument("-f", "--force", help="run all models even if the associated file already exists", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

args.SINGLE_PROVINCE = fpcup.geo.is_single_province(args.province)
args.YEAR = int(args.brp_filename.stem[-4:]) # Get the year from the BRP filename

# Set up a default output folder if a custom one was not provided
if args.output_dir is None:
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / f"brp{args.YEAR}-{args.crop.name}"


### Load constants
soildata = fpcup.soil.soil_types["ec1"]
agromanagement = args.crop.agromanagement_first_sowingdate(args.YEAR)


### Worker function; this runs PCSE once for one site
def run_pcse(i_row: tuple[int, fpcup._typing.Series]) -> bool | fpcup.model.RunDataBRP:
    """
    For a single BRP site, get the site-dependent data, wrap it into a RunData object, and run PCSE on it.
    Returns True if the results were succesfully written to file.
    Returns the corresponding RunData if a run failed.
    """
    i, row = i_row  # Unpack index/data pair
    latitude, longitude = row["latitude"], row["longitude"]

    # If desired, check if this run has been done already, and skip it if so
    if not args.force:
        run_id = fpcup.model.generate_run_id_BRP(brpyear=args.YEAR, plot_id=i, crop_name=args.crop.name, sowdate=agromanagement.crop_start_date)
        filename = args.output_dir / f"{run_id}.wout"
        if filename.exists():
            return False

    # Get weather data
    weatherdata = fpcup.weather.load_weather_data_NASAPower((latitude, longitude))

    # Combine input data
    run = fpcup.model.RunDataBRP(soildata=soildata, weatherdata=weatherdata, agromanagement=agromanagement, brpdata=row, brpyear=args.YEAR)
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
    # Feedback on constants
    if args.verbose:
        print(f"Save folder: {args.output_dir.absolute()}")
        print()

    # Make the output folder if it does not exist yet
    fpcup.io.makedirs(args.output_dir, exist_ok=True)

    # Load the BRP file
    brp = fpcup.io.read_geodataframe(args.brp_filename)
    if args.verbose:
        print(f"Loaded BRP data from {args.brp_filename.absolute()} -- {len(brp)} sites")

    # Select only the lines from the BRP file corresponding to the desired crop
    brp = brp.loc[brp["crop_species"] == args.crop.name]

    if args.verbose:
        print(f"Selected only plots growing {args.crop.name} -- {len(brp)} sites")
        print("Loaded agro management data:")
        print(agromanagement)

    # If we are only doing one province, select only the relevant lines from the BRP file
    if args.SINGLE_PROVINCE:
        brp = args.province.select_entries_in_province(brp)

        if args.verbose:
            print(f"Selected only plots in {args.province} -- {len(brp)} sites")

    # Split out the rows
    brp_rows = list(brp.iterrows())

    raise Exception

    ### Run the model
    model_statuses = fpcup.model.multiprocess_pcse(run_pcse, brp_rows, leave_progressbar=args.verbose)
    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose, use_existing=not args.force)
