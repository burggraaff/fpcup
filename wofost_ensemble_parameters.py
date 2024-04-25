"""
Run a PCSE ensemble for different values of input parameters to determine their effect.
All available soil types are tested.
A specified number of sites, roughly central to the Netherlands, are used.
By default, three crops (spring barley, green maize, winter wheat) are used; a single crop can be specified with -c.

Example:
    python wofost_ensemble_parameters.py rdmsol wav -n 100 -v

To simulate only multiple sowing dates, simply leave out the other parameters, e.g.:
    python wofost_ensemble_parameters.py -v -c barley -s 16 -d
"""
import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for different values of input parameters to determine their effect.")
parser.add_argument("parameter_names", help="parameter(s) to iterate over", type=str.upper, nargs="*")
parser.add_argument("-n", "--number", help="number of values per parameter; note the exponential increase in runs when doing multiple parameters", type=int, default=100)
parser.add_argument("-d", "--sowdates", help="run the simulation for multiple sowing dates (based on the crop's range)", action="store_true")
parser.add_argument("-c", "--crops", help="crop(s) to run simulations on", default=None)
parser.add_argument("-s", "--number_sites", help="number of sites; result may be lower due to rounding", type=int, default=16)
parser.add_argument("--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("--output_dir", help="folder to save PCSE outputs to (default: generated from parameters)", type=fpcup.io.Path, default=None)
parser.add_argument("-e", "--save_ensemble", help="save an ensemble summary at the end", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Generate output folder name from parameters
if args.output_dir is None:
    variable_names = args.parameter_names + ["DOS"] if args.sowdates else args.parameter_names
    variable_names = sorted(variable_names)
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / "-".join(variable_names)

### Load constants
YEAR = 2022
SOILTYPES = fpcup.soil.soil_types.values()
CONSTANTS = {}
variables = {"soildata": SOILTYPES}

### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()

    ### GENERAL SETUP
    # Check that enough inputs were provided
    assert len(args.parameter_names) > 0 or args.sowdates, "Please provide parameters to iterate over and/or use the -d flag to iterate over sowing dates."

    # Make the output folder if it does not exist yet
    fpcup.io.makedirs(args.output_dir, exist_ok=True)
    if args.verbose:
        print(f"Save folder: {args.output_dir.absolute()}")


    ### ITERABLE SETUP
    # Generate the parameter iterable
    if args.parameter_names:
        combined_parameters = fpcup.parameters.generate_ensemble_space(*args.parameter_names, n=args.number)
        variables = {"override": combined_parameters, **variables}

        if args.verbose:
            print(f"Iterating over the following parameters: {args.parameter_names}")

    # Get crop data
    if args.crops is None:  # Default
        args.crops = [fpcup.crop.crops["barley (spring)"], fpcup.crop.crops["maize (green)"], fpcup.crop.crops["wheat (winter)"]]
    elif isinstance(args.crops, str):  # Single crop
        args.crops = [fpcup.crop.select_crop(args.crops)]
    else:
        raise ValueError(f"Cannot determine target crop from input '{args.crops}'")

    # Mix in agromanagement
    if args.sowdates:  # Multiple sowing dates
        agromanagements = fpcup.tools.flatten_list(c.agromanagement_all_sowingdates(YEAR) for c in args.crops)
    else:  # Single sowing date
        agromanagements = [c.agromanagement_first_sowingdate(YEAR) for c in args.crops]

    if len(agromanagements) == 1:  # Single crop, single sowing date
        CONSTANTS = {"agromanagement": agromanagements[0], **CONSTANTS}
        if args.verbose:
            print("Generated 1 agromanagement calendar:")
            print(agromanagements[0])
            print()
    else:  # Multiple crops and/or multiple sowing dates
        variables = {"agromanagement": agromanagements, **variables}
        if args.verbose:
            print(f"Generated {len(agromanagements)} agromanagement calendars for {len(args.crops)} crops.")

    # Generate the coordinates
    coords = fpcup.site.generate_sites_space(latitude=(50.7, 53.6), longitude=(3.3, 7.3), n=args.number_sites)
    if args.verbose:
        print(f"Generated {len(coords)} coordinates.")

    # Pack the variables into one iterable of dictionaries
    variables = fpcup.tools.dict_product(variables)

    if args.verbose:
        print(f"Expected total runs: {args.number}**{len(args.parameter_names)} parameter values"
              f" * {len(agromanagements)} calendars"
              f" * {len(SOILTYPES)} soil types"
              f" * {len(coords)} sites"
              f" = {len(variables) * len(coords)} runs")


    ### RUN PCSE
    model_statuses = fpcup.model.run_pcse_site_ensemble(coords, variables, args.output_dir, run_data_constants=CONSTANTS, leave_progressbar=args.verbose)
    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    if args.save_ensemble:
        fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
