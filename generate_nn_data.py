"""
Run a PCSE ensemble for different values of input parameters to generate training data for a neural network.
One crop (out of barley/maize/wheat) must be specified.
All available soil types are tested.
A specified number of sites (-s), across the Netherlands, are tested.

Example:
    python generate_nn_data.py barley -s 100 -d 15 -y 5 -v
"""
import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run a PCSE ensemble for different values of input parameters to determine their effect.")
parser.add_argument("crop", help="crop to run simulations on", default=None)
# parser.add_argument("-n", "--number_parameters", help="number of values per parameter, across the range; note the exponential increase in runs when doing multiple parameters", type=int, default=25)
parser.add_argument("-d", "--number_sowdates", help="number of sowing dates, across the range", type=int, default=10)
parser.add_argument("-y", "--number_years", help="number of years, counting back from 2023", type=int, default=5, choices=range(1, 31))
parser.add_argument("-s", "--number_sites", help="number of sites", type=int, default=100)

parser.add_argument("--data_dir", help="folder to load PCSE data from", type=fpcup.io.Path, default=fpcup.settings.DEFAULT_DATA)
parser.add_argument("--output_dir", help="folder to save PCSE outputs to (default: generated from parameters)", type=fpcup.io.Path, default=None)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Get crop data
args.crop = fpcup.crop.select_crop(args.crop)

# Generate output folder name from parameters
if args.output_dir is None:
    args.output_dir = fpcup.settings.DEFAULT_OUTPUT / f"nn_{args.crop.name}"

### Load constants
BASEYEAR = 2023
SOILTYPES = fpcup.soil.soil_types.values()
PARAMETERS = [fpcup.parameters.RDMSOL, fpcup.parameters.WAV]


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()

    ### GENERAL SETUP
    # Make the output folder if it does not exist yet
    fpcup.io.makedirs(args.output_dir, exist_ok=True)
    if args.verbose:
        print(f"Save folder: {args.output_dir.absolute()}")


    ### ITERABLE SETUP
    if args.verbose:
        print()
        print(f"Iterating over {len(SOILTYPES)} soil types.")

    # Generate the parameter iterable
    parameter_ranges = {"RDMSOL": fpcup.parameters.RDMSOL.generate_space(40),
                        "WAV": fpcup.parameters.WAV.generate_space(10),
                        }
    if args.verbose:
        print()
        print("Iterating over the following parameters:")
        for key, prange in parameter_ranges.items():
            print(f"{key: <8} {len(prange): >4} values")

    parameter_ranges = fpcup.tools.dict_product(parameter_ranges)

    # Generate agromanagement calendars
    years = [BASEYEAR - i for i in range(args.number_years)][::-1]
    agromanagements = args.crop.agromanagement_N_sowingdates(years, args.number_sowdates, leave_progressbar=not args.verbose)
    if args.verbose:
        print()
        print(f"Iterating over {len(agromanagements)} agromanagement calendars, with sowing dates from {agromanagements[0].crop_start_date} to {agromanagements[-1].crop_start_date}.")

    # Pack the variables into one iterable of dictionaries
    variables = {"soildata": SOILTYPES,
                 "override": parameter_ranges,
                 "agromanagement": agromanagements,
                 }
    variables = fpcup.tools.dict_product(variables)


    # Generate the coordinates
    coords = fpcup.site.generate_sites_space(latitude=(50.7, 53.6), longitude=(3.3, 7.3), n=args.number_sites)
    if args.verbose:
        print()
        print(f"Generated {len(coords)} coordinates.")

    if args.verbose:
        print(f"Expected total runs: {len(variables) * len(coords):,}")


    ### RUN PCSE
    model_statuses = fpcup.model.run_pcse_site_ensemble(coords, variables, args.output_dir, leave_progressbar=args.verbose)
    failed_runs = fpcup.model.process_model_statuses(model_statuses, verbose=args.verbose)

    # Save an ensemble summary
    if args.save_ensemble:
        fpcup.io.save_ensemble_summary(args.output_dir, verbose=args.verbose)
