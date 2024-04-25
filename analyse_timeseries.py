"""
Load and plot the summary results from a previous PCSE ensemble run.

Example:
    python analyse_summary.py outputs/sites/ -v
"""
import fpcup

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Load and plot the results from a previous PCSE ensemble run.")
parser.add_argument("output_dir", help="folder to load PCSE outputs from", type=fpcup.io.Path)
parser.add_argument("-r", "--results_dir", help="folder to save plots into", type=fpcup.io.Path, default=fpcup.DEFAULT_RESULTS)
parser.add_argument("-p", "--province", help="province to select plots from (or all)", default=fpcup.geo.NETHERLANDS, type=fpcup.geo.process_input_province)
parser.add_argument("-y", "--replace_years", help="replace all years in the output with 2000", action="store_true")
parser.add_argument("-s", "--sample", help="load only a subsample of the outputs, for testing", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

args.SINGLE_PROVINCE = fpcup.geo.is_single_province(args.province)


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()
    ### Setup
    # Set up the input/output directories
    tag = args.output_dir.stem
    if args.verbose:
        print(f"Reading data from {args.output_dir.absolute()}")
        if args.sample:
            print("Only reading a subsample of the data.")
        print(f"Figures will be saved in {args.results_dir.absolute()}")

    # Space between setup and summary sections
    if args.verbose:
        print()

    ### Geo selection based on summary, if desired
    if args.SINGLE_PROVINCE:
        # Load the summary file(s)
        summary = fpcup.model.Summary.from_folder(args.output_dir, sample=args.sample, leave_progressbar=args.verbose)
    if args.verbose:
        print(f"Loaded summary file -- {len(summary)} rows")

        # Convert to GeoSummary
        summary = fpcup.model.GeoSummary.from_summary(summary)

    # If we are only doing one province, select only the relevant lines from the summary file
        summary = args.province.select_entries_in_province(summary)
        tag = f"{tag}-{args.province.abbreviation}"
        if args.verbose:
            print(f"Selected only sites in {args.province} -- {len(summary)} runs")

        run_ids = summary.index
    else:
        run_ids = None

    ### Load time series
    # If only one province is being done, load only the relevant files
    run_ids = summary.index if args.SINGLE_PROVINCE else None

    # Load the individual run results
    results = fpcup.io.load_ensemble_results_from_folder(args.output_dir, run_ids=run_ids, sample=args.sample, leave_progressbar=args.verbose)

    # Plot the individual runs
    filename_results = args.results_dir / f"WOFOST_{tag}-results.xxx"  # pdf or png
    fpcup.plotting.plot_wofost_ensemble_results(results, saveto=filename_results, replace_years=args.replace_years, title=f"Growth curves from {len(results)} WOFOST runs\n{tag}", leave_progressbar=args.verbose)
    if args.verbose:
        print(f"Saved batch results plot to {filename_results.absolute()}")
