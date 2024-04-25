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

    ### Summary
    # Load the combined input/output summary
    summary = fpcup.io.load_combined_ensemble_summary(args.output_dir, sample=args.sample, leave_progressbar=args.verbose)
    if args.verbose:
        print(f"Loaded summary file -- {len(summary)} rows")

    # Convert to GeoSummary
    summary = fpcup.model.GeoSummary(summary)

    # If we are only doing one province, select only the relevant lines from the summary file
    if args.SINGLE_PROVINCE:
        summary = args.province.select_entries_in_province(summary)
        tag = f"{tag}-{args.province.abbreviation}"
        if args.verbose:
            print(f"Selected only sites in {args.province} -- {len(summary)} rows")

    # Otherwise, add province information to all lines, dropping those that fall outside all provinces
    elif "province" not in summary.columns:
        fpcup.geo.add_provinces(summary, leave_progressbar=args.verbose)
        if args.verbose:
            print(f"Added province information -- {len(summary)} rows")

    # Check if area information is available (to be used in weights)
    AREA_AVAILABLE = ("area" in summary.columns)
    if args.verbose:
        un = "un" if not AREA_AVAILABLE else ""
        print(f"Plot areas {un}available; histograms and means will be {un}weighted")

    # Useful information
    if args.verbose:
        print(f"Figures will be saved as <name>-{tag}")

    # Plot summary results
    filename_summary = args.results_dir / f"WOFOST_{tag}-summary.pdf"
    fpcup.plotting.plot_wofost_summary(summary, weight_by_area=AREA_AVAILABLE, saveto=filename_summary, title=f"Summary of {len(summary)} WOFOST runs: {tag}", province=args.province)
    if args.verbose:
        print(f"Saved batch results plot to {filename_summary.absolute()}")

    # Calculate the mean per province of several variables, weighted by plot area if possible
    if not args.SINGLE_PROVINCE:
        filename_province_means = args.results_dir / f"WOFOST_{tag}-summary-byprovince.csv"
        filename_province_plot = args.results_dir / f"WOFOST_{tag}-summary-byprovince.pdf"

        # Save values to a text file
        summary_byprovince = fpcup.aggregate.aggregate_province(summary)
        fpcup.aggregate.save_aggregate_province(summary_byprovince, filename_province_means)
        if args.verbose:
            print(f"Saved provincial aggregate columns to {filename_province_means.absolute()}")

        # Geo plot
        fpcup.plotting.plot_wofost_summary_byprovince(summary_byprovince, title=f"Summary of {len(summary)} WOFOST runs (by province): {tag}", saveto=filename_province_plot)
        if args.verbose:
            print(f"Saved provincial aggregate plot to {filename_province_plot.absolute()}")

    # Space between summary and outputs sections
    if args.verbose:
        print()
