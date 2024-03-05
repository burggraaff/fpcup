"""
Load and plot the results from a previous PCSE ensemble run.
"""
from pathlib import Path

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Load and plot the results from a previous PCSE ensemble run.")
parser.add_argument("output_dir", help="folder to load PCSE outputs from", type=Path)
parser.add_argument("-y", "--replace_years", help="replace all years in the output with 2000", action="store_true")
parser.add_argument("--vector_max", help="number of runs at which to switch from vector (PDF) to raster (PNG) files", type=int, default=5000)
parser.add_argument("-p", "--province", help="province to select plots from (or all of the Netherlands)", default="Netherlands", choices=fpcup.geo.NAMES, type=fpcup.geo.process_input_province)
parser.add_argument("-s", "--sample", help="load only a subsample of the outputs, for testing", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

SINGLE_PROVINCE = (args.province != "Netherlands")

# Set up the input/output directories
tag = args.output_dir.stem
results_dir = Path.cwd() / "results"
if args.verbose:
    print(f"Reading data from {args.output_dir.absolute()}")
    if args.sample:
        print("Only reading a subsample of the data.")
    print(f"Figures will be saved in {results_dir.absolute()}")

# Space between setup and summary sections
if args.verbose:
    print()

# Load the summary file(s)
summary = fpcup.io.load_ensemble_summary_from_folder(args.output_dir, sample=args.sample, leave_progressbar=args.verbose)
if args.verbose:
    print(f"Loaded summary file -- {len(summary)} rows")

# Add province information if this is not available
if "province" not in summary.columns:
    fpcup.geo.add_provinces(summary, leave_progressbar=args.verbose)
    if args.verbose:
        print("Added province information")

# If we are only doing one province, select only the relevant lines from the summary file
if SINGLE_PROVINCE:
    summary = summary.loc[summary["province"] == args.province]
    tag = f"{tag}-{args.province}"
    if args.verbose:
        print(f"Selected only sites in {args.province} -- {len(summary)} rows")

# Check if area information is available (to be used in weights)
AREA_AVAILABLE = ("area" in summary.columns)
if args.verbose:
    un = "un" if not AREA_AVAILABLE else ""
    print(f"Plot areas {un}available; histograms and means will be {un}weighted")

# Useful information
if args.verbose:
    print(f"Figures will be saved as <name>-{tag}")

# Plot summary results
filename_summary = results_dir / f"WOFOST_{tag}-summary.pdf"
fpcup.plotting.plot_wofost_summary(summary, weight_by_area=AREA_AVAILABLE, saveto=filename_summary, title=f"Summary of {len(summary)} WOFOST runs: {tag}", province=args.province)
if args.verbose:
    print(f"Saved batch results plot to {filename_summary.absolute()}")

# Calculate the mean per province of several variables, weighted by plot area if possible
if not SINGLE_PROVINCE:
    filename_province_means = results_dir / f"WOFOST_{tag}-summary-byprovince.csv"
    filename_province_plot = results_dir / f"WOFOST_{tag}-summary-byprovince.pdf"

    summary_byprovince = fpcup.geo.aggregate_province(summary)
    fpcup.geo.save_aggregate_province(summary_byprovince, filename_province_means)
    if args.verbose:
        print(f"Saved provincial aggregate columns to {filename_province_means.absolute()}")

    fpcup.plotting.plot_wofost_summary_byprovince(summary_byprovince, title=f"Summary of {len(summary)} WOFOST runs (by province): {tag}", saveto=filename_province_plot)
    if args.verbose:
        print(f"Saved provincial aggregate plot to {filename_province_plot.absolute()}")

# Space between summary and outputs sections
if args.verbose:
    print()

# If only one province is being done, load only the relevant files
run_ids = summary.index if SINGLE_PROVINCE else None

# Load the individual run results
results = fpcup.io.load_ensemble_results_from_folder(args.output_dir, run_ids=run_ids, sample=args.sample, leave_progressbar=args.verbose)

# Determine file save format
USEVECTOR = (len(results) < args.vector_max)
format_lines = "pdf" if USEVECTOR else "png"
if args.verbose:
    print(f"Number of result files ({len(results)}) is {'smaller' if USEVECTOR else 'greater'} than maximum ({args.vector_max}) - saving line plots as {'vector' if USEVECTOR else 'bitmap'} files")
    print(f"Figure filenames will end in `_{tag}.{format_lines}`")

# Plot the individual runs
filename_results = results_dir / f"WOFOST_{tag}-results.{format_lines}"
fpcup.plotting.plot_wofost_ensemble_results(results, saveto=filename_results, replace_years=args.replace_years, title=f"Growth curves from {len(results)} WOFOST runs\n{tag}", leave_progressbar=args.verbose)
if args.verbose:
    print(f"Saved batch results plot to {filename_results.absolute()}")
