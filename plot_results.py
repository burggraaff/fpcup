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
areas = summary["area"] if AREA_AVAILABLE else None
if args.verbose:
    un = "un" if not AREA_AVAILABLE else ""
    print(f"Plot areas {un}available; histograms and means will be {un}weighted")

# Define aggregation functions
if AREA_AVAILABLE:
    aggregator = fpcup.analysis.weighted_mean_dict(summary)
else:
    aggregator = {key: "mean" for key in fpcup.analysis.KEYS_AGGREGATE}

# Useful information
if args.verbose:
    print(f"Figures will be saved as <name>-{tag}")

# Plot summary results
filename_summary = results_dir / f"WOFOST_{tag}-summary.pdf"

fpcup.plotting.plot_wofost_ensemble_summary(summary, weights=areas, saveto=filename_summary, title=f"Summary of {len(summary)} WOFOST runs: {tag}", province=args.province)
if args.verbose:
    print(f"Saved batch results plot to {filename_summary.absolute()}")

# Aggregate results by province
byprovince = summary.groupby("province")

# Calculate the mean per province of several variables, weighted by plot area if possible
if AREA_AVAILABLE:
    filename_means = results_dir / f"WOFOST_{tag}-weighted-mean.csv"
    if args.verbose:
        print("Calculating weighted means")

# Use a normal mean if there is no area information available
else:
    filename_means = results_dir / f"WOFOST_{tag}-mean.csv"
    if args.verbose:
        print("Could not calculate weighted means because there is no 'area' column -- defaulting to a regular mean")

byprovince_mean = byprovince[fpcup.analysis.KEYS_AGGREGATE].agg(aggregator)
byprovince_mean.to_csv(filename_means)
if args.verbose:
    print(f"Saved aggregate mean columns to {filename_means.absolute()}")

# Add geometry and plot the results
byprovince_mean = fpcup.geo.add_province_geometry(byprovince_mean, "area")
filename_aggregate = results_dir / f"WOFOST_{tag}-summary-aggregate.pdf"

fpcup.plotting.plot_wofost_ensemble_summary_aggregate(byprovince_mean, keys=fpcup.analysis.KEYS_AGGREGATE, title=f"Summary of {len(summary)} WOFOST runs (aggregate): {tag}", saveto=filename_aggregate)
if args.verbose:
    print(f"Saved aggregate mean plot to {filename_aggregate.absolute()}")

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
