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
parser.add_argument("-p", "--province", help="province to select plots from (or all)", default="All", choices=fpcup.province.province_names+["All"], type=fpcup.province.process_input_province)
parser.add_argument("-s", "--sample", help="load only a subsample of the outputs, for testing", action="store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Set up the input/output directories
tag = args.output_dir.stem
results_dir = Path.cwd() / "results"
if args.verbose:
    print(f"Reading data from {args.output_dir.absolute()}")
    if args.sample:
        print("Only reading a subsample of the data.")
    print(f"Figures will be saved in {results_dir.absolute()}")

# Load the summary file(s)
summary = fpcup.io.load_ensemble_summary_from_folder(args.output_dir, sample=args.sample, leave_progressbar=args.verbose)
if args.verbose:
    print(f"Loaded summary file with {len(summary)} rows.")

# Add province information if this is not available
if "province" not in summary.columns:
    fpcup.province.add_provinces(summary, leave_progressbar=args.verbose)
    if args.verbose:
        print("Added province information")

# If we are only doing one province, select only the relevant lines from the summary file
if args.province != "All":
    summary = summary.loc[summary["province"] == args.province]
    tag = f"{tag}-{args.province}"
    if args.verbose:
        print(f"Selected only plots in {args.province} -- {len(summary)} sites")

# Useful information
if args.verbose:
    print(f"Figures will be saved as <name>-{tag}")

# Plot summary results
keys_to_plot = ["LAIMAX", "TWSO", "CTRAT", "CEVST", "DOE", "DOM"]
filename_summary = results_dir / f"WOFOST_{tag}-summary.pdf"

fpcup.plotting.plot_wofost_ensemble_summary(summary, keys=keys_to_plot, saveto=filename_summary, title=f"Summary of {len(summary)} WOFOST runs: {tag}", province=args.province)
if args.verbose:
    print(f"Saved batch results plot to {filename_summary.absolute()}")

# Space between summary and outputs sections
if args.verbose:
    print()

# Load the individual run outputs
outputs = fpcup.io.load_ensemble_results_from_folder(args.output_dir, sample=args.sample, leave_progressbar=args.verbose)

# Determine file save format
usevector = (len(outputs) < args.vector_max)
format_lines = "pdf" if usevector else "png"
if args.verbose:
    print(f"Number of files ({len(outputs)}) is {'smaller' if usevector else 'greater'} than maximum ({args.vector_max}) - saving line plots as {'vector' if usevector else 'bitmap'} files")
    print(f"Figure filenames will end in `_{tag}.{format_lines}`")

# Plot the individual runs
filename_results = results_dir / f"WOFOST_{tag}-outputs.{format_lines}"
fpcup.plotting.plot_wofost_ensemble_results(outputs, saveto=filename_results, replace_years=args.replace_years, title=f"Outputs from {len(outputs)} WOFOST runs\n{tag}", leave_progressbar=args.verbose)
if args.verbose:
    print(f"Saved batch results plot to {filename_results.absolute()}")
