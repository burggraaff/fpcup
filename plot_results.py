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
parser.add_argument("-m", "--max", help="number of runs at which to rasterize and use generators", type=int, default=1000)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Set up the input/output directories
tag = args.output_dir.stem
results_dir = Path.cwd() / "results"
if args.verbose:
    print(f"Reading data from {args.output_dir.absolute()}")
    print(f"Figures will be saved in {results_dir.absolute()}")

# Settings
keys_outputs = ["DVS", "LAI", "TWSO", "TWLV", "TWST", "TWRT", "TAGP", "RD", "TRA", "SM", "WWLOW"]
keys_summary = ["LAIMAX", "TWSO", "CTRAT", "CEVST", "RD", "DOE", "DOM"]

nrfiles = len(list(args.output_dir.glob("*.wout")))
usevector = (nrfiles < args.max)

format_lines = "pdf" if usevector else "png"
if args.verbose:
    print(f"Expecting {nrfiles} files which is {'under' if usevector else 'over'} the maximum ({args.max}).")
    print(f"Figure filenames will end in `_{tag}.{format_lines}`")

# Load the results
outputs, summary = fpcup.io.load_ensemble_results_folder(args.output_dir, leave_progressbar=args.verbose, to_list=usevector)
summary.to_crs(fpcup.constants.CRS_AMERSFOORT, inplace=True)

# Plot the individual runs
filename_results = results_dir / f"WOFOST_{tag}-outputs.{format_lines}"
fpcup.plotting.plot_wofost_ensemble_results(outputs, keys=keys_outputs, saveto=filename_results, replace_years=args.replace_years, title=f"Outputs from {len(summary)} WOFOST runs\n{tag}", leave_progressbar=args.verbose)
if args.verbose:
    print(f"Saved batch results plot to {filename_results.absolute()}")

# Plot summary results
filename_summary = results_dir / f"WOFOST_{tag}-summary.pdf"

fpcup.plotting.plot_wofost_ensemble_summary(summary, keys=keys_summary, saveto=filename_summary, title=f"Summary of {len(summary)} WOFOST runs\n{tag}")
if args.verbose:
    print(f"Saved batch results plot to {filename_summary.absolute()}")
