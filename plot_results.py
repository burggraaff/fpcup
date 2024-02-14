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
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

# Set up the input/output directories
tag = args.output_dir.stem
results_dir = Path.cwd() / "results"
if args.verbose:
    print(f"Reading data from {args.output_dir.absolute()}")
    print(f"Figures will be saved in {results_dir.absolute()}")

# Load the results
outputs, summary = fpcup.io.load_ensemble_results_folder(args.output_dir, leave_progressbar=args.verbose)
summary.to_crs(fpcup.constants.CRS_AMERSFOORT, inplace=True)

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

# Plot summary results
keys_to_plot = ["LAIMAX", "TWSO", "CTRAT", "CEVST", "RD", "DOE", "DOM"]
filename_summary = results_dir / f"WOFOST_{tag}-summary.pdf"

fpcup.plotting.plot_wofost_ensemble_summary(summary, keys=keys_to_plot, saveto=filename_summary, title=f"Summary of {len(outputs)} WOFOST runs\n{tag}")
if args.verbose:
    print(f"Saved batch results plot to {filename_summary.absolute()}")
