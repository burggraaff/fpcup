"""
Load and plot the results from a previous PCSE ensemble run.
"""
from pathlib import Path

import fpcup

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser("Load and plot the results from a previous PCSE ensemble run.")
parser.add_argument("output_dir", help="folder to load PCSE outputs from")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-y", "--replace_years", help="replace all years in the output with 2000", action="store_true")
parser.add_argument("--vector_max", help="number of runs at which to switch from vector (PDF) to raster (PNG) files", type=int, default=5000)
args = parser.parse_args()

# Set up the input/output directories
output_dir = Path(args.output_dir)
tag = output_dir.stem
results_dir = Path.cwd() / "results"
if args.verbose:
    print(f"Reading data from {output_dir.absolute()}")
    print(f"Figures will be saved in {results_dir.absolute()}")

# Load the results
outputs, summary = fpcup.io.load_ensemble_results_folder(output_dir)

# Determine file save format
usevector = (len(outputs) < args.vector_max)
format_lines = "pdf" if usevector else "png"
if args.verbose:
    print(f"Number of files ({len(outputs)}) is {'smaller' if usevector else 'greater'} than maximum ({args.vector_max}) - saving line plots as {'vector' if usevector else 'bitmap'} files")
    print(f"Figure filenames will end in `_{tag}.{format_lines}`")

# Plot the individual runs
filename_results = results_dir / f"WOFOST_batch_{tag}.{format_lines}"
fpcup.plotting.plot_wofost_ensemble_results(outputs, saveto=filename_results, replace_years=args.replace_years)
if args.verbose:
    print(f"Saved batch results plot to {filename_results.absolute()}")

# Plot summary results