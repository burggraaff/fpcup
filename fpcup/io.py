"""
Functions for file input and output.

TO DO: Add the option to skip a run if run_id.wout / run_id.wsum exist (-f to force).
"""
from functools import cache
from os import makedirs
from pathlib import Path

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
from geopandas import GeoDataFrame, read_file as read_gpd

import pandas as pd

from tqdm import tqdm

from ._typing import Iterable, PathOrStr
from .model import Result

def save_ensemble_results(results: Iterable[Result], savefolder: PathOrStr, progressbar=True, leave_progressbar=True) -> None:
    """
    Save all Result DataFrames in `results` to files in a given `savefolder`.
    """
    # Get the total number of outputs if possible, for the progress bar
    # tqdm would normally do this internally, but we do it manually in case we want to use n somewhere else
    try:
        n = len(results)
    except TypeError:  # e.g. if `outputs` is a generator
        n = None

    # Loop over the outputs and save them to file
    for run in tqdm(results, total=n, desc="Saving output files", unit="files", disable=not progressbar, leave=leave_progressbar):
        run.to_file(savefolder)

def load_ensemble_results_single(filename: PathOrStr, index: str="day") -> pd.DataFrame:
    """
    Load a single ensemble results file.
    The resulting DataFrame will be assigned a run_id from its filename.
    """
    # Generate a run_id
    filename = Path(filename)
    run_id = filename.stem

    # Load the file
    results = pd.read_csv(filename).set_index(index)
    results.run_id = run_id

    return results

def load_ensemble_summary(filename: PathOrStr, index: str="run_id") -> pd.DataFrame:
    """
    Load a DataFrame containing summary results from a PCSE ensemble saved to a CSV file.
    """
    summary = pd.read_csv(filename).set_index(index)
    return summary

def load_ensemble_results_folder(folder: PathOrStr, index_results: str="day", index_summary: str="run_id", sample=False, progressbar=True, leave_progressbar=True) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Load all the output files in a given folder.
    The individual DataFrames will be assigned a run_id from their filenames.

    Also loads the ensemble summary file.

    If `sample` is True, only return the top 10 items (for testing purposes).

    To do: See if a dict {run_id: results} makes more sense. Or wrap everything into a single class?
    """
    # Get the filenames
    folder = Path(folder)
    filenames_results = [filename for filename in folder.glob("*.csv") if not filename.name.startswith("summary")]
    filename_summary = folder / "summary.csv"
    n_results = len(filenames_results)
    assert n_results > 0, f"No results files found in folder {folder.absolute()}"

    # Only return the top few items if `sample` is True
    if sample:
        filenames_results = filenames_results[:10]
        n_results = 10

    # Load the files
    summary = load_ensemble_summary(filename_summary, index=index_summary)
    results = [load_ensemble_results_single(filename, index=index_results) for filename in tqdm(filenames_results, total=n_results, desc="Loading results", unit="files", disable=not progressbar, leave=leave_progressbar)]

    return results, summary
