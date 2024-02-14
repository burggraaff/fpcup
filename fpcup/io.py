"""
Functions for file input and output.
"""
from functools import cache
from os import makedirs
from pathlib import Path

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
from geopandas import GeoDataFrame, read_file as read_gpd
from pyogrio.errors import DataSourceError

import pandas as pd

from tqdm import tqdm

from ._typing import Iterable, PathOrStr
from .model import Result, Summary

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

def load_ensemble_results_folder(folder: PathOrStr, sample=False, progressbar=True, leave_progressbar=True) -> tuple[list[Result], Summary]:
    """
    Load all the output files in a given folder.
    The individual Result DataFrames will be assigned a run_id from their filenames.

    Also loads the ensemble summary file.

    If `sample` is True, only return the top 10 items (for testing purposes).

    To do: See if a dict {run_id: results} makes more sense. Or wrap everything into a single class?
    """
    # Get the filenames
    folder = Path(folder)
    filenames_results = list(folder.glob("*.wout"))

    # Only return the top few items if `sample` is True
    if sample:
        filenames_results = filenames_results[:10]

    n_results = len(filenames_results)
    assert n_results > 0, f"No results files found in folder {folder.absolute()}"

    # Load the summary; from an ensemble file if possible
    try:
        summary = Summary.from_file(folder/"ensemble.wsum")
    except DataSourceError:
        summary = Summary.from_folder(folder, progressbar=progressbar, leave_progressbar=leave_progressbar)

    filenames_results = tqdm(filenames_results, total=n_results, desc="Loading results", unit="files", disable=not progressbar, leave=leave_progressbar)
    results = [Result.from_file(filename) for filename in filenames_results]

    return results, summary
