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
from .constants import CRS_AMERSFOORT
from .model import Result, Summary

_SAMPLE_LENGTH = 10

def save_ensemble_results(results: Iterable[Result], savefolder: PathOrStr, *,
                          progressbar=True, leave_progressbar=True) -> None:
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

def load_ensemble_summary_from_folder(folder: PathOrStr, *,
                                      crs=CRS_AMERSFOORT, sample=False, save_if_generated=True, progressbar=True, leave_progressbar=True) -> Summary:
    """
    For a given folder, try to load the ensemble summary file.
    If it is not available, load all individual summary files and combine them.
    """
    # Set up the folder and filename
    folder = Path(folder)
    ensemble_filename = folder/"ensemble.wsum"

    # Load the ensemble summary
    try:
        summary = Summary.from_file(ensemble_filename)

    # If the ensemble summary was not available, load individual files
    except DataSourceError:
        summary = Summary.from_folder(folder, progressbar=progressbar, leave_progressbar=leave_progressbar)

        # Save the ensemble to file if desired
        if save_if_generated and not sample:
            summary.to_file(ensemble_filename)

    # Subsample if desired
    if sample:
        summary = summary.head(_SAMPLE_LENGTH)

    # Convert to the desired CRS
    summary.to_crs(crs, inplace=True)

    # Sort by plot ID if available (BRP)
    if "plot_id" in summary.columns:
        summary.sort_values("plot_id", inplace=True)

    return summary

def load_ensemble_results_from_folder(folder: PathOrStr, *,
                                      extension=".wout", sample=False, progressbar=True, leave_progressbar=True) -> list[Result]:
    """
    Load all the output files in a given folder.
    The individual Result DataFrames will be assigned a run_id from their filenames.

    If `sample` is True, only return the top _SAMPLE_LENGTH (10) items (for testing purposes).

    To do: See if a dict {run_id: results} makes more sense. Or wrap everything into a single class?
    """
    # Get the filenames
    folder = Path(folder)
    filenames_results = list(folder.glob("*"+extension))

    # Only return the top few items if `sample` is True
    if sample:
        filenames_results = filenames_results[:_SAMPLE_LENGTH]

    n_results = len(filenames_results)
    assert n_results > 0, f"No results ({extension}) files found in folder {folder.absolute()}"

    # Load the summary; from an ensemble file if possible
    filenames_results = tqdm(filenames_results, total=n_results, desc="Loading results", unit="files", disable=not progressbar, leave=leave_progressbar)
    results = [Result.from_file(filename) for filename in filenames_results]

    return results
