"""
Functions for file input and output.
"""
from functools import cache, partial
from os import makedirs
from pathlib import Path

import geopandas as gpd
gpd.options.io_engine = "pyogrio"
from geopandas import GeoDataFrame

from pyogrio import read_dataframe as read_geodataframe, write_dataframe as write_geodataframe
from pyogrio.errors import DataSourceError

from tqdm import tqdm

from .constants import CRS_AMERSFOORT
from .model import InputSummary, Summary, TimeSeries
from .multiprocessing import multiprocess_file_io
from .typing import Iterable, Optional, PathOrStr

# Constants
_SAMPLE_LENGTH = 10
_SUMMARY_TYPES = (InputSummary, Summary)

def save_ensemble_summary(output_dir: PathOrStr, *,
                          use_existing=True, verbose=False) -> None:
    """
    Save an ensemble summary by loading the individual summary files.
    If desired (True by default), append to an existing file.
    Also prints some output to the user.
    """
    if verbose:
        print()

    # Generate the ensemble summaries
    output_dir = Path(output_dir)

    for summarytype in _SUMMARY_TYPES:
        # Load individual files
        summary = summarytype.from_folder(output_dir, use_existing=use_existing, leave_progressbar=verbose)

        # Save the modified ensemble summary to file
        summary_filename = output_dir / ("ensemble" + summarytype.suffix)
        summary.to_file(summary_filename)

        print(f"Saved ensemble summary ({len(summary)} runs) to {summary_filename.absolute()}")


def load_ensemble_summary_from_folder(folder: PathOrStr, *, sample=False, save_if_generated=True, **kwargs) -> tuple[InputSummary, Summary]:
    """
    For a given folder, try to load the ensemble input/output summary files.
    """
    # Load data
    summaries = [summarytype.from_folder(folder, **kwargs) for summarytype in _SUMMARY_TYPES]

    # Adjust if desired
    if sample:
        summaries = [s.head(_SAMPLE_LENGTH) for s in summaries]

    # Save to file if desired
    if save_if_generated and not sample:
        for s in summaries:
            ensemble_filename = folder / ("ensemble" + s.suffix)
            s.to_file(ensemble_filename)

    return summaries


def load_combined_ensemble_summary(folder: PathOrStr, *, sample=False, **kwargs) -> Summary:
    """
    For a given folder, load the ensemble input/output summary files and join them.
    """
    # Load data
    inputsummary, summary = load_ensemble_summary_from_folder(folder, sample=sample, **kwargs)

    # Drop known duplicate columns
    summary.drop(columns=["latitude", "longitude", "DOS"], inplace=True)
    summary = inputsummary.join(summary)

    return summary


_load_ensemble_result_simple = partial(TimeSeries.from_file, run_id=None, include_summary=False)
def load_ensemble_results_from_folder(folder: PathOrStr, run_ids: Optional[Iterable[PathOrStr]]=None, *,
                                      extension=".wout", sample=False,
                                      progressbar=True, leave_progressbar=True) -> list[TimeSeries]:
    """
    Load the result files in a given folder.
    By default, load all files in the folder. If `run_ids` is specified, load only those files.
    The individual Result DataFrames will be assigned a run_id from their filenames.

    If `sample` is True, only return the top _SAMPLE_LENGTH (10) items (for testing purposes).

    To do: See if a dict {run_id: results} makes more sense. Or wrap everything into a single EnsembleResult class?
    """
    # Get the filenames
    folder = Path(folder)

    # If filenames were not specified, load everything
    if run_ids is None:
        filenames = list(folder.glob(f"*{extension}"))
    # If filenames were specified, load only those
    else:
        filenames = [folder/f"{run_id}{extension}" for run_id in run_ids]

    # Only return the top few items if `sample` is True
    if sample:
        filenames = filenames[:_SAMPLE_LENGTH]

    n_results = len(filenames)
    assert n_results > 0, f"No results ({extension}) files found in folder {folder.absolute()}"

    # Load the files with an optional progressbar
    results = multiprocess_file_io(_load_ensemble_result_simple, filenames, n=n_results, progressbar=progressbar, leave_progressbar=leave_progressbar, desc="Loading PCSE outputs")

    return results
