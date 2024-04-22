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

from ._typing import Iterable, Optional, PathOrStr
from .constants import CRS_AMERSFOORT
from .model import InputSummary, Result, Summary, _SUFFIX_RUNDATA, _SUFFIX_OUTPUTS, _SUFFIX_SUMMARY
from .multiprocessing import multiprocess_file_io

# Constants
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

    for summarytype, suffix in zip([InputSummary, Summary], [_SUFFIX_RUNDATA, _SUFFIX_SUMMARY]):
        # Load individual files
        summary = summarytype.from_folder(output_dir, use_existing=use_existing, leave_progressbar=verbose)

        # Save the modified ensemble summary to file
        summary_filename = output_dir / ("ensemble" + suffix)
        summary.to_file(summary_filename)

        print(f"Saved ensemble summary ({len(summary)} runs) to {summary_filename.absolute()}")


def _load_ensemble_summary_from_folder_single(folder: PathOrStr, summarytype: type, suffix: str, *,
                                      crs=CRS_AMERSFOORT, sample=False, save_if_generated=True, progressbar=True, leave_progressbar=True) -> InputSummary | Summary:
    """
    For a given folder, try to load the ensemble input/output summary files.
    """
    # Set up the folder and filename
    folder = Path(folder)
    ensemble_filename = folder / ("ensemble" + suffix)

    summary = summarytype.from_folder(folder)

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


def load_ensemble_summary_from_folder(folder: PathOrStr, *,
                                      sample=False, save_if_generated=True, **kwargs) -> tuple[InputSummary, Summary]:
    """
    For a given folder, try to load the ensemble input/output summary files.
    """
    inputsummary = _load_ensemble_summary_from_folder_single(folder, InputSummary, _SUFFIX_RUNDATA, sample=sample, save_if_generated=save_if_generated, **kwargs)
    summary = _load_ensemble_summary_from_folder_single(folder, Summary, _SUFFIX_SUMMARY, sample=sample, save_if_generated=save_if_generated, **kwargs)

    return inputsummary, summary


_load_ensemble_result_simple = partial(Result.from_file, run_id=None, include_summary=False)
def load_ensemble_results_from_folder(folder: PathOrStr, run_ids: Optional[Iterable[PathOrStr]]=None, *,
                                      extension=".wout", sample=False,
                                      progressbar=True, leave_progressbar=True) -> list[Result]:
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
