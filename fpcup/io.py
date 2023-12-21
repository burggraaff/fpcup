"""
Functions for file input and output.
"""
from pathlib import Path
from typing import Iterable

import pandas as pd

from tqdm import tqdm

def save_ensemble_results(results: Iterable[pd.DataFrame], savefolder: (Path, str)):
    """
    Save all DataFrames in `outputs` to files in a given `savefolder`.
    Individual outputs are saved with their run_id as the filename.

    To do: Create custom DataFrame class that always has a run id (and other properties).
    """
    # Get the total number of outputs if possible, for the progress bar
    # tqdm would normally do this internally, but we do it manually in case we want to use n somewhere else
    try:
        n = len(results)
    except TypeError:  # e.g. if `outputs` is a generator
        n = None

    # Loop over the outputs and save them to file
    for run in tqdm(results, total=n, desc="Saving output files", unit="files"):
        filename = savefolder / f"{run.run_id}.csv"
        run.to_csv(filename)

def save_ensemble_summary(summary: pd.DataFrame, saveto: (Path, str)):
    """
    Save a DataFrame containing summary results from a PCSE ensemble run to a CSV file.
    """
    summary.to_csv(saveto)

def load_ensemble_results_single(filename: (Path, str), index="day") -> pd.DataFrame:
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

def load_ensemble_summary(filename: (Path, str), index="run_id") -> pd.DataFrame:
    """
    Load a DataFrame containing summary results from a PCSE ensemble saved to a CSV file.
    """
    summary = pd.read_csv(filename).set_index(index)
    return summary

def load_ensemble_results_folder(folder: (Path, str), index_results="day", index_summary="run_id", sample=False) -> tuple[list[pd.DataFrame], pd.DataFrame]:
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
    results = [load_ensemble_results_single(filename, index=index_results) for filename in tqdm(filenames_results, total=n_results, desc="Loading results", unit="files")]

    return results, summary
