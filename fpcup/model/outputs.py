"""
Everything related to Summary/Result classes that encapsulate WOFOST/PCSE output data.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe as read_geodataframe, write_dataframe as write_geodataframe

from pcse.models import Engine

from .rundata import SUFFIX_RUNDATA, RunData
from ..multiprocessing import multiprocess_file_io
from ..typing import Iterable, Optional, PathOrStr

### CONSTANTS
SUFFIX_OUTPUTS = ".wout"
SUFFIX_SUMMARY = ".wsum"


### SUMMARY CLASSES
class GeneralSummary(gpd.GeoDataFrame):
    """
    General class for Summary-like objects.
    Subclassed for inputs and outputs.
    """
    _suffix_default = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index.set_names("run_id", inplace=True)
        self.sort_index(inplace=True)

    @classmethod
    def from_file(cls, filename: PathOrStr):
        """
        Load a summary from a GeoJSON (.wrun / .wsum) file.
        """
        data = read_geodataframe(filename)
        data.set_index("run_id", inplace=True)
        return cls(data)

    @classmethod
    def from_ensemble(cls, summaries_individual: Iterable):
        """
        Combine many Summary objects into one through concatenation.
        """
        # Uses the regular Pandas concat function; when the outputs are GeoDataFrames (or Summaries), the result is a GeoDataFrame
        data = pd.concat(summaries_individual)
        return cls(data)

    @classmethod
    def from_folder(cls, folder: PathOrStr, extension: Optional[str]=None, *,
                    use_existing=True, progressbar=True, leave_progressbar=True):
        """
        Load an ensemble of Summary objects from a folder and combine them.
        """
        if extension is None:
            extension = "*" + cls._suffix_default

        # Find all summary files in the folder, except a previously existing ensemble one (if it exists)
        folder = Path(folder)
        filenames = list(folder.glob(extension))
        assert len(filenames) > 0, f"No files with extension '{extension}' were found in folder {folder.absolute()}"
        filename_ensemble = filenames[0].with_stem("ensemble")
        ENSEMBLE_EXISTS = (filename_ensemble in filenames)

        # If there is an existing summary file, use that and append the new files to it
        if use_existing and ENSEMBLE_EXISTS:
            ensemble = cls.from_file(filename_ensemble)  # Note that the ensemble may get loaded twice
            filenames = [f for f in filenames if f.stem not in ensemble.index]

            # If all files are in the ensemble already, simply return that
            if len(filenames) == 1:
                return ensemble

        # If desired, do not load the ensemble, but use all individual summaries instead
        elif not use_existing and ENSEMBLE_EXISTS:
            filenames.remove(filename_ensemble)

        # Load the files (with a tqdm progressbar if desired)
        summaries_individual = multiprocess_file_io(cls.from_file, filenames, progressbar=progressbar, leave_progressbar=leave_progressbar, desc="Loading summaries")

        return cls.from_ensemble(summaries_individual)

    def to_file(self, filename: PathOrStr, **kwargs) -> None:
        """
        Save to a GeoJSON file.
        The index has to be reset first to ensure it is written to file (annoying 'feature' of the GeoJSON driver).
        """
        self.reset_index(inplace=True)
        write_geodataframe(self, filename, driver="GeoJSON", **kwargs)
        self.set_index("run_id", inplace=True)


class InputSummary(GeneralSummary):
    """
    Store a summary of the inputs for a PCSE ensemble run.
    """
    _suffix_default = SUFFIX_RUNDATA

    @classmethod
    def from_rundata(cls, run_data: RunData):
        """
        Generate an Inputs objects from a RunData object.
        """
        return cls(index=[run_data.run_id], data=run_data.input_dict(), geometry=[run_data.geometry], crs=run_data.crs)


class Summary(GeneralSummary):
    """
    Stores a summary of the results from a PCSE ensemble run.
    """
    _suffix_default = SUFFIX_SUMMARY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index.set_names("run_id", inplace=True)
        self.sort_index(inplace=True)

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData, *,
                   crs: Optional[str]=None, **kwargs):
        """
        Generate a summary from a finished model, inserting the run_id as an index.
        """
        data = model.get_summary_output()

        # Get data from run_data
        data[0] = {**run_data.summary_dict(), **data[0]}
        index = [run_data.run_id]
        if crs is None:
            crs = run_data.crs

        return cls(data, index=index, crs=crs, **kwargs)


# RESULT (TIME SERIES) CLASSES
class Result(pd.DataFrame):
    """
    Stores the results from a single PCSE run.
    Essentially a DataFrame that is initialised from a PCSE model object and contains some useful additional variables.
    """
    _internal_names = pd.DataFrame._internal_names + ["run_id", "summary"]
    _internal_names_set = set(_internal_names)

    def __init__(self, data: Iterable, *,
                 run_id: str="", summary: Optional[Summary]=None, **kwargs):
        # Initialise the main DataFrame from the model output
        super().__init__(data, **kwargs)

        # Sort the results by time
        self.set_index("day", inplace=True)

        # Add the run ID and summary
        self.run_id = run_id
        self.summary = summary

    def __repr__(self) -> str:
        return ("-----\n"
                f"Run ID: {self.run_id}\n\n"
                f"Summary: {self.summary}\n\n"
                f"Data:\n{super().__repr__()}"
                "\n-----")

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData, **kwargs):
        """
        Initialise the main DataFrame from a model output.
        """
        output = model.get_output()

        # Get data from run_data if possible
        run_id = run_data.run_id

        # Save the summary output
        try:
            summary = Summary.from_model(model, run_data=run_data)
        except IndexError:
            summary = None

        return cls(output, run_id=run_id, summary=summary, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, *,
                  run_id: Optional[str]=None, include_summary=False, **kwargs):
        """
        Load an output file.
        If a run_id is not provided, use the filename stem.
        """
        filename = Path(filename)
        if run_id is None:
            run_id = filename.stem

        # Load the main data file
        data = pd.read_csv(filename)

        # Try to load the associated summary file
        if include_summary:
            summary_filename = filename.with_suffix(SUFFIX_SUMMARY)
            try:
                summary = Summary.from_file(summary_filename)
            except FileNotFoundError:
                summary = None
        else:
            summary = None

        return cls(data, run_id=run_id, summary=summary, **kwargs)

    def to_file(self, output_directory: PathOrStr, *,
                filename: Optional[PathOrStr]=None, **kwargs) -> None:
        """
        Save the results and summary to output files:
            output_directory / filename.wout - full results, CSV
            output_directory / filename.wsum - summary results, GeoJSON

        If no filename is provided, default to using the run ID.
        """
        output_directory = Path(output_directory)

        # Generate the output filenames from the user input (`filename`) or from the run id (default)
        # The suffix step cannot be done with just .with_suffix(".wout") in case the filename contains .
        if filename is not None:
            filename_base = output_directory / filename
        else:
            filename_base = output_directory / self.run_id
        filename_results = filename_base.with_suffix(filename_base.suffix + SUFFIX_OUTPUTS)
        filename_summary = filename_base.with_suffix(filename_base.suffix + SUFFIX_SUMMARY)

        # Save the outputs
        self.to_csv(filename_results, **kwargs)
        self.summary.to_file(filename_summary, **kwargs)
