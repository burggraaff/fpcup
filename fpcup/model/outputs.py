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
SUFFIX_TIMESERIES = ".wout"
SUFFIX_SUMMARY = ".wsum"


### SUMMARY CLASSES
class GeneralSummary(gpd.GeoDataFrame):
    """
    General class for Summary-like objects.
    Subclassed for inputs and outputs.
    """
    suffix_default = None

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
            extension = "*" + cls.suffix_default

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


class Summary(GeneralSummary):
    """
    Stores a summary of the results from a PCSE ensemble run.
    """
    suffix_default = SUFFIX_SUMMARY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index.set_names("run_id", inplace=True)
        self.sort_index(inplace=True)

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData, **kwargs):
        """
        Generate a summary from a finished model, inserting the run_id as an index.
        """
        data = model.get_summary_output()

        # Get data from run_data
        data[0] = {**run_data.summary_dict(), **data[0]}
        index = [run_data.run_id]

        return cls(data, index=index, **kwargs)


# TIME SERIES CLASSES
class TimeSeries(pd.DataFrame):
    """
    Stores the results from a single PCSE run; essentially a DataFrame with some small additions.
    """
    def __init__(self, data: Iterable, **kwargs):
        # Initialise the main DataFrame from the model output
        super().__init__(data, **kwargs)

        # Sort the results by time
        self.set_index("day", inplace=True)

    @classmethod
    def from_model(cls, model: Engine, **kwargs):
        """
        Initialise the main DataFrame from a model output.
        """
        output = model.get_output()
        return cls(output, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, **kwargs):
        """
        Load an output file.
        """
        data = pd.read_csv(filename)
        return cls(data, **kwargs)

    def to_file(self, *args, **kwargs) -> None:
        """
        Save the time series to a CSV file.
        """
        self.to_csv(*args, **kwargs)


### COMBINED OUTPUT CLASS
class Output:
    """
    Simple class that holds the TimeSeries and Summary for a PCSE run.
    Ensures that files get saved together.
    """
    ### OBJECT GENERATION AND INITIALISATION
    def __init__(self, run_id: str, summary: Summary, timeseries: TimeSeries):
        self.run_id = run_id
        self.summary = summary
        self.timeseries = timeseries

    @classmethod
    def from_model(cls, model: Engine, run_data: RunData):
        run_id = run_data.run_id
        summary = Summary.from_model(model, run_data=run_data)
        timeseries = TimeSeries.from_model(model)
        return cls(run_id, summary, timeseries)


    ### STRING REPRESENTATIONS
    def __repr__(self) -> str:
        return f"Finished run '{self.run_id}'"

    def __str__(self) -> str:
        return "\n".join([repr(self), str(self.summary), str(self.timeseries)])


    ### FILE INPUT / OUTPUT
    def to_file(self, output_dir: PathOrStr) -> None:
        # Generate the output filename
        output_dir = Path(output_dir)
        filename_summary = output_dir / (self.run_id + SUFFIX_SUMMARY)
        filename_timeseries = output_dir / (self.run_id + SUFFIX_TIMESERIES)

        # Write to file
        self.summary.to_file(filename_summary)
        self.timeseries.to_file(filename_timeseries)
