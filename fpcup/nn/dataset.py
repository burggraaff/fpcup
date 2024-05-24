"""
Data loaders for PCSE inputs and outputs.
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
from torch import Tensor, tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..io import load_combined_ensemble_summary
from ..model import InputSummary, Summary
from ..typing import Optional, PathOrStr


### CONSTANTS
# Temporary: keep it simple
CROP = "barley"
VARIETY = "Spring_barley_301"
SOILTYPE = "ec3"
pattern = f"*_{SOILTYPE}_B*"
pattern_suffix = pattern + ".wsum"


### TRANSFORM PCSE INPUTS/OUTPUTS TO NN SPECIFICATIONS
INPUTS = ["latitude", "longitude", "WAV", "RDMSOL", "sowyear", "sowdoy"]
OUTPUTS = ["DVS", "LAIMAX", "TAGP", "TWSO", "matyear", "matdoy"]
# Preprocess:
    # DOS -> year, doy

    # soiltype -> ???
    # crop -> ???
    # variety -> ???

def get_year(date) -> int:
    return date.year

def get_doy(date) -> int:
    return date.day_of_year

def add_input_columns(summary: pd.DataFrame) -> None:
    """
    Convert PCSE inputs (from rundata) to neural network variables.
    """
    summary["sowyear"] = summary["DOS"].apply(get_year)
    summary["sowdoy"] = summary["DOS"].apply(get_doy)

# Postprocess:
    # DOM -> lifetime

def add_output_columns(summary: pd.DataFrame) -> None:
    """
    Convert PCSE outputs (from summary) to neural network variables.
    """
    summary["matyear"] = summary["DOM"].apply(get_year)
    summary["matdoy"] = summary["DOM"].apply(get_doy)


### SAMPLING FUNCTIONS
def mcar(X: pd.DataFrame, y: pd.DataFrame, *,
         frac_test: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Missing Completely At Random.
    Temporarily copied over from imputation project.
    """
    X_train = X.sample(frac=1-frac_test)
    X_test = X.drop(index=X_train.index)
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    return X_train, X_test, y_train, y_test


### DATASET CLASSES
class PCSEEnsembleDatasetSmall(Dataset):
    """
    Handles the loading of PCSE ensemble input/output files that fit into a single file each.
    Useful for relatively small datasets.
    """
    ### Mandatory functions
    def __init__(self, data_dir: PathOrStr, *, transform=None, target_transform=None, **kwargs):
        # Basic setup
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # Load data
        self.summary = load_combined_ensemble_summary(data_dir, pattern=pattern, save_if_generated=False, **kwargs)
        add_input_columns(self.summary)
        add_output_columns(self.summary)
        self.summary = self.summary[INPUTS + OUTPUTS]

    def __len__(self) -> int:
        return len(self.summary)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        row = self.summary.iloc[i]
        input_data = row[INPUTS].to_list()
        summary_data = row[OUTPUTS].to_list()

        return tensor(input_data, dtype=torch.float32), tensor(summary_data, dtype=torch.float32)


    ### Output
    def __repr__(self) -> str:
        example_input, example_output = self[0]
        return f"{self.__class__.__name__}: length {len(self)}, input length {len(example_input)}, output length {len(example_output)}"


class PCSEEnsembleDataset(Dataset):
    """
    Handles the loading of PCSE ensemble input/output files in bulk.
    Useful for datasets that are too big to load into memory at once.

    Note: Current approach works for medium-sized data sets; bigger data sets will require larger changes in multiple places.

    To do:
        Use transforms for loading soil/crop data?
        Multiple data directories?
    """
    ### Mandatory functions
    def __init__(self, data_dir: PathOrStr, *, transform=None, target_transform=None):
        # Basic setup
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # Get summary filenames - makes sure only completed runs are loaded
        self.summary_files = list(self.data_dir.glob(pattern_suffix))
        self.input_files = [f.with_suffix(".wrun") for f in self.summary_files]

    def __len__(self) -> int:
        return len(self.summary_files)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        # Load input data
        input_filename = self.input_files[i]
        input_data = InputSummary.from_file(input_filename).iloc[0]
        sowyear = get_year(input_data["DOS"])
        sowdoy = get_doy(input_data["DOS"])

        input_data = input_data[["latitude", "longitude", "WAV", "RDMSOL"]].to_list() + [sowyear, sowdoy]

        # Load summary data
        summary_filename = self.summary_files[i]
        summary_data = Summary.from_file(summary_filename).iloc[0]

        matyear = get_year(summary_data["DOM"])
        matdoy = get_doy(summary_data["DOM"])

        summary_data = summary_data[["DVS", "LAIMAX", "TAGP", "TWSO"]].to_list() + [matyear, matdoy]

        return tensor(input_data, dtype=torch.float32), tensor(summary_data, dtype=torch.float32)


    ### Output
    def __repr__(self) -> str:
        example_input, example_output = self[0]
        return f"{self.__class__.__name__}: length {len(self)}, input length {len(example_input)}, output length {len(example_output)}"


### DATA I/O
def load_pcse_dataset(data_dir: PathOrStr, *,
                      frac_test: float=0.2,
                      **kwargs) -> tuple[DataLoader, DataLoader]:
    """
    From a given folder, load all PCSE input and output files, collate them, pre-process them, split them into training and testing, and re-scale them.
    """
    # Load data
    summary = load_combined_ensemble_summary(data_dir, pattern=pattern, save_if_generated=False, **kwargs)

    # Add columns
    add_input_columns(summary)
    add_output_columns(summary)

    # Split into inputs / outputs
    X, y = summary[INPUTS], summary[OUTPUTS]

    # Split into train/test
    X_train, X_test, y_train, y_test = mcar(X, y, frac_test=frac_test)

    # Convert to numpy float32
    X_train, X_test, y_train, y_test = [df.to_numpy(dtype=np.float32) for df in (X_train, X_test, y_train, y_test)]

    # Train and apply scalers
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Combine into Datasets
    data_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    data_test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return data_train, data_test, X_scaler, y_scaler


def outputs_to_dataframe(y: np.ndarray | Tensor, *, y_scaler: Optional[MinMaxScaler]=None) -> pd.DataFrame:
    """
    Take a tensor/array of outputs `y`, optionally rescale it, and convert it to a pandas DataFrame.
    """
    # Convert to numpy
    if isinstance(y, Tensor):
        y = y.numpy()

    # Rescale y if desired
    if y_scaler is not None:
        y = y_scaler.inverse_transform(y)

    # Add to a dataframe
    y = pd.DataFrame(data=y, columns=OUTPUTS)

    return y
