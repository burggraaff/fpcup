"""
Neural networks.
"""
import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .dataset import INPUTS, OUTPUTS
from ..tools import RUNNING_IN_IPYTHON
from ..typing import Callable, Optional

### DEFINE CONSTANTS
N_in = len(INPUTS)
N_out = len(OUTPUTS)

### SETUP
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
default_lossfunction = nn.L1Loss()


### HELPER FUNCTIONS
tqdm_batch = lambda dataloader, desc: tqdm(dataloader, desc=desc, unit="data", unit_scale=dataloader.batch_size, disable=RUNNING_IN_IPYTHON, leave=False)


### NEURAL NETWORK CLASSES
class PCSEEmulator(nn.Module):
    def __init__(self):
        # Basic initialisation
        super().__init__()

        # Network definition
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_in, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, 16),
            nn.ReLU(),
            nn.Linear(16, N_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear_relu_stack(x)
        return logits


### TRAINING / TESTING
def train_batch(model: nn.Module, loss_function: Callable, optimizer: torch.optim.Optimizer, X: Tensor, y: Tensor) -> float:
    """
    Train a given neural network `model` on data.
    One batch.
    """
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_function(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def test_batch(model: nn.Module, loss_function: Callable, optimizer: torch.optim.Optimizer, X: Tensor, y: Tensor) -> float:
    """
    Test a given neural network `model` on data.
    One batch.
    """
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_function(pred, y)

    return loss.item()


def predict_batch(model: nn.Module, X: Tensor) -> np.ndarray:
    """
    Use an existing model to predict y values for X values.
    One batch.
    """
    # Setup
    X = X.to(device)
    model.eval()  # Set to testing mode

    # Predict
    with torch.no_grad():
        pred = model(X)

    # Convert to Numpy
    pred = pred.numpy()

    return pred


def train_epoch(model: nn.Module, dataloader: DataLoader, loss_function: Callable, optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Train a given neural network `model` on data.
    One epoch.
    """
    # Setup
    model.train()  # Set to training mode

    # Loop over batches
    loss_per_batch = [train_batch(model, loss_function, optimizer, X, y) for (X, y) in tqdm_batch(dataloader, "Training")]
    loss_per_batch = np.array(loss_per_batch)

    return loss_per_batch


def test_epoch(model: nn.Module, dataloader: DataLoader, loss_function: Callable, optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Train a given neural network `model` on data.
    One epoch.
    """
    # Setup
    model.eval()  # Set to training mode

    # Loop over batches
    loss_per_batch = [test_batch(model, loss_function, optimizer, X, y) for (X, y) in tqdm_batch(dataloader, "Testing")]
    loss = np.mean(loss_per_batch)

    return loss


def train(model: nn.Module, training_data: DataLoader, *,
          loss_function: Callable=default_lossfunction, optimizer: Optional[torch.optim.Optimizer]=None,
          testing_data: Optional[DataLoader]=None, n_epochs: int=10) -> list[float]:
    """
    Train a given neural network `model` on data.
    n_epochs epochs (default: 10).
    """
    # Set up optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_train_epoch = []
    loss_test_epoch = []

    for i in trange(n_epochs, desc="Training", unit="epoch"):
        # Train
        loss_train = train_epoch(model, training_data, loss_function, optimizer)
        loss_train_epoch.append(loss_train)

        # Test
        if testing_data is not None:
            loss_test = test_epoch(model, testing_data, loss_function, optimizer)
        else:
            loss_test = np.nan
        loss_test_epoch.append(loss_test)

    loss_train_epoch = np.array(loss_train_epoch)
    loss_test_epoch = np.array(loss_test_epoch)

    return loss_train_epoch, loss_test_epoch


def predict(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    """
    Use an existing model to predict y values for X values.
    Goes through the full dataloader.
    """
    # Loop over batches
    pred = [predict_batch(model, X) for (X, y) in tqdm_batch(dataloader, "Predicting")]
    pred = np.concatenate(pred)

    return pred
