"""
Neural networks.
"""
import numpy as np
from tqdm import tqdm, trange

from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, Tensor, tensor
from torch.utils.data import DataLoader, Dataset

from fpcup.tools import RUNNING_IN_IPYTHON
from fpcup.typing import Callable, Optional

### DEFINE CONSTANTS


### SETUP
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


### HELPER FUNCTIONS
tqdm_batch = lambda dataloader, desc: tqdm(dataloader, desc=desc, unit="data", unit_scale=dataloader.batch_size, disable=RUNNING_IN_IPYTHON, leave=False)


### NEURAL NETWORK CLASSES
class PCSEEmulator(nn.Module):
    def __init__(self):
        # Basic initialisation
        super().__init__()

        # Network definition
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
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


def train(model: nn.Module, training_data: DataLoader, loss_function: Callable, optimizer: torch.optim.Optimizer, *,
          testing_data: Optional[DataLoader]=None, n_epochs: int=10) -> list[float]:
    """
    Train a given neural network `model` on data.
    n_epochs epochs (default: 10).
    """
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


def predict(model: nn.Module, X: np.ndarray, *,
            X_scaler: Optional[MinMaxScaler]=None, y_scaler: Optional[MinMaxScaler]=None) -> Tensor:
    """
    Use an existing model to predict y values for X values.
    """
    # Setup
    model.eval()  # Set to training mode

    # Rescale X if desired
    if X_scaler is not None:
        X = X_scaler.transform(X)

    X = tensor(X, device=device)

    # Predict
    with torch.no_grad():
        y = model(X)

    # Rescale y if desired
    if y_scaler is not None:
        y = y_scaler.inverse_transform(y)

    return y
