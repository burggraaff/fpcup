"""
Neural networks.
"""
import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn, Tensor, tensor
from torch.utils.data import DataLoader, Dataset

from fpcup.tools import RUNNING_IN_IPYTHON
from fpcup.typing import Callable, Optional

### DEFINE CONSTANTS


### SETUP
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


### NEURAL NETWORK CLASSES
class PCSEEmulator(nn.Module):
    def __init__(self):
        # Basic initialisation
        super().__init__()

        # Network definition
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
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
    loss_per_batch = [train_batch(model, loss_function, optimizer, X, y) for (X, y) in tqdm(dataloader, desc="Training", unit="data", unit_scale=dataloader.batch_size, disable=RUNNING_IN_IPYTHON, leave=False)]
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
    loss_per_batch = [test_batch(model, loss_function, optimizer, X, y) for (X, y) in tqdm(dataloader, desc="Testing", unit="data", unit_scale=dataloader.batch_size, disable=RUNNING_IN_IPYTHON, leave=False)]
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
