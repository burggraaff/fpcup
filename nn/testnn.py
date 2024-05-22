"""
Train a neural network on PCSE inputs/outputs.

Example:
    %run nn/testnn.py outputs/RDMSOL -v
"""
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from torch import nn, optim, tensor, Tensor
from torch.utils.data import DataLoader, Dataset, random_split

import fpcup
from fpcup.typing import PathOrStr

from dataset import PCSEEnsembleDataset, PCSEEnsembleDatasetSmall
from network import PCSEEmulator, device, train

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Train a neural network on PCSE inputs/outputs.")
parser.add_argument("output_dir", help="folder to load data (PCSE outputs) from", type=fpcup.io.Path)
parser.add_argument("-t", "--test_fraction", help="number of data (PCSE outputs) to reserve for testing", type=float, default=0.2)
parser.add_argument("-n", "--number_epochs", help="number of training epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=64)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--results_dir", help="folder to save plots into", type=fpcup.io.Path, default=fpcup.DEFAULT_RESULTS/"nn")
args = parser.parse_args()


### Constants
tag = args.output_dir.stem
lossfunc = nn.L1Loss()


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()

    ### SETUP
    # Load data
    dataset = PCSEEnsembleDatasetSmall(args.output_dir)
    if args.verbose:
        print("Loaded data set:")
        print(dataset)

    # Training / Testing data split
    fractions = [1-args.test_fraction, args.test_fraction]
    training_dataset, testing_dataset = random_split(dataset, fractions)
    training_data = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    testing_data = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=False)
    if args.verbose:
        print(f"Split data into training ({fractions[0]:.0%}) and testing ({fractions[1]:.0%}).")
        print(f"Batch size: {args.batch_size}")

    # Network
    model = PCSEEmulator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    ### TRAINING
    losses_train, losses_test = train(model, training_data, lossfunc, optimizer, testing_data=testing_data, n_epochs=args.number_epochs)


    ### PLOT
    fpcup.plotting.plot_loss_curve(losses_train, losses_test=losses_test, title=tag, saveto=f"nn_loss_{tag}.pdf")
