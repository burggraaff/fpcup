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

### Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description="Train a neural network on PCSE inputs/outputs.")
parser.add_argument("output_dir", help="folder to load data (PCSE outputs) from", type=fpcup.io.Path)
parser.add_argument("-t", "--test_fraction", help="number of data (PCSE outputs) to reserve for testing", type=float, default=0.2)
parser.add_argument("-n", "--number_epochs", help="number of training epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=64)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--results_dir", help="folder to save plots into", type=fpcup.io.Path, default=fpcup.DEFAULT_RESULTS/"nn")
args = parser.parse_args()


### Constants
tag = args.output_dir.stem

# Crop information
CROP_NAME = args.output_dir.stem.split("_")[-1]
CROP = fpcup.crop.select_crop(CROP_NAME)
SOILTYPE = "ec3"
pattern = f"*_{SOILTYPE}_{CROP.abbreviation}*"


### This gets executed only when the script is run normally; not by multiprocessing.
if __name__ == "__main__":
    fpcup.multiprocessing.freeze_support()

    ### SETUP
    # Load data
    summary_train, summary_test = fpcup.nn.dataset.load_pcse_summaries(args.output_dir, pattern=pattern, frac_test=args.test_fraction, leave_progressbar=args.verbose)

    training_dataset, testing_dataset, X_scaler, y_scaler = fpcup.nn.dataset.summaries_to_datasets(summary_train, summary_test)
    training_data = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    testing_data = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=False)

    if args.verbose:
        print(f"Split data into training ({1 - args.test_fraction:.0%}) and testing ({args.test_fraction:.0%}).")
        print(f"Batch size: {args.batch_size}")

    # Network
    model = fpcup.nn.network.PCSEEmulator().to(fpcup.nn.network.device)
    if args.verbose:
        print("Created model:")
        print(model)


    ### TRAINING
    losses_train, losses_test = fpcup.nn.network.train(model, training_data, testing_data=testing_data, n_epochs=args.number_epochs)


    ### PLOT LOSS CURVES
    saveto_loss = args.results_dir / f"{tag}-losscurve.pdf"
    fpcup.plotting.plot_loss_curve(losses_train, losses_test=losses_test, title=tag, saveto=saveto_loss)
    if args.verbose:
        print(f"Saved loss plot to {saveto_loss}")


    ### PERFORMANCE ASSESSMENT
    # Convert outputs to DataFrames
    y = fpcup.nn.dataset.outputs_to_dataframe(testing_dataset.tensors[1], y_scaler=y_scaler)
    pred = fpcup.nn.network.predict(model, testing_data)
    pred = fpcup.nn.dataset.outputs_to_dataframe(pred, y_scaler=y_scaler)

    # Calculate performance metrics
    metrics = fpcup.stats.compare_predictions(y, pred)

    # Scatter plot
    saveto_scatter = args.results_dir / f"{tag}-performance_scatter.pdf"
    fpcup.plotting.nn_scatter(y, pred, metrics=metrics, title=tag, saveto=saveto_scatter)
    if args.verbose:
        print(f"Saved scatter plot to {saveto_scatter}")

    # Histogram plot
    saveto_hist = args.results_dir / f"{tag}-performance_hist.pdf"
    fpcup.plotting.nn_histogram(y, pred, title=tag, saveto=saveto_hist)
    if args.verbose:
        print(f"Saved scatter plot to {saveto_hist}")
