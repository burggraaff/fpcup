"""
Functions for plotting data/results relating to the NN emulator for WOFOST (experimental).
"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, patheffects as mpe

from .common import symmetric_lims
from ..typing import Optional, PathOrStr


### TRAINING
def weighted_mean_loss(loss_per_batch: np.ndarray) -> np.ndarray:
    """
    Return the weighted mean loss per epoch, weighted with a sawtooth.
    """
    # Check dimensionality
    INPUT_IS_1D = (loss_per_batch.ndim == 1)
    if INPUT_IS_1D:
        loss_per_batch = loss_per_batch[np.newaxis, :]

    # Generate sawtooth
    n_batches = loss_per_batch.shape[1]
    sawtooth = np.arange(n_batches) + 1

    # Weighted mean
    loss_per_epoch = np.average(loss_per_batch, weights=sawtooth, axis=1)

    # Return 1D if the input was 1D
    if INPUT_IS_1D:
        loss_per_epoch = loss_per_epoch[0]

    return loss_per_epoch


c_train = "#4477AA"
c_test = "#CCBB44"
pe_epoch = [mpe.Stroke(linewidth=4, foreground="black"),
            mpe.Normal()]
def plot_loss_curve(losses_train: np.ndarray, *, losses_test: Optional[np.ndarray]=None,
                    title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Plot the loss curve per batch and per epoch.
    """
    # Constants
    n_epochs, n_batches = losses_train.shape
    epochs = np.arange(n_epochs + 1)
    batches = np.arange(losses_train.size) + 1

    # Training data: get loss per batch and per epoch
    loss_initial = [losses_train[0, 0]]
    losses_train_epoch = weighted_mean_loss(losses_train)
    losses_train_epoch = np.concatenate([loss_initial, losses_train_epoch])

    losses_train_batch = losses_train.ravel()

    # Testing data: dummy loss at epoch 0
    losses_test = np.insert(losses_test, 0, np.nan)

    # Variables for limits etc.
    if losses_test is not None:
        maxloss = np.nanmax([np.nanmax(losses_train), np.nanmax(losses_test)])
        minloss = np.nanmin([np.nanmin(losses_train), np.nanmin(losses_test)])
    else:
        maxloss = np.nanmax(losses_train)
        minloss = np.nanmin(losses_train)

    # Figure setup
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), layout="constrained")

    # Plot training loss per batch
    ax.plot(batches, losses_train_batch, color=c_train, zorder=0)

    ax.set_xlim(0, len(batches))
    ax.set_xlabel("Batch", color=c_train)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_ylim(minloss/1.05, maxloss*1.05)
    ax.grid(True, axis="y", ls="--")
    ax.grid(False, axis="x")

    # Plot training/testing loss per epoch
    ax2 = ax.twiny()
    ax2.plot(epochs, losses_train_epoch, color=c_train, path_effects=pe_epoch, label="Train", zorder=1)
    ax2.plot(epochs, losses_test, color=c_test, path_effects=pe_epoch, label="Test", zorder=1)

    ax2.set_xlim(0, n_epochs)
    ax2.set_xlabel("Epoch")
    ax2.grid(True, ls="--")
    ax2.legend(loc="best")

    # Switch x axes
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Final settings
    fig.suptitle(title)

    # Save and close
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    plt.close()



### ASSESSMENT
def nn_scatter(y: pd.DataFrame, pred: pd.DataFrame, *,
               metrics: Optional[pd.DataFrame]=None,
               title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Generate scatter plots of NN predictions vs the true values.
    Optionally include a textbox with pre-calculated performance metrics.
    """
    # Setup
    fig, axs = plt.subplots(nrows=1, ncols=len(y.columns), figsize=(15, 5), layout="constrained")

    # Plot individual parameters
    for ax, col in zip(axs, y.columns):
        ax.scatter(y[col], pred[col], s=4, color="black", alpha=0.5, rasterized=True, zorder=1)
        # ax.hexbin(y[col], pred[col], gridsize=25, mincnt=1, cmap="cividis")

    # Grid
    for ax in axs:
        ax.axline((0, 0), slope=1, color="0.5", zorder=2)
        ax.grid(True, color="0.5", linestyle="--", zorder=2)
        ax.axis("equal")

    # Metrics
    if metrics is not None:
        for ax, col in zip(axs, y.columns):
            metrics_col = metrics[col]
            # Format text
            metrics_text = "\n".join([rf"$R^2 = {metrics_col.loc['R²']:.2f}$",
                                      rf"MD = ${metrics_col.loc['MD']:+.1f}$",
                                      rf"MAD = ${metrics_col.loc['MAD']:.1f}$",
                                      rf"rMAD = {metrics_col.loc['relativeMAD']:.1%}",
                                      ])

            ax.text(0.95, 0.03, metrics_text, transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom", color="black", size=9, bbox={"facecolor": "white", "edgecolor": "black"})

    # Labels, lims
    for ax, col in zip(axs, y.columns):
        ax.set_title(col)

        vmin = np.nanmin([y[col].min(), pred[col].min()])
        vmax = np.nanmax([y[col].max(), pred[col].max()])
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)

    fig.supxlabel("WOFOST value", fontweight="bold")
    fig.supylabel("NN prediction", fontweight="bold")
    fig.suptitle(title)

    # Save and close
    if saveto is not None:
        plt.savefig(saveto)
    plt.close()


def nn_histogram(y: pd.DataFrame, pred: pd.DataFrame, *,
                 title: Optional[str]=None, saveto: Optional[PathOrStr]=None) -> None:
    """
    Generate histograms of the absolute and relative residuals between NN predictions vs true values.
    """
    # Calculate differences and relative differences
    diff = pred - y
    reldiff = diff / y * 100
    reldiff.replace([-np.inf, np.inf], np.nan, inplace=True)  # Mask infinities

    # Setup
    fig, axs = plt.subplots(nrows=2, ncols=len(y.columns), figsize=(10, 5), sharey=True, layout="constrained", gridspec_kw={"hspace": 0.2})

    # Plot histograms
    _hist_kwargs = {"bins": 51, "density": False, "color": "black"}
    diff.hist(ax=axs[0], **_hist_kwargs)
    reldiff.hist(ax=axs[1], **_hist_kwargs)

    # Labels, lims
    for ax in axs[0]:
        ax.set_xlabel(r"Absolute residual ($\hat{y} - y$)")
    for ax in axs[1]:
        ax.set_xlabel(r"Relative residual ($\hat{y}/y - 1$) [%]")

    for ax in axs.ravel():
        ax.set_xlim(*symmetric_lims(ax.get_xlim()))

    fig.suptitle(title)

    # Save and close
    if saveto is not None:
        plt.savefig(saveto)
    plt.close()
