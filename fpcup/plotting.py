"""
Functions for plotting data and results
"""
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def replace_year_in_datetime(date, newyear=2000):
    """
    For a datetime object yyyy-mm-dd, replace yyyy with newyear.
    """
    return date.replace(year=newyear)

def plot_wofost_ensemble(outputs, keys=None, saveto=None, replace_years=True, show=True):
    """
    Plot WOFOST ensemble results.
    """
    # If no keys were specified, get all of them
    if keys is None:
        keys = outputs[0].keys()

    # Plot curves for outputs
    fig, axs = plt.subplots(nrows=len(keys), sharex=True, figsize=(8,10))

    with tqdm(total=len(outputs), desc="Plotting results", unit="runs") as pbar:
        for df in outputs:
            # Remove the year information if desired, e.g. to compare year-by-year results
            if replace_years:
                time_axis = pd.to_datetime(df.index.to_series()).apply(replace_year_in_datetime)
            else:
                time_axis = df.index

            # Plot every key in the corresponding panel
            for ax, key in zip(axs, keys):
                ax.plot(time_axis, df[key], alpha=0.25)
            pbar.update()

    axs[-1].set_xlabel("Time")
    for ax, key in zip(axs, keys):
        ax.set_ylabel(key)
        ax.set_ylim(ymin=0)
        ax.grid()
    fig.align_ylabels()
    axs[0].set_title(f"Results from {len(outputs)} WOFOST runs")

    if saveto is not None:
        fig.savefig(saveto, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()