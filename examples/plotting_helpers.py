"""Helper functions for plotting in examples."""
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")


def plot_as_data(data, nbins=100, normalize_to=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if normalize_to is not None:
        # normalize to is the number of events of another dataset
        bins, edges = np.histogram(data, bins=nbins)
        bins = np.true_divide(bins, normalize_to)
        yerr = np.sqrt(bins) / np.sqrt(normalize_to)
    else:
        bins, edges = np.histogram(data, bins=nbins, density=True)
        yerr = np.sqrt(bins) / np.sqrt(len(data))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.errorbar(centers, bins, yerr=yerr, fmt="o", color="black", **kwargs)
    # plot with correct error bars
    return ax


def save_image(name, outdir):
    for ext in ["png", "pdf"]:
        plt.savefig(f"{outdir}/{name}.{ext}")
