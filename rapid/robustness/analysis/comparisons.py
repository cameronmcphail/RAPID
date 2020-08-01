"""Compares robustness values

Contains (1) a function for showing how a different set of scenarios
affects the robustness values and robustness rankings; and (2) a
function for showing how different robustness metrics affects the
robustness values and robustness rankings.

Also contains a helper function for creating basic visualisations of
the effects of scenarios and robustness metrics.
"""

import copy
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def scenarios_similarity(R):
    """Determines similarity in robustness from multiple scenario sets

    Robustness is a function of scenarios, decision alternatives, and
    a performance metric. 2 sets of scenarios can lead to a different
    calculation of robustness. This function measures the difference
    in 2 different ways:
    - Relative difference, delta (%); and
    - Kendall's Tau-b correlation, tau (unitless, [-1, 1]).

    Parameters
    ----------
    R : numpy.ndarray, shape=(m, n)
        Robustness values, R, for m decision alternatives
        and n scenario sets.

    Returns
    -------
    delta : numpy.ndarray, shape=(n, n)
        Average relative difference (%) in robustness for each pair of
        scenario sets. i.e. idx [0, 3] would be the relative difference
        between scenario sets 0 and 3 (and would be equal to [3, 0])
    tau : numpy.ndarray, shape=(n, n)
        Kendall's Tau-b correlation for each pair of scenario sets.
        i.e. idx [0, 3] would be the correlation between scenario
        sets 0 and 3 (and would be equal to [3, 0])
    """
    # Get the number of sets of scenarios
    n = R.shape[1]
    deltas = np.zeros((n, n))
    taus = np.zeros((n, n))

    for idx_1 in range(n):
        for idx_2 in range(idx_1, n):
            delta = np.divide(
                np.abs(R[:, idx_1] - R[:, idx_2]),
                (np.abs(R[:, idx_1] + R[:, idx_2])) / 2.)
            delta = np.average(delta) * 100.0
            deltas[idx_1, idx_2] = delta
            deltas[idx_2, idx_1] = delta
            tau, _ = stats.kendalltau(R[:, idx_1], R[:, idx_2], nan_policy='omit')
            taus[idx_1, idx_2] = tau
            taus[idx_2, idx_1] = tau

    return deltas, taus


def R_metric_similarity(R):
    """Determines similarity in robustness from multiple robustness metrics

    2 different robustness metrics can lead to a different
    calculation of robustness. This function measures the difference
    by using Kendall's Tau-b correlation, tau (unitless, [-1, 1]).

    Parameters
    ----------
    R : numpy.ndarray, shape=(m, n)
        Robustness values, R, for m decision alternatives
        and n robustness metrics.

    Returns
    -------
    tau : numpy.ndarray, shape=(n, n)
        Kendall's Tau-b correlation for each pair of robustness metrics.
        i.e. idx [0, 3] would be the correlation between R metrics
        0 and 3 (and would be equal to [3, 0])
    """
    # Get the number of sets of scenarios
    n = R.shape[1]
    taus = np.zeros((n, n))

    for idx_1 in range(n):
        for idx_2 in range(idx_1, n):
            tau, _ = stats.kendalltau(R[:, idx_1], R[:, idx_2], nan_policy='omit')
            taus[idx_1, idx_2] = tau
            taus[idx_2, idx_1] = tau

    return taus


def delta_plot(delta):
    """A helper fn for plotting the deltas

    Plots the deltas as a 2D heatmap grid.

    Parameters
    ----------
    delta : numpy.ndarray, shape=(n, n)
        Average relative difference (%) in robustness for each pair of
        scenario sets. i.e. idx [0, 3] would be the relative difference
        between scenario sets 0 and 3 (and would be equal to [3, 0])
    """
    ax = plt.subplot(111)
    cmap = 'rainbow'
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap.set_bad(cmap(0.0))
    norm = mpl.colors.LogNorm(vmin=1.0, vmax=100.0)
    im = ax.imshow(delta, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)
    plt.show()


def tau_plot(tau):
    """A helper fn for plotting the Kendall's Tau-b values

    Plots the tau values as a 2D heatmap grid.

    Parameters
    ----------
    tau : numpy.ndarray, shape=(n, n)
        Kendall's Tau-b correlation for each pair of scenario sets.
        i.e. idx [0, 3] would be the correlation between scenario
        sets 0 and 3 (and would be equal to [3, 0])
    """
    ax = plt.subplot(111)
    im = ax.imshow(tau, cmap='RdBu', vmin=-1.0, vmax=1.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)
    plt.show()
