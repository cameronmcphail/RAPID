"""Contains the T2 transformations (scenario subset selection).

This is related to the level of risk averseness of the decision-maker.
"""

import numpy as np


def all_scenarios(f):
    """Use all scenarios. Provided for completeness.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, n')
        The selected n' performance values
        In this case n' = n
    """
    return f


def worst_case(f):
    """Assume the worst-case scenario for each decision alternative.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, n')
        The selected n' performance values
        In this case n' = 1
    """
    worst_f = np.amin(f, 1, keepdims=True)
    return worst_f


def best_case(f):
    """Assume the best-case scenario for each decision alternative.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, n')
        The selected n' performance values
        In this case n' = 1
    """
    best_f = np.amax(f, 1, keepdims=True)
    return best_f


def worst_and_best_cases(f):
    """Work with the most extreme worst- and best-case scenarios.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, n')
        The selected n' performance values
        In this case n' = 2
    """
    worst_f = worst_case(f)
    best_f = best_case(f)
    _f = np.concatenate((worst_f, best_f), axis=1)
    return _f


def worst_half(f):
    """Work with the worst half of scenarios

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, n')
        The selected n' performance values
        In this case n' = 0.5*n (round up to nearest whole number)
    """
    sorted_f = np.sort(f)
    n = sorted_f.shape[1]  # Num of scenarios
    _n = int(n / 2. + 0.51)  # Half of the scenarios
    _f = sorted_f[:, :_n]
    return _f


def select_percentiles(f, percentiles):
    """Select particular percentiles of f for each decision alternative.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios
    percentiles : np.ndarray, shape=(n', ), dtype=float
        Which percentile of to select for each decision alternative.
        E.g. [0.2, 0.75] would get the 20th and 75th percentiles for
        each decision alternative. That is to say, the f values for
        each decision alternative where 20% and 75% of values are
        worse.

    Returns
    -------
    np.ndarray, shape=(m, n')
        The selected n' performance values
        n' is given by the percentiles parameter
    """
    _f = np.transpose(
        np.quantile(f, percentiles, axis=1, interpolation='nearest'))
    return _f
