"""Contains the T1 functions (performance value transformations).

In general, T1 will take one of 3 forms:
    1. Identity transform - for understanding actual performance
    2. Regret transform - for understanding cost of making wrong decision
    3. Satisficing transform - for understanding how often constraints
       are satisfied.
It is expected that after T1, the aim is to maximise performance.
i.e. Even for the identity transform, if minimising, then values
will be returned as negative.
"""

import numpy as np


def identity(f, maximise=True):
    """Keeps values the same unless minimising.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = _prepare_f(f)
    return _f if maximise else -_f


def regret_from_best_da(f, maximise=True):
    """T1: Regret from best decision alternative

    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = _prepare_f(f)
    _f = identity(_f, maximise=maximise)
    best_decision_alternatives = np.amax(_f, axis=0)
    regret = _f - best_decision_alternatives
    return regret


def satisficing_regret(f, threshold, maximise=True):
    """T1: Satisficing regret

    For a given decision alternative, this function compares its performance in
    each scenario to a threshold (threshold can be different for each decision
    alternative), and calculates the magnitude from the threshold (if it fails)
    or returns 0 if it meets the threshold
    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    threshold : np.ndarray, shape=(n, ) or float
        The values to compare the performance values to. i.e. The
        values you would regret not getting, relative to f.
        Can be a different value for each scenario or one value
        across all scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = _prepare_f(f)
    # Determine the regret for each solution in each scenario
    # This will be a number with the aim to be maximised.
    # If the solution has better performance than the threshold,
    # then it will be a positive number
    regret = regret_from_values(_f, threshold, maximise=maximise)
    # In satisficing regret, we only care about the magnitude of
    # failure IF there is a failure. So any performances that are
    # not failures are zeroed out.
    regret[regret > 0.] = 0.
    return regret


def regret_from_values(f, values, maximise=True):
    """T1: Regret from given values

    For a given decision alternative, this function compares its performance in
    each scenario to given performance value for that decision alternative
    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    values : np.ndarray, shape=(n, ) or float
        The values to compare the performance values to. i.e. The
        values you would regret not getting, relative to f.
        Can be a different value for each scenario or one value
        across all scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = _prepare_f(f)
    if isinstance(values, np.ndarray):
        v = values
    else:
        # If values is not different for each scenario, then we must
        # put it in the form of one value, repeated
        v = np.repeat(values, _f.shape[1])
    regret = np.subtract(_f, v)
    # Take into account whether f is to be minimised or maximised.
    regret = identity(regret, maximise=maximise)
    return regret


def regret_from_median(f, maximise=True):
    """T1: Regret from median values

    For a given decision alternative, this function compares its performance in
    each scenario to median performance for that decision alternative

    For a given decision alternative, this function compares its performance in
    each scenario to given performance value for that decision alternative
    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = _prepare_f(f)
    median_f = np.median(_f, axis=1, keepdims=True)
    regret = np.subtract(_f, median_f)
    regret = identity(regret, maximise=maximise)
    return regret


def satisfice(f, maximise=True, threshold=0.0, accept_equal=True):
    """Transform performance how many scenarios are satisficed

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).
    threshold : float, optional
        A minimum value where f >= threshold to be satisficed
        (The default is 0.0, which implies that any f value above 0 is
        of satisfactory performance).
    accept_equal : bool, optional
        Whether or not an f value equal to the threshold is acceptable.
        (The default is True, which implies a >= comparison, whereas
        False would imply a > comparison).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = _prepare_f(f)
    _f = identity(_f, maximise=maximise)
    c = threshold if maximise else -threshold
    _f = _f >= c if accept_equal else _f > c
    _f = np.where(_f, 1., 0.)
    return _f


def _prepare_f(f):
    """Ensures f is in the right form for t1 transformations.

    Converts to an np.ndarray if it isn't already.
    If is of shape (n, ), it converts it to shape (1, n)
    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.

    Returns
    -------
    np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    """
    _f = f if isinstance(f, np.ndarray) else np.asarray(f)
    if len(_f.shape) != 2:
        _f = np.reshape(_f, newshape=(1, -1))
    return _f
