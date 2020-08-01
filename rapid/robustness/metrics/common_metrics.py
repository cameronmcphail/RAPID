"""Contains common robustness metrics.

All metrics are explained in detail in:

McPhail, C., Maier, H. R., Kwakkel, J. H., Giuliani, M.,
Castelletti, A., & Westra, S. (2018).
Robustness metrics: How are they calculated, when should they be used
and why do they give different results?. Earth's Future, 6(2), 169-191.
https://doi.org/10.1002/2017EF000649
"""

import numpy as np

from .transforms import t1, t2, t3


def maximin(f, maximise=True):
    """Maximin metric (worst-case scenario)

    The maximin (minimax) metric was first used by Wald (1950).
    It is a very risk averse metric that assumes that the scenario
    that will occur is the scenario under which the performance
    is lowest.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.identity(f, maximise=maximise)
    _f = t2.worst_case(_f)
    R = t3.f_sum(_f)
    return R


def maximax(f, maximise=True):
    """Maximax metric (best-case scenario)

    Maximax is the opposite of the maximin metric (Wald, 1950). It
    is a metric with a low level of risk aversion that looks for the
    best possible performance that is possible in an decision
    alternative.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.identity(f, maximise=maximise)
    _f = t2.best_case(_f)
    R = t3.f_sum(_f)
    return R


def hurwicz(f, maximise=True, alpha=0.5):
    """Hurwicz's Optimism-Pessimism Rule

    Hurwicz’s optimism-pessimism rule (Hurwicz, 1953) uses a weighted
    sum of the maximin and maximax metrics that have previously been
    discussed. Like the previous metrics, the Hurwicz’s
    optimism-pessimism rule uses the distribution of performances for
    an individual decision alternative (i.e. it does not compare the
    performances of multiple decision alternatives). It has a
    parameter alpha that determines the relative degree of intrinsic
    risk aversion of the metric where 0<alpha<1 is the weighting of
    the maximin (high level of risk aversion) metric. In other words,
    alpha may be described as the proportion of high to low risk
    aversion for the decision-maker. Being composed of both the
    maximin and maximax metrics brings many of the characteristics of
    these metrics.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).
    alpha : float, optional
        The weighting to place on the worst-case scenario.
        (The default is 0.5, which implies an equal weighting of the
        best- and worst-case scenarios).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    # Define the weights for the worst- and best-cases.
    weights = np.asarray([alpha, 1. - alpha])
    _f = t1.identity(f, maximise=maximise)
    _f = t2.worst_and_best_cases(_f)
    R = t3.f_w_sum(_f, weights=weights)
    return R


def laplace(f, maximise=True):
    """Laplace's Principle of Insufficient Reason

    Laplace’s principle of insufficient reason
    (Laplace and Simon, 1951) states that in the absence of information
    on the relative probabilities of the scenarios, each scenario
    should be treated as equally likely. This is equivalent to assuming
    the mean performance across the distributions represents the
    expected value of robustness. Unlike the previously discussed
    metrics, Laplace’s principle of insufficient reason uses the
    performance values from every scenario rather than just using one
    or two performance values.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.identity(f, maximise=maximise)
    _f = t2.all_scenarios(_f)
    R = t3.f_mean(_f)
    return R


def minimax_regret(f, maximise=True):
    """Minimax Regret metric

    Rather than looking at individual decision alternatives, regret
    metrics including minimax regret (Savage, 1951) look for the
    regret of choosing a particular option. Specifically, minimax
    regret calculates the maximum regret that can be expected in
    any scenario. The regret for a decision alternative l_i in
    scenario s_j is calculated by comparing the performance
    f(x_i, s_j) to the best possible performance of any decision
    alternative in scenario s_j. For decision alternative x_i, the
    robustness value is the regret from the scenario with the
    greatest level of regret. In this case, the objective is to
    minimize the regret. Unlike other metrics which consider an
    individual decision alternative, the minimax regret metric is
    sensitive to the distributions of performance of two or more
    decision alternatives. However, it is only sensitive to the
    largest difference between the distributions.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.regret_from_best_da(f, maximise=maximise)
    _f = t2.worst_case(_f)
    R = t3.f_sum(_f)
    return R


def percentile_regret(f, maximise=True, percentile=0.1):
    """percentile regret metric

    This is derived from the 90th percentile minimax regret metric
    (Herman et al., 2015) which itself is a variant of the minimax
    metric (Savage, 1951) that was discussed previously. Regret is
    calculated using the same transformation as the minimax regret
    metric, and thus this metric also is used to compare two or more
    decision alternatives rather than only looking at an individual
    decision alternative. The expected amount of regret for decision
    alternative x_i is calculated using the kth percentile of regret
    rather than the maximum possible regret.

    This metric is thus more sensitive to the overall distribution of
    the performance when compared to the minimax regret metric.
    However, it is still most sensitive to only a small number of
    scenarios when compared to a metric such as Laplace’s principle of
    insufficient reason which uses the average of every scenario.


    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).
    percentile : float, optional
        Which percentile of regret values to use.
        (The default is 0.1, which implies the use of the 10th
        percentile. That is the f value at which only 10% of f values
        (for a decision alternative) are worse).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.regret_from_best_da(f, maximise=maximise)
    _f = t2.select_percentiles(_f, np.asarray([percentile]))
    R = t3.f_sum(_f)
    return R


def mean_variance(f, maximise=True):
    """Mean-variance metric

    The mean-variance metric (Kwakkel et al., 2016b) is similar to
    Laplace’s principle of insufficient reason in that it uses the
    mean to determine the expected value of the distribution of
    performances for an individual decision alternative. Unlike
    Laplace’s principle of insufficient reason, the mean-variance
    metric also considers the variability in the distribution of
    performances by using the standard deviation of performance
    values. This metric does face several challenges including
    that the influence of the mean and standard deviation will
    depend on their relative magnitude and thus the trade-off
    between mean and standard deviation is unknown
    (Kwakkel et al., 2016b).

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.identity(f, maximise=maximise)
    _f = t2.all_scenarios(_f)
    R = t3.f_mean_variance(_f)
    return R


def undesirable_deviations(f, maximise=True):
    """Undesirable deviations metric

    The undesirable deviations metric (Kwakkel et al., 2016b) is a
    variation on the approach used by Takriti & Ahmed (2004]). This
    metric only considers undesirable deviations (regret) away from
    the median performance value (which is considered the expected
    value).

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    # Do identity first, before regret, so that correct percentiles
    # can be determined.
    _f = t1.regret_from_median(f, maximise=maximise)
    _f = t2.worst_half(_f)
    R = t3.f_sum(_f)
    return R


def percentile_skew(f, maximise=True):
    """A calculation of skew based on percentiles

    The percentile-based skewness metric (Voudouris et al., 2014)
    considers the skewness of the distribution of performance values.
    This metric gives preference to decision alternatives where the
    performance values are skewed towards better performance values. It
    uses the 10th, 50th and 90th percentile values.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.identity(f, maximise=maximise)
    # This calculation of skew relies on the 10th, 50th and 90th
    # percentiles.
    percentiles = np.asarray([0.1, 0.5, 0.9])
    _f = t2.select_percentiles(_f, percentiles)
    R = t3.f_skew(_f)
    return R


def percentile_kurtosis(f, maximise=True):
    """A calculation of kurtosis based on percentiles

    A variation of Kurtosis was applied by Voudouris et al. (2014) to
    determine robustness. This metric indicates the “peakedness” of
    the distribution (Kwakkel et al., 2016). It uses the 10th, 25th,
    75th and 90th percentile performance values respectively for each
    decision alternative. Unlike the percentile-based skewness metric,
    this metric does not consider whether the distribution is skewed
    towards higher or lower performance values. A higher value implies
    that the performance values are more peaked around the median
    value.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.identity(f, maximise=maximise)
    # This calculation of skew relies on the 10th, 50th and 90th
    # percentiles.
    percentiles = np.asarray([0.1, 0.25, 0.75, 0.9])
    _f = t2.select_percentiles(_f, percentiles)
    R = t3.f_kurtosis(_f)
    return R


def starrs_domain(f, maximise=True, threshold=0.0, accept_equal=True):
    """Robustness based on proportion of scenarios meeting a threshold

    Unlike previous metrics, Starr’s domain criterion (Starr, 1963;
    Schneller and Sphicas, 1983) compares the distribution of
    performance values to a threshold value. This metric is most useful
    when the threshold is selected such that the level of performance
    above or below the threshold does not matter, but preferably the
    decision alternative will meet this threshold. For example, a
    system may have a threshold such that any decision alternative
    with a performance below the threshold is a fail and any
    performance above the threshold is a pass.

    Parameters
    ----------
    f : numpy.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool, optional
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
    numpy.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    _f = t1.satisfice(
        f,
        maximise=maximise,
        threshold=threshold,
        accept_equal=accept_equal)
    _f = t2.all_scenarios(_f)
    R = t3.f_mean(_f)
    return R
