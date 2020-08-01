"""Implements example from EM Workbench package.

Example contained in:

Kwakkel, J.H., 2017. The Exploratory Modeling Workbench: An open source
toolkit for exploratory modeling, scenario discovery, and
(multi-objective) robust decision making. Environ. Model. Softw. 96,
239–250. https://doi.org/10.1016/j.envsoft.2017.06.054

Altered to use custom robustness metrics from the RAPID package.
"""

import os
import functools
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from ema_workbench import (
    MultiprocessingEvaluator, Model, RealParameter, Constant, ScalarOutcome, Policy)
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.examples.lake_model import lake_problem
from rapid.robustness.metrics import t1, t2, t3, custom_R_metric
from rapid.robustness import metrics
from rapid.robustness import analysis


def get_custom_R_metrics():
    """Returns the custom robustness metrics from paper."""
    av_vulnerability_R = functools.partial(
        custom_R_metric(t1.identity, t2.select_percentiles, t3.f_identity),
        maximise=False,
        t2_kwargs={'percentiles': [0.25]})
    reliability_R = functools.partial(
        custom_R_metric(t1.satisfice, t2.all_scenarios, t3.f_mean),
        t1_kwargs={'threshold': 0.8},
        maximise=True)
    utility_R = functools.partial(
        custom_R_metric(t1.satisficing_regret, t2.select_percentiles, t3.f_identity),
        maximise=True,
        t1_kwargs={'threshold': 0.75},
        t2_kwargs={'percentiles': [0.5]})
    inertia_R = functools.partial(
        custom_R_metric(t1.identity, t2.select_percentiles, t3.f_identity),
        maximise=True,
        t2_kwargs={'percentiles': [0.5]})

    return [av_vulnerability_R, reliability_R, utility_R, inertia_R]


def get_custom_R_metrics_for_workbench():
    """Returns robustness metrics that can interact with ema_workbench

    It is simple to create robustness metrics (see get_custom_R_metrics())
    and ema_workbench requires them be specified in a particular way.
    Therefore, this function specifies them for the ema_workbench.

    Returns a list of robustness metrics to be used for the Lake Model.
    """
    # Note that we want to minimise max_P, so we define this in the
    # robustness metrics above (maximise=False), and this changes
    # the sign of the robustness metric, so that we can always
    # make the objective to MAXIMIZE robustness.
    R_metrics = get_custom_R_metrics()
    robustness_functions = [
        ScalarOutcome(
            'Av vulnerability R',
            kind=ScalarOutcome.MAXIMIZE,
            variable_name='max_P',
            function=R_metrics[0]),
        ScalarOutcome(
            'Reliability R',
            kind=ScalarOutcome.MAXIMIZE,
            variable_name='reliability',
            function=R_metrics[1]),
        ScalarOutcome(
            'Utility R',
            kind=ScalarOutcome.MAXIMIZE,
            variable_name='utility',
            function=R_metrics[2]),
        ScalarOutcome(
            'Inertia R',
            kind=ScalarOutcome.MAXIMIZE,
            variable_name='inertia',
            function=R_metrics[3])]

    return robustness_functions


def get_original_R_metrics():
    """Returns the Robustness metrics from original example."""
    robustness_functions = [
        ScalarOutcome(
            'mean p',
            kind=ScalarOutcome.MINIMIZE,
            variable_name='max_P',
            function=np.mean),
        ScalarOutcome(
            'std p',
            kind=ScalarOutcome.MINIMIZE,
            variable_name='max_P',
            function=np.std),
        ScalarOutcome(
            'sn reliability',
            kind=ScalarOutcome.MAXIMIZE,
            variable_name='reliability',
            function=signal_to_noise),
        ScalarOutcome(
            '10th percentile utility',
            kind=ScalarOutcome.MAXIMIZE,
            variable_name='reliability',
            function=functools.partial(np.percentile, q=10))]

    return robustness_functions


def signal_to_noise(data):
    """A robustness metric defined for the original example."""
    mean = np.mean(data)
    std = np.std(data)
    sn = mean/std
    return sn


def get_lake_model():
    """Returns a fully formulated model of the lake problem."""
    # instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem)
    lake_model.time_horizon = 100

    # specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]

    # set levers, one for each time step
    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in
                         range(lake_model.time_horizon)]

    # specify outcomes
    lake_model.outcomes = [ScalarOutcome('max_P',),
                           ScalarOutcome('utility'),
                           ScalarOutcome('inertia'),
                           ScalarOutcome('reliability')]

    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 150)]
    return lake_model


def optimize_lake_problem(use_original_R_metrics=False, demo=True):
    """Analysis of the Lake Problem.

    (1) Runs a multi-objective robust optimisation of the Lake Problem
        using both standard and custom robustness metrics;
    (2) analyses the effects of different sets of scenarios on the
        robustness values and robustness rankings;
    (3) plots these effects;
    (4) analyses the effects of different robustness metrics on the
        robustness values and robustness rankings; and
    (5) plots these effects.
    """
    filepath = './robust_results.h5'

    robustness_functions = (
        get_original_R_metrics()
        if use_original_R_metrics
        else get_custom_R_metrics_for_workbench())

    lake_model = get_lake_model()

    if not os.path.exists(filepath):
        n_scenarios = 10 if demo else 200  # for demo purposes only, should in practice be higher
        scenarios = sample_uncertainties(lake_model, n_scenarios)
        nfe = 1000 if demo else 50000  # number of function evaluations

        # Needed un Linux-based machines
        multiprocessing.set_start_method('spawn', True)

        # Run optimisation
        with MultiprocessingEvaluator(lake_model) as evaluator:
            robust_results = evaluator.robust_optimize(
                robustness_functions,
                scenarios,
                nfe=nfe,
                population_size=(10 if demo else 50),
                epsilons=[0.1,] * len(robustness_functions))
        print(robust_results)

    robust_results = pd.read_hdf(filepath, key='df')

    # Results are performance in each timestep, followed by robustness
    # we only care about the robustness, so we get that
    col_names = robust_results.columns.values.tolist()
    col_names = col_names[-len(robustness_functions):]

    # Plot the robustness results
    sns.pairplot(robust_results, vars=col_names, diag_kind='kde')
    # plt.show()

    # Extract the decision alternatives from the results
    # We need to extract the decision alternatives
    decision_alternatives = robust_results.iloc[:, :-4].values
    decision_alternatives = [
        Policy(
            idx,
            **{
                str(idx): value
                for idx, value
                in enumerate(decision_alternatives[idx].tolist())})
        for idx in range(decision_alternatives.shape[0])]

    # Find the influence of scenarios. Here we are creating 5
    # sets of 100 scenarios each, all using the same sampling
    # method.
    scenarios_per_set = 100
    n_sets = 5
    n_scenarios = scenarios_per_set * n_sets
    scenarios = sample_uncertainties(lake_model, n_scenarios)

    # Simulate optimal solutions across all scenarios
    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.perform_experiments(
            scenarios=scenarios, policies=decision_alternatives)
    # We will just look at the vulnerability ('max_P') for this example
    f = np.reshape(results[1]['max_P'], newshape=(-1, n_scenarios))
    # Split the results into the different sets of scenarios
    split_f = np.split(f, n_sets, axis=1)
    # Calculate robustness for each set of scenarios
    # Note that each split_f[set_idx] is a 2D array, with each row being
    # a decision alternative, and each column a scenario
    R_metric = get_custom_R_metrics()[0]
    R = [R_metric(split_f[set_idx]) for set_idx in range(n_sets)]
    R = np.transpose(R)

    # Calculate similarity in robustness from different scenario sets
    delta, tau = analysis.scenarios_similarity(R)
    # Plot the deltas using a helper function
    analysis.delta_plot(delta)
    # Plot the Kendall's tau-b values using a helper function
    analysis.tau_plot(tau)


    # We now want to test the effects of different robustness metrics,
    # across all of the 100 scenarios. We first define a few new
    # robustness metrics (in addition to our original R metric for
    # the vulnerability). For this example we use some classic metrics
    R_metrics = [
        R_metric,  # The original robustness metric
        functools.partial(metrics.maximax, maximise=False),
        functools.partial(metrics.laplace, maximise=False),
        functools.partial(metrics.minimax_regret, maximise=False),
        functools.partial(metrics.percentile_kurtosis, maximise=False)
    ]

    # Calculate robustness for each robustness metric
    R = np.transpose([R_metric(f) for R_metric in R_metrics])

    # Calculate similarity in robustness from different robustness metrics
    tau = analysis.R_metric_similarity(R)
    # Plot the Kendall's tau-b values using a helper function
    analysis.tau_plot(tau)


if __name__ == '__main__':
    optimize_lake_problem(
        use_original_R_metrics=False)
