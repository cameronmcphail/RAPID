"""Contains helper code for creating custom robustness metrics.

Guidance for creating custom robustness metrics contained in:
TODO

"""

from .transforms import t1, t2, t3


class custom_R_metric:
    """Create a custom robustness metric

    """
    def __init__(self, t1_func, t2_func, t3_func):
        """Initialize the custom Robustness metric
        """
        self.t1_func = t1_func
        self.t2_func = t2_func
        self.t3_func = t3_func

    def __call__(
            self,
            f,
            maximise=True,
            t1_kwargs=None,
            t2_kwargs=None,
            t3_kwargs=None):
        """Calculate robustness from given values

        Parameters
        ----------
        f : numpy.ndarray, shape=(m, n)
            Performance values, f, for m decision alternatives
            and n scenarios.
        maximise : bool, optional
            Is the performance metric to be maximised or minimised.
            (The default is True, which implies high values of f are better
            than low values of f).
        t1_kwargs, t2_kwargs, t3_kwargs : dict, optional
            The keyword arguments required for these transfromations

        Returns
        -------
        numpy.ndarray, shape=(m, ) OR float if m=1
            The robustness value for each of the m decision alternatives
        """
        if t1_kwargs is None:
            t1_kwargs = {}
        if t2_kwargs is None:
            t2_kwargs = {}
        if t3_kwargs is None:
            t3_kwargs = {}
        transformed_f = self.t1_func(f, maximise=maximise, **t1_kwargs)
        selected_f = self.t2_func(transformed_f, **t2_kwargs)
        R = self.t3_func(selected_f, **t3_kwargs)
        if R.shape[0] == 1:
            R = R[0]
        return R


def callable_transformation(transformation, kwargs):
    """Allows kwargs to be given to transformation before calling it."""
    func = lambda f: transformation(f, **kwargs)
    return func


def guidance_to_R():
    """Guides the user to produce a custom robustness metric.

    Returns
    -------
    class
        A callable class that is the custom robustness metric.
    """
    print('\n\n******')
    print('Create a custom robustness metric')
    print('******\n')
    t1_func = None
    t2_func = None
    t3_func = None

    threshold = ''
    while threshold not in ['y', 'n']:
        print('Does a meaningful threshold for the level of performance exist? (y/n)')
        print('\tE.g. supply must be greater than demand, or')
        print('\t     cost must be kept within a budget?')
        threshold = input()
        if threshold not in ['y', 'n']:
            print('\nError: answer must be "y" or "n".')

    most_important = ''
    if threshold == 'n':
        while most_important not in ['a', 'b']:
            print('\nIs it most important to (a) make the best decision, or (b) avoid making the wrong decision? (a or b)')
            most_important = input()
            if most_important not in ['a', 'b']:
                print('\nError: answer must be "a" or "b".')
        if most_important == 'a':
            t1_func = t1.identity
        else:
            t1_func = t1.regret_from_best_da
    else:
        print('\n** Make sure that the threshold is included in t1_kwargs when calculating robustness! **')
        while most_important not in ['a', 'b']:
            print('\nIs it most important to (a) minimise the magnitude of failure, or (b) maximise number of scenarios with acceptable performance? (a or b)')
            most_important = input()
            if most_important not in ['a', 'b']:
                print('\nError: answer must be "a" or "b".')
        if most_important == 'a':
            t1_func = t1.satisficing_regret
        else:
            t1_func = t1.satisfice
            t3_func = t3.f_mean

    indication_of_f = ''
    if t3_func is None:
        while indication_of_f not in ['a', 'b']:
            print('\nIs it most important to (a) get an indication of the level of performance or (b) the range of performance? (a or b)')
            indication_of_f = input()
            if indication_of_f not in ['a', 'b']:
                print('\nError: answer must be "a" or "b".')
        if indication_of_f == 'a':
            t3_func = t3.f_mean
        else:
            t3_func = t3.f_range

    if indication_of_f in ['', 'b']:
        print('\nSelect an upper and lower percentile to reflect the level of risk aversion/tolerance.')
        print('(i.e. between 0%% and 100%% reflecting maximum risk aversion and maximum risk tolerance, respectively).')
        print('First, enter the upper percentile (e.g. enter "87" for 87%):')
        upper_bnd = float(input()) / 100.
        print('\nNow enter the lower bound:')
        lower_bnd = float(input()) / 100.
        t2_func = callable_transformation(t2.select_percentiles, {'percentiles': [lower_bnd, upper_bnd]})
    else:
        print('\nSelect an percentile to reflect the level of risk aversion/tolerance.')
        print('(i.e. between 0% and 100% reflecting maximum risk aversion and maximum risk tolerance, respectively):')
        percentile = float(input()) / 100.
        t2_func = callable_transformation(t2.select_percentiles, {'percentiles': [percentile]})

    R_metric = custom_R_metric(t1_func, t2_func, t3_func)
    return R_metric
