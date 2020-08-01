"""Helper functions for calculating robustness from performance values"""

import numpy as np


def f_to_R(f_df, R_dict):
    """Calculates robustness from performance values.

    Uses a set of performance values, `f`, determined from simulations
    across multiple decision alternatives, `l`, different scenarios,
    `s`, and calculates robustness, `R`, using a variety of given
    robustness metrics.

    Parameters
    ----------
    f_df : pandas.DataFrame
        A dataframe of performance values, `f`, with indexes for the
        scenario, `s`, and decision alternative, `l`.
        Must include any other variables to be used during calculation
        of robustness (referred to as 'varX_name' below).
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...,
                   '<var1_name>', '<var2_name>', ...]`
    R_dict : dict of dict
        A mapping of robustness metric (`R`) names to information
        about those robustness metrics including
        'f': string
            the corresponding performance metric to use
        'maximise': bool
            whether the aim of that performance metric is to be maximised
        'threshold': None or string
            the name of the column in `f_df` containing thresholds
            OR None if not using a threshold or threshold is given in kwargs
        'func': func
            the robustness metric function
        'kwargs': dict
            keyword arguments required for calculating R
            e.g. {'t1_kwargs': {'threshold': 5.2}}
                 would pass a threshold kwarg to the t1 transformation
                 if using the custom_R_metric
        Note that all performance metric names must be listed here.
        E.g. `{'<R1_name>': {'f': <f1_name>, 'maximise': <bool>, 'threshold': None, 'func': <func>, 'kwargs': {'kwarg1': <arg>}},
               '<R2_name>': {'f': <f1_name>, 'maximise': <bool>, 'threshold': 'critical', 'func': <func>, 'kwargs': {}}, ...}`

    Returns
    -------
    pandas.DataFrame
        A dataframe of robustness values, `R`, with indexes for the
        decision alternative, `l`, and a column for the performance
        metric name, ``f_name``.
        Columns: `['l_idx', 'f_name', '<R1_name>', '<R2_name>', ...]`
    """
    # Extract the names of the f metrics
    f_metrics = [R_dict[R_name]['f'] for R_name in R_dict]
    df_cols = [col for col in f_df.columns]
    # Check that the same metrics are in f_maximise
    for f_metric in f_metrics:
        assert f_metric in df_cols

    # Sort dataframe to ensure consistency
    f_df = sort_f_df(f_df)

    # Get the scenario and decision alternative idxs, and
    # check that the s_idx and l_idx indexes are valid.
    s_idxs, l_idxs = get_f_df_details(f_df)

    # Loop through performance metrics
    R = {}
    for R_metric in R_dict:
        f_metric = R_dict[R_metric]['f']
        # Check that required data exists
        kwargs = R_dict[R_metric]['kwargs']
        if R_dict[R_metric]['threshold'] is not None:
            assert R_dict[R_metric]['threshold'] in df_cols
            if 't1_kwargs' not in kwargs:
                kwargs['t1_kwargs'] = {}
            kwargs['t1_kwargs']['threshold'] = np.reshape(
                f_df.iloc[:, f_df.columns.get_loc(R_dict[R_metric]['threshold'])].values,
                newshape=(l_idxs.size, s_idxs.size))
        f = np.reshape(
            f_df.iloc[:, f_df.columns.get_loc(f_metric)].values,
            newshape=(l_idxs.size, s_idxs.size))
        kwargs['maximise'] = R_dict[R_metric]['maximise']
        R[R_metric] = R_dict[R_metric]['func'](f, **kwargs)
    return R


def sort_f_df(f_df):
    """Sorts f_df by s_idx first then by l_idx.

    E.g. for scenario 0, see all decision alternatives in order,
    then scenario 1, scenario 2, etc.

    Parameters
    ----------
    f_df : pandas.DataFrame
        A dataframe of performance values, `f`, with indexes for the
        scenario, `s`, and decision alternative, `l`.
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...]`
    """
    # This will sort first by s_idx then by l_idx, both from 0 to ...
    f_df.sort_values(['l_idx', 's_idx'], ascending=[True, True])
    return f_df


def get_f_df_details(f_df):
    """Gets the unique s_idx and l_idx values in f_df.

    Also checks that for each s_idx, each unique l_idx exists
    (and vice versa).

    Parameters
    ----------
    f_df : pandas.DataFrame
        A dataframe of performance values, `f`, with indexes for the
        scenario, `s`, and decision alternative, `l`.
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...]`

    Returns
    -------
    s_idxs : list of int
        Number of scenario (`s`) idxs
    l_idxs : list of int
        List decision alternative (`l`) idxs
    """
    # Check that df is sorted
    f_df = sort_f_df(f_df)

    s_idxs = f_df['s_idx'].unique()
    l_idxs = f_df['l_idx'].unique()

    for s_idx in s_idxs:
        relevant_rows = f_df.loc[f_df['s_idx'] == s_idx]
        relevant_l_idxs = relevant_rows['l_idx'].values
        assert np.allclose(relevant_l_idxs, l_idxs)

    return s_idxs, l_idxs
