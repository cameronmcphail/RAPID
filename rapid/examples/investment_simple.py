"""Runs a simple investment example

From Wikipedia page: Regret (decision theory)
https://en.wikipedia.org/wiki/Regret_(decision_theory)
Accessed 18/10/2019

"Suppose an investor has to choose between investing in
stocks, bonds or the money market, and the total return
depends on what happens to interest rates. The following
table shows some possible returns:

----------------------------------------------------------------------------------------------
|| Return       || Interest rates rise | Static rates | Interest rates fall || Worst return ||
||--------------||---------------------|--------------|---------------------||--------------||
|| Stocks       ||                  -4 |            4 |                  12 ||           -4 ||
|| Bonds        ||                  -2 |            3 |                   8 ||           -2 ||
|| Money market ||                   3 |            2 |                   1 ||            1 ||
||--------------||---------------------|--------------|---------------------||----------------
|| Best return  ||                   3 |            4 |                  12 ||
------------------------------------------------------------------------------

The crude maximin choice based on returns would be to invest
in the money market, ensuring a return of at least 1. ...

The regret table for this example, constructed by subtracting
actual returns from best returns, is as follows:

----------------------------------------------------------------------------------------------
|| Regret       || Interest rates rise | Static rates | Interest rates fall || Worst regret ||
||--------------||---------------------|--------------|---------------------||--------------||
|| Stocks       ||                   7 |            0 |                   0 ||            7 ||
|| Bonds        ||                   5 |            1 |                   4 ||            5 ||
|| Money market ||                   0 |            2 |                  11 ||           11 ||
----------------------------------------------------------------------------------------------

Therefore, using a minimax choice based on regret, the best
course would be to invest in bonds. ...
"""

import pandas as pd

from rapid.robustness.metrics import t1, t2, t3, custom_R_metric
from rapid.robustness.evaluator import f_to_R


def investment_simple_example():
    """Runs investment example explained above"""
    # Define the returns AND some made-up critical thresholds
    # Critical thresholds are used by the Starr's Domain metric
    df = pd.DataFrame.from_dict({
        's_idx': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'l_idx': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'return': [-4, 4, 12, -2, 3, 8, 3, 2, 1],
        'critical': [3, 1, 0, 3, 1, 0, 3, 1, 0]
    })
    info = {
        'Maximin': {
            'f': 'return',
            'maximise': True,
            'threshold': None,
            'func': custom_R_metric(t1.identity, t2.worst_case, t3.f_mean),
            'kwargs': {}},
        'Minimax regret': {
            'f': 'return',
            'maximise': True,
            'threshold': None,
            'func': custom_R_metric(t1.regret_from_best_da, t2.worst_case, t3.f_mean),
            'kwargs': {}},
        'Starr\'s Domain': {
            'f': 'return',
            'maximise': True,
            'threshold': 'critical',
            'func': custom_R_metric(t1.satisfice, t2.worst_case, t3.f_mean),
            'kwargs': {}},
        # 'Custom R': {
        #     'f': 'return',
        #     'maximise': True,
        #     'threshold': None,
        #     'func': guidance_to_R(),
        #     'kwargs': {'t1_kwargs': {'threshold': 1.5}}}
    }

    R = f_to_R(df, info)

    for metric in R:
        print('Robustness metric: {}'.format(metric))
        print('Robustness scores: {}\n'.format(R[metric]))
