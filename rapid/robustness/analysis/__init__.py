"""Contains functions that help analyse robustness values.

This includes functions that help analyse the impact of scenarios on
robustness values and robustness rankings, as well as functions that
help analyse the impact of different robustness metrics on the
robustness values and rankings.

This modeule also contains functions to help visualise these impacts.
"""
from .comparisons import (
    scenarios_similarity,
    R_metric_similarity,
    delta_plot,
    tau_plot
)
