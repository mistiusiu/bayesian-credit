"""Table 11 evaluation (SPEC §2.12): predictive metrics, policy metrics and the report.

* :mod:`.predictive` — Brier, AUC, ECE / reliability, log loss, MAE / RMSE, transition
  log-likelihood and row error.
* :mod:`.policy` — cumulative reward / regret, rolling reward, selection frequency,
  decline rate, profitability vs historical, inclusion, convergence.
* :mod:`.report` — :class:`EvaluationReport` (all nine Table 11 dimensions).
"""

from __future__ import annotations

from . import policy, predictive, report
from .policy import (
    LOG_COLUMNS,
    arm_mix,
    convergence_epoch,
    cumulative_regret,
    cumulative_reward,
    decline_rate,
    inclusion_metrics,
    profitability_vs_historical,
    rolling_reward,
    selection_frequency,
)
from .predictive import (
    brier,
    expected_calibration_error,
    log_loss,
    mae,
    reliability_table,
    rmse,
    roc_auc,
    roc_curve,
    transition_log_likelihood,
    transition_row_error,
)
from .report import DIMENSIONS, EvaluationReport

__all__ = [
    "DIMENSIONS",
    "LOG_COLUMNS",
    "EvaluationReport",
    "arm_mix",
    "brier",
    "convergence_epoch",
    "cumulative_regret",
    "cumulative_reward",
    "decline_rate",
    "expected_calibration_error",
    "inclusion_metrics",
    "log_loss",
    "mae",
    "policy",
    "predictive",
    "profitability_vs_historical",
    "reliability_table",
    "report",
    "rmse",
    "roc_auc",
    "roc_curve",
    "rolling_reward",
    "selection_frequency",
    "transition_log_likelihood",
    "transition_row_error",
]
