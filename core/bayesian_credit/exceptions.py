"""Exception hierarchy for the framework.

§3.5 of the proposal requires "comprehensive exception handling to manage numerical
instability and mathematical boundary conditions inherent in Bayesian inference".
Every module raises one of these rather than a bare ``ValueError`` so callers (API,
simulation runner) can distinguish a modelling boundary condition from a bug.
"""

from __future__ import annotations


class BayesianCreditError(Exception):
    """Base class for all framework errors."""


class ConfigurationError(BayesianCreditError):
    """A configuration object is internally inconsistent."""


class NumericalInstabilityError(BayesianCreditError):
    """A computation hit a mathematical boundary (e.g. method-of-moments with
    variance >= mean(1-mean), overflow in an online variance update, division by a
    zero exposure)."""


class InvalidPosteriorError(BayesianCreditError):
    """A Beta/Gaussian posterior has non-positive or non-finite parameters."""


class InvalidTransitionMatrixError(BayesianCreditError):
    """A Markov transition matrix violates row-stochasticity, absorbing-state or
    structural-zero constraints, or (I - Q) is singular."""


class LeakageError(BayesianCreditError):
    """Information from a later lifecycle stage (interim/terminal) was offered to a
    component that may only see pre-decision context (Appendix F.3)."""


class CircuitBreakerOpenError(BayesianCreditError):
    """A policy update or adaptation was attempted while the PSI circuit breaker is
    open and no human acknowledgement has been recorded."""


class DataSourceError(BayesianCreditError):
    """The Lending Club file / synthetic generator could not produce a valid LoanTable."""


class CheckpointError(BayesianCreditError):
    """A simulation checkpoint is missing, corrupt or incompatible with the run config."""


class GovernanceError(BayesianCreditError):
    """Invalid human-in-the-loop operation (unknown or already acknowledged review,
    acknowledging a breaker that is not open, unknown review action)."""
