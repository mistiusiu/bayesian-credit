"""Risk models (proposal §3.3): the Two-Stage Beta-Beta model and the MCTM.

* :class:`HierarchicalBetaBernoulli` — Stage 1, ``p_{g,a} | D ~ Beta(α_{g,a} + d, β_{g,a} + n)``
* :class:`BetaMoMRecovery` — Stage 2, ``R_{g,a} | D = 1 ~ Beta(α^R, β^R)`` (method of moments)
* :class:`TwoStageBetaBeta` / :class:`RiskEstimate` — the facade used by rewards and bandits
* :class:`PolicyEffectModel` / :class:`PolicyEffectTilt` — structured leaf prior (policy effect)
* :class:`MarkovChainTransitionMatrix` — credit-state chain over ``S = {C, D, R, L}``
"""

from .beta import BetaPosterior
from .hierarchical import HierarchicalBetaBernoulli, beta_binomial_kappa, beta_hdi
from .mctm import (
    START_CURRENT,
    START_DELINQUENT,
    MarkovChainTransitionMatrix,
    State,
    TransitionStructure,
    allowed_transitions,
    validate_transition_matrix,
)
from .policy_effect import PolicyEffectModel, PolicyEffectTilt, TiltTable
from .recovery import BetaMoMRecovery, RunningMoments
from .two_stage import RiskEstimate, TwoStageBetaBeta

__all__ = [
    "START_CURRENT",
    "START_DELINQUENT",
    "BetaMoMRecovery",
    "BetaPosterior",
    "HierarchicalBetaBernoulli",
    "MarkovChainTransitionMatrix",
    "PolicyEffectModel",
    "PolicyEffectTilt",
    "RiskEstimate",
    "RunningMoments",
    "State",
    "TiltTable",
    "TransitionStructure",
    "TwoStageBetaBeta",
    "allowed_transitions",
    "beta_binomial_kappa",
    "beta_hdi",
    "validate_transition_matrix",
]
