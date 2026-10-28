"""Decision layer: Gaussian / contextual Thompson Sampling and baselines (SPEC §2.8).

``build_policy(name, arm_space, segmenter, cfg, prior_provider)`` constructs a policy by
the names used in ``SimulationConfig.policies``; ``policy_from_state`` rebuilds any
policy of this package from its ``state()`` dict (checkpoint/resume).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import BanditConfig
from ..data.arms import ArmSpace
from ..data.segmentation import Segmenter
from ..exceptions import ConfigurationError
from .arm_state import (
    POSTERIOR_SCALES,
    GaussianArmPosterior,
    GaussianArmSet,
    posterior_variance,
    ucb_index,
    welford_update,
)
from .base import (
    POOLED_GROUP,
    ArmPriorProvider,
    ArmSetPolicy,
    Decision,
    GroupKeyResolver,
    Policy,
    normalise_eligible,
)
from .baselines import UCB1, EpsilonGreedy, Greedy, HistoricalPolicy, RandomPolicy
from .contextual import ContextualThompsonSampling
from .thompson import GaussianThompsonSampling

#: Names accepted by :func:`build_policy` (``"thompson_flat"`` forces the flat TS).
POLICY_NAMES = (
    "thompson",
    "thompson_flat",
    "epsilon_greedy",
    "ucb1",
    "greedy",
    "random",
    "historical",
)

#: class name -> class, for :func:`policy_from_state`.
POLICY_CLASSES: dict[str, type[ArmSetPolicy]] = {
    cls.__name__: cls
    for cls in (
        GaussianThompsonSampling,
        ContextualThompsonSampling,
        EpsilonGreedy,
        UCB1,
        Greedy,
        RandomPolicy,
        HistoricalPolicy,
    )
}


def build_policy(
    name: str,
    arm_space: ArmSpace,
    segmenter: Segmenter | None = None,
    cfg: BanditConfig | None = None,
    prior_provider: ArmPriorProvider | None = None,
) -> ArmSetPolicy:
    """Policy factory.

    ``"thompson"`` is :class:`ContextualThompsonSampling` when a segmenter is given and
    :class:`GaussianThompsonSampling` otherwise; ``"thompson_flat"`` is always flat. The
    baselines are contextual (per group) when a segmenter is given. ε and the UCB
    constant come from ``cfg.epsilon`` / ``cfg.ucb_c``; the policy's ``name`` is
    ``name`` so decisions are labelled with the requested policy name.
    """
    cfg = cfg if cfg is not None else BanditConfig()
    key = str(name).strip().lower()
    common: dict[str, Any] = {"cfg": cfg, "prior_provider": prior_provider, "name": key}
    if key == "thompson":
        if segmenter is not None:
            return ContextualThompsonSampling(arm_space, segmenter, **common)
        return GaussianThompsonSampling(arm_space, **common)
    if key == "thompson_flat":
        return GaussianThompsonSampling(arm_space, **common)
    if key == "epsilon_greedy":
        return EpsilonGreedy(arm_space, segmenter, epsilon=cfg.epsilon, **common)
    if key == "ucb1":
        return UCB1(arm_space, segmenter, c=cfg.ucb_c, **common)
    if key == "greedy":
        return Greedy(arm_space, segmenter, **common)
    if key == "random":
        return RandomPolicy(arm_space, segmenter, **common)
    if key == "historical":
        return HistoricalPolicy(arm_space, segmenter, **common)
    raise ConfigurationError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")


def policy_from_state(
    state: Mapping[str, Any],
    arm_space: ArmSpace,
    segmenter: Segmenter | None = None,
    prior_provider: ArmPriorProvider | None = None,
) -> ArmSetPolicy:
    """Rebuild any bandit of this package from ``policy.state()`` (dispatch on ``class``)."""
    cls_name = state.get("class")
    cls = POLICY_CLASSES.get(str(cls_name))
    if cls is None:
        raise ConfigurationError(f"unknown policy class {cls_name!r} in state")
    return cls.from_state(state, arm_space, segmenter=segmenter, prior_provider=prior_provider)


__all__ = [
    "POLICY_CLASSES",
    "POLICY_NAMES",
    "POOLED_GROUP",
    "POSTERIOR_SCALES",
    "UCB1",
    "ArmPriorProvider",
    "ArmSetPolicy",
    "ContextualThompsonSampling",
    "Decision",
    "EpsilonGreedy",
    "GaussianArmPosterior",
    "GaussianArmSet",
    "GaussianThompsonSampling",
    "Greedy",
    "GroupKeyResolver",
    "HistoricalPolicy",
    "Policy",
    "RandomPolicy",
    "build_policy",
    "normalise_eligible",
    "policy_from_state",
    "posterior_variance",
    "ucb_index",
    "welford_update",
]
