"""Gaussian Thompson Sampling over lending policies (§3.3; SPEC §2.8).

At each decision epoch the agent draws ``θ̃_k ~ N(μ_k, σ_k²)`` for every eligible arm
and selects ``A_t = argmax_k θ̃_k``; after the reward is observed the chosen arm's
posterior is updated with the incremental mean / online variance of
:mod:`bayesian_credit.bandits.arm_state`. ``BanditConfig.posterior_scale`` switches the
draw between the proposal's ``N(μ, σ²)`` and the standard-error form ``N(μ, σ²/n)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import BanditConfig
from ..data.arms import ArmSpace
from ..data.segmentation import Segmenter
from .arm_state import GaussianArmSet
from .base import ArmPriorProvider, ArmSetPolicy


def thompson_choose(
    arms: GaussianArmSet,
    idx: np.ndarray,
    rng: np.random.Generator,
    cfg: BanditConfig,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Draw θ̃_k for the eligible arms ``idx`` and return the argmax (ties → lowest index).

    Scores are the draws (NaN for ineligible arms). Consumes exactly one
    ``rng.normal`` call of size ``len(idx)``.
    """
    theta = arms.sample(rng, idx, cfg.posterior_scale, cfg.min_sigma)
    scores = np.full(len(arms), np.nan)
    scores[idx] = theta
    return int(idx[int(np.argmax(theta))]), scores, {}


class GaussianThompsonSampling(ArmSetPolicy):
    """Flat Gaussian Thompson Sampling: one arm set (group ``"*"``) for all borrowers.

    θ_k ~ N(μ_k, σ_k²), A_t = argmax θ̃_k (§3.3). Warm start of the single arm set:
    ``prior_provider("*")`` for ``warm_start="risk_model"`` (flat prior if no provider),
    otherwise N(``cfg.prior_mu``, ``cfg.prior_sigma``²). It also serves as the pooled
    parent level of :class:`~bayesian_credit.bandits.contextual.ContextualThompsonSampling`.
    """

    default_name = "thompson"
    supports_segmenter = False

    def __init__(
        self,
        arm_space: ArmSpace,
        *,
        cfg: BanditConfig | None = None,
        prior_provider: ArmPriorProvider | None = None,
        warm_start: str | None = None,
        name: str | None = None,
        pin_null_arm: bool = True,
    ):
        super().__init__(
            arm_space,
            None,
            cfg=cfg,
            prior_provider=prior_provider,
            warm_start=warm_start,
            name=name,
            pin_null_arm=pin_null_arm,
        )

    @classmethod
    def _construct(
        cls,
        arm_space: ArmSpace,
        segmenter: Segmenter | None,
        cfg: BanditConfig,
        prior_provider: ArmPriorProvider | None,
        name: str | None,
        pin_null_arm: bool,
        params: dict[str, Any],
    ) -> GaussianThompsonSampling:
        return cls(
            arm_space,
            cfg=cfg,
            prior_provider=prior_provider,
            name=name,
            pin_null_arm=pin_null_arm,
            **params,
        )

    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        return thompson_choose(arms, idx, rng, self.cfg)
