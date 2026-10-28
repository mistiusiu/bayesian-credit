"""Contextual Thompson Sampling with hierarchical cold start (§3.3; SPEC §2.8; G.3).

The bandit is made *contextual* by the hierarchy: the borrower context ``X_t`` is mapped
to a group ``g`` by the :class:`~bayesian_credit.data.segmentation.Segmenter` and each
group runs its own Gaussian Thompson Sampling over the arm space (Table 7: pre-decision
context, arms = amount × rate × term, posterior sampling for exploration). A pooled flat
:class:`~bayesian_credit.bandits.thompson.GaussianThompsonSampling` receives every
reward as well — it is the hierarchy's parent level, used to warm-start new groups under
``warm_start="global"`` and for the Simpson's-paradox diagnostic.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from ..config import BanditConfig
from ..data.arms import ArmSpace
from ..data.segmentation import Segmenter
from .arm_state import GaussianArmSet
from .base import ArmPriorProvider, ArmSetPolicy
from .thompson import GaussianThompsonSampling, thompson_choose


class ContextualThompsonSampling(ArmSetPolicy):
    """One Gaussian Thompson Sampling arm set per borrower group ``g`` (§3.3, G.3).

    New groups are created lazily on first visit and warm-started per
    ``cfg.warm_start``: ``"risk_model"`` → ``prior_provider(g)`` (the Two-Stage
    Beta-Beta reward distribution "used as the sampling distribution for the automated
    decisions"); ``"global"`` → copy of the pooled parent's current posteriors;
    ``"flat"`` → N(``cfg.prior_mu``, ``cfg.prior_sigma``²). Without a provider,
    ``"risk_model"`` falls back to ``"global"`` and decisions carry
    ``meta["warm_start_fallback"] = True``. Decisions carry
    ``meta = {"group_new": bool, "n_group_obs": int, "seg_version": int}``.
    """

    default_name = "thompson"
    requires_segmenter = True

    def __init__(
        self,
        arm_space: ArmSpace,
        segmenter: Segmenter,
        *,
        cfg: BanditConfig | None = None,
        prior_provider: ArmPriorProvider | None = None,
        warm_start: str | None = None,
        name: str | None = None,
        pin_null_arm: bool = True,
    ):
        super().__init__(
            arm_space,
            segmenter,
            cfg=cfg,
            prior_provider=prior_provider,
            warm_start=warm_start,
            name=name,
            pin_null_arm=pin_null_arm,
        )

    def _make_pooled(self) -> GaussianThompsonSampling:
        return GaussianThompsonSampling(
            self.arm_space,
            cfg=dataclasses.replace(self.cfg, warm_start="flat"),
            name=f"{self.name}:pooled",
            pin_null_arm=self.pin_null_arm,
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

    def simpsons_check(self, observed_only: bool = True) -> pd.DataFrame:
        """Per group: pooled-best vs group-best arm by posterior mean, and the mean gap.

        Columns: ``group, n_group_obs, pooled_best_arm, group_best_arm, pooled_best_mu,
        group_best_mu, group_mu_of_pooled_best, group_n_of_pooled_best, gap, reversal``.
        See :meth:`ArmSetPolicy.simpsons_check`.
        """
        return super().simpsons_check(observed_only=observed_only)
