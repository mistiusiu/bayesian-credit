"""Baseline lending policies (SPEC §2.8; G.3 "evaluated against appropriate baselines").

All baselines share the per-group arm-set machinery of
:class:`~bayesian_credit.bandits.base.ArmSetPolicy` (contextual when given a
Segmenter, flat otherwise), so they get warm start, selection counts, ``arm_table``,
snapshots and freeze semantics for free, record every reward they are given, and always
respect the eligibility mask (the null arm is always eligible).

* :class:`EpsilonGreedy` — explore uniformly over eligible arms with probability ε,
  otherwise exploit the highest posterior mean (Chapelle & Li, 2011 comparison set).
* :class:`UCB1` — untried eligible arms first, then ``mu + c·σ·sqrt(2 ln t / m)``.
* :class:`Greedy` — always the highest posterior mean (no exploration).
* :class:`RandomPolicy` — uniform over eligible *lending* arms.
* :class:`HistoricalPolicy` — the loan's historically observed policy (Table 8 reference).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import BanditConfig
from ..data.arms import ArmSpace
from ..data.segmentation import Segmenter
from ..exceptions import ConfigurationError
from .arm_state import GaussianArmSet
from .base import (
    ArmPriorProvider,
    ArmSetPolicy,
    argmax_eligible,
    decline_arm,
    masked_scores,
    valid_arm,
)


class EpsilonGreedy(ArmSetPolicy):
    """ε-greedy (§2.2.2: the comparator of Chapelle & Li, 2011; G.3 baseline): with
    probability ε a uniform eligible arm (null arm included), else ``argmax_k μ_k`` over
    eligible arms (ties → lowest index).

    Scores are the posterior means (q-values); ``meta["explore"]`` flags exploration.
    Every call consumes one uniform draw (plus one integer draw when exploring).
    """

    default_name = "epsilon_greedy"

    def __init__(
        self,
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        *,
        cfg: BanditConfig | None = None,
        prior_provider: ArmPriorProvider | None = None,
        warm_start: str | None = None,
        name: str | None = None,
        pin_null_arm: bool = True,
        epsilon: float | None = None,
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
        self.epsilon = float(self.cfg.epsilon if epsilon is None else epsilon)
        if not 0.0 <= self.epsilon <= 1.0:
            raise ConfigurationError(f"epsilon must be in [0, 1], got {self.epsilon}")

    def _params(self) -> dict[str, Any]:
        return {"epsilon": self.epsilon}

    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        explore = bool(rng.random() < self.epsilon)
        if explore:
            arm = int(idx[int(rng.integers(idx.size))])
        else:
            arm = argmax_eligible(arms.mu, idx)
        return arm, masked_scores(arms.mu, mask), {"explore": explore}


class UCB1(ArmSetPolicy):
    """UCB1 (§2.2.1 "Upper Confidence Bound methods"; Auer et al., 2002) adapted to
    delayed, currency-valued rewards.

    Untried eligible arms (no selections in the group) are played first, lowest index
    first. Otherwise the arm maximising ``μ_k + c·σ_k·sqrt(2·ln t / m_k)`` is played,
    where ``t`` is the number of decisions made in the group and ``m_k`` the number of
    times arm ``k`` was selected there. **[decision]** (i) The bonus is scaled by the
    arm's reward standard deviation σ_k (prior σ until two observations) because textbook
    UCB1 assumes rewards in [0, 1] whereas rewards here are in currency units. (ii) Play
    counts rather than observation weights size the bonus: feedback arrives months after
    the decision (and interim observations are fractional), and counting only observed
    rewards would replay the same pending arm for every borrower until its first reward
    arrives. With immediate unit-weight feedback both choices reduce to textbook UCB1.
    """

    default_name = "ucb1"

    def __init__(
        self,
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        *,
        cfg: BanditConfig | None = None,
        prior_provider: ArmPriorProvider | None = None,
        warm_start: str | None = None,
        name: str | None = None,
        pin_null_arm: bool = True,
        c: float | None = None,
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
        self.c = float(self.cfg.ucb_c if c is None else c)
        if self.c < 0:
            raise ConfigurationError(f"UCB exploration constant must be >= 0, got {self.c}")

    def _params(self) -> dict[str, Any]:
        return {"c": self.c}

    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        t_total = max(int(arms.selections.sum()), 1)
        index = arms.ucb(
            t_total,
            self.c,
            idx,
            counts=arms.selections[idx],
            scale_by_std=True,
            min_sigma=self.cfg.min_sigma,
        )
        scores = np.full(len(arms), np.nan)
        scores[idx] = index
        j = int(np.argmax(index))  # +inf for untried arms ⇒ lowest-index untried first
        return int(idx[j]), scores, {"untried": bool(np.isinf(index[j]))}


class Greedy(ArmSetPolicy):
    """Pure exploitation (§2.2.1 "deterministic greedy strategies"): ``argmax_k μ_k`` over
    eligible arms (ties → lowest index).

    With ``warm_start="risk_model"`` this is "act on the risk model's expected reward,
    then on observed means" — the no-exploration reference for regret comparisons.
    """

    default_name = "greedy"

    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, np.ndarray, dict[str, Any]]:
        return argmax_eligible(arms.mu, idx), masked_scores(arms.mu, mask), {}


class RandomPolicy(ArmSetPolicy):
    """Equal allocation (§2.2.1): uniform over eligible *lending* arms; the null arm only if
    nothing else is eligible.

    Excluding the decline arm makes this the "lend at random terms" reference rather than
    a policy that mostly declines. Consumes one integer draw per decision.
    """

    default_name = "random"

    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, None, dict[str, Any]]:
        null = self.arm_space.null_arm
        candidates = idx if null is None else idx[idx != null]
        if candidates.size == 0:
            candidates = idx
        return int(candidates[int(rng.integers(candidates.size))]), None, {}


class HistoricalPolicy(ArmSetPolicy):
    """The historically observed lending policy (``y_historical_arm``; Table 8 "historical
    policy used as the observed reference"; G.3 baseline).

    Returns the ``historical_arm`` passed to :meth:`select`; if it is missing (None, NaN,
    -1, out of range) or ineligible under the affordability guard, it declines (null
    arm). ``update`` records rewards in the arm table but never changes behaviour.
    """

    default_name = "historical"

    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, None, dict[str, Any]]:
        h = valid_arm(historical_arm, self.arm_space.n_arms)
        if h is not None and mask[h]:
            return h, None, {"historical_arm": h}
        meta = {"historical_arm": h, "historical_missing": h is None}
        return decline_arm(self.arm_space, mask), None, meta
