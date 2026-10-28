"""Bayesian environment of the replay simulation (SPEC §2.11; proposal §3.6 "Bayesian
simulation", G.3).

The environment is the simulation's model of the world: the Two-Stage Beta-Beta risk
posterior ``(p_{g,a}, R_{g,a})`` (§3.3), the MCTM default-timing model and the compound
reward ``R_t = Revenue_t − E[Loss]_t − Funding_t``. It answers three questions:

* **what is arm k worth for group g?** - :meth:`BayesianEnvironment.expected` (closed form,
  cached per ``(g, k)``) and :meth:`prior_provider` (the bandits' ``"risk_model"`` warm
  start: μ_k⁰, σ_k⁰ for every arm);
* **what reward does a decision deliver at resolution?** - :meth:`draw_terminal_reward`,
  one posterior-predictive draw ``(p, R) ~ (Beta, Beta)`` pushed through
  ``reward_fn.sample`` (feedback mode ``"simulated"``);
* **what does the borrower's repayment so far signal?** - :meth:`interim_reward`, the
  compound interim surrogate of ``reward_fn.interim`` for an observed credit state.

It also **learns from reality**: :meth:`observe_outcome` updates the risk posterior with
the *realized historical* outcome of every loan as it resolves (``y_group``,
``y_historical_arm``, ``y_default``, ``y_recovery_rate``), independently of what any policy
chose. A second instance fitted on *all* resolved loans (train + stream) is the
**hindsight** environment, used only as the oracle / regret yardstick.

Caching. ``risk``/``expected``/``interim`` values are pure functions of the model state and
are cached; every state change (:meth:`observe_outcomes`, :meth:`refresh`,
:meth:`resegment`, :meth:`restore`) clears the caches, so a cached value always equals a
fresh computation (needed for bit-for-bit resume). Two exact shortcuts keep the replay
fast (both are tested to reproduce the reference implementations bit for bit):

* :meth:`risk` reuses per-group building blocks: Stage 1 is the hierarchy's own
  ``posterior(g, a)`` (leaf priors are arm-specific under the structured leaf prior, so
  there is nothing to share across the arms of a group), and a leaf with fewer than
  ``min_obs`` recoveries resolves exactly like its ``(group, term)`` node
  (``BetaMoMRecovery`` fallback chain), which is resolved once per ``(group, term)``;
* with an MCTM, ``reward_fn.interim(...).mean`` depends on the risk model only through
  ``E[R]``: :meth:`interim_rewards` caches the risk-free parts per
  ``(arm, state, month, months_paid)`` across epochs and evaluates
  ``(realized + ρ·future) − exposure·q·(1 − E[R]) − ρ·funding`` in the same operation order.

**Prepayment.** With ``RewardConfig.prepayment_adjustment = "empirical"`` and an
:class:`~bayesian_credit.rewards.realization.InterestRealization` table (the reference fit's,
like the MCTM), every expected / sampled / interim reward of ``(g, a)`` uses the realisation
factor ``ρ(grade(g), term(a))`` (:meth:`realization`). ρ is a pure function of the group key
and the arm, so every ``(group, arm)`` cache stays exact; the hindsight yardstick uses the
same table, hence the same formula as the learning signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import FrameworkConfig
from ..data.arms import ArmSpace
from ..data.panel import LoanStatePanel
from ..data.segmentation import Segmenter
from ..models.mctm import MarkovChainTransitionMatrix
from ..models.two_stage import RiskEstimate, TwoStageBetaBeta
from ..rewards import amortization as amort
from ..rewards.compound import CompoundRewardFunction, RewardDistribution
from ..rewards.realization import InterestRealization, RealizationLookup
from ..types import State

__all__ = ["BayesianEnvironment", "MIN_PRIOR_SIGMA", "month_index_of"]

#: Floor on the warm-start σ_k⁰ (the decline arm's reward is exactly 0).
MIN_PRIOR_SIGMA = 1e-6


def month_index_of(values: Any) -> np.ndarray:
    """Calendar month index ``year·12 + month − 1`` (float; NaN for missing dates)."""
    dt = pd.to_datetime(pd.Series(values) if not isinstance(values, pd.Series) else values)
    out = (dt.dt.year * 12 + dt.dt.month - 1).to_numpy(dtype=float, na_value=np.nan)
    return out


class BayesianEnvironment:
    """Posterior-predictive reward generator with online risk updates (§3.3, §3.6, G.3).

    Parameters
    ----------
    two_stage:
        Fitted :class:`TwoStageBetaBeta` (default level ``p`` and recovery ``R``).
    mctm:
        :class:`MarkovChainTransitionMatrix` for default timing (``None`` → uniform timing).
    reward_fn:
        :class:`CompoundRewardFunction`; its ``cfg.risk_aversion`` λ defines :meth:`score`.
    arm_space:
        The :class:`ArmSpace` (arm 0 = decline, reward identically 0).
    segmenter:
        The segmentation the risk model's group keys follow (kept for bookkeeping and
        serialisation; callers pass group keys explicitly; also gives the grade of a group
        for the prepayment factor).
    interest_realization:
        :class:`~bayesian_credit.rewards.realization.InterestRealization` table (prepayment);
        ignored when ``reward_fn.cfg.prepayment_adjustment`` is ``None``.
    """

    def __init__(
        self,
        two_stage: TwoStageBetaBeta,
        mctm: MarkovChainTransitionMatrix | None,
        reward_fn: CompoundRewardFunction,
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        *,
        interest_realization: InterestRealization | None = None,
    ):
        self.two_stage = two_stage
        self.mctm = mctm
        self.reward_fn = reward_fn
        self.arm_space = arm_space
        self.segmenter = segmenter
        self.interest_realization = interest_realization
        self._realize = RealizationLookup.from_config(reward_fn.cfg, interest_realization, arm_space, segmenter)
        self.n_observed = 0
        self.n_refreshes = 0
        self._policies = list(arm_space.policies)
        self._null = arm_space.null_arm
        self._risk_cache: dict[tuple[str, int], RiskEstimate] = {}
        self._exp_cache: dict[tuple[str, int], RewardDistribution] = {}
        self._interim_cache: dict[tuple[str, int, int, int, int], float] = {}
        self._prior_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._r_fallback: dict[tuple[str, int | None], tuple[Any, str]] = {}
        self._er_cache: dict[tuple[str, int], float] = {}
        #: risk-independent interim parts per (arm, state, month, months_paid); survives
        #: invalidate() because it depends only on the policy schedule, the MCTM and cfg.
        self._parts_cache: dict[int, tuple[float, float, float]] = {}

    # ----- construction -----------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        table: pd.DataFrame,
        cfg: FrameworkConfig,
        *,
        arm_space: ArmSpace | None = None,
        segmenter: Segmenter | None = None,
        reward_fn: CompoundRewardFunction | None = None,
        mctm: MarkovChainTransitionMatrix | None = None,
        snapshot_date: Any = None,
        groups: Sequence[str] | pd.Series | None = None,
        interest_realization: InterestRealization | None = None,
    ) -> BayesianEnvironment:
        """Fit the risk model on the resolved rows of ``table`` (``y_*`` columns needed).

        ``groups`` overrides ``table["y_group"]`` (e.g. a refined segmentation). Without an
        ``mctm`` one is fitted on the reconstructed state panel of ``table``. The Stage-1
        structured leaf prior (policy effect) is fitted on the same rows over ``arm_space``;
        ``snapshot_date`` (default ``cfg.data.snapshot_date``) defines its elapsed-term loans.
        ``interest_realization`` is the prepayment table (not re-estimated here).
        """
        arm_space = arm_space or ArmSpace.from_config(cfg.arms)
        segmenter = segmenter or Segmenter.from_config(cfg.segmentation)
        reward_fn = reward_fn or CompoundRewardFunction(cfg.reward)
        data = table if groups is None else table.assign(y_group=pd.Series(groups, index=table.index).astype(str))
        snapshot = snapshot_date or cfg.data.snapshot_date
        two_stage = TwoStageBetaBeta.from_config(cfg, arm_space).fit(data, snapshot_date=snapshot)
        if mctm is None:
            panel = LoanStatePanel.from_table(table, cfg.mctm, snapshot)
            mctm = MarkovChainTransitionMatrix.from_config(cfg.mctm).fit_counts(panel.counts)
        return cls(two_stage, mctm, reward_fn, arm_space, segmenter, interest_realization=interest_realization)

    # ----- prepayment ---------------------------------------------------------------------------
    def realization(self, group: str, arm: int) -> float:
        """Prepayment factor ``ρ(grade(group), term(arm))`` (1.0 when disabled / no table)."""
        return self._realize(group, arm)

    # ----- caches -----------------------------------------------------------------------------
    def invalidate(self) -> None:
        """Drop every cached value (called after any model state change)."""
        self._risk_cache.clear()
        self._exp_cache.clear()
        self._interim_cache.clear()
        self._prior_cache.clear()
        self._r_fallback.clear()
        self._er_cache.clear()

    # ----- posterior quantities ------------------------------------------------------------------
    def risk(self, group: str, arm: int) -> RiskEstimate:
        """``RiskEstimate`` of ``(group, arm)`` - identical to ``two_stage.risk(g, a)`` (cached)."""
        key = (str(group), int(arm))
        hit = self._risk_cache.get(key)
        if hit is None:
            hit = self._compute_risk(key[0], key[1])
            self._risk_cache[key] = hit
        return hit

    def _compute_risk(self, g: str, a: int) -> RiskEstimate:
        """``TwoStageBetaBeta.risk`` with per-group reuse (see module docstring).

        Stage 1: the hierarchy's ``posterior(g, a)`` (the arm-specific, possibly tilted,
        leaf prior updated with the leaf counts). Stage 2: a leaf with fewer than
        ``min_obs`` recoveries falls through to the ``(group, term)`` chain, which is
        resolved once per ``(group, term of a)``.
        """
        dm = self.two_stage.default_model
        rm = self.two_stage.recovery_model
        p = dm.posterior(g, a)
        d, n = dm.counts(g, a)
        n_rec = rm.n_obs(g, a)
        if n_rec >= rm.cfg.min_obs:
            rec, src = rm.resolve(g, a)
        else:
            fkey = (g, rm.term_of(a))
            fb = self._r_fallback.get(fkey)
            if fb is None:
                fb = rm.resolve(g, a)  # leaf insufficient -> exactly the (group, term) chain's result
                self._r_fallback[fkey] = fb
            rec, src = fb
        return RiskEstimate(
            group=g, arm=a, p_default=p, recovery=rec, n_default_obs=d + n, n_recovery_obs=n_rec, recovery_source=src
        )

    def expected_recovery(self, group: str, arm: int) -> float:
        """``E[R | D = 1]`` of ``(group, arm)`` (cached)."""
        key = (str(group), int(arm))
        hit = self._er_cache.get(key)
        if hit is None:
            hit = self.risk(*key).recovery.mean
            self._er_cache[key] = hit
        return hit

    def expected(self, group: str, arm: int) -> RewardDistribution:
        """Closed-form reward distribution of ``arm`` for ``group`` (cached; decline → 0)."""
        key = (str(group), int(arm))
        hit = self._exp_cache.get(key)
        if hit is None:
            policy = self._policies[key[1]]
            if policy.is_null:
                hit = RewardDistribution.zero()
            else:
                hit = self.reward_fn.expected(policy, self.risk(*key), self.mctm, realization=self._realize(*key))
            self._exp_cache[key] = hit
        return hit

    def score(self, group: str, arm: int) -> float:
        """Risk-adjusted score ``mean − λ·std`` (λ = ``reward_fn.cfg.risk_aversion``)."""
        return self.expected(group, arm).score(self.reward_fn.cfg.risk_aversion)

    def expected_table(self, groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mean, std, score)`` arrays of shape ``(len(groups), n_arms)``."""
        k = self.arm_space.n_arms
        lam = self.reward_fn.cfg.risk_aversion
        mean = np.zeros((len(groups), k))
        std = np.zeros((len(groups), k))
        for i, g in enumerate(groups):
            for a in range(k):
                d = self.expected(g, a)
                mean[i, a] = d.mean
                std[i, a] = d.std
        return mean, std, mean - lam * std

    def prior_provider(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        """Warm start ``(μ_k⁰, σ_k⁰)`` for every arm (SPEC §2.8 ``warm_start="risk_model"``):
        the closed-form expected reward and its posterior std (floored at 1e-6)."""
        g = str(group)
        hit = self._prior_cache.get(g)
        if hit is None:
            k = self.arm_space.n_arms
            mu = np.empty(k)
            sigma = np.empty(k)
            for a in range(k):
                d = self.expected(g, a)
                mu[a] = d.mean
                sigma[a] = d.std
            hit = (mu, np.maximum(sigma, MIN_PRIOR_SIGMA))
            self._prior_cache[g] = hit
        return hit[0].copy(), hit[1].copy()

    # ----- rewards ---------------------------------------------------------------------------------
    def draw_terminal_reward(self, group: str, arm: int, rng: np.random.Generator) -> float:
        """One posterior-predictive terminal reward: ``(p, R)`` drawn from the Stage-1 / Stage-2
        posteriors of ``(group, arm)``, then ``reward_fn.sample(policy, p, R, mctm)``.

        The decline arm returns 0.0 without consuming random numbers.
        """
        policy = self._policies[int(arm)]
        if policy.is_null:
            return 0.0
        est = self.risk(group, arm)
        p = est.p_default.sample(rng)
        r = est.recovery.sample(rng)
        return float(self.reward_fn.sample(policy, p, r, self.mctm, realization=self._realize(group, arm)))

    def draw_terminal_rewards(
        self, groups: Sequence[str] | np.ndarray, arms: Sequence[int] | np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Vectorised :meth:`draw_terminal_reward` for many (group, arm) pairs.

        Draws ``p`` for all lending rows in one ``rng.beta`` call, then ``R`` in a second
        call (row order), so the stream consumed is a deterministic function of the input.
        Each row evaluates ``ρ·a − p·((ρ·b + c) − c·R)`` exactly like ``reward_fn.sample``.
        """
        arms = np.asarray(arms, dtype=np.int64)
        groups = np.asarray(groups, dtype=object)
        n = arms.size
        out = np.zeros(n)
        lend = np.flatnonzero(arms != (-1 if self._null is None else self._null))
        if lend.size == 0:
            return out
        a_p = np.empty(lend.size)
        b_p = np.empty(lend.size)
        a_r = np.empty(lend.size)
        b_r = np.empty(lend.size)
        fixed = np.empty(lend.size)
        k_br = np.empty(lend.size)
        c = np.empty(lend.size)
        terms_cache: dict[int, Any] = {}
        realize = self._realize
        for j, i in enumerate(lend.tolist()):
            a = int(arms[i])
            g = groups[i]
            est = self.risk(g, a)
            a_p[j], b_p[j] = est.p_default.alpha, est.p_default.beta
            a_r[j], b_r[j] = est.recovery.alpha, est.recovery.beta
            t = terms_cache.get(a)
            if t is None:
                t = self.reward_fn.terms(self._policies[a], self.mctm)
                terms_cache[a] = t
            rho = realize(g, a)
            fixed[j] = rho * t.fixed
            k_br[j] = rho * t.interest_at_risk + t.exposure
            c[j] = t.exposure
        p = rng.beta(a_p, b_p)
        r = rng.beta(a_r, b_r)
        out[lend] = fixed - p * (k_br - c * r)
        return out

    def interim_reward(self, group: str, arm: int, state: State | int, month: int, months_paid: int) -> float:
        """Mean of ``reward_fn.interim`` for a loan in ``state`` at age ``month`` after
        ``months_paid`` installments, under ``(group, arm)`` (cached; decline → 0)."""
        key = (str(group), int(arm), int(state), int(month), int(months_paid))
        hit = self._interim_cache.get(key)
        if hit is None:
            policy = self._policies[key[1]]
            if policy.is_null:
                hit = 0.0
            else:
                hit = float(
                    self.reward_fn.interim(
                        policy, State(key[2]), key[3], key[4], self.risk(key[0], key[1]), self.mctm,
                        realization=self._realize(key[0], key[1]),
                    ).mean
                )
            self._interim_cache[key] = hit
        return hit

    def _interim_parts(self, arm: int, state: int, month: int, months_paid: int) -> tuple[float, float, float, float]:
        """Risk-free parts of ``reward_fn.interim`` with an MCTM: ``(realized, future,
        exposure·q, funding)`` such that ``mean = (realized + ρ·future) − exposure·q·(1 − E[R])
        − ρ·funding`` (same expressions and operation order as ``CompoundRewardFunction.interim``)."""
        policy = self._policies[arm]
        L, r, T = float(policy.amount), float(policy.rate), int(policy.term)
        sched = amort.schedule_arrays(L, r, T)
        month_i = int(min(max(int(month), 0), T))
        paid = int(min(max(int(months_paid), 0), T))
        funding = self.reward_fn.terms(policy, self.mctm).funding_cost
        realized = float(sched.cum_interest[paid])
        exposure = float(sched.balances[paid])
        s = State(int(state))
        if s is State.RECOVERED:
            return realized, 0.0, 0.0, funding
        if s is State.LOSS:
            return realized, 0.0, exposure, funding
        remaining = T - month_i
        q = float(self.mctm.p_loss_given_state(s, remaining))  # type: ignore[union-attr]
        if remaining <= 0:
            gross_future, b_rem = 0.0, 0.0
        else:
            i_future = sched.interest[month_i:]
            gross_future = float(i_future.sum())
            b_rem = float(i_future @ self.reward_fn.timing_cdf(remaining, self.mctm, s))
        future = gross_future - q * b_rem
        return realized, future, exposure * q, funding

    def interim_rewards(
        self,
        groups: Sequence[str] | np.ndarray,
        arms: Sequence[int] | np.ndarray,
        states: Sequence[int] | np.ndarray,
        months: Sequence[int] | np.ndarray,
        months_paid: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Vectorised :meth:`interim_reward` (bit-identical; see module docstring)."""
        arms = np.asarray(arms, dtype=np.int64)
        n = arms.size
        if n == 0:
            return np.zeros(0)
        groups = np.asarray(groups, dtype=object)
        states = np.asarray(states, dtype=np.int64)
        months = np.asarray(months, dtype=np.int64)
        paid = np.asarray(months_paid, dtype=np.int64)
        if self.mctm is None:
            f = self.interim_reward
            return np.fromiter(
                (f(g, a, s, m, p) for g, a, s, m, p in zip(groups.tolist(), arms.tolist(), states.tolist(),
                                                            months.tolist(), paid.tolist(), strict=True)),
                dtype=float,
                count=n,
            )
        # risk-free parts per unique (arm, state, month, months_paid)
        code = ((arms * 8 + states) * 4096 + np.clip(months, -1, 4094) + 1) * 4096 + np.clip(paid, -1, 4094) + 1
        ucode, inv = np.unique(code, return_inverse=True)
        parts = np.empty((ucode.size, 4))
        cache = self._parts_cache
        for j, c in enumerate(ucode.tolist()):
            hit = cache.get(c)
            if hit is None:
                p_ = c % 4096 - 1
                rest = c // 4096
                m_ = rest % 4096 - 1
                rest //= 4096
                hit = self._interim_parts(rest // 8, rest % 8, m_, p_)
                cache[c] = hit
            parts[j] = hit
        # E[R] and the prepayment factor ρ per unique (group, arm)
        k = self.arm_space.n_arms
        gcodes, guniq = pd.factorize(pd.Series(groups, dtype=object))
        pair = gcodes.astype(np.int64) * k + arms
        upair, pinv = np.unique(pair, return_inverse=True)
        er_u = np.array([self.expected_recovery(guniq[c // k], c % k) for c in upair.tolist()], dtype=float)
        rho_u = np.array([self._realize(guniq[c // k], c % k) for c in upair.tolist()], dtype=float)
        rho = rho_u[pinv]
        realized, future, kq, fund = parts[inv, 0], parts[inv, 1], parts[inv, 2], parts[inv, 3]
        rf = realized + rho * future
        out = rf - kq * (1.0 - er_u[pinv]) - rho * fund
        if self._null is not None:
            out[arms == self._null] = 0.0
        return out

    # ----- learning from reality -----------------------------------------------------------------------
    def observe_outcome(self, row: Mapping[str, Any] | pd.Series, group: str | None = None) -> bool:
        """Update the risk posterior with one *realized historical* outcome.

        Uses ``y_historical_arm``, ``y_default`` and ``y_recovery_rate`` of ``row`` and its
        group (``group`` argument, else ``row["y_group"]``). Unresolved rows are ignored.
        Returns whether an update happened.
        """
        get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
        g = group if group is not None else get("y_group")
        changed = self._observe(g, get("y_historical_arm"), get("y_default"), get("y_recovery_rate"))
        if changed:
            self.invalidate()
        return changed

    def _observe(self, group: Any, arm: Any, default: Any, recovery: Any) -> bool:
        if group is None or arm is None or default is None:
            return False
        try:
            d = float(default)
            a = int(arm)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(d) or a < 0:
            return False
        rec = None
        if recovery is not None:
            try:
                rv = float(recovery)
                rec = rv if math.isfinite(rv) else None
            except (TypeError, ValueError):
                rec = None
        self.two_stage.update(str(group), a, d == 1.0, rec)
        self.n_observed += 1
        return True

    def observe_outcomes(
        self,
        groups: Sequence[str] | np.ndarray,
        arms: Sequence[int] | np.ndarray,
        defaults: Sequence[float] | np.ndarray,
        recovery_rates: Sequence[float] | np.ndarray | None = None,
    ) -> int:
        """Batch :meth:`observe_outcome` (row order); caches are cleared once. Returns the
        number of outcomes absorbed."""
        groups = np.asarray(groups, dtype=object)
        arms = np.asarray(arms)
        defaults = np.asarray(defaults, dtype=float)
        rec = np.full(defaults.size, np.nan) if recovery_rates is None else np.asarray(recovery_rates, dtype=float)
        n = 0
        for g, a, d, r in zip(groups.tolist(), arms.tolist(), defaults.tolist(), rec.tolist(), strict=True):
            n += self._observe(g, a, d, r)
        if n:
            self.invalidate()
        return n

    def observe_table(self, df: pd.DataFrame, groups: Sequence[str] | None = None) -> int:
        """:meth:`observe_outcomes` over a LoanTable slice."""
        g = df["y_group"].to_numpy(dtype=object) if groups is None else np.asarray(groups, dtype=object)
        return self.observe_outcomes(
            g,
            df["y_historical_arm"].to_numpy(),
            df["y_default"].to_numpy(dtype=float, na_value=np.nan),
            df["y_recovery_rate"].to_numpy(dtype=float, na_value=np.nan),
        )

    def refresh(self) -> None:
        """Re-derive the Stage-1 child priors from the current counts (H2 in docs/MATH.md)."""
        self.two_stage.refresh_priors()
        self.n_refreshes += 1
        self.invalidate()

    def resegment(self, segmenter: Segmenter, two_stage: TwoStageBetaBeta) -> None:
        """Adopt a refined segmentation together with a risk model re-fitted under it."""
        self.segmenter = segmenter
        self.two_stage = two_stage
        self._realize = self._realize.with_segmenter(segmenter)
        self.invalidate()

    # ----- serialisation ---------------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Plain-dict state (risk model, MCTM, segmenter, counters)."""
        return {
            "type": "BayesianEnvironment",
            "version": 1,
            "two_stage": self.two_stage.to_dict(),
            "mctm": None if self.mctm is None else self.mctm.to_dict(),
            "segmenter": None if self.segmenter is None else self.segmenter.to_dict(),
            "n_observed": int(self.n_observed),
            "n_refreshes": int(self.n_refreshes),
            "interest_realization": None if self.interest_realization is None else self.interest_realization.to_dict(),
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        self.two_stage = TwoStageBetaBeta.from_dict(snap["two_stage"])
        self.mctm = None if snap.get("mctm") is None else MarkovChainTransitionMatrix.from_dict(snap["mctm"])
        self.segmenter = None if snap.get("segmenter") is None else Segmenter.from_dict(snap["segmenter"])
        self.n_observed = int(snap.get("n_observed", 0))
        self.n_refreshes = int(snap.get("n_refreshes", 0))
        table = snap.get("interest_realization")
        self.interest_realization = None if not table else InterestRealization.from_dict(table)
        self._realize = RealizationLookup.from_config(
            self.reward_fn.cfg, self.interest_realization, self.arm_space, self.segmenter
        )
        self._parts_cache.clear()
        self.invalidate()

    @classmethod
    def from_snapshot(
        cls, snap: Mapping[str, Any], reward_fn: CompoundRewardFunction, arm_space: ArmSpace
    ) -> BayesianEnvironment:
        env = cls(TwoStageBetaBeta.from_dict(snap["two_stage"]), None, reward_fn, arm_space)
        env.restore(snap)
        return env

    def __repr__(self) -> str:
        return (
            f"BayesianEnvironment(observed={self.n_observed}, refreshes={self.n_refreshes}, "
            f"segmenter={self.segmenter!r}, risk={self.two_stage!r})"
        )
