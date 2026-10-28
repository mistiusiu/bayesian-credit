r"""Compound risk-adjusted reward (proposal §3.3, p. 29; SPEC §2.7).

    R_t    = Revenue_t − Expected Loss_t − Funding Cost_t
    E[L_t] = L_t · P(D | X, A_t) · (1 − E[R | X, A_t])

Closed form (:meth:`CompoundRewardFunction.expected`)
----------------------------------------------------
Let a policy be (L, r, T) with monthly schedule interest I_m and opening balances B_{m−1}
(m = 1..T), and let F(m) be the default-timing CDF (``mctm.default_timing_cdf(T)``, else the
uniform m/T — defaults happen mid-life on average). Define

    gross_interest = Σ_m I_m
    b              = Σ_m I_m · F(m)                 interest lost if the loan defaults: a
                                                    defaulter at month τ pays interest only
                                                    for m < τ, so the expected interest is
                                                    Σ_m I_m (1 − p·F(m))
    funding_cost   = funding_rate · Σ_m B_{m−1} / 12          (= funding_rate · L · WAL)
    ε              = 1                              exposure_basis = "full" (proposal:
                                                    E[L_t] = L_t·P(D)·(1 − E[R]))
                   = Σ_m f(m) B_{m−1} / L           exposure_basis = "outstanding": expected
                                                    outstanding fraction at default, f = pmf
    c              = ε · L
    a              = gross_interest − funding_cost

    Reward(p, R) = gross_interest − funding_cost − b·p − c·p·(1 − R) = a − p·h(R),
    h(R)         = (b + c) − c·R

With p ~ Beta (Stage 1 posterior) and R ~ Beta (Stage 2 posterior) independent:

    E[Reward]   = a − (b + c)·E[p] + c·E[p]·E[R]
    E[h]        = (b + c) − c·E[R]
    E[h²]       = (b + c)² − 2(b + c)·c·E[R] + c²·E[R²]
    Var[Reward] = Var[p·h(R)] = E[p²]·E[h²] − (E[p]·E[h])²

using E[p²], E[R²] = :attr:`BetaPosterior.second_moment`. The variance is over posterior
(parameter) uncertainty, which is what Thompson Sampling and the risk-adjusted score
``mean − λ·std`` need. The null policy (decline) has reward exactly 0 (mean 0, std 0).

Prepayment (``realization`` ρ, **[decision]**, docs/MATH.md §2.8)
-----------------------------------------------------------------
Borrowers who repay early pay only part of the scheduled interest and tie up capital for
less time. With ρ ∈ (0, 1] the realised share of the scheduled interest (estimated by
:class:`~bayesian_credit.rewards.realization.InterestRealization` per grade and term), the
expected-interest term and the funding cost are both multiplied by ρ::

    Revenue      = ρ · Σ_m I_m (1 − p·F(m)) = ρ·gross_interest − ρ·b·p
    Funding      = ρ · funding_cost
    Reward(p, R) = ρ·a − p·((ρ·b + c) − c·R),     i.e. a → ρ·a, b → ρ·b, c unchanged

so every formula above holds with ``(a, b) → (ρ·a, ρ·b)``. ``realization = 1`` (default) is
the contractual schedule; the realised ``terminal()`` reward is never adjusted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import RewardConfig
from ..models.mctm import MarkovChainTransitionMatrix
from ..models.two_stage import RiskEstimate
from ..types import State
from . import amortization as amort

__all__ = [
    "COMPONENT_KEYS",
    "CompoundRewardFunction",
    "RewardDistribution",
    "RewardTerms",
    "reward_moments",
]

#: Keys of :attr:`RewardDistribution.components` produced by ``expected()``.
COMPONENT_KEYS: tuple[str, ...] = (
    "gross_interest",
    "expected_interest",
    "expected_loss",
    "funding_cost",
    "exposure_factor",
)
_TERMS_CACHE_MAX = 250_000


@dataclass(frozen=True)
class RewardDistribution:
    """Posterior distribution summary of a policy's reward: mean, std and its components."""

    mean: float
    std: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def var(self) -> float:
        return self.std * self.std

    def score(self, risk_aversion: float = 0.0) -> float:
        """Risk-adjusted score ``mean − λ·std`` (λ = the "risk appetite" knob)."""
        return self.mean - risk_aversion * self.std

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "components": dict(self.components)}

    def __float__(self) -> float:
        return float(self.mean)

    @classmethod
    def zero(cls) -> RewardDistribution:
        return cls(0.0, 0.0, {k: 0.0 for k in COMPONENT_KEYS})


@dataclass(frozen=True)
class RewardTerms:
    """Policy-level constants of the closed form: ``gross_interest``, ``funding_cost``,
    ``interest_at_risk`` (b), ``exposure`` (c = ε·L) and ``exposure_factor`` (ε)."""

    gross_interest: float = 0.0
    funding_cost: float = 0.0
    interest_at_risk: float = 0.0
    exposure: float = 0.0
    exposure_factor: float = 0.0

    @property
    def fixed(self) -> float:
        """``a = gross_interest − funding_cost``."""
        return self.gross_interest - self.funding_cost


def _moments_scalar(
    a: float, b: float, c: float, p_mean: float, p_second: float, r_mean: float, r_second: float
) -> tuple[float, float]:
    k = b + c
    mean = a - k * p_mean + c * p_mean * r_mean
    e_h = k - c * r_mean
    e_h2 = k * k - 2.0 * k * c * r_mean + c * c * r_second
    var = p_second * e_h2 - (p_mean * e_h) ** 2
    return mean, (var if var > 0.0 else 0.0)


def reward_moments(
    a: Any, b: Any, c: Any, p_mean: Any, p_second: Any, r_mean: Any, r_second: Any
) -> tuple[Any, Any]:
    """Vectorised ``(E[Reward], Var[Reward])`` of ``a − p·((b + c) − c·R)`` with p ⫫ R.

    Inputs broadcast against each other; scalars in → floats out, arrays in → arrays of the
    broadcast shape out (see the module docstring for the algebra).
    """
    args = (a, b, c, p_mean, p_second, r_mean, r_second)
    if all(isinstance(v, (int, float)) for v in args):  # scalar fast path (hot loop)
        return _moments_scalar(*(float(v) for v in args))
    zero = np.zeros(np.broadcast_shapes(*(np.shape(v) for v in args)))
    k = np.add(b, c)
    mean = zero + (a - k * p_mean + c * p_mean * r_mean)
    e_h = k - c * r_mean
    e_h2 = k * k - 2.0 * k * c * r_mean + c * c * r_second
    var = zero + np.maximum(p_second * e_h2 - (p_mean * e_h) ** 2, 0.0)
    if zero.ndim == 0:
        return float(mean), float(var)
    return mean, var


def _realization(value: Any) -> float:
    """Validated prepayment factor ρ (finite, > 0)."""
    rho = float(value)
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError(f"realization must be a finite number > 0, got {value!r}")
    return rho


def _policy_params(policy: Any) -> tuple[float, float, int, bool]:
    L, r, T = float(policy.amount), float(policy.rate), int(policy.term)
    return L, r, T, (L <= 0 or T <= 0)


def _get(row: Mapping[str, Any] | pd.Series, key: str) -> float:
    value = row.get(key) if hasattr(row, "get") else None
    if value is None:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


class CompoundRewardFunction:
    """``R_t = Revenue_t − Expected Loss_t − Funding Cost_t`` with the closed form above.

    ``cfg.funding_rate`` (annual cost of funds), ``cfg.risk_aversion`` (λ in
    ``mean − λ·std``) and ``cfg.exposure_basis`` ("full" | "outstanding"). Per-policy
    constants (:class:`RewardTerms`) are cached on ``(L, r, T, mctm.fingerprint, cfg)``.
    ``expected`` / ``sample`` / ``interim`` take the prepayment factor ``realization`` (ρ,
    default 1); callers look it up with a
    :class:`~bayesian_credit.rewards.realization.RealizationLookup` when
    ``cfg.prepayment_adjustment == "empirical"``.
    """

    def __init__(self, cfg: RewardConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else RewardConfig()
        self.cfg.validate()
        self._terms_cache: dict[tuple[Any, ...], RewardTerms] = {}
        self._uniform_cache: dict[int, np.ndarray] = {}
        self._future_cache: dict[tuple[Any, ...], tuple[float, float]] = {}

    # ----- default timing ---------------------------------------------------------------
    def timing_cdf(
        self, term: int, mctm: MarkovChainTransitionMatrix | None = None, start: State | int = State.CURRENT
    ) -> np.ndarray:
        """``F(m)``, m = 1..term: ``mctm.default_timing_cdf(term, start)`` or uniform m/term."""
        term = int(term)
        if term < 1:
            raise ValueError("term must be >= 1")
        if mctm is not None:
            return mctm.default_timing_cdf(term, State(int(start)))
        F = self._uniform_cache.get(term)
        if F is None:
            F = np.arange(1, term + 1, dtype=float) / term
            F.setflags(write=False)
            self._uniform_cache[term] = F
        return F

    def timing_pmf(
        self, term: int, mctm: MarkovChainTransitionMatrix | None = None, start: State | int = State.CURRENT
    ) -> np.ndarray:
        return np.diff(self.timing_cdf(term, mctm, start), prepend=0.0)

    # ----- closed form --------------------------------------------------------------------
    def terms(self, policy: Any, mctm: MarkovChainTransitionMatrix | None = None) -> RewardTerms:
        """Cached :class:`RewardTerms` of a policy (all zero for the null policy)."""
        L, r, T, null = _policy_params(policy)
        if null:
            return RewardTerms()
        key = (L, r, T, None if mctm is None else mctm.fingerprint, self.cfg.funding_rate, self.cfg.exposure_basis)
        cached = self._terms_cache.get(key)
        if cached is not None:
            return cached
        sched = amort.schedule_arrays(L, r, T)
        F = self.timing_cdf(T, mctm)
        b = float(sched.interest @ F)
        funding = self.cfg.funding_rate * sched.sum_opening_balance / 12.0
        if self.cfg.exposure_basis == "outstanding":
            f = np.diff(F, prepend=0.0)
            eps = float(f @ sched.opening_balance) / L
        else:
            eps = 1.0
        terms = RewardTerms(
            gross_interest=sched.total_interest,
            funding_cost=funding,
            interest_at_risk=b,
            exposure=eps * L,
            exposure_factor=eps,
        )
        if len(self._terms_cache) >= _TERMS_CACHE_MAX:
            self._terms_cache.clear()
        self._terms_cache[key] = terms
        return terms

    def expected(
        self,
        policy: Any,
        risk: RiskEstimate,
        mctm: MarkovChainTransitionMatrix | None = None,
        realization: float = 1.0,
    ) -> RewardDistribution:
        """Closed-form reward distribution of ``policy`` under ``risk`` (see module docstring).

        ``realization`` = ρ multiplies the expected interest and the funding cost
        (prepayment); ``components["gross_interest"]`` stays the contractual schedule and
        ``components["realization"]`` reports ρ.
        """
        t = self.terms(policy, mctm)
        if _policy_params(policy)[3]:
            return RewardDistribution.zero()
        rho = _realization(realization)
        p, rec = risk.p_default, risk.recovery
        ep, er = p.mean, rec.mean
        mean, var = _moments_scalar(
            rho * t.fixed, rho * t.interest_at_risk, t.exposure, ep, p.second_moment, er, rec.second_moment
        )
        components = {
            "gross_interest": t.gross_interest,
            "expected_interest": rho * (t.gross_interest - t.interest_at_risk * ep),
            "expected_loss": t.exposure * ep * (1.0 - er),
            "funding_cost": rho * t.funding_cost,
            "exposure_factor": t.exposure_factor,
            "realization": rho,
        }
        return RewardDistribution(float(mean), float(math.sqrt(var)), components)

    def sample(
        self,
        policy: Any,
        p_draws: Any,
        R_draws: Any,
        mctm: MarkovChainTransitionMatrix | None = None,
        realization: float = 1.0,
    ) -> np.ndarray:
        """Monte Carlo rewards ``ρ·a − p·((ρ·b + c) − c·R)`` for paired draws (vectorised)."""
        p = np.asarray(p_draws, dtype=float)
        R = np.asarray(R_draws, dtype=float)
        if _policy_params(policy)[3]:
            return np.zeros(np.broadcast_shapes(p.shape, R.shape))
        rho = _realization(realization)
        t = self.terms(policy, mctm)
        return rho * t.fixed - p * ((rho * t.interest_at_risk + t.exposure) - t.exposure * R)

    def score(self, dist: RewardDistribution) -> float:
        """``mean − cfg.risk_aversion · std``."""
        return dist.score(self.cfg.risk_aversion)

    # ----- realised (terminal) reward -----------------------------------------------------
    def terminal(self, row: Mapping[str, Any] | pd.Series) -> float:
        """Realised historical reward of a resolved loan::

            y_realized_revenue − y_realized_loss − funding_rate · Σ_{m ≤ k} B_{m−1} / 12

        with the loan's own (funded_amnt [loan_amnt if missing], int_rate, term) schedule and
        ``k = y_months_on_book`` capped at the term — funding is charged on the balance-months
        actually on book (= ``funding_rate · L · WAL`` for a loan held to maturity). Returns
        NaN when a required field is missing.
        """
        L = _get(row, "funded_amnt")
        if not math.isfinite(L):
            L = _get(row, "loan_amnt")
        r, T = _get(row, "int_rate"), _get(row, "term")
        rev, loss = _get(row, "y_realized_revenue"), _get(row, "y_realized_loss")
        mob = _get(row, "y_months_on_book")
        if not all(math.isfinite(v) for v in (L, r, T, rev, loss, mob)):
            return math.nan
        T_i = int(round(T))
        balance_months = amort.cumulative_opening_balance(L, r, T_i, min(max(mob, 0.0), T_i))
        return float(rev - loss - self.cfg.funding_rate * balance_months / 12.0)

    def terminal_many(self, table: pd.DataFrame) -> np.ndarray:
        """Vectorised :meth:`terminal` over a LoanTable (closed-form balance sums)."""
        def col(name: str) -> np.ndarray:
            if name not in table.columns:
                return np.full(len(table), np.nan)
            return pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)

        L = col("funded_amnt")
        L = np.where(np.isfinite(L), L, col("loan_amnt"))
        r, T = col("int_rate"), np.round(col("term"))
        rev, loss, mob = col("y_realized_revenue"), col("y_realized_loss"), col("y_months_on_book")
        ok = np.isfinite(L) & np.isfinite(r) & np.isfinite(T) & np.isfinite(rev) & np.isfinite(loss) & np.isfinite(mob)
        out = np.full(len(table), np.nan)
        if ok.any():
            k = np.minimum(np.maximum(mob[ok], 0.0), T[ok])
            bm = amort.cumulative_opening_balance(L[ok], r[ok], T[ok], k)
            out[ok] = rev[ok] - loss[ok] - self.cfg.funding_rate * bm / 12.0
        return out

    # ----- interim (surrogate) reward ---------------------------------------------------
    def _interim_p_loss(
        self, state: State, remaining: int, risk: RiskEstimate, mctm: MarkovChainTransitionMatrix | None
    ) -> float:
        if mctm is not None:
            return mctm.p_loss_given_state(state, remaining)
        ep = risk.p_default.mean
        return ep if state is State.CURRENT else max(ep, 0.5)

    def _future_terms(
        self, L: float, r: float, T: int, month: int, state: State, mctm: MarkovChainTransitionMatrix | None
    ) -> tuple[float, float]:
        """Cached ``(Σ_{m>month} I_m, Σ_{m>month} I_m·F_rem(m − month))``."""
        key = (L, r, T, month, int(state), None if mctm is None else mctm.fingerprint)
        cached = self._future_cache.get(key)
        if cached is None:
            remaining = T - month
            if remaining <= 0:
                cached = (0.0, 0.0)
            else:
                i_future = amort.schedule_arrays(L, r, T).interest[month:]
                cached = (float(i_future.sum()), float(i_future @ self.timing_cdf(remaining, mctm, state)))
            if len(self._future_cache) >= _TERMS_CACHE_MAX:
                self._future_cache.clear()
            self._future_cache[key] = cached
        return cached

    def interim(
        self,
        policy: Any,
        state: State | int,
        month: int,
        months_paid: int,
        risk: RiskEstimate,
        mctm: MarkovChainTransitionMatrix | None = None,
        realization: float = 1.0,
    ) -> RewardDistribution:
        """Interim compound signal of a loan observed in ``state`` at loan age ``month``
        after ``months_paid`` installments (§1.7.2 "interim repayment behaviour").

        Let ``B = B_{months_paid}`` be the principal still outstanding (equal to ``B_month``
        for a borrower who is up to date), ``funding`` the full-life funding cost and
        ``realized = Σ_{m ≤ months_paid} I_m``.

        * RECOVERED: ``realized − funding`` (std 0).
        * LOSS: ``realized − B·(1 − E[R]) − funding`` (std ``B·std(R)``).
        * CURRENT / DELINQUENT: with ``q = mctm.p_loss_given_state(state, T − month)``
          (without an MCTM: ``q = E[p]`` for C and ``max(E[p], 0.5)`` for D) and ``F_rem`` the
          timing CDF over the remaining ``T − month`` months starting from ``state``::

              future   = Σ_{m > month} I_m (1 − q·F_rem(m − month))
              loss     = B · q · (1 − E[R])
              mean     = realized + future − loss − funding

          std: the closed form with ``b' = Σ_{m>month} I_m F_rem``, ``c' = B`` and the default
          probability ``p' ~ Beta(q·κ, (1 − q)·κ)`` (κ = α + β of the Stage-1 posterior, i.e.
          that posterior re-centred at q), independent of R.

        Interest missed between ``months_paid`` and ``month`` is not counted (conservative).
        At ``month = 0`` in C without an MCTM this equals :meth:`expected` for the "full"
        exposure basis.

        Prepayment: with ``realization`` = ρ the expected *future* interest
        ``Σ_{m > month} I_m (1 − q·F_rem)`` and the funding cost are multiplied by ρ (the
        interest already received is not), so at ``month = 0`` the identity with
        :meth:`expected` holds for every ρ.
        """
        L, r, T, null = _policy_params(policy)
        if null:
            return RewardDistribution.zero()
        rho = _realization(realization)
        s = State(int(state))
        sched = amort.schedule_arrays(L, r, T)
        month_i = int(min(max(int(month), 0), T))
        paid = int(min(max(int(months_paid), 0), T))
        funding = rho * self.terms(policy, mctm).funding_cost
        realized = float(sched.cum_interest[paid])
        exposure = float(sched.balances[paid])
        rec = risk.recovery
        er, er2 = rec.mean, rec.second_moment
        future = 0.0
        if s is State.RECOVERED:
            q, loss, var = 0.0, 0.0, 0.0
        elif s is State.LOSS:
            q = 1.0
            loss = exposure * (1.0 - er)
            var = exposure * exposure * rec.var
        else:
            remaining = T - month_i
            q = float(self._interim_p_loss(s, remaining, risk, mctm))
            gross_future, b_rem = self._future_terms(L, r, T, month_i, s, mctm)
            future = rho * (gross_future - q * b_rem)
            loss = exposure * q * (1.0 - er)
            kappa = risk.p_default.concentration
            q2 = q * (1.0 - q) / (kappa + 1.0) + q * q
            _, var = _moments_scalar(0.0, rho * b_rem, exposure, q, q2, er, er2)
        mean = realized + future - loss - funding
        components = {
            "gross_interest": sched.total_interest,
            "expected_interest": realized + future,
            "expected_loss": loss,
            "funding_cost": funding,
            "exposure_factor": exposure / L,
            "realized_interest": realized,
            "p_loss": q,
            "realization": rho,
        }
        return RewardDistribution(float(mean), float(math.sqrt(max(float(var), 0.0))), components)

    def __repr__(self) -> str:
        c = self.cfg
        return (
            f"CompoundRewardFunction(funding_rate={c.funding_rate}, risk_aversion={c.risk_aversion}, "
            f"exposure_basis={c.exposure_basis!r})"
        )
