"""Counterfactual shadow lending engine (SPEC §2.10; proposal §3.6, G.4, Table 8).

The shadow engine runs *alongside* the historical lending policy: for a borrower context
``X`` it evaluates every candidate policy ``A_k = (L_k, r_k, term)`` of the arm space under
the posterior of the Two-Stage Beta-Beta model (§3.3) and the MCTM default timing, and
recommends the arm with the highest risk-adjusted expected reward ``mean − λ·std`` among
the admissible arms. Against the loan's historical policy (Table 8: "historical policy used
as the observed reference") it reports the expected uplift, the uplift against the
*realized* historical reward where the loan has resolved, and whether the borrower is
*lendable* at all (some admissible lending arm with ``P(reward > 0) ≥ lendable_threshold``).

Per borrower group ``g`` the engine precomputes, once, a table over all arms
(:meth:`ShadowLendingEngine.group_table`):

* closed form (``CompoundRewardFunction.expected``, with the prepayment factor
  ``realization`` = ρ(grade(g), term) when ``RewardConfig.prepayment_adjustment`` is on):
  ``reward_mean``, ``reward_std``, ``risk_adjusted``; Stage-1 ``p_default_mean`` + 95 % HDI;
  Stage-2 ``expected_recovery``;
* Monte Carlo over the posterior predictive (``n_samples`` draws of ``(p, R)`` per arm, all
  arms in one vectorised ``rng.beta`` call): ``reward_q05 / q50 / q95`` and
  ``prob_positive = P(reward > 0)``. The generator is seeded per group
  (``default_rng([seed, crc32(g)])``) so tables do not depend on evaluation order.

**[decision] Admissible arms.** The lending guardrails (``ArmSpace.eligible``: payment-to-
income, amount-to-income, total DTI and the rate corridor around the borrower's reference
rate, :class:`~bayesian_credit.data.reference_rates.ReferenceRates`) constrain
*counterfactual* offers; the loan's historical policy is always admissible (it was actually
extended). The recommendation therefore maximises the score over ``eligible ∪
{historical arm}``, which makes ``uplift_expected = score(recommended) −
score(historical) ≥ 0`` by construction (with the default λ = 0 the score is the expected
reward, so the uplift is the expected-profit uplift).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.arms import ArmSpace, LendingPolicy
from ..data.reference_rates import ReferenceRates
from ..data.schema import BorrowerContext, LeakageGuard
from ..data.segmentation import Segmenter
from ..evaluation.policy import arm_mix
from ..models.hierarchical import beta_hdi
from ..models.mctm import MarkovChainTransitionMatrix
from ..models.two_stage import TwoStageBetaBeta
from ..rewards.compound import CompoundRewardFunction
from ..rewards.realization import InterestRealization, RealizationLookup
from ..types import Stage

__all__ = [
    "ARM_TABLE_COLUMNS",
    "OUTCOME_CLASSES",
    "PER_LOAN_COLUMNS",
    "RISK_AVERSION_SWEEP",
    "PortfolioSummary",
    "ShadowLendingEngine",
    "ShadowReport",
    "eligibility_matrix",
]

#: Columns of :meth:`ShadowLendingEngine.group_table` (one row per arm, grid order).
ARM_TABLE_COLUMNS = (
    "arm",
    "amount",
    "rate",
    "term",
    "is_null",
    "installment",
    "label",
    "p_default_mean",
    "p_default_hdi_low",
    "p_default_hdi_high",
    "expected_recovery",
    "reward_mean",
    "reward_std",
    "reward_q05",
    "reward_q50",
    "reward_q95",
    "prob_positive",
    "risk_adjusted",
    "realization",
)
#: Risk-aversion values of the portfolio's ``by_risk_aversion`` sweep (λ in ``mean − λ·std``).
RISK_AVERSION_SWEEP: tuple[float, ...] = (0.0, 0.5, 1.0)

#: Per-loan columns of :meth:`ShadowLendingEngine.evaluate_portfolio`.
PER_LOAN_COLUMNS = (
    "loan_id",
    "group",
    "grade",
    "historical_arm",
    "historical_expected",
    "historical_realized",
    "recommended_arm",
    "recommended_expected",
    "uplift_expected",
    "lendable",
    "declined",
    "rescued",
    "upsized",
    "downsized",
    "repriced_up",
    "repriced_down",
    "outcome_class",
    "reference_rate",
    "n_eligible_lending",
)

#: Mutually exclusive outcome classes, in precedence order.
OUTCOME_CLASSES = ("declined", "rescued", "upsized", "downsized", "repriced_down", "repriced_up", "unchanged")

_CHUNK = 25_000
_MONEY_DECIMALS = 2
_PROB_DECIMALS = 6


def eligibility_matrix(arm_space: ArmSpace, annual_inc: Any, dti: Any = None, reference_rate: Any = None) -> np.ndarray:
    """Vectorised :meth:`ArmSpace.eligible` for many borrowers: ``(n, n_arms)`` bool
    (same rule: :meth:`ArmSpace.eligible_many`). ``dti`` (percent) and ``reference_rate``
    are optional aligned sequences; a missing input skips its guardrail."""
    return arm_space.eligible_many(
        pd.Series(annual_inc) if not isinstance(annual_inc, (pd.Series, np.ndarray)) else annual_inc,
        dti,
        reference_rate,
    )


def _r(x: Any, nd: int) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _json_clean(value: Any) -> Any:
    """Recursively round floats (6 decimals), NaN → None, numpy → python."""
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _r(value, 6)
    return value


def _policy_of(value: Any) -> LendingPolicy | None:
    if value is None:
        return None
    if isinstance(value, LendingPolicy):
        return value
    if isinstance(value, Mapping):
        return LendingPolicy(float(value["amount"]), float(value["rate"]), int(value["term"]))
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return LendingPolicy(float(value[0]), float(value[1]), int(value[2]))
    raise TypeError(f"cannot interpret {value!r} as a lending policy (amount, rate, term)")


# ======================================================================================
# Reports
# ======================================================================================
@dataclass
class ShadowReport:
    """Counterfactual evaluation of one borrower (SPEC §2.10; Table 8)."""

    group: str
    arms: pd.DataFrame
    eligible_mask: np.ndarray
    recommended_arm: int
    recommended_policy: LendingPolicy
    lendable: bool
    historical_arm: int | None = None
    historical_policy: LendingPolicy | None = None
    historical_expected_reward: float | None = None
    historical_realized_reward: float | None = None
    uplift_expected: float | None = None
    uplift_vs_realized: float | None = None
    loan_id: Any = None
    issue_d: Any = None
    context: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0
    risk_aversion: float = 0.0
    lendable_threshold: float = 0.6
    max_payment_to_income: float | None = 0.20
    reference_rate: float | None = None
    eligibility: dict[str, Any] = field(default_factory=dict)

    @property
    def recommended(self) -> pd.Series:
        return self.arms.iloc[self.recommended_arm]

    @property
    def declined(self) -> bool:
        return bool(self.recommended_policy.is_null)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (money rounded to cents, probabilities to 6 decimals)."""
        money = {"amount", "installment", "reward_mean", "reward_std", "reward_q05", "reward_q50", "reward_q95", "risk_adjusted"}
        rows = []
        for rec in self.arms.to_dict("records"):
            row: dict[str, Any] = {}
            for k, v in rec.items():
                if k in ("arm", "term"):
                    row[k] = int(v)
                elif k in ("is_null", "eligible"):
                    row[k] = bool(v)
                elif k == "label":
                    row[k] = str(v)
                else:
                    row[k] = _r(v, _MONEY_DECIMALS if k in money else _PROB_DECIMALS)
            rows.append(row)
        ctx = {}
        for k, v in self.context.items():
            if isinstance(v, (float, np.floating)):
                ctx[k] = None if not math.isfinite(float(v)) else float(v)
            elif isinstance(v, (np.integer,)):
                ctx[k] = int(v)
            else:
                ctx[k] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
        issue = self.issue_d
        return {
            "loan_id": None if self.loan_id is None else (int(self.loan_id) if isinstance(self.loan_id, (int, np.integer)) else self.loan_id),
            "issue_d": None if issue is None or pd.isna(issue) else pd.Timestamp(issue).strftime("%Y-%m"),
            "group": self.group,
            "context": ctx,
            "context_columns": sorted(self.context),
            "leakage_safe": True,
            "n_samples": int(self.n_samples),
            "risk_aversion": float(self.risk_aversion),
            "lendable_threshold": float(self.lendable_threshold),
            "max_payment_to_income": None if self.max_payment_to_income is None else float(self.max_payment_to_income),
            "eligibility": {
                "reference_rate": _r(self.reference_rate, _PROB_DECIMALS),
                "n_eligible": int(np.count_nonzero(self.eligible_mask)),
                **{k: (_r(v, _PROB_DECIMALS) if isinstance(v, (float, np.floating)) else v) for k, v in self.eligibility.items()
                   if k not in ("reference_rate", "n_eligible")},
            },
            "arms": rows,
            "recommended_arm": int(self.recommended_arm),
            "recommended_policy": self.recommended_policy.to_dict(),
            "recommended_label": self.recommended_policy.label(),
            "declined": self.declined,
            "historical_policy": None if self.historical_policy is None else self.historical_policy.to_dict(),
            "historical_arm": None if self.historical_arm is None else int(self.historical_arm),
            "historical_expected_reward": _r(self.historical_expected_reward, _MONEY_DECIMALS),
            "historical_realized_reward": _r(self.historical_realized_reward, _MONEY_DECIMALS),
            "uplift_expected": _r(self.uplift_expected, _MONEY_DECIMALS),
            "uplift_vs_realized": _r(self.uplift_vs_realized, _MONEY_DECIMALS),
            "lendable": bool(self.lendable),
        }


@dataclass
class PortfolioSummary:
    """Portfolio-level counterfactual comparison (SPEC §2.10; G.4 counterfactual
    evaluation, financial-inclusion metrics)."""

    n_loans: int
    n_lendable: int
    expected_profit_historical: float
    expected_profit_shadow: float
    uplift: float
    uplift_risk_adjusted: float
    realized_profit_historical: float | None
    n_realized: int
    counts: dict[str, int]
    by_grade: pd.DataFrame
    by_group: pd.DataFrame
    inclusion: pd.DataFrame
    top_loans: pd.DataFrame
    risk_aversion: float = 0.0
    n_samples: int = 0
    lendable_threshold: float = 0.6
    seconds: float = 0.0
    recommendation_mix: dict[str, Any] = field(default_factory=dict)
    eligibility: dict[str, Any] = field(default_factory=dict)
    uplift_decomposition: dict[str, Any] = field(default_factory=dict)
    by_risk_aversion: dict[str, Any] = field(default_factory=dict)

    @property
    def share_lendable(self) -> float:
        return self.n_lendable / self.n_loans if self.n_loans else float("nan")

    @property
    def uplift_pct(self) -> float:
        base = abs(self.expected_profit_historical)
        return 100.0 * self.uplift / base if base > 0 else float("nan")

    @staticmethod
    def _frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        out = []
        for rec in df.to_dict("records"):
            row = {}
            for k, v in rec.items():
                if isinstance(v, dict):
                    row[k] = {kk: int(vv) for kk, vv in v.items()}
                elif isinstance(v, (bool, np.bool_)):
                    row[k] = bool(v)
                elif isinstance(v, (int, np.integer)):
                    row[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    row[k] = _r(v, 6)
                elif v is None or (isinstance(v, float) and math.isnan(v)):
                    row[k] = None
                else:
                    row[k] = str(v) if not isinstance(v, str) else v
            out.append(row)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_loans": int(self.n_loans),
            "n_lendable": int(self.n_lendable),
            "share_lendable": _r(self.share_lendable, 6),
            "expected_profit_historical": _r(self.expected_profit_historical, 2),
            "expected_profit_shadow": _r(self.expected_profit_shadow, 2),
            "uplift": _r(self.uplift, 2),
            "uplift_pct": _r(self.uplift_pct, 4),
            "uplift_risk_adjusted": _r(self.uplift_risk_adjusted, 2),
            "realized_profit_historical": _r(self.realized_profit_historical, 2),
            "n_realized": int(self.n_realized),
            "counts": {k: int(v) for k, v in self.counts.items()},
            "by_grade": self._frame_records(self.by_grade),
            "by_group": self._frame_records(self.by_group),
            "inclusion": self._frame_records(self.inclusion),
            "top_loans": self._frame_records(self.top_loans),
            "risk_aversion": float(self.risk_aversion),
            "n_samples": int(self.n_samples),
            "lendable_threshold": float(self.lendable_threshold),
            "seconds": round(float(self.seconds), 3),
            "recommendation_mix": _json_clean(self.recommendation_mix),
            "eligibility": _json_clean(self.eligibility),
            "uplift_decomposition": _json_clean(self.uplift_decomposition),
            "by_risk_aversion": _json_clean(self.by_risk_aversion),
        }


# ======================================================================================
# Engine
# ======================================================================================
class ShadowLendingEngine:
    """Shadow lending engine: per-borrower and portfolio counterfactuals (SPEC §2.10).

    ``ShadowLendingEngine(two_stage, mctm, reward_fn, arm_space, segmenter, n_samples=2000)``;
    ``evaluate(context, historical_policy=None, realized_reward=None, rng=None)`` →
    :class:`ShadowReport`; ``evaluate_loan(row)``; ``evaluate_portfolio(table, max_loans,
    seed)`` → ``(per_loan DataFrame, PortfolioSummary)`` (vectorised through the per-group
    arm tables; ~200k loans in a few seconds).
    """

    def __init__(
        self,
        two_stage: TwoStageBetaBeta,
        mctm: MarkovChainTransitionMatrix | None,
        reward_fn: CompoundRewardFunction,
        arm_space: ArmSpace,
        segmenter: Segmenter,
        n_samples: int = 2000,
        *,
        seed: int = 0,
        lendable_threshold: float | None = None,
        hdi_mass: float = 0.95,
        reference_rates: ReferenceRates | None = None,
        interest_realization: InterestRealization | None = None,
    ):
        if int(n_samples) < 10:
            raise ValueError("n_samples must be >= 10")
        self.reference_rates = reference_rates
        self.interest_realization = interest_realization
        self._realize = RealizationLookup.from_config(reward_fn.cfg, interest_realization, arm_space, segmenter)
        self.two_stage = two_stage
        self.mctm = mctm
        self.reward_fn = reward_fn
        self.arm_space = arm_space
        self.segmenter = segmenter
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.hdi_mass = float(hdi_mass)
        self.lendable_threshold = float(
            reward_fn.cfg.lendable_threshold if lendable_threshold is None else lendable_threshold
        )
        self.risk_aversion = float(reward_fn.cfg.risk_aversion)
        self._tables: dict[str, dict[str, np.ndarray]] = {}
        self._arm_frame = arm_space.to_frame()

    @classmethod
    def from_fitted(cls, fitted: Any, n_samples: int = 2000, seed: int = 0) -> ShadowLendingEngine:
        """Build from a :class:`~bayesian_credit.pipeline.FittedModels` (with its reference
        rates, when the fit has them)."""
        return cls(
            fitted.two_stage, fitted.mctm, fitted.reward_fn, fitted.arm_space, fitted.segmenter, n_samples, seed=seed,
            reference_rates=getattr(fitted, "reference_rates", None),
            interest_realization=getattr(fitted, "interest_realization", None),
        )

    def realization(self, group: str, arm: int) -> float:
        """Prepayment factor ``ρ(grade(group), term(arm))`` used by the arm tables."""
        return self._realize(group, arm)

    # ----- eligibility ---------------------------------------------------------------------------
    def reference_rate_of(self, context: BorrowerContext | Mapping[str, Any]) -> float | None:
        """The borrower's reference rate (``None`` without reference rates)."""
        if self.reference_rates is None:
            return None
        return self.reference_rates.for_context(context)

    def eligible_for(
        self, context: BorrowerContext | Mapping[str, Any], reference_rate: float | None = None
    ) -> tuple[np.ndarray, float | None]:
        """``(mask, reference_rate)``: the guardrail mask of one borrower (pre-decision
        ``annual_inc``, ``dti`` and the reference rate of its sub-grade)."""
        get = context.get if hasattr(context, "get") else (lambda k, d=None: d)
        ref = reference_rate if reference_rate is not None else self.reference_rate_of(context)
        inc, dti = get("annual_inc"), get("dti")
        mask = self.arm_space.eligible(
            None if inc is None else inc,
            dti=None if dti is None else dti,
            reference_rate=ref,
        )
        return mask, (None if ref is None or not math.isfinite(float(ref)) else float(ref))

    def clear_cache(self) -> None:
        self._tables.clear()

    # ----- per-group arm tables ----------------------------------------------------------------
    def _compute_arrays(self, group: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
        k = self.arm_space.n_arms
        policies = self.arm_space.policies
        lam = self.risk_aversion
        a_p, b_p, a_r, b_r = (np.ones(k) for _ in range(4))
        mean, std = np.zeros(k), np.zeros(k)
        fixed, kbc, c = np.zeros(k), np.zeros(k), np.zeros(k)
        rho_arr = np.ones(k)
        lend = np.array([not p.is_null for p in policies], dtype=bool)
        for a, pol in enumerate(policies):
            est = self.two_stage.risk(group, a)
            a_p[a], b_p[a] = est.p_default.alpha, est.p_default.beta
            a_r[a], b_r[a] = est.recovery.alpha, est.recovery.beta
            if pol.is_null:
                continue
            rho = self._realize(group, a)
            rho_arr[a] = rho
            dist = self.reward_fn.expected(pol, est, self.mctm, realization=rho)
            mean[a], std[a] = dist.mean, dist.std
            t = self.reward_fn.terms(pol, self.mctm)
            fixed[a], kbc[a], c[a] = rho * t.fixed, rho * t.interest_at_risk + t.exposure, t.exposure
        lo, hi = beta_hdi(a_p, b_p, self.hdi_mass)
        q05, q50, q95, ppos = (np.zeros(k) for _ in range(4))
        idx = np.flatnonzero(lend)
        if idx.size:
            n = self.n_samples
            p = rng.beta(a_p[idx, None], b_p[idx, None], size=(idx.size, n))
            r = rng.beta(a_r[idx, None], b_r[idx, None], size=(idx.size, n))
            rewards = fixed[idx, None] - p * (kbc[idx, None] - c[idx, None] * r)
            qs = np.quantile(rewards, [0.05, 0.5, 0.95], axis=1)
            q05[idx], q50[idx], q95[idx] = qs[0], qs[1], qs[2]
            ppos[idx] = (rewards > 0.0).mean(axis=1)
        return {
            "p_default_mean": a_p / (a_p + b_p),
            "p_default_hdi_low": lo,
            "p_default_hdi_high": hi,
            "expected_recovery": a_r / (a_r + b_r),
            "reward_mean": mean,
            "reward_std": std,
            "reward_q05": q05,
            "reward_q50": q50,
            "reward_q95": q95,
            "prob_positive": ppos,
            "risk_adjusted": mean - lam * std,
            "realization": rho_arr,
        }

    def _group_rng(self, group: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(str(group).encode("utf-8"))])

    def group_arrays(self, group: str) -> dict[str, np.ndarray]:
        """Cached per-arm arrays of ``group`` (see :data:`ARM_TABLE_COLUMNS`)."""
        g = str(group)
        hit = self._tables.get(g)
        if hit is None:
            hit = self._compute_arrays(g, self._group_rng(g))
            for v in hit.values():
                v.setflags(write=False)
            self._tables[g] = hit
        return hit

    def group_table(self, group: str, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """All arms for ``group`` as a DataFrame (:data:`ARM_TABLE_COLUMNS`).

        With ``rng`` the Monte-Carlo columns are re-drawn from that generator (not cached).
        """
        arrays = self.group_arrays(group) if rng is None else self._compute_arrays(str(group), rng)
        df = self._arm_frame.loc[:, ["arm", "amount", "rate", "term", "is_null", "installment", "label"]].copy()
        for col in ARM_TABLE_COLUMNS[7:]:
            df[col] = np.asarray(arrays[col], dtype=float)
        return df

    # ----- single borrower -----------------------------------------------------------------------
    def _context(self, context: BorrowerContext | Mapping[str, Any]) -> BorrowerContext:
        if isinstance(context, BorrowerContext):
            return context
        feats = dict(context)
        loan_id = feats.pop("loan_id", None)
        LeakageGuard.assert_allowed(feats.keys(), Stage.PRE_DECISION)  # raises LeakageError
        return BorrowerContext(loan_id=loan_id, features=feats)

    def evaluate(
        self,
        context: BorrowerContext | Mapping[str, Any],
        historical_policy: LendingPolicy | Mapping[str, Any] | tuple | None = None,
        realized_reward: float | None = None,
        rng: np.random.Generator | None = None,
        *,
        historical_arm: int | None = None,
        reference_rate: float | None = None,
    ) -> ShadowReport:
        """Evaluate every arm for one borrower (pre-decision context only).

        ``historical_policy`` (amount, rate, term) is mapped onto its nearest arm unless
        ``historical_arm`` is given. ``rng`` re-draws the Monte-Carlo quantiles instead of
        using the cached group table. Eligibility applies the arm space's guardrails to the
        context's ``annual_inc`` / ``dti`` and the reference rate (``reference_rate``, else
        the engine's :class:`ReferenceRates` lookup of the context's sub-grade).
        """
        ctx = self._context(context)
        group = self.segmenter.key_for(ctx)
        table = self.group_table(group, rng)
        mask, ref = self.eligible_for(ctx, reference_rate)
        lending_mask = mask & self.arm_space.is_lending
        n_eligible_lending = int(np.count_nonzero(lending_mask))
        hist_pol = _policy_of(historical_policy)
        h: int | None = None
        if historical_arm is not None and 0 <= int(historical_arm) < self.arm_space.n_arms:
            h = int(historical_arm)
        elif hist_pol is not None:
            h = self.arm_space.nearest(hist_pol.amount, hist_pol.rate, hist_pol.term)
            if h < 0:
                h = None
        historical_admitted = False
        if h is not None:
            historical_admitted = not bool(mask[h])
            mask[h] = True
            if hist_pol is None:
                hist_pol = self.arm_space[h]
        score = table["risk_adjusted"].to_numpy(dtype=float)
        rec = int(np.argmax(np.where(mask, score, -np.inf)))
        mean = table["reward_mean"].to_numpy(dtype=float)
        ppos = table["prob_positive"].to_numpy(dtype=float)
        lending = ~table["is_null"].to_numpy(dtype=bool)
        lendable = bool(np.any(mask & lending & (ppos >= self.lendable_threshold)))
        hist_exp = float(mean[h]) if h is not None else None
        uplift = float(score[rec] - score[h]) if h is not None else None
        realized = None
        if realized_reward is not None:
            rv = float(realized_reward)
            realized = rv if math.isfinite(rv) else None
        table = table.assign(eligible=mask)
        return ShadowReport(
            group=group,
            arms=table,
            eligible_mask=mask,
            recommended_arm=rec,
            recommended_policy=self.arm_space[rec],
            lendable=lendable,
            historical_arm=h,
            historical_policy=hist_pol,
            historical_expected_reward=hist_exp,
            historical_realized_reward=realized,
            uplift_expected=uplift,
            uplift_vs_realized=None if realized is None else float(mean[rec] - realized),
            loan_id=ctx.loan_id,
            issue_d=ctx.issue_date,
            context=dict(ctx.features),
            n_samples=self.n_samples,
            risk_aversion=self.risk_aversion,
            lendable_threshold=self.lendable_threshold,
            max_payment_to_income=self.arm_space.max_payment_to_income,
            reference_rate=ref,
            eligibility={
                "reference_rate": ref,
                "n_eligible": int(np.count_nonzero(mask)),
                "n_eligible_lending": n_eligible_lending,
                "historical_admitted": historical_admitted,
                "guardrails": dict(self.arm_space.guardrails),
                "rate_corridor_applied": ref is not None and self.arm_space.rate_corridor is not None,
            },
        )

    def evaluate_loan(self, row: Mapping[str, Any] | pd.Series, rng: np.random.Generator | None = None) -> ShadowReport:
        """:meth:`evaluate` for a LoanTable row: the historical policy is the loan's
        ``(loan_amnt, int_rate, term)`` (arm ``y_historical_arm`` when present) and the
        realized reward ``reward_fn.terminal(row)`` (when the loan has resolved)."""
        ctx = BorrowerContext.from_row(row)
        hist = None
        get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
        if get("loan_amnt") is not None and get("int_rate") is not None and get("term") is not None:
            hist = LendingPolicy(float(get("loan_amnt")), float(get("int_rate")), int(get("term")))
        h = get("y_historical_arm")
        h_arm = None
        if h is not None and not pd.isna(h) and int(h) >= 0:
            h_arm = int(h)
        realized = None
        if get("y_realized_loss") is not None:
            rv = self.reward_fn.terminal(row)
            realized = rv if math.isfinite(rv) else None
        return self.evaluate(ctx, hist, realized, rng, historical_arm=h_arm)

    # ----- portfolio ---------------------------------------------------------------------------------
    def evaluate_portfolio(
        self,
        table: pd.DataFrame,
        max_loans: int | None = None,
        seed: int = 0,
        *,
        risk_aversions: tuple[float, ...] | list[float] | None = RISK_AVERSION_SWEEP,
    ) -> tuple[pd.DataFrame, PortfolioSummary]:
        """Shadow vs historical policy for every loan of ``table`` (vectorised).

        Returns the per-loan frame (:data:`PER_LOAN_COLUMNS`) and a :class:`PortfolioSummary`.
        ``max_loans`` draws a seeded random subset (``rng.choice`` without replacement).
        ``risk_aversions`` adds the ``by_risk_aversion`` sweep: the same arm tables and
        eligibility re-scored with ``mean − λ·std`` for each λ (``None`` / empty: no sweep).
        """
        import time

        t0 = time.perf_counter()
        df = table
        if max_loans is not None and len(df) > int(max_loans):
            pick = np.sort(np.random.default_rng(seed).choice(len(df), size=int(max_loans), replace=False))
            df = df.iloc[pick]
        df = df.reset_index(drop=True)
        n = len(df)
        k = self.arm_space.n_arms
        null = self.arm_space.null_arm
        groups = self.segmenter.assign(df).astype(str).to_numpy(dtype=object) if n else np.empty(0, dtype=object)
        codes, uniq = pd.factorize(pd.Series(groups, dtype=object))
        tabs = [self.group_arrays(g) for g in uniq]
        score = np.stack([t["risk_adjusted"] for t in tabs]) if tabs else np.zeros((0, k))
        mean = np.stack([t["reward_mean"] for t in tabs]) if tabs else np.zeros((0, k))
        std = np.stack([t["reward_std"] for t in tabs]) if tabs else np.zeros((0, k))
        ppos = np.stack([t["prob_positive"] for t in tabs]) if tabs else np.zeros((0, k))
        lams = [float(x) for x in (risk_aversions or ())]
        sweep_scores = [mean - lam * std for lam in lams]
        sweep = [{"profit": 0.0, "n_lend": 0, "amount": 0.0, "rate": 0.0, "n36": 0, "n_max_amount": 0, "n_max_rate": 0}
                 for _ in lams]

        if "y_historical_arm" in df.columns:
            hist = df["y_historical_arm"].to_numpy(dtype=float, na_value=np.nan)
            hist = np.where(np.isfinite(hist), hist, -1).astype(np.int64)
        else:
            hist = self.arm_space.nearest_many(
                df["loan_amnt"].to_numpy(dtype=float), df["int_rate"].to_numpy(dtype=float), df["term"].to_numpy(dtype=float)
            )
        valid_h = (hist >= 0) & (hist < k)
        h_safe = np.where(valid_h, hist, 0)
        inc = df["annual_inc"].to_numpy(dtype=float, na_value=np.nan) if "annual_inc" in df.columns else np.full(n, np.nan)
        dti = df["dti"].to_numpy(dtype=float, na_value=np.nan) if "dti" in df.columns else None
        ref = self.reference_rates.for_frame(df) if self.reference_rates is not None else None
        lending = self.arm_space.is_lending

        rec = np.zeros(n, dtype=np.int64)
        lendable = np.zeros(n, dtype=bool)
        n_elig = np.zeros(n, dtype=np.int64)
        # uplift decomposition: best admissible arm keeping the historical term and amount
        # (repricing only), keeping the historical term (repricing + resizing), unrestricted
        part = {"reprice": np.zeros(n), "reprice_resize": np.zeros(n)}
        terms_g, amounts_g = self.arm_space.terms, self.arm_space.amounts
        for lo in range(0, n, _CHUNK):
            hi = min(lo + _CHUNK, n)
            mask = self.arm_space.eligible_many(
                inc[lo:hi], None if dti is None else dti[lo:hi], None if ref is None else ref[lo:hi]
            )
            n_elig[lo:hi] = np.count_nonzero(mask & lending[None, :], axis=1)
            vh = np.flatnonzero(valid_h[lo:hi])
            mask[vh, hist[lo:hi][vh]] = True
            s = score[codes[lo:hi]]
            rec[lo:hi] = np.argmax(np.where(mask, s, -np.inf), axis=1)
            lendable[lo:hi] = np.any(mask & lending[None, :] & (ppos[codes[lo:hi]] >= self.lendable_threshold), axis=1)
            hh = h_safe[lo:hi]
            same_term = (terms_g[None, :] == terms_g[hh][:, None]) | ~valid_h[lo:hi, None]
            same_amount = (amounts_g[None, :] == amounts_g[hh][:, None]) | ~valid_h[lo:hi, None]
            null_col = np.zeros(k, dtype=bool)
            if null is not None:
                null_col[null] = True
            m_rows = mean[codes[lo:hi]]
            rows_l = np.arange(hi - lo)
            for key, keep in (("reprice", same_term & same_amount), ("reprice_resize", same_term)):
                pick = np.argmax(np.where(mask & (keep | null_col[None, :]), s, -np.inf), axis=1)
                part[key][lo:hi] = m_rows[rows_l, pick]
            for acc, sc in zip(sweep, sweep_scores, strict=True):
                pick = np.argmax(np.where(mask, sc[codes[lo:hi]], -np.inf), axis=1)
                lend_l = lending[pick]
                acc["profit"] += float(m_rows[rows_l, pick].sum())
                acc["n_lend"] += int(lend_l.sum())
                acc["amount"] += float(amounts_g[pick][lend_l].sum())
                acc["rate"] += float(self.arm_space.rates[pick][lend_l].sum())
                acc["n36"] += int((terms_g[pick][lend_l] == 36).sum())
                if self.arm_space.amount_grid:
                    acc["n_max_amount"] += int((amounts_g[pick][lend_l] >= max(self.arm_space.amount_grid)).sum())
                    acc["n_max_rate"] += int((self.arm_space.rates[pick][lend_l] >= max(self.arm_space.rate_grid) - 1e-12).sum())

        rows = np.arange(n)
        hist_exp = np.where(valid_h, mean[codes, h_safe], np.nan)
        rec_exp = mean[codes, rec]
        uplift = np.where(valid_h, score[codes, rec] - score[codes, h_safe], np.nan)
        has_outcomes = "y_realized_loss" in df.columns
        realized = self.reward_fn.terminal_many(df) if has_outcomes else np.full(n, np.nan)
        amounts, rates = self.arm_space.amounts, self.arm_space.rates
        declined = rec == (null if null is not None else -1)
        served = ~declined & valid_h
        rescued = (realized < 0) & ~declined & (rec_exp > 0)
        upsized = served & (amounts[rec] > amounts[h_safe]) & (rec_exp > 0)
        downsized = served & (amounts[rec] < amounts[h_safe])
        repriced_up = served & (rates[rec] > rates[h_safe])
        repriced_down = served & (rates[rec] < rates[h_safe])
        outcome = np.full(n, "unchanged", dtype=object)
        for name, flag in reversed(
            [("declined", declined), ("rescued", rescued), ("upsized", upsized), ("downsized", downsized),
             ("repriced_down", repriced_down), ("repriced_up", repriced_up)]
        ):
            outcome[flag] = name
        grade = df["grade"].astype(str).to_numpy(dtype=object) if "grade" in df.columns else np.full(n, "unknown", dtype=object)
        per_loan = pd.DataFrame(
            {
                "loan_id": df["loan_id"].to_numpy() if "loan_id" in df.columns else rows,
                "group": pd.array(groups, dtype="string"),
                "grade": pd.array(grade, dtype="string"),
                "historical_arm": hist,
                "historical_expected": hist_exp,
                "historical_realized": realized,
                "recommended_arm": rec,
                "recommended_expected": rec_exp,
                "uplift_expected": uplift,
                "lendable": lendable,
                "declined": declined,
                "rescued": rescued,
                "upsized": upsized,
                "downsized": downsized,
                "repriced_up": repriced_up,
                "repriced_down": repriced_down,
                "outcome_class": pd.array(outcome, dtype="string"),
                "reference_rate": np.full(n, np.nan) if ref is None else ref,
                "n_eligible_lending": n_elig,
            }
        )
        summary = self._summarise(per_loan, time.perf_counter() - t0)
        hist_total = float(np.nansum(hist_exp))
        base = abs(hist_total)

        def pct(total: float) -> float | None:
            return 100.0 * (total - hist_total) / base if base > 0 else None

        reprice, resize, full = (float(np.sum(part["reprice"])), float(np.sum(part["reprice_resize"])),
                                 float(np.nansum(rec_exp)))
        summary.uplift_decomposition = {
            "expected_profit_historical": hist_total,
            "repricing_only": {"expected_profit": reprice, "uplift": reprice - hist_total, "uplift_pct": pct(reprice)},
            "repricing_and_resizing": {"expected_profit": resize, "uplift": resize - hist_total, "uplift_pct": pct(resize)},
            "unrestricted": {"expected_profit": full, "uplift": full - hist_total, "uplift_pct": pct(full)},
            "note": "best admissible arm keeping the historical term and amount / the historical term / unrestricted",
        }
        by_lam: dict[str, Any] = {}
        for lam, acc in zip(lams, sweep, strict=True):
            nl = acc["n_lend"]
            by_lam[str(lam)] = {
                "risk_aversion": lam,
                "expected_profit_shadow": acc["profit"],
                "expected_profit_historical": hist_total,
                "uplift": acc["profit"] - hist_total,
                "uplift_pct": pct(acc["profit"]),
                "mean_amount_recommended": acc["amount"] / nl if nl else None,
                "mean_rate_recommended": acc["rate"] / nl if nl else None,
                "share_36m_recommended": acc["n36"] / nl if nl else None,
                "share_at_max_amount_recommended": acc["n_max_amount"] / nl if nl else None,
                "share_at_max_rate_recommended": acc["n_max_rate"] / nl if nl else None,
                "decline_share": 1.0 - nl / n if n else None,
            }
        summary.by_risk_aversion = by_lam
        return per_loan, summary

    def recommendation_mix(self, per_loan: pd.DataFrame) -> dict[str, Any]:
        """Where the recommendations land, next to the historical policy: shares by term,
        rate bucket and amount bucket (:func:`~bayesian_credit.evaluation.policy.arm_mix`),
        mean rate / amount, share at the top of the grid and the gap to the reference rate."""
        amounts, rates, terms = self.arm_space.amounts, self.arm_space.rates, self.arm_space.terms
        n = len(per_loan)
        max_rate = max(self.arm_space.rate_grid) if self.arm_space.rate_grid else None
        max_amount = max(self.arm_space.amount_grid) if self.arm_space.amount_grid else None
        rec = per_loan["recommended_arm"].to_numpy(dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
        declined = per_loan["declined"].to_numpy(dtype=bool) if n else np.zeros(0, dtype=bool)
        hist = per_loan["historical_arm"].to_numpy(dtype=np.int64) if n else np.zeros(0, dtype=np.int64)
        valid_h = (hist >= 0) & (hist < self.arm_space.n_arms)
        h = np.where(valid_h, hist, 0)
        recommended = arm_mix(amounts[rec], rates[rec], terms[rec], ~declined, max_rate=max_rate, max_amount=max_amount)
        historical = arm_mix(amounts[h][valid_h], rates[h][valid_h], terms[h][valid_h], max_rate=max_rate, max_amount=max_amount)
        ref = per_loan["reference_rate"].to_numpy(dtype=float) if "reference_rate" in per_loan.columns else np.full(n, np.nan)
        use = ~declined & np.isfinite(ref)
        gap = rates[rec][use] - ref[use]
        hist_use = valid_h & np.isfinite(ref)
        hist_gap = rates[h][hist_use] - ref[hist_use]
        corridor = self.arm_space.rate_corridor
        return {
            "recommended": recommended,
            "historical": historical,
            "mean_rate_recommended": recommended["mean_rate"],
            "mean_rate_historical": historical["mean_rate"],
            "mean_amount_recommended": recommended["mean_amount"],
            "mean_amount_historical": historical["mean_amount"],
            "share_36m_recommended": recommended["share_36m"],
            "share_36m_historical": historical["share_36m"],
            "share_at_max_rate_recommended": recommended["share_at_max_rate"],
            "share_at_max_amount_recommended": recommended["share_at_max_amount"],
            "reference": {
                "n": int(use.sum()),
                "mean_reference_rate": float(ref[use].mean()) if use.any() else None,
                "mean_rate_minus_reference": float(gap.mean()) if gap.size else None,
                "share_above_reference": float(np.mean(gap > 1e-9)) if gap.size else None,
                "share_within_corridor": float(np.mean(np.abs(gap) <= corridor + 1e-9)) if gap.size and corridor is not None else None,
                "historical_mean_rate_minus_reference": float(hist_gap.mean()) if hist_gap.size else None,
            },
        }

    def _summarise(self, pl: pd.DataFrame, seconds: float) -> PortfolioSummary:
        amounts, rates = self.arm_space.amounts, self.arm_space.rates
        n = len(pl)
        realized = pl["historical_realized"].to_numpy(dtype=float)
        known = np.isfinite(realized)
        counts = {c: int((pl["outcome_class"] == c).sum()) for c in OUTCOME_CLASSES}
        flags = {f"n_{c}": int(pl[c].sum()) for c in ("declined", "rescued", "upsized", "downsized", "repriced_up", "repriced_down")}
        counts.update(flags)
        work = pl.assign(
            _hist_amount=amounts[pl["historical_arm"].clip(lower=0).to_numpy()],
            _hist_rate=rates[pl["historical_arm"].clip(lower=0).to_numpy()],
            _rec_amount=np.where(pl["declined"], np.nan, amounts[pl["recommended_arm"].to_numpy()]),
            _rec_rate=np.where(pl["declined"], np.nan, rates[pl["recommended_arm"].to_numpy()]),
            _realized=pl["historical_realized"],
        )

        def per(key: str) -> pd.DataFrame:
            if n == 0:
                return pd.DataFrame()
            g = work.groupby(key, sort=True, dropna=False)
            out = g.agg(
                n=("loan_id", "size"),
                expected_profit_historical=("historical_expected", "sum"),
                expected_profit_shadow=("recommended_expected", "sum"),
                uplift_risk_adjusted=("uplift_expected", "sum"),
                realized_profit_historical=("_realized", lambda s: float(s.sum()) if s.notna().any() else np.nan),
                n_realized=("_realized", "count"),
                share_lendable=("lendable", "mean"),
                decline_rate_shadow=("declined", "mean"),
            ).reset_index()
            out["uplift"] = out["expected_profit_shadow"] - out["expected_profit_historical"]
            cls_counts = work.groupby([key, "outcome_class"], sort=True, dropna=False).size().unstack(fill_value=0)
            out["counts"] = [
                {c: int(cls_counts.loc[v, c]) if (v in cls_counts.index and c in cls_counts.columns) else 0 for c in OUTCOME_CLASSES}
                for v in out[key]
            ]
            return out

        by_grade = per("grade")
        by_group = per("group")
        if len(by_group):
            by_group = by_group.sort_values("n", ascending=False, kind="stable").reset_index(drop=True)
        if n:
            inclusion = (
                work.groupby("grade", sort=True, dropna=False)
                .agg(
                    n=("loan_id", "size"),
                    share_lendable=("lendable", "mean"),
                    decline_rate_shadow=("declined", "mean"),
                    mean_rate_historical=("_hist_rate", "mean"),
                    mean_rate_shadow=("_rec_rate", "mean"),
                    mean_amount_historical=("_hist_amount", "mean"),
                    mean_amount_shadow=("_rec_amount", "mean"),
                )
                .reset_index()
            )
            gap = (pl["recommended_expected"] - pl["historical_expected"]).abs()
            top = pl.assign(_gap=gap).sort_values("_gap", ascending=False, kind="stable").head(50)
            top = top.assign(
                historical_label=[self.arm_space[int(a)].label() if a >= 0 else "unmapped" for a in top["historical_arm"]],
                recommended_label=[self.arm_space[int(a)].label() for a in top["recommended_arm"]],
            ).drop(columns="_gap")
        else:
            inclusion = pd.DataFrame()
            top = pl.head(0)
        hist_total = float(np.nansum(pl["historical_expected"].to_numpy(dtype=float)))
        shadow_total = float(np.nansum(pl["recommended_expected"].to_numpy(dtype=float)))
        n_elig = pl["n_eligible_lending"].to_numpy(dtype=float) if "n_eligible_lending" in pl.columns else np.zeros(0)
        refs = pl["reference_rate"].to_numpy(dtype=float) if "reference_rate" in pl.columns else np.zeros(0)
        eligibility = {
            "guardrails": dict(self.arm_space.guardrails),
            "reference_rates": self.reference_rates is not None,
            "share_with_reference_rate": float(np.isfinite(refs).mean()) if refs.size else None,
            "mean_n_eligible_lending": float(n_elig.mean()) if n_elig.size else None,
            "share_single_fallback_arm": float(np.mean(n_elig <= 1)) if n_elig.size else None,
        }
        return PortfolioSummary(
            n_loans=n,
            n_lendable=int(pl["lendable"].sum()),
            expected_profit_historical=hist_total,
            expected_profit_shadow=shadow_total,
            uplift=shadow_total - hist_total,
            uplift_risk_adjusted=float(np.nansum(pl["uplift_expected"].to_numpy(dtype=float))),
            realized_profit_historical=float(realized[known].sum()) if known.any() else None,
            n_realized=int(known.sum()),
            counts=counts,
            by_grade=by_grade,
            by_group=by_group,
            inclusion=inclusion,
            top_loans=top,
            risk_aversion=self.risk_aversion,
            n_samples=self.n_samples,
            lendable_threshold=self.lendable_threshold,
            seconds=seconds,
            recommendation_mix=self.recommendation_mix(pl),
            eligibility=eligibility,
        )
