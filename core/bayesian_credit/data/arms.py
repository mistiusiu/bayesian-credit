"""Lending policies and the arm space of the bandit.

§3.3: "A lending policy is defined as a combination of loan exposure and pricing
parameters A_k = (L_k, r_k) ... Let 𝒜 = {A_1, ..., A_K} denote the set of candidate
lending policies, where each policy corresponds to one arm". Table 7 adds loan term.
Arm 0 is the **null arm** (decline) so the system can choose *whether* to lend (§1.5).

**[decision] Lending guardrails** (``ArmSpaceConfig``; docs/MATH.md assumptions G1-G5).
An arm ``a = (L_a, r_a, T_a)`` is *eligible* for a borrower when every guardrail whose
input is known holds:

* affordability  ``installment_a ≤ max_payment_to_income · annual_inc / 12``  (0.20, the
  payment-to-income cap of unsecured instalment credit);
* exposure       ``L_a ≤ max_amount_to_income · annual_inc``                  (0.50);
* total debt     ``dti / 100 + installment_a · 12 / annual_inc ≤ max_total_dti`` (0.43, the
  Qualified-Mortgage DTI ceiling of 12 CFR 1026.43(e)(2)(vi); Lending Club's ``dti`` is in
  percent and excludes mortgage payments);
* pricing        ``|r_a − reference_rate| ≤ rate_corridor``                  (0.05), where the
  reference rate is a pure pre-decision function of the borrower's sub-grade
  (:class:`~bayesian_credit.data.reference_rates.ReferenceRates`).

See ``ArmSpaceConfig`` for the sources of the defaults.

The null arm is always eligible. If no lending arm survives, the single lending arm whose
rate is closest to the reference rate (any rate when it is unknown) with the smallest
installment is kept, so a minimal counter-offer always exists next to "decline".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import ArmSpaceConfig
from ..exceptions import ConfigurationError


def annuity_installment(amount: float, annual_rate: float, term_months: int) -> float:
    """Standard amortising monthly installment; 0 for the null policy."""
    if term_months <= 0 or amount <= 0:
        return 0.0
    m = annual_rate / 12.0
    if m <= 0:
        return amount / term_months
    f = (1 + m) ** term_months
    return amount * m * f / (f - 1)


@dataclass(frozen=True, order=True)
class LendingPolicy:
    """A_k = (L_k, r_k, term). ``rate`` is an annual fraction (0.13 == 13%)."""

    amount: float
    rate: float
    term: int

    @property
    def is_null(self) -> bool:
        return self.amount <= 0 or self.term <= 0

    @property
    def installment(self) -> float:
        return annuity_installment(self.amount, self.rate, self.term)

    def to_dict(self) -> dict[str, Any]:
        return {"amount": self.amount, "rate": self.rate, "term": self.term}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LendingPolicy:
        return cls(float(d["amount"]), float(d["rate"]), int(d["term"]))

    def label(self) -> str:
        if self.is_null:
            return "decline"
        return f"${self.amount/1000:g}k @ {self.rate*100:.0f}% / {self.term}m"


NULL_POLICY = LendingPolicy(0.0, 0.0, 0)

#: ``ArmSpaceConfig`` defaults of the eligibility guardrails (single source of truth).
_DEFAULT_GUARDRAILS: dict[str, float | None] = {
    name: getattr(ArmSpaceConfig, name)
    for name in ("max_payment_to_income", "rate_corridor", "max_amount_to_income", "max_total_dti")
}
#: Tolerance on rate comparisons (grid rates such as 0.07 + 0.05 are not exact in binary).
_RATE_TOL = 1e-9


def _guardrail(value: Any, name: str) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not v > 0 or not np.isfinite(v):
        raise ConfigurationError(f"{name} must be a positive number or None, got {value!r}")
    return v


def _as_float_vector(values: Any, n: int | None = None, name: str = "values") -> np.ndarray:
    """1-D float array (``None`` / non-numeric → NaN); broadcast a scalar to ``n``."""
    if values is None:
        out = np.full(1 if n is None else n, np.nan)
    else:
        if isinstance(values, pd.Series):
            ser = values
        elif isinstance(values, np.ndarray):
            ser = pd.Series(values.reshape(-1))
        elif np.isscalar(values):
            ser = pd.Series([values], dtype=object)
        else:
            ser = pd.Series(list(values), dtype=object)
        if not pd.api.types.is_numeric_dtype(ser.dtype) or pd.api.types.is_bool_dtype(ser.dtype):
            ser = pd.to_numeric(ser.astype(object), errors="coerce")
        out = ser.to_numpy(dtype=float, na_value=np.nan).reshape(-1)
    if n is not None:
        if out.size == 1 and n != 1:
            out = np.full(n, out[0])
        elif out.size != n:
            raise ValueError(f"{name} has {out.size} values, expected {n}")
    return out


class ArmSpace:
    """Finite set of candidate lending policies; index == arm id.

    Arms are stored in grid order (term-major, then amount, then rate) after the
    optional null arm at index 0, so selection-frequency heatmaps are trivial to build.
    """

    def __init__(
        self,
        policies: Sequence[LendingPolicy],
        max_payment_to_income: float | None = _DEFAULT_GUARDRAILS["max_payment_to_income"],
        *,
        rate_corridor: float | None = _DEFAULT_GUARDRAILS["rate_corridor"],
        max_amount_to_income: float | None = _DEFAULT_GUARDRAILS["max_amount_to_income"],
        max_total_dti: float | None = _DEFAULT_GUARDRAILS["max_total_dti"],
    ):
        if not policies:
            raise ConfigurationError("ArmSpace needs at least one policy")
        self._policies: tuple[LendingPolicy, ...] = tuple(policies)
        nulls = [i for i, p in enumerate(self._policies) if p.is_null]
        if len(nulls) > 1 or (nulls and nulls[0] != 0):
            raise ConfigurationError("at most one null policy is allowed and it must be arm 0")
        self.null_arm: int | None = 0 if nulls else None
        self.max_payment_to_income = _guardrail(max_payment_to_income, "max_payment_to_income")
        self.rate_corridor = _guardrail(rate_corridor, "rate_corridor")
        self.max_amount_to_income = _guardrail(max_amount_to_income, "max_amount_to_income")
        self.max_total_dti = _guardrail(max_total_dti, "max_total_dti")
        self.amounts = np.array([p.amount for p in self._policies], dtype=float)
        self.rates = np.array([p.rate for p in self._policies], dtype=float)
        self.terms = np.array([p.term for p in self._policies], dtype=int)
        self.installments = np.array([p.installment for p in self._policies], dtype=float)
        self.is_lending = np.array([not p.is_null for p in self._policies], dtype=bool)
        self.amount_grid = tuple(sorted({p.amount for p in self._policies if not p.is_null}))
        self.rate_grid = tuple(sorted({p.rate for p in self._policies if not p.is_null}))
        self.term_grid = tuple(sorted({p.term for p in self._policies if not p.is_null}))

    # ----- construction -------------------------------------------------------------
    @classmethod
    def from_grid(
        cls,
        amounts: Sequence[float],
        rates: Sequence[float],
        terms: Sequence[int],
        include_null: bool = True,
        max_payment_to_income: float | None = _DEFAULT_GUARDRAILS["max_payment_to_income"],
        *,
        rate_corridor: float | None = _DEFAULT_GUARDRAILS["rate_corridor"],
        max_amount_to_income: float | None = _DEFAULT_GUARDRAILS["max_amount_to_income"],
        max_total_dti: float | None = _DEFAULT_GUARDRAILS["max_total_dti"],
    ) -> ArmSpace:
        policies: list[LendingPolicy] = [NULL_POLICY] if include_null else []
        for t in sorted(set(int(x) for x in terms)):
            for a in sorted(set(float(x) for x in amounts)):
                for r in sorted(set(float(x) for x in rates)):
                    policies.append(LendingPolicy(a, r, t))
        return cls(
            policies,
            max_payment_to_income=max_payment_to_income,
            rate_corridor=rate_corridor,
            max_amount_to_income=max_amount_to_income,
            max_total_dti=max_total_dti,
        )

    @classmethod
    def from_config(cls, cfg: ArmSpaceConfig) -> ArmSpace:
        cfg.validate()
        return cls.from_grid(
            cfg.amounts,
            cfg.rates,
            cfg.terms,
            cfg.include_null,
            cfg.max_payment_to_income,
            rate_corridor=cfg.rate_corridor,
            max_amount_to_income=cfg.max_amount_to_income,
            max_total_dti=cfg.max_total_dti,
        )

    @property
    def guardrails(self) -> dict[str, float | None]:
        """The eligibility guardrails in force (``None`` = switched off)."""
        return {
            "max_payment_to_income": self.max_payment_to_income,
            "rate_corridor": self.rate_corridor,
            "max_amount_to_income": self.max_amount_to_income,
            "max_total_dti": self.max_total_dti,
        }

    # ----- container protocol -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._policies)

    def __getitem__(self, arm: int) -> LendingPolicy:
        return self._policies[int(arm)]

    def __iter__(self):
        return iter(self._policies)

    @property
    def n_arms(self) -> int:
        return len(self._policies)

    @property
    def policies(self) -> tuple[LendingPolicy, ...]:
        return self._policies

    @property
    def lending_arms(self) -> np.ndarray:
        """Indices of all non-null arms."""
        return np.array([i for i, p in enumerate(self._policies) if not p.is_null], dtype=int)

    # ----- mapping historical decisions onto the grid --------------------------------
    def nearest(self, amount: float, rate: float, term: int) -> int:
        """Arm closest to an observed (loan_amnt, int_rate, term).

        Term must match exactly when that term exists in the grid; distance is measured
        in grid-normalised units so amount and rate contribute comparably.
        """
        return int(self.nearest_many(np.array([amount]), np.array([rate]), np.array([term]))[0])

    def nearest_many(
        self, amounts: np.ndarray, rates: np.ndarray, terms: np.ndarray
    ) -> np.ndarray:
        amounts = np.asarray(amounts, dtype=float)
        rates = np.asarray(rates, dtype=float)
        terms = np.asarray(terms, dtype=float)
        lend = self.lending_arms
        a_scale = (max(self.amount_grid) - min(self.amount_grid)) or 1.0
        r_scale = (max(self.rate_grid) - min(self.rate_grid)) or 1.0
        da = (amounts[:, None] - self.amounts[lend][None, :]) / a_scale
        dr = (rates[:, None] - self.rates[lend][None, :]) / r_scale
        dist = da**2 + dr**2
        term_mismatch = terms[:, None] != self.terms[lend][None, :]
        # a term mismatch is only tolerated when the term is not in the grid at all
        has_term = np.isin(terms, self.term_grid)
        dist = np.where(term_mismatch & has_term[:, None], np.inf, dist)
        out = lend[np.argmin(dist, axis=1)]
        out = np.where(np.isnan(amounts) | np.isnan(rates), -1, out)
        return out.astype(int)

    # ----- lending guardrails ---------------------------------------------------------
    def eligible(
        self,
        annual_inc: float | None,
        *,
        dti: float | None = None,
        reference_rate: float | None = None,
    ) -> np.ndarray:
        """Boolean mask of the arms that pass every lending guardrail (module docstring).

        A guardrail whose input is unknown (income missing / non-positive, ``dti`` missing
        or negative, ``reference_rate`` missing) or that is switched off (``None``) is
        skipped, so unknown income and no reference rate leave every arm eligible. The null
        arm is always eligible; when no lending arm survives, the lending arm with the rate
        closest to ``reference_rate`` and the smallest installment is kept.
        """
        return self.eligible_many([annual_inc], None if dti is None else [dti],
                                  None if reference_rate is None else [reference_rate])[0]

    def eligible_many(
        self,
        annual_inc: Any,
        dti: Any = None,
        reference_rate: Any = None,
    ) -> np.ndarray:
        """Vectorised :meth:`eligible`: ``(n_borrowers, n_arms)`` boolean matrix.

        ``annual_inc``, ``dti`` (percent) and ``reference_rate`` are aligned 1-D sequences
        (``None`` = unknown for every borrower); row ``i`` equals
        ``eligible(annual_inc[i], dti=dti[i], reference_rate=reference_rate[i])``.
        """
        inc = _as_float_vector(annual_inc)
        n, k = inc.size, self.n_arms
        mask = np.ones((n, k), dtype=bool)
        known_inc = np.isfinite(inc) & (inc > 0)
        safe_inc = np.where(known_inc, inc, 1.0)
        if self.max_payment_to_income is not None:
            ok = self.installments[None, :] <= (self.max_payment_to_income * safe_inc / 12.0)[:, None]
            mask &= ok | ~known_inc[:, None]
        if self.max_amount_to_income is not None:
            ok = self.amounts[None, :] <= (self.max_amount_to_income * safe_inc)[:, None]
            mask &= ok | ~known_inc[:, None]
        if dti is not None and self.max_total_dti is not None:
            d = _as_float_vector(dti, n, "dti")
            known = known_inc & np.isfinite(d) & (d >= 0)
            total = np.where(known, d, 0.0)[:, None] / 100.0 + self.installments[None, :] * 12.0 / safe_inc[:, None]
            mask &= (total <= self.max_total_dti) | ~known[:, None]
        ref = None
        if reference_rate is not None:
            ref = _as_float_vector(reference_rate, n, "reference_rate")
        if ref is not None and self.rate_corridor is not None:
            known = np.isfinite(ref)
            gap = np.abs(self.rates[None, :] - np.where(known, ref, 0.0)[:, None])
            mask &= (gap <= self.rate_corridor + _RATE_TOL) | ~known[:, None]
        if self.null_arm is not None:
            mask[:, self.null_arm] = True
        lend = self.is_lending
        if lend.any():  # (a space of only the null arm keeps it: it is always eligible)
            empty = np.flatnonzero(~(mask & lend[None, :]).any(axis=1))
            if empty.size:
                mask[empty, self.fallback_arms(None if ref is None else ref[empty], size=empty.size)] = True
        return mask

    def fallback_arms(self, reference_rate: Any = None, size: int | None = None) -> np.ndarray:
        """The lending arm kept when no lending arm passes the guardrails, per borrower:
        among the lending arms whose rate is closest to the reference rate (all lending
        arms when it is unknown), the one with the smallest installment (ties → lowest
        index)."""
        lend = np.flatnonzero(self.is_lending)
        if reference_rate is None:
            m = 1 if size is None else int(size)
            ref = np.full(m, np.nan)
        else:
            ref = _as_float_vector(reference_rate)
        out = np.empty(ref.size, dtype=np.int64)
        known = np.isfinite(ref)
        inst = self.installments[lend]
        cheapest = lend[int(np.argmin(inst))]
        out[~known] = cheapest
        if known.any():
            gap = np.abs(self.rates[lend][None, :] - ref[known][:, None])
            closest = gap <= gap.min(axis=1, keepdims=True) + _RATE_TOL
            cost = np.where(closest, inst[None, :], np.inf)
            out[known] = lend[np.argmin(cost, axis=1)]
        return out

    # ----- views ---------------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "arm": np.arange(self.n_arms),
                "amount": self.amounts,
                "rate": self.rates,
                "term": self.terms,
                "is_null": [p.is_null for p in self._policies],
                "installment": self.installments,
                "label": [p.label() for p in self._policies],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {"policies": [p.to_dict() for p in self._policies], **self.guardrails}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmSpace:
        """Inverse of :meth:`to_dict`; a guardrail missing from ``d`` (an arm space saved
        before it existed) takes the ``ArmSpaceConfig`` default."""
        return cls(
            [LendingPolicy.from_dict(p) for p in d["policies"]],
            max_payment_to_income=d.get("max_payment_to_income", _DEFAULT_GUARDRAILS["max_payment_to_income"]),
            rate_corridor=d.get("rate_corridor", _DEFAULT_GUARDRAILS["rate_corridor"]),
            max_amount_to_income=d.get("max_amount_to_income", _DEFAULT_GUARDRAILS["max_amount_to_income"]),
            max_total_dti=d.get("max_total_dti", _DEFAULT_GUARDRAILS["max_total_dti"]),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArmSpace) and other._policies == self._policies

    def __hash__(self) -> int:
        return hash(self._policies)
