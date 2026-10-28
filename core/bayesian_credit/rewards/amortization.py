"""Level-payment (annuity) amortisation helpers used by the compound reward (SPEC §2.7).

For amount L, annual rate r (a fraction, 0.13 = 13%) and term T months, with monthly rate
i = r/12::

    M        = L·i(1+i)^T / ((1+i)^T − 1)            installment (M = L/T when r = 0)
    B_m      = L(1+i)^m − M((1+i)^m − 1)/i            balance after m payments; B_0 = L, B_T = 0
    I_m      = i·B_{m−1}                              interest paid in month m
    P_m      = B_{m−1} − B_m                          principal repaid in month m
    WAL      = Σ_m m·P_m / (12 L) = Σ_m B_{m−1} / (12 L)        weighted-average life (years)
    Σ_m I_m  = i·Σ_m B_{m−1}   ⇒   WAL = total_interest / (r·L)   (r > 0)

Scalar calls are memoised with :func:`functools.lru_cache` on the ``(L, r, T)`` tuple (the
simulation evaluates the same arm grid millions of times) and return read-only arrays.
``installment``, ``total_interest``, ``weighted_average_life_years`` and
``cumulative_opening_balance`` also accept numpy arrays (broadcast, closed forms) for
vectorised use over many loans. A null policy (``L ≤ 0`` or ``T ≤ 0``) has an empty
schedule and zero everything.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data.arms import annuity_installment

__all__ = [
    "SCHEDULE_CACHE_SIZE",
    "AmortizationSchedule",
    "cumulative_opening_balance",
    "installment",
    "outstanding_balances",
    "schedule",
    "schedule_arrays",
    "total_interest",
    "weighted_average_life_years",
]

SCHEDULE_CACHE_SIZE = 16_384


@dataclass(frozen=True, eq=False)
class AmortizationSchedule:
    """Cached schedule of one ``(L, r, T)`` loan. Arrays are read-only.

    ``month``/``payment``/``interest``/``principal``/``opening_balance``/``closing_balance``
    have length T (month m = 1..T, ``opening_balance[m-1] = B_{m−1}``). ``balances`` has
    length T + 1 (``B_0 … B_T``); ``cum_interest[k] = Σ_{m≤k} I_m`` and
    ``cum_opening_balance[k] = Σ_{m≤k} B_{m−1}`` have length T + 1 with a leading 0.
    """

    amount: float
    rate: float
    term: int
    installment: float
    month: np.ndarray
    payment: np.ndarray
    interest: np.ndarray
    principal: np.ndarray
    opening_balance: np.ndarray
    closing_balance: np.ndarray
    balances: np.ndarray
    cum_interest: np.ndarray
    cum_opening_balance: np.ndarray
    total_interest: float
    sum_opening_balance: float

    @property
    def wal_years(self) -> float:
        """Weighted-average life ``Σ_m B_{m−1} / (12 L)`` in years (0 for a null policy)."""
        return self.sum_opening_balance / (12.0 * self.amount) if self.amount > 0 and self.term > 0 else 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "month": self.month,
                "payment": self.payment,
                "interest": self.interest,
                "principal": self.principal,
                "opening_balance": self.opening_balance,
                "balance": self.closing_balance,
            }
        )


def _ro(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _scalar_key(amount: Any, rate: Any, term: Any) -> tuple[float, float, int]:
    L, r, t = float(amount), float(rate), float(term)
    if not (math.isfinite(L) and math.isfinite(r) and math.isfinite(t)):
        raise ValueError(f"amortisation inputs must be finite, got ({amount}, {rate}, {term})")
    if r < 0:
        raise ValueError(f"rate must be >= 0, got {rate}")
    if not t.is_integer():
        raise ValueError(f"term must be a whole number of months, got {term}")
    return L, r, int(t)


def _is_scalar(*values: Any) -> bool:
    return all(np.ndim(v) == 0 for v in values)


@functools.lru_cache(maxsize=SCHEDULE_CACHE_SIZE)
def _schedule_cached(amount: float, rate: float, term: int) -> AmortizationSchedule:
    if amount <= 0 or term <= 0:
        empty = _ro(np.zeros(0))
        zero1 = _ro(np.zeros(1))
        return AmortizationSchedule(
            max(amount, 0.0), rate, max(term, 0), 0.0, _ro(np.zeros(0, dtype=np.int64)), empty,
            empty, empty, empty, empty, zero1, zero1, zero1, 0.0, 0.0,
        )
    i = rate / 12.0
    m = np.arange(term + 1, dtype=float)
    M = annuity_installment(amount, rate, term)
    if i > 0:
        growth = (1.0 + i) ** m
        B = amount * growth - M * (growth - 1.0) / i
    else:
        B = amount * (1.0 - m / term)
    B[-1] = 0.0
    B = np.maximum(B, 0.0)
    opening, closing = B[:-1].copy(), B[1:].copy()
    interest = i * opening
    principal = opening - closing
    payment = interest + principal
    cum_int = np.concatenate(([0.0], np.cumsum(interest)))
    cum_open = np.concatenate(([0.0], np.cumsum(opening)))
    return AmortizationSchedule(
        amount=amount,
        rate=rate,
        term=term,
        installment=float(M),
        month=_ro(np.arange(1, term + 1, dtype=np.int64)),
        payment=_ro(payment),
        interest=_ro(interest),
        principal=_ro(principal),
        opening_balance=_ro(opening),
        closing_balance=_ro(closing),
        balances=_ro(B),
        cum_interest=_ro(cum_int),
        cum_opening_balance=_ro(cum_open),
        total_interest=float(cum_int[-1]),
        sum_opening_balance=float(cum_open[-1]),
    )


def schedule_arrays(amount: float, rate: float, term: int) -> AmortizationSchedule:
    """Cached :class:`AmortizationSchedule` for one loan (scalar inputs)."""
    return _schedule_cached(*_scalar_key(amount, rate, term))


def schedule(amount: float, rate: float, term: int) -> pd.DataFrame:
    """Schedule as a DataFrame: ``month, payment, interest, principal, opening_balance,
    balance`` (``balance`` = closing balance B_m)."""
    return schedule_arrays(amount, rate, term).to_frame()


@functools.lru_cache(maxsize=SCHEDULE_CACHE_SIZE)
def _installment_cached(amount: float, rate: float, term: int) -> float:
    return float(annuity_installment(amount, rate, term))


def _broadcast(*values: Any) -> list[np.ndarray]:
    return np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in values))


def _installment_vec(L: np.ndarray, r: np.ndarray, T: np.ndarray) -> np.ndarray:
    i = r / 12.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g = (1.0 + i) ** T
        M = np.where(i > 0, L * i * g / (g - 1.0), L / T)
    return np.where((L > 0) & (T > 0), M, 0.0)


def installment(amount: Any, rate: Any, term: Any) -> float | np.ndarray:
    """Level monthly payment M (0 for a null policy). Same formula as
    :func:`bayesian_credit.data.arms.annuity_installment`."""
    if _is_scalar(amount, rate, term):
        return _installment_cached(*_scalar_key(amount, rate, term))
    return _installment_vec(*_broadcast(amount, rate, term))


def total_interest(amount: Any, rate: Any, term: Any) -> float | np.ndarray:
    """``Σ_m I_m`` (array inputs use the closed form ``T·M − L``)."""
    if _is_scalar(amount, rate, term):
        return schedule_arrays(amount, rate, term).total_interest
    L, r, T = _broadcast(amount, rate, term)
    M = _installment_vec(L, r, T)
    return np.where((L > 0) & (T > 0), T * M - L, 0.0)


def weighted_average_life_years(amount: Any, rate: Any, term: Any) -> float | np.ndarray:
    """WAL in years ``= Σ_m B_{m−1} / (12 L)`` (array inputs: ``total_interest / (r·L)``,
    or ``(T + 1)/24`` when r = 0)."""
    if _is_scalar(amount, rate, term):
        return schedule_arrays(amount, rate, term).wal_years
    L, r, T = _broadcast(amount, rate, term)
    ti = total_interest(L, r, T)
    with np.errstate(divide="ignore", invalid="ignore"):
        wal = np.where(r > 0, ti / (r * L), (T + 1.0) / 24.0)
    return np.where((L > 0) & (T > 0), wal, 0.0)


def outstanding_balances(amount: float, rate: float, term: int) -> np.ndarray:
    """Read-only ``[B_0, B_1, …, B_T]`` — principal outstanding after m payments."""
    return schedule_arrays(amount, rate, term).balances


def cumulative_opening_balance(amount: Any, rate: Any, term: Any, months: Any) -> float | np.ndarray:
    """``Σ_{m=1}^{k} B_{m−1}`` with ``k = clip(months, 0, T)`` — balance-months on book.

    ``funding_rate · cumulative_opening_balance(...) / 12`` is the funding cost of the months
    actually on book (= ``funding_rate · L · WAL`` for k = T). Array inputs use the closed
    form ``L·S − (M/i)(S − k)`` with ``S = ((1+i)^k − 1)/i`` (``L(k − k(k−1)/(2T))`` for r = 0).
    """
    if _is_scalar(amount, rate, term, months):
        sched = schedule_arrays(amount, rate, term)
        mob = float(months)
        if not math.isfinite(mob):
            raise ValueError(f"months must be finite, got {months}")
        k = int(min(max(math.floor(mob), 0), sched.term))
        return float(sched.cum_opening_balance[k])
    L, r, T, k = _broadcast(amount, rate, term, months)
    valid = (L > 0) & (T > 0)
    k = np.clip(np.floor(k), 0.0, np.where(T > 0, T, 0.0))
    i = r / 12.0
    M = _installment_vec(L, r, T)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        S = np.where(i > 0, ((1.0 + i) ** k - 1.0) / i, k)
        amortising = L * S - (M / i) * (S - k)
        straight = L * (k - k * (k - 1.0) / (2.0 * T))
        out = np.where(i > 0, amortising, straight)
    return np.where(valid, out, 0.0)
