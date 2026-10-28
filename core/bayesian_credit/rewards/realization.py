"""Prepayment: the share of the scheduled interest a lender actually collects.

**[decision]** (docs/MATH.md §2.8). The closed-form reward (:mod:`.compound`) values the full
contractual schedule ``Σ_m I_m`` of a non-defaulting loan, but most Lending Club borrowers
repay early: fully-paid 36-month loans realise ≈ 82 % of their scheduled interest and
60-month loans ≈ 72 %, so an unadjusted reward overvalues long terms. The
:class:`InterestRealization` table measures this on the training window::

    ρ_i      = clip(total_rec_int_i / scheduled_interest(funded_amnt_i, int_rate_i, term_i), 0, 1)
    ρ(g, T)  = mean of ρ_i over the fully-paid loans of grade g and term T  (n ≥ min_count)
    fallback   (grade, term) → term → global → 1 (no table);  every entry clipped to [0.3, 1.0]

using only loans whose term has elapsed by the snapshot when enough exist (recent vintages'
fully-paid loans are the early payers). The reward multiplies the expected-interest term and
the funding cost by ``ρ(grade(g), T_a)`` — prepayment shortens both the interest stream and
the balance-time that has to be funded (``CompoundRewardFunction.expected(...,
realization=ρ)``). ρ depends only on the borrower group's grade and the arm's term, so every
``(group, arm)`` cache of the simulation stays exact. :class:`RealizationLookup` binds a
table to an arm space and a segmentation and returns ``ρ`` for a ``(group, arm)`` pair
(``1.0`` when ``RewardConfig.prepayment_adjustment`` is ``None`` or there is no table).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..data.outcomes import add_months
from ..data.segmentation import SEPARATOR
from . import amortization as amort

__all__ = [
    "PREPAYMENT_ADJUSTMENTS",
    "InterestRealization",
    "RealizationLookup",
    "group_grade",
]

#: Accepted values of ``RewardConfig.prepayment_adjustment``.
PREPAYMENT_ADJUSTMENTS = ("empirical", None)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _clean(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def group_grade(group: Any, segmenter: Any = None) -> str | None:
    """Lending Club grade of a borrower-group key.

    With a segmenter: the ``grade`` component of the key, else the first letter of its
    ``sub_grade`` component (``None`` when the segmentation has neither). Without one: the
    first component when it is a single letter (the default ``grade|purpose|home`` keys).
    """
    if group is None:
        return None
    parts = str(group).split(SEPARATOR)
    dims = tuple(getattr(segmenter, "dimensions", ()) or ())
    if dims and len(parts) == len(dims):
        if "grade" in dims:
            value = parts[dims.index("grade")].strip()
            return value or None
        if "sub_grade" in dims:
            value = parts[dims.index("sub_grade")].strip()
            return value[:1] or None
        return None
    head = parts[0].strip()
    return head if len(head) == 1 and head.isalpha() else None


class InterestRealization:
    """Realised share ``ρ`` of the scheduled interest of fully-paid loans by (grade, term)
    with fallbacks to term and global (module docstring)."""

    def __init__(
        self,
        by_grade_term: Mapping[str, Mapping[Any, float]] | None = None,
        by_term: Mapping[Any, float] | None = None,
        global_rho: float | None = None,
        *,
        counts: Mapping[str, int] | None = None,
        n_fit: int = 0,
        sample: str | None = None,
        min_count: int = 30,
        clip: tuple[float, float] = (0.3, 1.0),
    ) -> None:
        self.by_grade_term: dict[str, dict[int, float]] = {
            str(g): {int(t): float(v) for t, v in terms.items()} for g, terms in (by_grade_term or {}).items()
        }
        self.by_term: dict[int, float] = {int(t): float(v) for t, v in (by_term or {}).items()}
        self.global_rho: float | None = None if global_rho is None else float(global_rho)
        self.counts: dict[str, int] = {str(k): int(v) for k, v in (counts or {}).items()}
        self.n_fit = int(n_fit)
        self.sample = sample
        self.min_count = int(min_count)
        self.clip = (float(clip[0]), float(clip[1]))

    # ----- estimation -------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        *,
        snapshot_date: Any = None,
        min_count: int = 30,
        clip: tuple[float, float] = (0.3, 1.0),
        min_elapsed: int = 1_000,
    ) -> InterestRealization:
        """Estimate ρ on the fully-paid loans of ``df`` (``y_resolved`` and ``y_default == 0``).

        Needs ``total_rec_int``, ``int_rate``, ``term`` and ``funded_amnt`` (``loan_amnt``
        when missing); ``grade`` (or ``sub_grade``) for the grade cells. With ``issue_d`` and
        ``snapshot_date`` only loans whose term has elapsed by the snapshot are used when at
        least ``min_elapsed`` exist. A cell needs ``min_count`` loans, else the next level.
        """
        lo, hi = float(clip[0]), float(clip[1])
        n = len(df)
        resolved = np.ones(n, dtype=bool)
        if "y_resolved" in df.columns:
            r = df["y_resolved"]
            resolved = r.to_numpy(dtype=bool) if r.dtype == bool else (r.to_numpy(dtype=float, na_value=0.0) != 0.0)
        y = _num(df, "y_default")
        paid = resolved & (y == 0.0)
        L = _num(df, "funded_amnt")
        L = np.where(np.isfinite(L) & (L > 0), L, _num(df, "loan_amnt"))
        rate, term, rec_int = _num(df, "int_rate"), _num(df, "term"), _num(df, "total_rec_int")
        ok = paid & np.isfinite(L) & (L > 0) & np.isfinite(rate) & (rate > 0) & np.isfinite(term) & (term > 0)
        ok &= np.isfinite(rec_int) & (rec_int >= 0)
        sample = "fully_paid"
        if snapshot_date is not None and "issue_d" in df.columns:
            end = add_months(df["issue_d"].reset_index(drop=True), term)
            elapsed = (end <= pd.Timestamp(snapshot_date)).to_numpy(dtype=bool)
            if int((ok & elapsed).sum()) >= int(min_elapsed):
                ok &= elapsed
                sample = "fully_paid_elapsed"
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return cls(n_fit=0, sample=sample, min_count=min_count, clip=(lo, hi))
        sched = np.asarray(amort.total_interest(L[idx], rate[idx], np.round(term[idx])), dtype=float)
        good = sched > 0
        idx, sched = idx[good], sched[good]
        rho_i = np.clip(rec_int[idx] / sched, 0.0, 1.0)
        terms = np.round(term[idx]).astype(np.int64)
        if "grade" in df.columns:
            grades = df["grade"].astype("string").to_numpy(dtype=object, na_value=None)[idx]
        elif "sub_grade" in df.columns:
            grades = np.array([None if v is None else str(v)[:1] for v in
                               df["sub_grade"].astype("string").to_numpy(dtype=object, na_value=None)[idx]], dtype=object)
        else:
            grades = np.full(idx.size, None, dtype=object)
        frame = pd.DataFrame({"grade": grades, "term": terms, "rho": rho_i})
        counts: dict[str, int] = {"global": int(len(frame))}
        by_term: dict[int, float] = {}
        for t, sub in frame.groupby("term", sort=True):
            counts[f"term:{int(t)}"] = int(len(sub))
            if len(sub) >= min_count:
                by_term[int(t)] = float(np.clip(sub["rho"].mean(), lo, hi))
        by_gt: dict[str, dict[int, float]] = {}
        for (g, t), sub in frame.dropna(subset=["grade"]).groupby(["grade", "term"], sort=True):
            counts[f"{g}|{int(t)}"] = int(len(sub))
            if len(sub) >= min_count:
                by_gt.setdefault(str(g), {})[int(t)] = float(np.clip(sub["rho"].mean(), lo, hi))
        global_rho = float(np.clip(frame["rho"].mean(), lo, hi))
        return cls(by_gt, by_term, global_rho, counts=counts, n_fit=int(len(frame)), sample=sample,
                   min_count=min_count, clip=(lo, hi))

    # ----- lookup -----------------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return self.global_rho is None and not self.by_term and not self.by_grade_term

    def factor(self, grade: Any = None, term: Any = None) -> float:
        """``ρ(grade, term)`` with the (grade, term) → term → global → 1.0 fallback."""
        t = None
        if term is not None:
            try:
                tf = float(term)
                t = int(round(tf)) if math.isfinite(tf) and tf > 0 else None
            except (TypeError, ValueError):
                t = None
        g = None if grade is None else (str(grade).strip() or None)
        if g is not None and t is not None:
            hit = self.by_grade_term.get(g, {}).get(t)
            if hit is not None:
                return hit
        if t is not None and t in self.by_term:
            return self.by_term[t]
        return 1.0 if self.global_rho is None else self.global_rho

    def source(self, grade: Any = None, term: Any = None) -> str:
        """Level the factor of ``(grade, term)`` comes from: ``grade_term``, ``term``,
        ``global`` or ``none``."""
        t = None if term is None else int(round(float(term)))
        g = None if grade is None else str(grade)
        if g is not None and t is not None and t in self.by_grade_term.get(g, {}):
            return "grade_term"
        if t is not None and t in self.by_term:
            return "term"
        return "none" if self.global_rho is None else "global"

    def table(self) -> pd.DataFrame:
        """Rows ``grade, term, rho, n, level`` (``grade`` NA for term/global rows)."""
        rows: list[tuple[Any, Any, float, int, str]] = []
        for g in sorted(self.by_grade_term):
            for t in sorted(self.by_grade_term[g]):
                rows.append((g, t, self.by_grade_term[g][t], self.counts.get(f"{g}|{t}", 0), "grade_term"))
        for t in sorted(self.by_term):
            rows.append((pd.NA, t, self.by_term[t], self.counts.get(f"term:{t}", 0), "term"))
        if self.global_rho is not None:
            rows.append((pd.NA, pd.NA, self.global_rho, self.counts.get("global", 0), "global"))
        cols = list(zip(*rows, strict=True)) if rows else [[], [], [], [], []]
        return pd.DataFrame(
            {
                "grade": pd.array(cols[0], dtype="string"),
                "term": pd.array(cols[1], dtype="Int64"),
                "rho": np.asarray(cols[2], dtype=float),
                "n": np.asarray(cols[3], dtype=np.int64),
                "level": pd.array(cols[4], dtype="string"),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Compact JSON view for ``FittedModels.metrics`` / the API."""
        return {
            "global": _clean(self.global_rho),
            "by_term": {str(t): v for t, v in sorted(self.by_term.items())},
            "by_grade_term": {g: {str(t): v for t, v in sorted(ts.items())} for g, ts in sorted(self.by_grade_term.items())},
            "n_fit": self.n_fit,
            "sample": self.sample,
            "min_count": self.min_count,
            "clip": list(self.clip),
        }

    # ----- serialisation -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "InterestRealization",
            "version": 1,
            "by_grade_term": {g: {str(t): float(v) for t, v in ts.items()} for g, ts in self.by_grade_term.items()},
            "by_term": {str(t): float(v) for t, v in self.by_term.items()},
            "global": _clean(self.global_rho),
            "counts": dict(self.counts),
            "n_fit": self.n_fit,
            "sample": self.sample,
            "min_count": self.min_count,
            "clip": list(self.clip),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> InterestRealization:
        data = dict(data or {})
        clip = data.get("clip") or (0.3, 1.0)
        return cls(
            data.get("by_grade_term"),
            data.get("by_term"),
            data.get("global"),
            counts=data.get("counts"),
            n_fit=int(data.get("n_fit", 0)),
            sample=data.get("sample"),
            min_count=int(data.get("min_count", 30)),
            clip=(float(clip[0]), float(clip[1])),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InterestRealization) and other.to_dict() == self.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        terms = ", ".join(f"{t}m {v:.3f}" for t, v in sorted(self.by_term.items()))
        return f"InterestRealization({terms or 'empty'}; n_fit={self.n_fit})"


class RealizationLookup:
    """``(group, arm) → ρ`` for an arm space and a segmentation (memoised).

    ``ρ = table.factor(group_grade(group, segmenter), arm_space[arm].term)``; ``1.0`` for
    the null arm, arms outside the space, a disabled adjustment or a missing table.
    """

    def __init__(
        self,
        table: InterestRealization | None,
        arm_space: Any,
        segmenter: Any = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.table = table
        self.arm_space = arm_space
        self.segmenter = segmenter
        self.enabled = bool(enabled) and table is not None and not table.is_empty
        self._memo: dict[tuple[str, int], float] = {}

    @classmethod
    def from_config(cls, reward_cfg: Any, table: InterestRealization | None, arm_space: Any,
                    segmenter: Any = None) -> RealizationLookup:
        """Enabled when ``reward_cfg.prepayment_adjustment == "empirical"``."""
        mode = getattr(reward_cfg, "prepayment_adjustment", "empirical")
        return cls(table, arm_space, segmenter, enabled=mode == "empirical")

    def __call__(self, group: Any, arm: int) -> float:
        if not self.enabled:
            return 1.0
        key = (str(group), int(arm))
        hit = self._memo.get(key)
        if hit is None:
            a = key[1]
            if 0 <= a < self.arm_space.n_arms and not self.arm_space[a].is_null:
                hit = self.table.factor(group_grade(key[0], self.segmenter), int(self.arm_space[a].term))
            else:
                hit = 1.0
            self._memo[key] = hit
        return hit

    def for_term(self, group: Any, term: Any) -> float:
        """ρ for an arbitrary term (e.g. an issued policy that is not on the arm grid)."""
        if not self.enabled:
            return 1.0
        return self.table.factor(group_grade(group, self.segmenter), term)

    def with_segmenter(self, segmenter: Any) -> RealizationLookup:
        """A fresh lookup over another segmentation (memo cleared)."""
        return RealizationLookup(self.table, self.arm_space, segmenter, enabled=self.enabled)
