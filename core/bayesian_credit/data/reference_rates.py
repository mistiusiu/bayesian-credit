"""Reference interest rate of a borrower — the anchor of the pricing guardrail.

**[decision]** (docs/MATH.md, assumption G4). The rate corridor of the arm space
(``ArmSpaceConfig.rate_corridor``) admits only arms priced within ±corridor of a
*reference rate*. The reference rate must be a pure pre-decision function of the
borrower: it is the training-window mean ``int_rate`` of the borrower's ``sub_grade``
(Lending Club's own risk tier, a ``PRE_DECISION`` column), falling back to the mean of its
``grade`` and then to the global mean when the sub-grade / grade was never seen. The table
is fitted once on the training loans (``issue_d < train_end``) and stored with the fit
(``reference_rates.json``), so the shadow engine and the simulation use the same anchor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import DataSourceError

__all__ = ["ReferenceRates"]


def _key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


class ReferenceRates:
    """Mean ``int_rate`` by ``sub_grade`` → ``grade`` → global (pre-decision lookup table).

    ``fit(train_df)``; ``for_context(context)`` for one borrower (a
    :class:`~bayesian_credit.data.schema.BorrowerContext`, a mapping or a row);
    ``for_frame(df)`` for many (``np.ndarray``, NaN only when nothing was fitted). Rates are
    fractions (0.1299). The grade of a context without ``grade`` is the first character of
    its ``sub_grade``.
    """

    def __init__(
        self,
        by_sub_grade: Mapping[str, float] | None = None,
        by_grade: Mapping[str, float] | None = None,
        global_rate: float | None = None,
        *,
        counts: Mapping[str, int] | None = None,
        n_fit: int = 0,
    ):
        self.by_sub_grade: dict[str, float] = {str(k): float(v) for k, v in (by_sub_grade or {}).items()}
        self.by_grade: dict[str, float] = {str(k): float(v) for k, v in (by_grade or {}).items()}
        self.global_rate: float | None = None if global_rate is None else float(global_rate)
        self.counts: dict[str, int] = {str(k): int(v) for k, v in (counts or {}).items()}
        self.n_fit = int(n_fit)

    # ----- estimation -------------------------------------------------------------------
    @classmethod
    def fit(cls, df: pd.DataFrame, *, rate_col: str = "int_rate") -> ReferenceRates:
        """Mean ``rate_col`` per ``sub_grade`` and per ``grade`` over the rows of ``df``
        with a finite rate (outcomes are not used; the rate is known at issuance)."""
        if rate_col not in df.columns:
            raise DataSourceError(f"cannot fit reference rates: column {rate_col!r} is missing")
        rate = pd.to_numeric(df[rate_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        ok = np.isfinite(rate)
        if not ok.any():
            return cls()
        frame = pd.DataFrame({"rate": rate[ok]})
        by_sub: dict[str, float] = {}
        by_grade: dict[str, float] = {}
        counts: dict[str, int] = {}
        if "sub_grade" in df.columns:
            sub = df["sub_grade"].astype("string").to_numpy(dtype=object, na_value=None)[ok]
            frame["sub_grade"] = [_key(v) for v in sub]
        if "grade" in df.columns:
            grd = df["grade"].astype("string").to_numpy(dtype=object, na_value=None)[ok]
            frame["grade"] = [_key(v) for v in grd]
        elif "sub_grade" in frame.columns:
            frame["grade"] = [None if v is None else v[0] for v in frame["sub_grade"]]
        if "sub_grade" in frame.columns:
            agg = frame.dropna(subset=["sub_grade"]).groupby("sub_grade", sort=True)["rate"].agg(["mean", "size"])
            by_sub = {str(k): float(v) for k, v in agg["mean"].items()}
            counts.update({f"sub_grade:{k}": int(v) for k, v in agg["size"].items()})
        if "grade" in frame.columns:
            agg = frame.dropna(subset=["grade"]).groupby("grade", sort=True)["rate"].agg(["mean", "size"])
            by_grade = {str(k): float(v) for k, v in agg["mean"].items()}
            counts.update({f"grade:{k}": int(v) for k, v in agg["size"].items()})
        return cls(by_sub, by_grade, float(frame["rate"].mean()), counts=counts, n_fit=int(ok.sum()))

    # ----- lookup -----------------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return self.global_rate is None and not self.by_grade and not self.by_sub_grade

    def lookup(self, sub_grade: Any = None, grade: Any = None) -> float | None:
        """Reference rate for a (sub_grade, grade) pair with the sub-grade → grade → global
        fallback (``None`` only when nothing was fitted)."""
        sg, gr = _key(sub_grade), _key(grade)
        if sg is not None and sg in self.by_sub_grade:
            return self.by_sub_grade[sg]
        if gr is None and sg is not None:
            gr = sg[0]
        if gr is not None and gr in self.by_grade:
            return self.by_grade[gr]
        return self.global_rate

    def for_context(self, context: Any) -> float | None:
        """Reference rate of one borrower: ``context`` is a ``BorrowerContext`` (its
        features), a mapping or a ``pd.Series`` holding ``sub_grade`` and/or ``grade``."""
        feats = getattr(context, "features", context)
        get = feats.get if hasattr(feats, "get") else (lambda k, d=None: getattr(feats, k, d))
        return self.lookup(get("sub_grade", None), get("grade", None))

    def for_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised :meth:`for_context` over the rows of ``df`` (NaN when nothing was fitted)."""
        n = len(df)
        out = np.full(n, np.nan if self.global_rate is None else self.global_rate)
        if n == 0:
            return out
        filled = np.zeros(n, dtype=bool)
        sub = None
        if "sub_grade" in df.columns:
            sub = df["sub_grade"].astype("string").reset_index(drop=True).str.strip()
            vals = sub.map(self.by_sub_grade).to_numpy(dtype=float, na_value=np.nan)
            hit = np.isfinite(vals)
            out[hit] = vals[hit]
            filled |= hit
        grade = None
        if "grade" in df.columns:
            grade = df["grade"].astype("string").reset_index(drop=True).str.strip()
            if sub is not None:
                grade = grade.where(grade.notna() & (grade != ""), sub.str.slice(0, 1))
        elif sub is not None:
            grade = sub.str.slice(0, 1)
        if grade is not None:
            vals = grade.map(self.by_grade).to_numpy(dtype=float, na_value=np.nan)
            hit = np.isfinite(vals) & ~filled
            out[hit] = vals[hit]
        return out

    # ----- serialisation -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ReferenceRates",
            "version": 1,
            "by_sub_grade": dict(self.by_sub_grade),
            "by_grade": dict(self.by_grade),
            "global_rate": self.global_rate,
            "counts": dict(self.counts),
            "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ReferenceRates:
        data = dict(data or {})
        g = data.get("global_rate")
        return cls(
            data.get("by_sub_grade"),
            data.get("by_grade"),
            None if g is None or (isinstance(g, float) and math.isnan(g)) else float(g),
            counts=data.get("counts"),
            n_fit=int(data.get("n_fit", 0)),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReferenceRates) and other.to_dict() == self.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        g = "n/a" if self.global_rate is None else f"{self.global_rate:.4f}"
        return f"ReferenceRates(sub_grades={len(self.by_sub_grade)}, grades={len(self.by_grade)}, global={g})"
