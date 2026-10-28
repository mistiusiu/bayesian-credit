"""Borrower segmentation — the "group g" of the hierarchical model.

§3.1/§3.2: outcomes are estimated "across different population segments" to guard
against aggregation bias and Simpson's paradox; the PSI Sentinel can prompt the human in
the loop to *refine* or *coarsen* the grouping (§3.3 last paragraph). A
:class:`Segmenter` is a pure function of pre-decision columns, so group keys are safe
to use as context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..config import SegmentationConfig
from ..exceptions import ConfigurationError, LeakageError
from ..types import Stage
from .schema import LeakageGuard

SEPARATOR = "|"

_PURPOSE_KEEP = {"debt_consolidation", "credit_card"}


def purpose_group(purpose: pd.Series) -> pd.Series:
    p = purpose.astype("string").str.lower()
    return p.where(p.isin(_PURPOSE_KEEP), "other").fillna("other")


def _band(values: pd.Series, edges: Sequence[float], labels: Sequence[str]) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    out = pd.cut(v, bins=list(edges), labels=list(labels), right=False, include_lowest=True)
    return out.astype("string").fillna("unknown")


def fico_band(fico_low: pd.Series) -> pd.Series:
    return _band(fico_low, [-np.inf, 660, 700, 740, 780, np.inf], ["<660", "660-699", "700-739", "740-779", "780+"])


def dti_band(dti: pd.Series) -> pd.Series:
    return _band(dti, [-np.inf, 10, 20, 30, np.inf], ["<10", "10-20", "20-30", "30+"])


def income_band(annual_inc: pd.Series) -> pd.Series:
    return _band(annual_inc, [-np.inf, 40_000, 75_000, 120_000, np.inf], ["<40k", "40-75k", "75-120k", "120k+"])


def emp_band(emp_length: pd.Series) -> pd.Series:
    return _band(emp_length, [-np.inf, 2, 6, 10, np.inf], ["<2", "2-5", "6-9", "10+"])


def _home_ownership(values: pd.Series) -> pd.Series:
    v = values.astype("string").str.upper()
    return v.where(v.isin({"RENT", "OWN", "MORTGAGE"}), "OTHER").fillna("OTHER")


def _passthrough(col: str):
    def f(values: pd.Series) -> pd.Series:
        return values.astype("string").fillna("unknown")

    f.__name__ = col
    return f


#: dimension -> (source pre-decision column, derivation)
DIMENSIONS: dict[str, tuple[str, Any]] = {
    "grade": ("grade", _passthrough("grade")),
    "sub_grade": ("sub_grade", _passthrough("sub_grade")),
    "purpose_group": ("purpose", purpose_group),
    "home_ownership": ("home_ownership", _home_ownership),
    "fico_band": ("fico_range_low", fico_band),
    "dti_band": ("dti", dti_band),
    "income_band": ("annual_inc", income_band),
    "emp_band": ("emp_length", emp_band),
    "verification_status": ("verification_status", _passthrough("verification_status")),
    "application_type": ("application_type", _passthrough("application_type")),
}


class Segmenter:
    """Maps borrower context -> group key ``"B|debt_consolidation|RENT"``."""

    def __init__(
        self,
        dimensions: Sequence[str],
        refinement_order: Sequence[str] = SegmentationConfig().refinement_order,
    ):
        dims = tuple(dimensions)
        if not dims:
            raise ConfigurationError("Segmenter needs at least one dimension")
        unknown = [d for d in dims if d not in DIMENSIONS]
        if unknown:
            raise ConfigurationError(f"unsupported segmentation dimensions: {unknown}")
        for d in dims:
            src = DIMENSIONS[d][0]
            if LeakageGuard.stage_of(src) != Stage.PRE_DECISION:
                raise LeakageError(f"dimension {d} derives from non pre-decision column {src}")
        self.dimensions = dims
        self.refinement_order = tuple(refinement_order)

    @classmethod
    def from_config(cls, cfg: SegmentationConfig) -> Segmenter:
        return cls(cfg.dimensions, cfg.refinement_order)

    # ----- derivation -----------------------------------------------------------------
    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with one derived column per dimension (same index as ``df``)."""
        out = pd.DataFrame(index=df.index)
        for d in self.dimensions:
            src, fn = DIMENSIONS[d]
            if src not in df.columns:
                out[d] = pd.Series("unknown", index=df.index, dtype="string")
            else:
                out[d] = fn(df[src]).astype("string")
        return out

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """Group key per row."""
        derived = self.derive(df)
        key = derived[self.dimensions[0]].astype("string")
        for d in self.dimensions[1:]:
            key = key + SEPARATOR + derived[d].astype("string")
        return key.rename("y_group")

    def key_for(self, context: Mapping[str, Any]) -> str:
        """Group key for a single borrower (BorrowerContext or mapping of features)."""
        feats = getattr(context, "features", context)
        row = pd.DataFrame([dict(feats)])
        return str(self.assign(row).iloc[0])

    # ----- human-in-the-loop re-segmentation ------------------------------------------
    def refine(self) -> Segmenter:
        """Append the next unused dimension from ``refinement_order`` (finer groups)."""
        for d in self.refinement_order:
            if d not in self.dimensions:
                return Segmenter((*self.dimensions, d), self.refinement_order)
        return self  # nothing left to refine

    def coarsen(self) -> Segmenter:
        """Drop the last dimension (coarser groups); never below one dimension."""
        if len(self.dimensions) == 1:
            return self
        return Segmenter(self.dimensions[:-1], self.refinement_order)

    # ----- serialisation ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"dimensions": list(self.dimensions), "refinement_order": list(self.refinement_order)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segmenter:
        return cls(d["dimensions"], d.get("refinement_order", SegmentationConfig().refinement_order))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Segmenter) and other.dimensions == self.dimensions

    def __hash__(self) -> int:
        return hash(self.dimensions)

    def __repr__(self) -> str:
        return f"Segmenter({list(self.dimensions)})"
