"""Population Stability Index for the PSI Sentinel (Appendix G.5, Table 9; SPEC §2.9).

    PSI = Σ_i (a_i − e_i) · ln(a_i / e_i)

over the buckets ``i`` of a feature, where ``e_i`` is the bucket share in the reference
(model-development) population and ``a_i`` the share in the current window. Every term is
non-negative. Table 9 reads PSI < 0.10 as little or no shift, 0.10–0.25 as moderate
shift and > 0.25 as significant shift.

Bucketing (fitted on the reference window, persisted by :meth:`PSICalculator.to_dict`):

* numeric features — bin edges at the reference quantiles (deciles for ``n_bins=10``),
  de-duplicated with ``np.unique``; the outer edges are extended to ±inf so every current
  value has a bucket. Bins are left-closed ``[e_i, e_{i+1})``. If the quantiles collapse to
  a single bin (a heavy point mass), the single split that best balances the reference
  mass is used **[decision]**. Numeric features with fewer than 3 distinct reference
  values are treated as categorical over their (numeric) levels.
* categorical/string/bool features — the levels seen in the reference plus an
  ``"__other__"`` bucket for levels never seen before.
* every feature also has a ``"__missing__"`` bucket **[decision]**, so a change in
  missingness is itself detectable (it contributes 0 when both windows have none).

Shares are smoothed as ``(share + eps)`` and renormalised before the log, so there is no
``log(0)`` and ``PSI(reference, reference) = 0`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import PSIConfig
from ..exceptions import ConfigurationError

OTHER_BUCKET = "__other__"
MISSING_BUCKET = "__missing__"
TABLE_COLUMNS = ("bucket", "expected_share", "actual_share", "contribution")


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _finite_or_none(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None


@dataclass
class _FeatureBins:
    """Fitted bucketing of one feature (kind = numeric | discrete | categorical)."""

    name: str
    kind: str
    edges: list[float] = field(default_factory=list)  # numeric: inner edges (outer = ±inf)
    levels: list[Any] = field(default_factory=list)  # discrete: floats; categorical: strings
    expected_counts: list[int] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        if self.kind == "numeric":
            bounds = [-math.inf, *self.edges, math.inf]
            out = []
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
                left = "(-inf" if lo == -math.inf else f"[{_fmt(lo)}"
                right = "inf)" if hi == math.inf else f"{_fmt(hi)})"
                out.append(f"{left}, {right}")
            return [*out, MISSING_BUCKET]
        if self.kind == "discrete":
            return [_fmt(v) for v in self.levels] + [OTHER_BUCKET, MISSING_BUCKET]
        return [str(v) for v in self.levels] + [OTHER_BUCKET, MISSING_BUCKET]

    def counts(self, values: pd.Series) -> np.ndarray:
        """Bucket counts of ``values`` in label order."""
        if self.kind == "numeric":
            x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            missing = np.isnan(x)
            bins = np.searchsorted(np.asarray(self.edges, dtype=float), x[~missing], side="right")
            out = np.bincount(bins, minlength=len(self.edges) + 1).astype(np.int64)
            return np.append(out, int(missing.sum()))
        if self.kind == "discrete":
            x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            missing = np.isnan(x)
            levels = np.asarray(self.levels, dtype=float)
            hits = x[~missing, None] == levels[None, :]
            per_level = hits.sum(axis=0).astype(np.int64)
            other = int((~hits.any(axis=1)).sum())
            return np.concatenate([per_level, [other, int(missing.sum())]]).astype(np.int64)
        s = values.astype("string")
        missing = s.isna().to_numpy(dtype=bool)
        present = s[~missing]
        vc = present.value_counts()
        per_level = np.array([int(vc.get(lvl, 0)) for lvl in self.levels], dtype=np.int64)
        other = int(len(present) - per_level.sum())
        return np.concatenate([per_level, [other, int(missing.sum())]]).astype(np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "edges": [float(e) for e in self.edges],
            "levels": list(self.levels),
            "expected_counts": [int(c) for c in self.expected_counts],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> _FeatureBins:
        kind = d["kind"]
        levels = list(d.get("levels", []))
        if kind == "discrete":
            levels = [float(v) for v in levels]
        elif kind == "categorical":
            levels = [str(v) for v in levels]
        return cls(
            name=str(d["name"]),
            kind=kind,
            edges=[float(e) for e in d.get("edges", [])],
            levels=levels,
            expected_counts=[int(c) for c in d.get("expected_counts", [])],
        )


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _fit_bins(name: str, s: pd.Series, n_bins: int) -> _FeatureBins:
    if _is_numeric(s):
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = np.sort(x[np.isfinite(x)])
        uniq = np.unique(finite)
        if uniq.size >= 3:
            edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
            inner = edges[1:-1]
            if inner.size == 0:
                # quantiles collapsed (point mass): best-balanced single split
                cand = uniq[1:]
                left = np.searchsorted(finite, cand, side="left") / finite.size
                inner = np.array([cand[int(np.argmax(np.minimum(left, 1.0 - left)))]])
            fb = _FeatureBins(name, "numeric", edges=[float(e) for e in inner])
        else:
            fb = _FeatureBins(name, "discrete", levels=[float(v) for v in uniq])
    else:
        st = s.astype("string")
        levels = sorted(str(v) for v in st.dropna().unique())
        fb = _FeatureBins(name, "categorical", levels=levels)
    fb.expected_counts = fb.counts(s).tolist()
    return fb


@dataclass
class PSIResult:
    """PSI = Σ (a_i − e_i)·ln(a_i/e_i) of one current window against the reference, per
    feature and aggregated (G.5, Table 9).

    ``tables[f]`` has columns ``bucket, expected_share, actual_share, contribution``
    (smoothed shares; contributions sum to ``per_feature[f]``). Features absent from the
    current window are skipped and listed in ``missing_features``. An empty window gives
    NaN PSI values.
    """

    per_feature: dict[str, float]
    tables: dict[str, pd.DataFrame]
    aggregate_max: float
    aggregate_mean: float
    n_reference: int
    n_current: int
    missing_features: list[str] = field(default_factory=list)

    def aggregate(self, how: str = "max") -> float:
        if how == "max":
            return self.aggregate_max
        if how == "mean":
            return self.aggregate_mean
        raise ConfigurationError(f"aggregate must be 'max' or 'mean', got {how!r}")

    def to_dict(self, include_tables: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "per_feature": {k: _finite_or_none(v) for k, v in self.per_feature.items()},
            "aggregate_max": _finite_or_none(self.aggregate_max),
            "aggregate_mean": _finite_or_none(self.aggregate_mean),
            "n_reference": int(self.n_reference),
            "n_current": int(self.n_current),
            "missing_features": list(self.missing_features),
        }
        if include_tables:
            out["tables"] = {
                f: [
                    {
                        "bucket": str(r.bucket),
                        "expected_share": _finite_or_none(r.expected_share),
                        "actual_share": _finite_or_none(r.actual_share),
                        "contribution": _finite_or_none(r.contribution),
                    }
                    for r in t.itertuples(index=False)
                ]
                for f, t in self.tables.items()
            }
        return out


class PSICalculator:
    """Fits reference buckets and scores current windows with the PSI (G.5, Table 9).

    ``PSICalculator(features, n_bins=10, eps=1e-4).fit(reference_df)`` then
    ``psi(current_df) -> PSIResult``. Only the fitted buckets and reference counts are
    kept, so a checkpointed calculator (:meth:`to_dict`) needs no reference data.
    """

    def __init__(self, features: Iterable[str], n_bins: int = 10, eps: float = 1e-4):
        self.features: tuple[str, ...] = tuple(features)
        if not self.features:
            raise ConfigurationError("PSICalculator needs at least one feature")
        if int(n_bins) < 2:
            raise ConfigurationError("n_bins must be >= 2")
        if not eps > 0:
            raise ConfigurationError("eps must be > 0")
        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.bins: dict[str, _FeatureBins] = {}
        self.unavailable: list[str] = []
        self.n_reference = 0

    @classmethod
    def from_config(cls, cfg: PSIConfig) -> PSICalculator:
        return cls(cfg.features, n_bins=cfg.n_bins, eps=cfg.eps)

    @property
    def fitted(self) -> bool:
        return bool(self.bins)

    def fit(self, reference_df: pd.DataFrame) -> PSICalculator:
        """Fit buckets on the reference window (features absent from it are skipped)."""
        if reference_df is None or len(reference_df) == 0:
            raise ConfigurationError("cannot fit PSI buckets on an empty reference window")
        self.bins = {}
        self.unavailable = []
        for f in self.features:
            if f not in reference_df.columns:
                self.unavailable.append(f)
                continue
            self.bins[f] = _fit_bins(f, reference_df[f], self.n_bins)
        if not self.bins:
            raise ConfigurationError(f"none of the PSI features {list(self.features)} is in the reference")
        self.n_reference = int(len(reference_df))
        return self

    def _smooth(self, counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        shares = counts / total if total > 0 else np.zeros(counts.size)
        shares = shares + self.eps
        return shares / shares.sum()

    def psi(self, current_df: pd.DataFrame) -> PSIResult:
        """PSI of ``current_df`` against the reference, per feature and aggregated."""
        if not self.fitted:
            raise ConfigurationError("PSICalculator.fit() must be called before psi()")
        n_current = 0 if current_df is None else int(len(current_df))
        per_feature: dict[str, float] = {}
        tables: dict[str, pd.DataFrame] = {}
        missing = list(self.unavailable)
        for f, fb in self.bins.items():
            if current_df is None or f not in current_df.columns:
                missing.append(f)
                continue
            expected = self._smooth(np.asarray(fb.expected_counts, dtype=float))
            if n_current == 0:
                actual = np.full(expected.size, np.nan)
                contrib = np.full(expected.size, np.nan)
                value = float("nan")
            else:
                actual = self._smooth(fb.counts(current_df[f]).astype(float))
                contrib = (actual - expected) * np.log(actual / expected)
                value = float(contrib.sum())
            per_feature[f] = value
            tables[f] = pd.DataFrame(
                {
                    "bucket": pd.Series(fb.labels, dtype="string"),
                    "expected_share": expected,
                    "actual_share": actual,
                    "contribution": contrib,
                },
                columns=list(TABLE_COLUMNS),
            )
        finite = [v for v in per_feature.values() if math.isfinite(v)]
        agg_max = float(max(finite)) if finite else float("nan")
        agg_mean = float(np.mean(finite)) if finite else float("nan")
        return PSIResult(
            per_feature=per_feature,
            tables=tables,
            aggregate_max=agg_max,
            aggregate_mean=agg_mean,
            n_reference=self.n_reference,
            n_current=n_current,
            missing_features=missing,
        )

    # ----- serialisation -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "n_bins": self.n_bins,
            "eps": self.eps,
            "n_reference": self.n_reference,
            "unavailable": list(self.unavailable),
            "bins": {f: fb.to_dict() for f, fb in self.bins.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PSICalculator:
        calc = cls(d["features"], n_bins=d.get("n_bins", 10), eps=d.get("eps", 1e-4))
        calc.bins = {str(f): _FeatureBins.from_dict(b) for f, b in d.get("bins", {}).items()}
        calc.unavailable = [str(f) for f in d.get("unavailable", [])]
        calc.n_reference = int(d.get("n_reference", 0))
        return calc

    def bucket_labels(self, feature: str) -> Sequence[str]:
        return self.bins[feature].labels
