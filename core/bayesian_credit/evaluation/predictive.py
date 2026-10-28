"""Predictive-accuracy metrics of the risk layer (Table 11; SPEC §2.12) — pure numpy.

Conventions shared by every function:

* inputs may be lists, numpy arrays or pandas Series (nullable dtypes included);
* pairs where either value is missing / non-finite are dropped before computing;
* when nothing is left — or the metric is undefined (e.g. AUC with a single class) — the
  result is ``NaN`` instead of an exception. Mismatched lengths raise ``ValueError``.

Binary metrics expect ``y ∈ {0, 1}`` and probabilities ``p ∈ [0, 1]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "brier",
    "expected_calibration_error",
    "log_loss",
    "mae",
    "reliability_table",
    "rmse",
    "roc_auc",
    "roc_curve",
    "transition_log_likelihood",
    "transition_row_error",
]


def _to_float(values: Any) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        s = pd.Series(values)
        if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            s = pd.to_numeric(s, errors="coerce")
        return s.to_numpy(dtype=float, na_value=np.nan).ravel()
    arr = np.asarray(values)
    if arr.dtype == object:
        arr = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(arr, dtype=float).ravel()


def _pairs(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    a, b = _to_float(y), _to_float(p)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.size} targets vs {b.size} predictions")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def _binary_pairs(y: Any, p: Any, probabilities: bool = True) -> tuple[np.ndarray, np.ndarray]:
    yy, pp = _pairs(y, p)
    if yy.size and not np.all((yy == 0.0) | (yy == 1.0)):
        raise ValueError("binary targets must be 0/1")
    if probabilities and pp.size and (pp.min() < 0.0 or pp.max() > 1.0):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    return yy, pp


def brier(y: Any, p: Any) -> float:
    """Brier score ``mean((p − y)²)``."""
    yy, pp = _binary_pairs(y, p)
    if yy.size == 0:
        return float("nan")
    return float(np.mean((pp - yy) ** 2))


def log_loss(y: Any, p: Any, eps: float = 1e-12) -> float:
    """Binary cross-entropy ``−mean(y log p + (1 − y) log(1 − p))`` with p clipped to
    ``[eps, 1 − eps]``."""
    yy, pp = _binary_pairs(y, p)
    if yy.size == 0:
        return float("nan")
    pp = np.clip(pp, eps, 1.0 - eps)
    return float(-np.mean(yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp)))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(float)
    starts = ends - counts + 1.0
    return ((starts + ends) / 2.0)[inverse]


def roc_auc(y: Any, p: Any) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get half credit).

    ``AUC = (Σ ranks(positives) − n₊(n₊ + 1)/2) / (n₊ n₋)``; NaN unless both classes occur.
    ``p`` may be any score (not necessarily a probability).
    """
    yy, pp = _binary_pairs(y, p, probabilities=False)
    n_pos = int(yy.sum())
    n_neg = int(yy.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(pp)
    u = ranks[yy == 1.0].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(fpr, tpr, thresholds)`` at every distinct score, starting at ``(0, 0, +inf)``
    (equivalent to ``sklearn.metrics.roc_curve(..., drop_intermediate=False)``). Empty
    arrays when either class is absent."""
    yy, pp = _binary_pairs(y, p, probabilities=False)
    n_pos = yy.sum()
    n_neg = yy.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    order = np.argsort(-pp, kind="mergesort")
    ps, ys = pp[order], yy[order]
    last = np.r_[np.nonzero(np.diff(ps))[0], ps.size - 1]  # last index of each distinct score
    tps = np.cumsum(ys)[last]
    fps = (last + 1) - tps
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    return fpr, tpr, np.r_[np.inf, ps[last]]


def reliability_table(y: Any, p: Any, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability bins on [0, 1]: ``bin_low, bin_high, n, mean_pred, frac_pos``.

    Every bin is listed; empty bins have ``n = 0`` and NaN means. ``p = 1`` falls in the
    last bin.
    """
    if int(n_bins) < 1:
        raise ValueError("n_bins must be >= 1")
    n_bins = int(n_bins)
    yy, pp = _binary_pairs(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.floor(pp * n_bins).astype(np.int64), 0, n_bins - 1)
    n = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=pp, minlength=n_bins)
    sum_y = np.bincount(idx, weights=yy, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_pred = np.where(n > 0, sum_p / np.maximum(n, 1), np.nan)
        frac_pos = np.where(n > 0, sum_y / np.maximum(n, 1), np.nan)
    return pd.DataFrame(
        {
            "bin_low": edges[:-1],
            "bin_high": edges[1:],
            "n": n.astype(np.int64),
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
        }
    )


def expected_calibration_error(y: Any, p: Any, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """``ECE = Σ_b (n_b / N)·|frac_pos_b − mean_pred_b|`` over equal-width bins, plus the
    reliability table (:func:`reliability_table`). ECE is NaN for empty input."""
    table = reliability_table(y, p, n_bins)
    total = int(table["n"].sum())
    if total == 0:
        return float("nan"), table
    filled = table["n"].to_numpy() > 0
    gaps = np.abs(table["frac_pos"].to_numpy()[filled] - table["mean_pred"].to_numpy()[filled])
    return float(np.sum(table["n"].to_numpy()[filled] / total * gaps)), table


def mae(y: Any, y_pred: Any) -> float:
    """Mean absolute error (e.g. realised vs expected recovery rate)."""
    a, b = _pairs(y, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def rmse(y: Any, y_pred: Any) -> float:
    """Root-mean-squared error."""
    a, b = _pairs(y, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def _matrix_pair(P: Any, counts: Any) -> tuple[np.ndarray, np.ndarray]:
    PP = np.asarray(P, dtype=float)
    C = np.asarray(counts, dtype=float)
    if PP.shape != C.shape:
        raise ValueError(f"shape mismatch: P {PP.shape} vs counts {C.shape}")
    if C.size and (not np.all(np.isfinite(C)) or np.any(C < 0)):
        raise ValueError("transition counts must be finite and non-negative")
    return PP, C


def transition_log_likelihood(P: Any, counts: Any, normalize: bool = False) -> float:
    """Holdout transition log-likelihood ``Σ_{ij} n_ij log P_ij`` over cells with
    ``n_ij > 0`` (``-inf`` if such a cell has ``P_ij = 0``). ``normalize=True`` divides by
    the number of transitions (mean log-likelihood per transition). NaN when there are no
    transitions."""
    PP, C = _matrix_pair(P, counts)
    total = float(C.sum()) if C.size else 0.0
    if total <= 0:
        return float("nan")
    mask = C > 0
    if np.any(PP[mask] <= 0):
        return float("-inf")
    ll = float(np.sum(C[mask] * np.log(PP[mask])))
    return ll / total if normalize else ll


def transition_row_error(P: Any, counts: Any) -> np.ndarray:
    """Per-row L1 distance ``Σ_j |P_ij − n_ij / n_i|`` between the predicted and empirical
    transition rows; NaN for rows without observed transitions (e.g. absorbing rows)."""
    PP, C = _matrix_pair(P, counts)
    if C.ndim != 2:
        return np.full(C.shape[0] if C.ndim else 0, np.nan)
    n = C.sum(axis=1)
    out = np.full(C.shape[0], np.nan)
    rows = n > 0
    out[rows] = np.abs(PP[rows] - C[rows] / n[rows, None]).sum(axis=1)
    return out
