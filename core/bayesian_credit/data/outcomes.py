"""Outcome derivation, temporal split and dataset summary (SPEC §1.2, Appendix F.2).

``derive_outcomes`` turns the TERMINAL / INTERIM columns of a LoanTable into the ``y_*``
columns used by the Two-Stage Beta-Beta model (§3.3), the reward function and the replay
simulation. Terminal information therefore only ever enters the framework through these
columns, which :class:`~bayesian_credit.data.schema.LeakageGuard` classifies as
``TERMINAL`` (except ``y_group`` / ``y_historical_arm``, which are functions of
pre-decision / policy columns).

Definitions (``cfg`` is a :class:`~bayesian_credit.config.DataConfig`):

* ``y_resolved``  - loan_status in ``STATUS_RESOLVED`` (Fully Paid / Charged Off / Default
  and the two "Does not meet the credit policy" variants).
* ``y_default``   - 1.0 for ``STATUS_DEFAULTED``, 0.0 for ``STATUS_FULLY_PAID``, NaN otherwise.
* ``y_recovery_rate`` - defaults only. ``recovery_basis="principal"`` (default):
  ``clip((total_rec_prncp + recoveries - collection_recovery_fee) / funded_amnt, 0, 1)``,
  so ``E[L] = L * P(D) * (1 - E[R])`` is exactly the expected principal loss.
  ``"post_chargeoff"``: ``(recoveries - fee) / (funded_amnt - total_rec_prncp)`` clipped to
  [0, 1] (1.0 when nothing was outstanding).
* ``y_realized_revenue`` = total_rec_int + total_rec_late_fee (revenue to date).
* ``y_realized_loss`` = max(funded_amnt - total_rec_prncp - (recoveries - fee), 0) for
  defaults, 0 for fully paid, NaN while unresolved.
* ``y_months_on_book`` = whole months issue_d -> last_pymnt_d; when last_pymnt_d is
  missing, issue_d -> snapshot for unresolved loans and the floor (1) for resolved ones;
  clipped to [1, term].
* ``y_resolution_date`` - fully paid: last_pymnt_d; default: last_pymnt_d (or issue_d when
  no payment was ever made) + ``chargeoff_lag_months``; never later than the snapshot
  (a charge-off observed at the snapshot happened by then); NaT while unresolved.
* ``y_historical_arm`` - ``ArmSpace.nearest_many(loan_amnt, int_rate, term)``.
* ``y_group`` - ``Segmenter.assign`` key.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..config import ArmSpaceConfig, DataConfig, SegmentationConfig
from ..exceptions import DataSourceError
from .arms import ArmSpace
from .loader import STRING_DTYPE
from .schema import (
    ID_COLUMN,
    OUTCOME_COLUMNS,
    STATUS_DEFAULTED,
    STATUS_FULLY_PAID,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED,
)
from .segmentation import Segmenter

#: Columns ``derive_outcomes`` needs.
REQUIRED_FOR_OUTCOMES: tuple[str, ...] = (
    "loan_status",
    "issue_d",
    "last_pymnt_d",
    "term",
    "loan_amnt",
    "int_rate",
    "funded_amnt",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
)

_ARM_CHUNK = 100_000  # rows per nearest-arm batch (bounds the n x K distance matrix)


# ---------------------------------------------------------------------------------------
# month arithmetic (shared with the panel and the synthetic generator)
# ---------------------------------------------------------------------------------------
def month_index(dates: Any) -> np.ndarray:
    """Months since 1970-01 as float (NaN for NaT); only differences are meaningful."""
    arr = _as_datetime64(dates)
    out = arr.astype("datetime64[M]").astype(np.int64).astype(float)
    out[np.isnat(arr)] = np.nan
    return out


def months_between(start: Any, end: Any) -> np.ndarray:
    """Whole calendar months from ``start`` to ``end`` (float array, NaN if either is NaT).

    LC dates are month-precision (``Dec-2015`` == 2015-12-01), so this is the number of
    monthly payment dates between issuance and ``end``.
    """
    a = month_index(start)
    b = month_index(end)
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)
    return b - a


def add_months(dates: Any, months: Any) -> pd.Series:
    """Month-start dates ``months`` after ``dates`` (NaT / NaN propagate)."""
    idx = month_index(dates)
    total = idx + np.asarray(months, dtype=float)
    ok = np.isfinite(total)
    res = np.where(ok, total, 0).astype(np.int64).astype("datetime64[M]").astype("datetime64[ns]")
    res[~ok] = np.datetime64("NaT", "ns")
    return pd.Series(res, dtype="datetime64[ns]")


def _as_datetime64(values: Any) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.Index)) and pd.api.types.is_datetime64_dtype(values.dtype):
        return values.to_numpy()
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.datetime64):
        return np.atleast_1d(values)
    if isinstance(values, (pd.Timestamp, str)) or np.isscalar(values):
        values = [values]
    return np.atleast_1d(pd.DatetimeIndex(pd.to_datetime(np.asarray(values))).to_numpy())


def snapshot_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise DataSourceError("snapshot_date must be a valid date")
    return ts


# ---------------------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------------------
def status_flags(status: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fully_paid, defaulted, known) boolean arrays for a loan_status column."""
    s = status.astype(STRING_DTYPE)
    paid = s.isin(STATUS_FULLY_PAID).to_numpy(dtype=bool)
    default = s.isin(STATUS_DEFAULTED).to_numpy(dtype=bool)
    known = s.isin(STATUS_RESOLVED | STATUS_UNRESOLVED).to_numpy(dtype=bool)
    return paid, default, known


def recovery_rate(df: pd.DataFrame, basis: str = "principal") -> np.ndarray:
    """Recovery rate for every row (callers mask to defaults)."""
    funded = _funded(df)
    rec_prncp = df["total_rec_prncp"].to_numpy(dtype=float)
    net = _net_recoveries(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        if basis == "principal":
            rr = (np.nan_to_num(rec_prncp) + net) / funded
        elif basis == "post_chargeoff":
            outstanding = funded - np.nan_to_num(rec_prncp)
            rr = np.where(outstanding > 0, net / outstanding, 1.0)
        else:
            raise DataSourceError("recovery_basis must be 'principal' or 'post_chargeoff'")
    rr = np.clip(rr, 0.0, 1.0)
    return np.where(np.isfinite(funded) & (funded > 0), rr, np.nan)


def _funded(df: pd.DataFrame) -> np.ndarray:
    funded = df["funded_amnt"].to_numpy(dtype=float)
    return np.where(np.isfinite(funded), funded, df["loan_amnt"].to_numpy(dtype=float))


def _net_recoveries(df: pd.DataFrame) -> np.ndarray:
    rec = np.nan_to_num(df["recoveries"].to_numpy(dtype=float))
    fee = np.nan_to_num(df["collection_recovery_fee"].to_numpy(dtype=float))
    return rec - fee


def months_on_book(df: pd.DataFrame, snapshot: pd.Timestamp, resolved: np.ndarray) -> np.ndarray:
    """``y_months_on_book`` (see module docstring)."""
    m_last = months_between(df["issue_d"], df["last_pymnt_d"])
    m_snap = months_between(df["issue_d"], np.full(len(df), snapshot))
    mob = np.where(np.isnan(m_last), np.where(resolved, 1.0, m_snap), m_last)
    term = df["term"].to_numpy(dtype=float)
    return np.clip(mob, 1.0, term)


def historical_arms(df: pd.DataFrame, arm_space: ArmSpace) -> np.ndarray:
    """``ArmSpace.nearest_many`` for each historical (loan_amnt, int_rate, term).

    Evaluated once per distinct triple (LC repeats amounts / rates heavily) and in chunks,
    which bounds the n x K distance matrix; results are identical to a direct call.
    """
    keys = pd.MultiIndex.from_arrays(
        [df["loan_amnt"].to_numpy(dtype=float), df["int_rate"].to_numpy(dtype=float), df["term"].to_numpy(dtype=float)]
    )
    codes, uniques = pd.factorize(keys)
    ua = uniques.get_level_values(0).to_numpy(dtype=float)
    ur = uniques.get_level_values(1).to_numpy(dtype=float)
    ut = uniques.get_level_values(2).to_numpy(dtype=float)
    arms = np.empty(len(uniques), dtype=np.int64)
    for lo in range(0, len(uniques), _ARM_CHUNK):
        hi = lo + _ARM_CHUNK
        arms[lo:hi] = arm_space.nearest_many(ua[lo:hi], ur[lo:hi], ut[lo:hi])
    out = np.full(len(df), -1, dtype=np.int64)  # NaN amount / rate -> -1 (as nearest_many)
    ok = codes >= 0
    out[ok] = arms[codes[ok]]
    return out


def derive_outcomes(
    df: pd.DataFrame,
    arm_space: ArmSpace | None = None,
    segmenter: Segmenter | None = None,
    cfg: DataConfig | None = None,
    *,
    snapshot_date: Any = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the SPEC §1.2 ``y_*`` columns (re)computed.

    ``arm_space`` / ``segmenter`` default to the ``ArmSpaceConfig`` / ``SegmentationConfig``
    defaults; ``snapshot_date`` overrides ``cfg.snapshot_date``. Unknown loan_status values
    are treated as unresolved (with a warning).
    """
    cfg = cfg or DataConfig()
    missing = [c for c in REQUIRED_FOR_OUTCOMES if c not in df.columns]
    if missing:
        raise DataSourceError(f"derive_outcomes needs columns {missing}")
    arm_space = arm_space or ArmSpace.from_config(ArmSpaceConfig())
    segmenter = segmenter or Segmenter.from_config(SegmentationConfig())
    snapshot = snapshot_timestamp(snapshot_date if snapshot_date is not None else cfg.snapshot_date)
    lag = int(cfg.chargeoff_lag_months)

    paid, default, known = status_flags(df["loan_status"])
    if (~known).any():
        unknown = sorted(set(df.loc[~known, "loan_status"].astype(str)))
        warnings.warn(f"unknown loan_status values treated as unresolved: {unknown}", stacklevel=2)
    resolved = paid | default

    out = df.drop(columns=[c for c in OUTCOME_COLUMNS if c in df.columns])
    out = out.reset_index(drop=True)

    funded = _funded(df)
    rec_prncp = np.nan_to_num(df["total_rec_prncp"].to_numpy(dtype=float))
    net = _net_recoveries(df)

    out["y_resolved"] = resolved
    out["y_default"] = np.where(default, 1.0, np.where(paid, 0.0, np.nan))
    out["y_recovery_rate"] = np.where(default, recovery_rate(df, cfg.recovery_basis), np.nan)
    interest = df["total_rec_int"].to_numpy(dtype=float)
    late_fee = np.nan_to_num(df["total_rec_late_fee"].to_numpy(dtype=float))
    out["y_realized_revenue"] = interest + late_fee
    loss = np.maximum(funded - rec_prncp - net, 0.0)
    out["y_realized_loss"] = np.where(default, loss, np.where(paid, 0.0, np.nan))
    mob = months_on_book(df, snapshot, resolved)
    out["y_months_on_book"] = mob

    last = pd.to_datetime(df["last_pymnt_d"]).reset_index(drop=True)
    issue = pd.to_datetime(df["issue_d"]).reset_index(drop=True)
    paid_date = last.where(last.notna(), add_months(issue, mob))
    default_anchor = last.where(last.notna(), issue)
    default_date = add_months(default_anchor, np.full(len(df), lag))
    res = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    res[paid] = paid_date[paid]
    res[default] = default_date[default]
    res = res.where(~(res > snapshot), snapshot)
    out["y_resolution_date"] = res.astype("datetime64[ns]")

    out["y_historical_arm"] = historical_arms(df, arm_space)
    out["y_group"] = segmenter.assign(df).reset_index(drop=True).astype(STRING_DTYPE)
    return out


# ---------------------------------------------------------------------------------------
# split & summary
# ---------------------------------------------------------------------------------------
def temporal_split(
    df: pd.DataFrame, train_end: Any, *, stream_end: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, stream): ``issue_d < train_end`` vs ``train_end <= issue_d [<= stream_end]``.

    The default protocol (SPEC §2.12) fits on loans issued before 2015-01-01 and streams
    2015-01 .. 2018-12.
    """
    cut = snapshot_timestamp(train_end)
    issue = pd.to_datetime(df["issue_d"])
    train_mask = (issue < cut).to_numpy()
    stream_mask = ~train_mask
    if stream_end is not None:
        stream_mask &= (issue <= snapshot_timestamp(stream_end)).to_numpy()
    return (
        df.loc[train_mask].reset_index(drop=True),
        df.loc[stream_mask].reset_index(drop=True),
    )


def _float_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def summarize(df: pd.DataFrame, cfg: DataConfig | None = None) -> dict[str, Any]:
    """JSON-friendly dataset summary used by the API (``/datasets/{id}/``).

    Keys: n, issue_min, issue_max, status_counts, n_resolved, resolved_share, n_defaults,
    default_rate (among resolved), mean_recovery (defaults), default_rate_by_grade,
    default_rate_by_issue_year, mean_int_rate, mean_loan_amnt, term_mix. Uses the
    ``y_*`` columns when present, otherwise derives them from loan_status.
    """
    cfg = cfg or DataConfig()
    n = int(len(df))
    out: dict[str, Any] = {"n": n}
    if n == 0:
        out.update(
            issue_min=None, issue_max=None, status_counts={}, n_resolved=0, resolved_share=None,
            n_defaults=0, default_rate=None, mean_recovery=None, default_rate_by_grade={},
            default_rate_by_issue_year={}, mean_int_rate=None, mean_loan_amnt=None, term_mix={},
        )
        return out
    issue = pd.to_datetime(df["issue_d"])
    out["issue_min"] = issue.min().date().isoformat() if issue.notna().any() else None
    out["issue_max"] = issue.max().date().isoformat() if issue.notna().any() else None
    counts = df["loan_status"].astype(STRING_DTYPE).value_counts(dropna=False)
    out["status_counts"] = {str(k): int(v) for k, v in counts.items()}
    if "y_default" in df.columns:
        y = df["y_default"].to_numpy(dtype=float)
    else:
        paid, default, _ = status_flags(df["loan_status"])
        y = np.where(default, 1.0, np.where(paid, 0.0, np.nan))
    resolved = np.isfinite(y)
    out["n_resolved"] = int(resolved.sum())
    out["resolved_share"] = _float_or_none(resolved.mean())
    out["n_defaults"] = int(np.nansum(y))
    out["default_rate"] = _float_or_none(np.nanmean(y)) if resolved.any() else None
    if "y_recovery_rate" in df.columns:
        rr = df["y_recovery_rate"].to_numpy(dtype=float)
    elif all(c in df.columns for c in ("funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee")):
        rr = np.where(y == 1.0, recovery_rate(df, cfg.recovery_basis), np.nan)
    else:
        rr = np.array([np.nan])
    out["mean_recovery"] = _float_or_none(np.nanmean(rr)) if np.isfinite(rr).any() else None
    frame = pd.DataFrame({"y": y, "year": issue.dt.year.to_numpy()})
    if "grade" in df.columns:
        frame["grade"] = df["grade"].astype(STRING_DTYPE).to_numpy()
        by_grade = frame.dropna(subset=["y"]).groupby("grade")["y"].mean()
        out["default_rate_by_grade"] = {str(k): float(v) for k, v in by_grade.items()}
    else:
        out["default_rate_by_grade"] = {}
    by_year = frame.dropna(subset=["y"]).groupby("year")["y"].mean()
    out["default_rate_by_issue_year"] = {str(int(k)): float(v) for k, v in by_year.items()}
    out["mean_int_rate"] = _float_or_none(df["int_rate"].mean()) if "int_rate" in df.columns else None
    out["mean_loan_amnt"] = _float_or_none(df["loan_amnt"].mean()) if "loan_amnt" in df.columns else None
    if "term" in df.columns:
        mix = df["term"].value_counts(normalize=True)
        out["term_mix"] = {str(int(k)): float(v) for k, v in mix.items()}
    else:
        out["term_mix"] = {}
    if ID_COLUMN in df.columns:
        out["n_unique_loans"] = int(df[ID_COLUMN].nunique())
    return out
