"""LoanStatePanel - monthly credit-state paths reconstructed from the LC snapshot (SPEC §2.6).

Lending Club publishes one snapshot row per loan, not a monthly panel. Appendix G.2 needs
transition counts between the MCTM states S = {C, D, R, L} (§3.3, Table 6), so each
loan's path is reconstructed deterministically from ``loan_status``, ``issue_d``,
``last_pymnt_d``, ``term`` and ``total_rec_late_fee`` (**[decision]** heuristic):

* **Fully Paid, never late** (``total_rec_late_fee == 0``) - C for each month to payoff
  (months issue -> last payment, clipped to [1, term]); then one R month under
  ``TransitionStructure.EXTENDED``; under ``PROPOSAL`` payoff of a never-delinquent loan is
  right-censoring (no C -> R count, P[C,R] = 0).
* **Fully Paid with late fees** - Table 6 defines *Recovered* as "repayment/recovery
  following delinquency", and a loan that paid late fees was certainly delinquent at some
  point; a fully paid one is therefore literally repayment following delinquency. The
  episode is encoded immediately before payoff and resolves D -> R under **both**
  structures: C for the first k-1 of its k payment months, D in the payoff month, then R
  (C -> C x (k - 2), C -> D, D -> R; just D -> R when k = 1).
* **Current with late fees** - still on book, so the delinquency certainly *cured*: one
  C -> C is replaced by C -> D + D -> C, i.e. a single D month inserted at the midpoint of
  its performing run (needs >= 2 C months). D -> C is thus estimated from the population
  where a cure is unambiguous and D -> R from the one where repayment after delinquency
  is unambiguous.
* **Current** otherwise - C for the months on book (issue -> last payment, else ->
  snapshot; clipped to [1, term]); censored in C.
* **Charged Off / Default** - C for each paid month (C -> C x (m_last - 1)), then D for
  ``chargeoff_lag_months`` (C -> D, D -> D x (lag - 1)), then L (D -> L). No payment ever
  -> D from month 1.
* **In Grace Period / Late (16-30 days)** - C for each paid month, then one D month
  (censored in D).
* **Late (31-120 days)** - C for each paid month, then D for the months since the last
  payment (clipped to [1, lag]): C -> D, D -> D x (k - 1); censored in D.

Paths are stored as run-length *segments* ``[(state, run_length), ...]`` (<= 4 per loan),
never as a dense month grid. A loan with a path of M months contributes M - 1 transitions,
so ``counts.sum() == n_loan_months - n_loans``; rows R and L of ``counts`` are zero
(absorbing - nothing is counted out of them).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import DataConfig, MCTMConfig
from ..exceptions import DataSourceError
from ..types import State, TransitionStructure
from .loader import STRING_DTYPE
from .outcomes import months_between, snapshot_timestamp
from .schema import ID_COLUMN, STATUS_CURRENT, STATUS_DEFAULTED, STATUS_FULLY_PAID

N_STATES = len(State)
MAX_SEGMENTS = 4
_C, _D, _R, _L = int(State.CURRENT), int(State.DELINQUENT), int(State.RECOVERED), int(State.LOSS)

# internal status classes
ST_UNKNOWN, ST_PAID, ST_DEFAULT, ST_CURRENT, ST_GRACE, ST_LATE = 0, 1, 2, 3, 4, 5
_GRACE_STATUSES = frozenset({"In Grace Period", "Late (16-30 days)"})
_LATE_STATUSES = frozenset({"Late (31-120 days)"})

PANEL_COLUMNS: tuple[str, ...] = ("loan_status", "issue_d", "last_pymnt_d", "term", "total_rec_late_fee")
DEFAULT_SNAPSHOT: str = DataConfig().snapshot_date

Segments = list[tuple[int, int]]


def _status_class_scalar(status: Any) -> int:
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return ST_UNKNOWN
    s = str(status)
    if s in STATUS_FULLY_PAID:
        return ST_PAID
    if s in STATUS_DEFAULTED:
        return ST_DEFAULT
    if s in STATUS_CURRENT:
        return ST_CURRENT
    if s in _GRACE_STATUSES:
        return ST_GRACE
    if s in _LATE_STATUSES:
        return ST_LATE
    return ST_UNKNOWN


def classify_status(status: pd.Series) -> np.ndarray:
    """Map loan_status to the panel's internal status classes (int8)."""
    s = status.astype(STRING_DTYPE)
    out = np.full(len(s), ST_UNKNOWN, dtype=np.int8)
    out[s.isin(STATUS_FULLY_PAID).to_numpy(dtype=bool)] = ST_PAID
    out[s.isin(STATUS_DEFAULTED).to_numpy(dtype=bool)] = ST_DEFAULT
    out[s.isin(STATUS_CURRENT).to_numpy(dtype=bool)] = ST_CURRENT
    out[s.isin(_GRACE_STATUSES).to_numpy(dtype=bool)] = ST_GRACE
    out[s.isin(_LATE_STATUSES).to_numpy(dtype=bool)] = ST_LATE
    return out


# ---------------------------------------------------------------------------------------
# vectorised segment construction
# ---------------------------------------------------------------------------------------
def build_segments(
    df: pd.DataFrame,
    lag: int,
    structure: TransitionStructure,
    snapshot: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(states, lengths, status_class)``; states/lengths are ``(n, 4)`` arrays.

    Segments are left-aligned; unused slots have length 0. Rows with an unknown status
    get an empty path.
    """
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"LoanStatePanel needs columns {missing}")
    if lag < 1:
        raise DataSourceError("chargeoff_lag_months must be >= 1")
    n = len(df)
    cls = classify_status(df["loan_status"])
    term = df["term"].to_numpy(dtype=np.int64)
    m_last = months_between(df["issue_d"], df["last_pymnt_d"])
    m_snap = months_between(df["issue_d"], np.full(n, snapshot))
    has_last = np.isfinite(m_last)
    m_last0 = np.where(has_last, m_last, 0.0)
    fee = np.nan_to_num(df["total_rec_late_fee"].to_numpy(dtype=float)) > 0
    extended = TransitionStructure(structure) == TransitionStructure.EXTENDED

    n_paid = np.clip(np.where(has_last, m_last, 1.0), 1, term)
    n_cur = np.clip(np.where(has_last, m_last, np.nan_to_num(m_snap, nan=1.0)), 1, term)
    n_def = np.clip(m_last0, 0, term)
    k_late = np.clip(np.where(has_last, m_snap - m_last, m_snap), 1, lag)
    k_late = np.nan_to_num(k_late, nan=1.0)

    states = np.zeros((n, MAX_SEGMENTS), dtype=np.int8)
    lengths = np.zeros((n, MAX_SEGMENTS), dtype=np.int64)

    # fully paid: never late -> [C k] (+ [R 1] under EXTENDED; PROPOSAL censors);
    # paid late fees -> repayment following delinquency [C k-1][D 1][R 1] under both structures
    paid = cls == ST_PAID
    paid_fee = paid & fee
    paid_clean = paid & ~fee
    k_paid = n_paid.astype(np.int64)
    states[paid, 0], lengths[paid, 0] = _C, np.where(fee, k_paid - 1, k_paid)[paid]
    states[paid_fee, 1], lengths[paid_fee, 1] = _D, 1
    states[paid_fee, 2], lengths[paid_fee, 2] = _R, 1
    states[paid_clean, 3] = _R
    if extended:
        lengths[paid_clean, 3] = 1

    # current: a late fee means a delinquency that certainly cured -> [C a][D 1][C b]
    cur = cls == ST_CURRENT
    n_c = n_cur.astype(np.int64)
    cured = cur & fee & (n_c >= 2)
    first = np.where(cured, n_c // 2, n_c)
    states[cur, 0], lengths[cur, 0] = _C, first[cur]
    states[cur, 1], lengths[cur, 1] = _D, cured[cur].astype(np.int64)
    states[cur, 2], lengths[cur, 2] = _C, np.where(cured, n_c - n_c // 2, 0)[cur]

    # defaults: C x paid months, D x lag, L
    dflt = cls == ST_DEFAULT
    states[dflt, 0], lengths[dflt, 0] = _C, n_def[dflt].astype(np.int64)
    states[dflt, 1], lengths[dflt, 1] = _D, lag
    states[dflt, 2], lengths[dflt, 2] = _L, 1

    # delinquent but unresolved
    grace = cls == ST_GRACE
    states[grace, 0], lengths[grace, 0] = _C, n_def[grace].astype(np.int64)
    states[grace, 1], lengths[grace, 1] = _D, 1
    late = cls == ST_LATE
    states[late, 0], lengths[late, 0] = _C, n_def[late].astype(np.int64)
    states[late, 1], lengths[late, 1] = _D, k_late[late].astype(np.int64)

    # left-align non-empty segments
    order = np.argsort(lengths == 0, axis=1, kind="stable")
    states = np.take_along_axis(states, order, axis=1)
    lengths = np.take_along_axis(lengths, order, axis=1)
    return states, lengths, cls


def _counts_from_arrays(states: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    idx_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for j in range(states.shape[1]):
        m = lengths[:, j] > 0
        s = states[m, j].astype(np.int64)
        idx_parts.append(s * N_STATES + s)
        w_parts.append((lengths[m, j] - 1).astype(float))
    for j in range(states.shape[1] - 1):
        m = (lengths[:, j] > 0) & (lengths[:, j + 1] > 0)
        idx_parts.append(states[m, j].astype(np.int64) * N_STATES + states[m, j + 1].astype(np.int64))
        w_parts.append(np.ones(int(m.sum())))
    if not idx_parts:
        return np.zeros((N_STATES, N_STATES))
    counts = np.bincount(
        np.concatenate(idx_parts), weights=np.concatenate(w_parts), minlength=N_STATES * N_STATES
    )
    return counts.reshape(N_STATES, N_STATES)


def _last_states(states: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    n_seg = (lengths > 0).sum(axis=1)
    idx = np.maximum(n_seg - 1, 0)
    last = states[np.arange(len(states)), idx].astype(np.int64)
    return np.where(n_seg > 0, last, -1)


def paths_to_counts(segments: Sequence[tuple[int, int]] | Iterable[Sequence[tuple[int, int]]]) -> np.ndarray:
    """4x4 transition counts of one path ``[(state, run), ...]`` or of many such paths."""
    seq = list(segments)
    if not seq:
        return np.zeros((N_STATES, N_STATES))
    first = seq[0]
    paths = [seq] if (len(first) == 2 and np.isscalar(first[0])) else seq
    counts = np.zeros((N_STATES, N_STATES))
    for path in paths:
        prev: int | None = None
        for state, run in path:
            s, r = int(state), int(run)
            if r <= 0:
                continue
            if prev is not None:
                counts[prev, s] += 1
            counts[s, s] += r - 1
            prev = s
    return counts


# ---------------------------------------------------------------------------------------
# scalar path (fast pure-python twin of build_segments, used by interim_state)
# ---------------------------------------------------------------------------------------
def _get(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    if isinstance(row, pd.Series):
        return row.get(key)
    return getattr(row, key, None)


def _month_number(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(value)
    return ts.year * 12 + ts.month - 1


def row_segments(
    row: Any,
    lag: int = 4,
    structure: TransitionStructure = TransitionStructure.PROPOSAL,
    snapshot: Any = DEFAULT_SNAPSHOT,
) -> Segments:
    """Reconstructed path of a single loan as ``[(state, run_length), ...]``."""
    cls = _status_class_scalar(_get(row, "loan_status"))
    if cls == ST_UNKNOWN:
        return []
    issue = _month_number(_get(row, "issue_d"))
    if issue is None:
        return []
    last = _month_number(_get(row, "last_pymnt_d"))
    snap = _month_number(snapshot)
    term = int(_get(row, "term"))
    fee_raw = _get(row, "total_rec_late_fee")
    fee = fee_raw is not None and not pd.isna(fee_raw) and float(fee_raw) > 0
    m_last = None if last is None else last - issue
    m_snap = snap - issue

    def clip(v: float, lo: float, hi: float) -> int:
        return int(min(max(v, lo), hi))

    segs: Segments = []
    if cls == ST_PAID:
        k = clip(1 if m_last is None else m_last, 1, term)
        if fee:  # repayment following delinquency: D in the payoff month, then R (both structures)
            return ([(_C, k - 1)] if k > 1 else []) + [(_D, 1), (_R, 1)]
        segs = [(_C, k)]
        if TransitionStructure(structure) == TransitionStructure.EXTENDED:
            segs.append((_R, 1))
        return segs
    if cls == ST_CURRENT:
        n_c = clip(m_snap if m_last is None else m_last, 1, term)
        if fee and n_c >= 2:  # cured delinquency
            return [(_C, n_c // 2), (_D, 1), (_C, n_c - n_c // 2)]
        return [(_C, n_c)]
    n_def = clip(0 if m_last is None else m_last, 0, term)
    if n_def > 0:
        segs.append((_C, n_def))
    if cls == ST_DEFAULT:
        segs += [(_D, lag), (_L, 1)]
    elif cls == ST_GRACE:
        segs.append((_D, 1))
    else:  # ST_LATE
        k = m_snap if m_last is None else m_snap - m_last
        segs.append((_D, clip(k, 1, lag)))
    return segs


def _state_at(segs: Segments, month: int, paid_off: bool) -> tuple[State, int]:
    if month < 1:
        raise ValueError("month is 1-based (loan-month 1 = first month after issuance)")
    pos = 0
    paid = 0
    prev: int | None = None
    for state, run in segs:
        if state == _R and prev == _D:
            paid += 1  # the delinquent payoff month: installment honoured with the payoff
        if month <= pos + run:
            if state == _C:
                paid += month - pos
            return State(state), paid
        if state == _C:
            paid += run
        pos += run
        prev = state
    if paid_off:
        return State.RECOVERED, paid
    if not segs:
        return State.CURRENT, 0
    return State(segs[-1][0]), paid


def interim_state(
    row: Any,
    month: int,
    cfg: MCTMConfig | None = None,
    snapshot_date: Any = None,
) -> tuple[State, int]:
    """State at loan-month ``month`` (1-based) and the number of installments paid so far.

    Same heuristic as the panel. ``months_paid`` counts the C months so far; for a fully
    paid loan that paid late fees the delinquent payoff month (D) is counted once the loan
    reaches R, so a repaid loan always ends with ``months_paid`` = its number of payment
    months. After the end of the reconstructed path the loan stays in its absorbing state
    (fully paid -> RECOVERED under both structures, since payoff is observed; charged off
    -> LOSS); censored loans (Current / late at the snapshot) carry their last observed
    state forward.
    """
    cfg = cfg or MCTMConfig()
    snap = snapshot_date if snapshot_date is not None else DEFAULT_SNAPSHOT
    segs = row_segments(row, cfg.chargeoff_lag_months, TransitionStructure.EXTENDED, snap)
    paid_off = _status_class_scalar(_get(row, "loan_status")) == ST_PAID
    return _state_at(segs, int(month), paid_off)


# ---------------------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------------------
@dataclass
class LoanStatePanel:
    """Transition counts of the reconstructed monthly credit-state panel (SPEC §2.6, G.2).

    ``counts[i, j]`` = number of loan-months in state ``i`` followed by state ``j``
    (indices are :class:`~bayesian_credit.types.State` values), the sufficient statistic
    of the MCTM's Dirichlet-multinomial rows (§3.3). ``paths`` (only with
    ``keep_paths=True``) holds each loan's run-length segments for holdout evaluation.
    """

    counts: np.ndarray
    n_loans: int
    n_loan_months: int
    n_censored: int
    counts_by_cohort: dict[int, np.ndarray]
    structure: TransitionStructure = TransitionStructure.PROPOSAL
    chargeoff_lag_months: int = 4
    snapshot_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(DEFAULT_SNAPSHOT))
    n_skipped: int = 0
    paths: pd.DataFrame | None = None

    # ----- construction ---------------------------------------------------------------
    @classmethod
    def from_table(
        cls,
        df: pd.DataFrame,
        cfg: MCTMConfig | None = None,
        snapshot_date: Any = None,
        *,
        keep_paths: bool = False,
    ) -> LoanStatePanel:
        """Reconstruct every loan's path and aggregate transition counts."""
        cfg = cfg or MCTMConfig()
        snapshot = snapshot_timestamp(snapshot_date if snapshot_date is not None else DEFAULT_SNAPSHOT)
        structure = TransitionStructure(cfg.structure)
        lag = int(cfg.chargeoff_lag_months)
        states, lengths, status_cls = build_segments(df, lag, structure, snapshot)
        counts = _counts_from_arrays(states, lengths)
        path_len = lengths.sum(axis=1)
        has_path = path_len > 0
        last = _last_states(states, lengths)
        n_loans = int(has_path.sum())
        n_months = int(path_len.sum())
        n_censored = int((has_path & ((last == _C) | (last == _D))).sum())

        years = pd.to_datetime(df["issue_d"]).dt.year.to_numpy()
        by_cohort: dict[int, np.ndarray] = {}
        for year in np.unique(years[has_path]):
            m = years == year
            by_cohort[int(year)] = _counts_from_arrays(states[m], lengths[m])

        paths = None
        if keep_paths:
            ids = df[ID_COLUMN].to_numpy() if ID_COLUMN in df.columns else np.arange(len(df))
            seg_lists = [
                [(int(s), int(r)) for s, r in zip(srow, lrow, strict=True) if r > 0]
                for srow, lrow in zip(states.tolist(), lengths.tolist(), strict=True)
            ]
            paths = pd.DataFrame({ID_COLUMN: ids, "segments": seg_lists})

        panel = cls(
            counts=counts,
            n_loans=n_loans,
            n_loan_months=n_months,
            n_censored=n_censored,
            counts_by_cohort=by_cohort,
            structure=structure,
            chargeoff_lag_months=lag,
            snapshot_date=snapshot,
            n_skipped=int((status_cls == ST_UNKNOWN).sum()),
            paths=paths,
        )
        panel.validate()
        return panel

    def validate(self) -> None:
        """Structural invariants: absorbing rows empty, identity, PROPOSAL zeros."""
        c = self.counts
        if c.shape != (N_STATES, N_STATES) or (c < 0).any():
            raise DataSourceError("transition counts must be a non-negative 4x4 array")
        if c[_R].sum() != 0 or c[_L].sum() != 0:
            raise DataSourceError("absorbing states R and L must have no outgoing counts")
        if int(round(c.sum())) != self.n_loan_months - self.n_loans:
            raise DataSourceError("total transitions must equal n_loan_months - n_loans")
        if self.structure == TransitionStructure.PROPOSAL and (c[_C, _R] != 0 or c[_C, _L] != 0):
            raise DataSourceError("PROPOSAL structure requires zero C->R and C->L counts")

    # ----- reuse ------------------------------------------------------------------------
    @property
    def config(self) -> MCTMConfig:
        return MCTMConfig(structure=self.structure, chargeoff_lag_months=self.chargeoff_lag_months)

    def segments_for(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(states, lengths)`` segment arrays for ``df`` under this panel's settings."""
        states, lengths, _ = build_segments(df, self.chargeoff_lag_months, self.structure, self.snapshot_date)
        return states, lengths

    def transition_counts_for(self, df: pd.DataFrame) -> np.ndarray:
        """Counts for another (e.g. holdout) table under the same heuristic / snapshot."""
        return _counts_from_arrays(*self.segments_for(df))

    def interim_state(self, row: Any, month: int) -> tuple[State, int]:
        """:func:`interim_state` with this panel's lag and snapshot."""
        return interim_state(row, month, self.config, self.snapshot_date)

    def interim_states(self, df: pd.DataFrame, month: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`interim_state` for every row of ``df``.

        Returns ``(states, months_paid)`` int arrays; ``month`` is a scalar or per-row array.
        """
        months = np.broadcast_to(np.asarray(month, dtype=np.int64), (len(df),))
        if (months < 1).any():
            raise ValueError("month is 1-based (loan-month 1 = first month after issuance)")
        states, lengths, cls = build_segments(
            df, self.chargeoff_lag_months, TransitionStructure.EXTENDED, self.snapshot_date
        )
        rows = np.arange(len(df))
        cum = np.cumsum(lengths, axis=1)
        start = cum - lengths
        mm = months[:, None]
        inside = (mm > start) & (mm <= cum) & (lengths > 0)
        found = inside.any(axis=1)
        j = np.argmax(inside, axis=1)
        c_len = np.where(states == _C, lengths, 0)
        c_cum = np.cumsum(c_len, axis=1)
        st = states[rows, j].astype(np.int64)
        paid_in = c_cum[rows, j] - c_len[rows, j] + np.where(st == _C, months - start[rows, j], 0)
        total_c = c_cum[:, -1]
        last = _last_states(states, lengths)
        beyond_state = np.where(cls == ST_PAID, _R, np.where(last < 0, _C, last))
        out_state = np.where(found, st, beyond_state)
        # repayment following delinquency: the D payoff month counts once R is reached
        prev = np.concatenate([np.full((len(df), 1), -1, dtype=states.dtype), states[:, :-1]], axis=1)
        has_dr = ((states == _R) & (prev == _D) & (lengths > 0)).any(axis=1)
        reached_r = (found & (st == _R)) | ~found
        out_paid = np.where(found, paid_in, total_c) + (has_dr & reached_r)
        return out_state.astype(np.int64), out_paid.astype(np.int64)

    # ----- views ------------------------------------------------------------------------
    def empirical_matrix(self) -> np.ndarray:
        """Row-normalised counts (MLE); absorbing rows are identity, empty rows NaN."""
        c = self.counts.astype(float)
        totals = c.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(totals > 0, c / totals, np.nan)
        p[_R] = np.eye(N_STATES)[_R]
        p[_L] = np.eye(N_STATES)[_L]
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [s.label for s in State],
            "structure": self.structure.value,
            "chargeoff_lag_months": self.chargeoff_lag_months,
            "snapshot_date": self.snapshot_date.date().isoformat(),
            "counts": self.counts.tolist(),
            "n_loans": self.n_loans,
            "n_loan_months": self.n_loan_months,
            "n_censored": self.n_censored,
            "n_skipped": self.n_skipped,
            "counts_by_cohort": {str(k): v.tolist() for k, v in self.counts_by_cohort.items()},
        }
