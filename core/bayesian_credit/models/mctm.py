"""Markov Chain Transition Matrix over credit states (proposal §3.3, pp. 27-28; SPEC §2.5).

State space ``S = {C, D, R, L}`` (Current, Delinquent, Recovered, Absorbed Loss)::

        ⎡ p_CC  p_CD  0     0    ⎤
    P = ⎢ p_DC  p_DD  p_DR  p_DL ⎥        P_RR = P_LL = 1   (R, L absorbing)
        ⎢ 0     0     1     0    ⎥
        ⎣ 0     0     0     1    ⎦

    π_t = π_0 Pᵗ = [π_C(t), π_D(t), π_R(t), π_L(t)]
    P(Default, t) = π_D(t) + π_L(t)
    lim_{k→∞} π_k = [0, 0, π_R^∞, π_L^∞]
    P(R | D, t),  P(L | D, t)

Absorption uses the fundamental matrix of the transient block ``Q = P[{C,D},{C,D}]``:
``N = (I − Q)⁻¹``, absorption probabilities ``B = N·R_block`` and expected steps to
absorption ``t = N·1``. At maturity the "delinquents collapse" (proposal p. 28): C → R and D
splits between R and L in the ratio ``p_DR : p_DL`` (:meth:`maturity_collapse`).
"""

from __future__ import annotations

import hashlib
from typing import Any, Self

import numpy as np
import pandas as pd

from ..config import MCTMConfig
from ..exceptions import ConfigurationError, InvalidTransitionMatrixError
from ..types import ABSORBING_STATES, TRANSIENT_STATES, State, TransitionStructure

__all__ = [
    "N_STATES",
    "START_CURRENT",
    "START_DELINQUENT",
    "MarkovChainTransitionMatrix",
    "State",
    "TransitionStructure",
    "allowed_transitions",
    "validate_transition_matrix",
]

N_STATES = len(State)
_C, _D, _R, _L = (int(s) for s in (State.CURRENT, State.DELINQUENT, State.RECOVERED, State.LOSS))
_TRANSIENT = tuple(int(s) for s in TRANSIENT_STATES)
_ABSORBING = tuple(int(s) for s in ABSORBING_STATES)


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


#: π_0 for a newly originated loan (all mass in Current).
START_CURRENT: np.ndarray = _readonly(np.array([1.0, 0.0, 0.0, 0.0]))
#: π_0 for a loan that is currently delinquent.
START_DELINQUENT: np.ndarray = _readonly(np.array([0.0, 1.0, 0.0, 0.0]))


def allowed_transitions(structure: TransitionStructure | str) -> np.ndarray:
    """Boolean 4×4 mask of transitions that are not structurally zero.

    ``PROPOSAL``: C → {C, D}; D → {C, D, R, L}; R → R; L → L (exactly the §3.3 matrix).
    ``EXTENDED``: additionally C → R (prepayment / scheduled payoff).
    """
    s = TransitionStructure(structure)
    mask = np.zeros((N_STATES, N_STATES), dtype=bool)
    mask[_C, [_C, _D]] = True
    mask[_D, :] = True
    mask[_R, _R] = True
    mask[_L, _L] = True
    if s is TransitionStructure.EXTENDED:
        mask[_C, _R] = True
    return _readonly(mask)


def validate_transition_matrix(
    matrix: Any, structure: TransitionStructure | str = TransitionStructure.PROPOSAL, atol: float = 1e-8
) -> np.ndarray:
    """Return a cleaned copy of ``matrix`` or raise :class:`InvalidTransitionMatrixError`.

    Checks: shape 4×4, finite, non-negative, rows sum to one, absorbing rows are
    ``[0,0,1,0]`` / ``[0,0,0,1]`` and structural zeros of ``structure`` are zero (all within
    ``atol``). Rows are renormalised to remove floating-point noise.
    """
    P = np.array(matrix, dtype=float)
    if P.shape != (N_STATES, N_STATES):
        raise InvalidTransitionMatrixError(f"transition matrix must be 4x4, got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise InvalidTransitionMatrixError("transition matrix has non-finite entries")
    if np.any(P < -atol):
        raise InvalidTransitionMatrixError("transition matrix has negative entries")
    sums = P.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > atol):
        raise InvalidTransitionMatrixError(f"rows must sum to 1, got {sums.tolist()}")
    for s in _ABSORBING:
        expected = np.zeros(N_STATES)
        expected[s] = 1.0
        if np.any(np.abs(P[s] - expected) > atol):
            raise InvalidTransitionMatrixError(f"{State(s).name} must be absorbing, row = {P[s].tolist()}")
    zeros = ~allowed_transitions(structure)
    if np.any(np.abs(P[zeros]) > atol):
        bad = [f"{State(i).name}->{State(j).name}" for i, j in zip(*np.nonzero(zeros & (np.abs(P) > atol)), strict=True)]
        raise InvalidTransitionMatrixError(
            f"structural zeros violated for {TransitionStructure(structure).value}: {bad}"
        )
    P = np.where(zeros, 0.0, np.clip(P, 0.0, None))
    return P / P.sum(axis=1, keepdims=True)


def _as_distribution(pi: Any) -> np.ndarray:
    """State (or int) → one-hot; array-like → validated distribution(s) of shape (4,) or (n, 4)."""
    if isinstance(pi, (State, int, np.integer)) and not isinstance(pi, bool):
        out = np.zeros(N_STATES)
        out[int(State(int(pi)))] = 1.0
        return out
    arr = np.asarray(pi, dtype=float)
    if arr.shape[-1:] != (N_STATES,) or arr.ndim > 2:
        raise ValueError(f"state distribution must have shape (4,) or (n, 4), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(np.abs(arr.sum(axis=-1) - 1.0) > 1e-8):
        raise ValueError("state distribution must be non-negative and sum to 1")
    return arr


def _start_key(start: Any) -> Any:
    """Cheap cache key for a start state / distribution (hot path of the interim reward)."""
    if isinstance(start, (State, int, np.integer)) and not isinstance(start, bool):
        return int(start)
    if start is START_CURRENT:
        return int(State.CURRENT)
    if start is START_DELINQUENT:
        return int(State.DELINQUENT)
    pi = _as_distribution(start)
    if pi.ndim != 1:
        raise ValueError("a single start distribution is required")
    hot = np.flatnonzero(pi == 1.0)
    return int(hot[0]) if hot.size == 1 else tuple(float(v) for v in pi)


class MarkovChainTransitionMatrix:
    """Bayesian MCTM with Dirichlet rows (proposal §3.3; SPEC §2.5).

    Transient rows C, D have ``Dirichlet(prior_strength + counts)`` posteriors over their
    *allowed* cells (:func:`allowed_transitions`); structurally-zero cells get no prior mass,
    ignore any counts (reported in :attr:`diagnostics`) and are forced to 0. Absorbing rows
    are fixed at ``[0,0,1,0]`` and ``[0,0,0,1]``. :attr:`matrix` is the posterior mean.
    ``prior_strength = 0`` gives the MLE; a transient row without counts then "stays put"
    (C → C = 1 or D → D = 1) and is flagged in ``diagnostics["empty_rows"]``.

    Note: ``π_L(t)`` is non-decreasing in t (L is absorbing), but ``π_D(t) + π_L(t)`` is
    only monotone for realistic matrices (it can fall when delinquents cure fast).
    """

    def __init__(
        self,
        structure: TransitionStructure | str = TransitionStructure.PROPOSAL,
        prior_strength: float = 1.0,
    ) -> None:
        self.structure = TransitionStructure(structure)
        prior_strength = float(prior_strength)
        if not np.isfinite(prior_strength) or prior_strength < 0:
            raise ConfigurationError("prior_strength must be a finite number >= 0")
        self.prior_strength = prior_strength
        self.allowed = allowed_transitions(self.structure)
        self.diagnostics: dict[str, Any] = {}
        self.fit_counts(np.zeros((N_STATES, N_STATES)))  # prior mean until fitted
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: MCTMConfig) -> MarkovChainTransitionMatrix:
        return cls(cfg.structure, cfg.prior_strength)

    @classmethod
    def from_matrix(
        cls, P: Any, structure: TransitionStructure | str = TransitionStructure.PROPOSAL
    ) -> MarkovChainTransitionMatrix:
        """Wrap a known matrix (validated). It carries no Dirichlet counts, so
        :meth:`sample_matrix` returns it unchanged."""
        obj = cls(structure=structure, prior_strength=0.0)
        obj._set_matrix(P)
        obj._counts = np.zeros((N_STATES, N_STATES))
        obj._alpha = None
        obj._source = "matrix"
        obj._fitted = True
        obj.diagnostics = {"empty_rows": [], "ignored_structural_counts": 0.0, "ignored_absorbing_counts": 0.0}
        return obj

    # ----- estimation ---------------------------------------------------------------------
    def fit_counts(self, counts: Any) -> Self:
        """Posterior-mean matrix from a 4×4 transition count matrix (rows = from-state)."""
        c = np.array(counts, dtype=float)
        if c.shape != (N_STATES, N_STATES):
            raise ValueError(f"counts must be 4x4, got {c.shape}")
        if not np.all(np.isfinite(c)) or np.any(c < 0):
            raise InvalidTransitionMatrixError("transition counts must be finite and non-negative")
        alpha = np.zeros((N_STATES, N_STATES))
        P = np.zeros((N_STATES, N_STATES))
        empty: list[str] = []
        ignored = 0.0
        for i in _TRANSIENT:
            ok = self.allowed[i]
            ignored += float(c[i, ~ok].sum())
            alpha[i, ok] = self.prior_strength + c[i, ok]
            total = alpha[i].sum()
            if total > 0:
                P[i] = alpha[i] / total
            else:
                P[i, i] = 1.0
                empty.append(State(i).name)
        for s in _ABSORBING:
            P[s, s] = 1.0
        absorbing_off = float(sum(c[s].sum() - c[s, s] for s in _ABSORBING))
        self._set_matrix(P)
        self._counts = c
        self._alpha = alpha
        self._source = "counts"
        self._fitted = True
        self.diagnostics = {
            "empty_rows": empty,
            "ignored_structural_counts": ignored,
            "ignored_absorbing_counts": absorbing_off,
        }
        return self

    def _set_matrix(self, P: Any) -> None:
        clean = _readonly(validate_transition_matrix(P, self.structure))
        self._P = clean
        self._cache: dict[tuple[Any, ...], Any] = {}
        digest = hashlib.blake2b(clean.tobytes() + self.structure.value.encode(), digest_size=8)
        self._fingerprint = digest.hexdigest()

    # ----- accessors ------------------------------------------------------------------------
    @property
    def matrix(self) -> np.ndarray:
        """Posterior-mean transition matrix (read-only view)."""
        return self._P

    @property
    def counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def dirichlet_alpha(self) -> np.ndarray | None:
        """Dirichlet parameters of the transient rows (None for :meth:`from_matrix`)."""
        return None if self._alpha is None else self._alpha.copy()

    @property
    def fingerprint(self) -> str:
        """Stable hash of (matrix, structure); changes whenever the matrix changes."""
        return self._fingerprint

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_transitions(self) -> float:
        return float(self._counts.sum())

    def sample_matrix(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of P from the row-wise Dirichlet posterior (transition uncertainty).

        Only cells with positive Dirichlet mass are drawn; structural zeros stay 0 and the
        absorbing rows are fixed.
        """
        P = np.array(self._P)
        if self._alpha is None:
            return P
        for i in _TRANSIENT:
            a = self._alpha[i]
            pos = a > 0
            if not pos.any():
                continue
            row = np.zeros(N_STATES)
            row[pos] = rng.dirichlet(a[pos])
            P[i] = row
        return P

    # ----- propagation ----------------------------------------------------------------------
    def propagate(self, pi0: Any, t: int) -> np.ndarray:
        """``π_t = π_0 Pᵗ`` (``pi0`` may be a :class:`State`, a (4,) or an (n, 4) array)."""
        t = int(t)
        if t < 0:
            raise ValueError("t must be >= 0")
        return _as_distribution(pi0) @ np.linalg.matrix_power(self._P, t)

    def trajectory(self, pi0: Any, horizon: int) -> np.ndarray:
        """``[π_0, π_1, …, π_horizon]`` as an array of shape (horizon + 1, 4)."""
        horizon = int(horizon)
        if horizon < 0:
            raise ValueError("horizon must be >= 0")
        pi = _as_distribution(pi0)
        if pi.ndim != 1:
            raise ValueError("trajectory needs a single distribution")
        out = np.empty((horizon + 1, N_STATES))
        out[0] = pi
        for k in range(1, horizon + 1):
            out[k] = out[k - 1] @ self._P
        return out

    def p_default(self, t: int, pi0: Any = START_CURRENT) -> float:
        """``P(Default, t) = π_D(t) + π_L(t)``."""
        pi = self.propagate(pi0, t)
        return float(pi[_D] + pi[_L])

    def p_default_curve(self, horizon: int, pi0: Any = START_CURRENT) -> np.ndarray:
        """``π_D(t) + π_L(t)`` for t = 0..horizon."""
        traj = self.trajectory(pi0, horizon)
        return traj[:, _D] + traj[:, _L]

    # ----- absorption -----------------------------------------------------------------------
    def fundamental_matrix(self) -> np.ndarray:
        """``N = (I − Q)⁻¹``; raises :class:`InvalidTransitionMatrixError` if singular."""
        key = ("N",)
        if key not in self._cache:
            Q = self._P[np.ix_(_TRANSIENT, _TRANSIENT)]
            IQ = np.eye(len(_TRANSIENT)) - Q
            det = float(np.linalg.det(IQ))
            if not np.isfinite(det) or abs(det) < 1e-12:
                raise InvalidTransitionMatrixError(
                    "I - Q is singular: a transient state never reaches an absorbing state "
                    f"(Q = {Q.tolist()})"
                )
            self._cache[key] = _readonly(np.linalg.inv(IQ))
        return self._cache[key]

    def absorption_probabilities(self) -> np.ndarray:
        """``B = N·R_block`` — shape (2, 2): rows (C, D), columns (R, L)."""
        return self.fundamental_matrix() @ self._P[np.ix_(_TRANSIENT, _ABSORBING)]

    def expected_time_to_absorption(self) -> np.ndarray:
        """``N·1`` — expected number of steps to absorption from (C, D)."""
        return self.fundamental_matrix().sum(axis=1)

    def stationary(self, pi0: Any = START_CURRENT) -> np.ndarray:
        """Limit ``lim π_k = [0, 0, π_R^∞, π_L^∞]`` (from Current by default)."""
        pi = _as_distribution(pi0)
        if pi.ndim != 1:
            raise ValueError("stationary needs a single distribution")
        B = self.absorption_probabilities()
        out = np.zeros(N_STATES)
        out[_R] = pi[_R] + pi[_C] * B[0, 0] + pi[_D] * B[1, 0]
        out[_L] = pi[_L] + pi[_C] * B[0, 1] + pi[_D] * B[1, 1]
        return out

    def p_recover_given_delinquent(self, t: int | None = None) -> float:
        """``P(R | D, t)``: in R after t steps starting delinquent (t=None → the limit)."""
        if t is None:
            return float(self.absorption_probabilities()[1, 0])
        return float(self.propagate(START_DELINQUENT, t)[_R])

    def p_loss_given_delinquent(self, t: int | None = None) -> float:
        """``P(L | D, t)``: in L after t steps starting delinquent (t=None → the limit)."""
        if t is None:
            return float(self.absorption_probabilities()[1, 1])
        return float(self.propagate(START_DELINQUENT, t)[_L])

    # ----- maturity and default timing ------------------------------------------------------
    def maturity_collapse(self, pi: Any) -> np.ndarray:
        """At maturity the delinquents collapse: C → R; D splits ``p_DR : p_DL`` (50/50 if
        both are 0). Works on (4,) or (n, 4) arrays."""
        arr = np.asarray(pi, dtype=float)
        p_dr, p_dl = self._P[_D, _R], self._P[_D, _L]
        total = p_dr + p_dl
        w_r, w_l = (p_dr / total, p_dl / total) if total > 0 else (0.5, 0.5)
        out = np.zeros_like(arr)
        out[..., _R] = arr[..., _R] + arr[..., _C] + w_r * arr[..., _D]
        out[..., _L] = arr[..., _L] + w_l * arr[..., _D]
        return out

    def default_timing_cdf(self, horizon: int, start: Any = START_CURRENT) -> np.ndarray:
        """``F(m) = P(absorbed into L by m | in L at maturity)`` for m = 1..horizon.

        Trajectory from ``start``; the maturity collapse is applied at ``m = horizon``, so
        ``F(m) = π_L(m) / π_L^collapsed(horizon)`` for ``m < horizon`` and ``F(horizon) = 1``.
        If ``π_L^collapsed(horizon) = 0`` the uniform ``m/horizon`` is returned. Read-only,
        cached per (horizon, start).
        """
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        key = ("cdf", horizon, _start_key(start))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        pi0 = _as_distribution(start)
        traj = self.trajectory(pi0, horizon)
        final = float(self.maturity_collapse(traj[horizon])[_L])
        if final <= 0.0:
            F = np.arange(1, horizon + 1, dtype=float) / horizon
        else:
            F = traj[1:, _L] / final
            F[-1] = 1.0
            F = np.clip(np.maximum.accumulate(F), 0.0, 1.0)
        self._cache[key] = _readonly(F)
        return self._cache[key]

    def default_timing_pmf(self, horizon: int, start: Any = START_CURRENT) -> np.ndarray:
        """``f(m) = F(m) − F(m − 1)`` (F(0) = 0); sums to 1."""
        F = self.default_timing_cdf(horizon, start)
        return np.diff(F, prepend=0.0)

    def p_loss_given_state(self, state: State | int, remaining_months: int) -> float:
        """Probability of ending in L within ``remaining_months`` from ``state``, including
        the maturity collapse at the end (used by the interim reward)."""
        s = State(int(state))
        m = int(remaining_months)
        if m < 0:
            raise ValueError("remaining_months must be >= 0")
        key = ("ploss", int(s), m)
        cached = self._cache.get(key)
        if cached is None:
            pi = self.propagate(s, m)
            cached = float(self.maturity_collapse(pi)[_L])
            self._cache[key] = cached
        return cached

    # ----- goodness of fit ------------------------------------------------------------------
    def log_likelihood(self, counts: Any) -> float:
        """``Σ counts·log P`` over cells with counts > 0; ``-inf`` if any such cell has P = 0
        (0.0 for an all-zero count matrix)."""
        c = np.asarray(counts, dtype=float)
        if c.shape != (N_STATES, N_STATES):
            raise ValueError(f"counts must be 4x4, got {c.shape}")
        mask = c > 0
        if np.any(self._P[mask] <= 0):
            return float("-inf")
        return float(np.sum(c[mask] * np.log(self._P[mask])))

    def row_prediction_error(self, counts: Any) -> np.ndarray:
        """L1 distance between each transient row (C, D) and its empirical row; NaN if empty."""
        c = np.asarray(counts, dtype=float)
        if c.shape != (N_STATES, N_STATES):
            raise ValueError(f"counts must be 4x4, got {c.shape}")
        out = np.full(len(_TRANSIENT), np.nan)
        for k, i in enumerate(_TRANSIENT):
            n = c[i].sum()
            if n > 0:
                out[k] = float(np.abs(self._P[i] - c[i] / n).sum())
        return out

    # ----- reporting / serialisation --------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """16 rows ``from_state, to_state, probability, count, prior, allowed``."""
        rows = []
        for i in State:
            for j in State:
                transient = int(i) in _TRANSIENT
                allowed = bool(self.allowed[int(i), int(j)])
                rows.append(
                    {
                        "from_state": i.label,
                        "to_state": j.label,
                        "probability": float(self._P[int(i), int(j)]),
                        "count": float(self._counts[int(i), int(j)]),
                        "prior": self.prior_strength if (transient and allowed and self._alpha is not None) else 0.0,
                        "allowed": allowed,
                    }
                )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MarkovChainTransitionMatrix",
            "version": 1,
            "structure": self.structure.value,
            "prior_strength": self.prior_strength,
            "source": self._source,
            "fitted": self._fitted,
            "counts": self._counts.tolist(),
            "matrix": self._P.tolist(),
            "diagnostics": {k: (list(v) if isinstance(v, list) else v) for k, v in self.diagnostics.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarkovChainTransitionMatrix:
        if data.get("source") == "matrix":
            return cls.from_matrix(data["matrix"], data["structure"])
        obj = cls(data["structure"], data.get("prior_strength", 1.0))
        obj.fit_counts(data["counts"])
        obj._fitted = bool(data.get("fitted", True))
        return obj

    def __repr__(self) -> str:
        P = self._P
        return (
            f"MarkovChainTransitionMatrix({self.structure.value}, prior={self.prior_strength:g}, "
            f"n={self.n_transitions:g}, p_CD={P[_C, _D]:.4f}, p_DR={P[_D, _R]:.4f}, p_DL={P[_D, _L]:.4f})"
        )
