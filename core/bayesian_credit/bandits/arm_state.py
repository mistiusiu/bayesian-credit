"""Gaussian posterior over an arm's expected risk-adjusted reward (§3.3; SPEC §2.8).

The proposal models the belief over the expected reward of lending policy ``A_k`` as
``θ_k ~ N(μ_k, σ_k²)`` and, after observing a reward ``r`` for the selected arm, updates

    μ_new = μ + (r − μ) / (n + 1)                              (incremental mean)
    n     ← n + 1
    σ²    ← ((n − 1)·σ²_old + (r − μ_old)(r − μ_new)) / n       (online variance)

**Welford form.** With ``M2 = Σ_i (r_i − μ)²`` the variance recursion is Welford's
algorithm: ``M2 ← M2 + (r − μ_old)(r − μ_new)`` and ``σ² = M2 / (n − 1)``. Reading the
``n`` of the variance recursion as the *pre-update* count makes the two forms
algebraically identical, because ``M2_old = (n_old − 1)·σ²_old``; it is the unbiased
(ddof = 1) estimator. (Reading it as the post-update count would give the ddof = 0
recursion ``M2 / n``; the two differ by ``n/(n−1) → 1``.) For ``n ≤ 1`` the sample
variance is undefined and σ² stays at the prior value.

**Weighted generalisation [decision].** Interim repayment signals arrive as fractional
observations of weight ``w ∈ [0, 1]`` (``BanditConfig.interim_weight``). Treating ``w`` as
a frequency weight (West, 1979)::

    n  ← n + w
    μ  ← μ + w (r − μ_old) / n
    M2 ← M2 + w (r − μ_old)(r − μ_new)
    σ² = M2 / (n − 1)     (n > 1; prior σ² otherwise)

For ``w = 1`` this is exactly the proposal's recursion, so ``μ`` equals the sample mean
and ``σ²`` the ddof = 1 sample variance of the rewards; for general weights ``μ`` is the
weighted mean and ``M2 = Σ w_i (r_i − μ)²``. ``w = 0`` is a no-op. The prior mean carries
no pseudo-count: it governs exploration of an arm until its first observation.

**Posterior scale.** ``"proposal"`` samples ``θ̃ ~ N(μ, σ²)`` exactly as written in §3.3;
``"standard_error"`` samples ``N(μ, σ²/max(n, 1))`` (the sampling distribution of the mean,
which concentrates as ``n`` grows). The standard deviation is floored at ``min_sigma``.

This module holds the single implementation of that arithmetic (:func:`welford_update`,
:func:`posterior_variance`, :func:`ucb_index`), used both by the scalar
:class:`GaussianArmPosterior` value object and by the vectorised :class:`GaussianArmSet`
(one posterior per arm, stored column-wise so Thompson draws are one ``rng.normal`` call).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError, InvalidPosteriorError, NumericalInstabilityError

#: Admissible values of ``BanditConfig.posterior_scale``.
POSTERIOR_SCALES = ("proposal", "standard_error")


def welford_update(
    mu: float, m2: float, n: float, reward: float, weight: float = 1.0
) -> tuple[float, float, float]:
    """One (weighted) Welford step; returns the new ``(mu, m2, n)``.

    Raises :class:`NumericalInstabilityError` for a non-finite reward, a negative or
    non-finite weight, or an overflow of the running moments. Inputs are never mutated,
    so a failed update leaves the caller's state untouched.
    """
    r = float(reward)
    w = float(weight)
    if not math.isfinite(r):
        raise NumericalInstabilityError(f"non-finite reward {reward!r} offered to a bandit arm")
    if not math.isfinite(w) or w < 0.0:
        raise NumericalInstabilityError(f"observation weight must be finite and >= 0, got {weight!r}")
    mu, m2, n = float(mu), float(m2), float(n)
    if w == 0.0:
        return mu, m2, n
    n_new = n + w
    delta = r - mu
    mu_new = mu + w * delta / n_new
    m2_new = m2 + w * delta * (r - mu_new)
    if not (math.isfinite(mu_new) and math.isfinite(m2_new)):
        raise NumericalInstabilityError(
            f"overflow in the online mean/variance update (reward={r!r}, mu={mu!r}, m2={m2!r})"
        )
    # (r - mu_old)(r - mu_new) >= 0 always; the clip only absorbs round-off.
    return mu_new, max(m2_new, 0.0), n_new


def posterior_variance(m2: float, n: float, prior_sigma2: float) -> float:
    """σ² = M2/(n − 1) for n > 1, else the prior variance (sample variance undefined)."""
    return float(m2) / (float(n) - 1.0) if n > 1.0 else float(prior_sigma2)


def _check_scale(scale: str) -> None:
    if scale not in POSTERIOR_SCALES:
        raise ConfigurationError(f"posterior scale must be one of {POSTERIOR_SCALES}, got {scale!r}")


def ucb_index(
    mu: np.ndarray | float,
    counts: np.ndarray | float,
    t_total: float,
    c: float = 2.0,
    scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Upper-confidence index ``mu + c·scale·sqrt(2·ln(t_total)/count)`` (UCB1 style).

    ``count <= 0`` gives ``+inf`` (untried arms first). ``scale = 1`` is the textbook
    UCB1 bonus for rewards in [0, 1]; pass a reward standard deviation to make the index
    invariant to the currency unit of the reward.
    """
    mu_a = np.asarray(mu, dtype=float)
    cnt = np.asarray(counts, dtype=float)
    log_t = math.log(max(float(t_total), 1.0))
    safe = np.where(cnt > 0, cnt, 1.0)
    bonus = float(c) * np.asarray(scale, dtype=float) * np.sqrt(2.0 * log_t / safe)
    return np.where(cnt > 0, mu_a + bonus, np.inf)


@dataclass
class GaussianArmPosterior:
    """θ_k ~ N(μ_k, σ_k²): belief over one lending policy's expected reward (§3.3).

    ``update`` implements the proposal's incremental mean and online variance exactly for
    unit weights (via Welford's ``M2``) and the frequency-weighted generalisation for
    fractional interim observations; see the module docstring for the algebra.
    ``prior_mu`` / ``prior_sigma2`` default to the initial ``mu`` / ``sigma2``.
    """

    mu: float = 0.0
    sigma2: float = 1.0
    n: float = 0.0
    m2: float = 0.0
    prior_mu: float | None = None
    prior_sigma2: float | None = None

    def __post_init__(self) -> None:
        self.mu = float(self.mu)
        self.sigma2 = float(self.sigma2)
        self.n = float(self.n)
        self.m2 = float(self.m2)
        self.prior_mu = self.mu if self.prior_mu is None else float(self.prior_mu)
        self.prior_sigma2 = self.sigma2 if self.prior_sigma2 is None else float(self.prior_sigma2)
        values = (self.mu, self.sigma2, self.n, self.m2, self.prior_mu, self.prior_sigma2)
        if not all(math.isfinite(v) for v in values):
            raise InvalidPosteriorError(f"Gaussian arm posterior has non-finite parameters {values}")
        if self.sigma2 < 0 or self.prior_sigma2 < 0 or self.n < 0 or self.m2 < 0:
            raise InvalidPosteriorError("sigma2, prior_sigma2, n and m2 must be non-negative")

    @classmethod
    def from_prior(cls, mu: float, sigma: float) -> GaussianArmPosterior:
        """Fresh posterior N(mu, sigma²) with no observations."""
        return cls(mu=mu, sigma2=float(sigma) ** 2)

    # ----- learning --------------------------------------------------------------------
    def update(self, r: float, weight: float = 1.0) -> None:
        """Absorb reward ``r`` with observation weight ``weight`` (0 ⇒ no-op)."""
        mu, m2, n = welford_update(self.mu, self.m2, self.n, r, weight)
        self.mu, self.m2, self.n = mu, m2, n
        self.sigma2 = posterior_variance(m2, n, self.prior_sigma2)

    # ----- sampling / indices ------------------------------------------------------------
    def std(self, scale: str = "proposal", min_sigma: float = 1e-6) -> float:
        """Std of the sampling distribution: σ (``"proposal"``) or σ/√max(n,1)."""
        _check_scale(scale)
        var = self.sigma2 if scale == "proposal" else self.sigma2 / max(self.n, 1.0)
        return max(math.sqrt(max(var, 0.0)), float(min_sigma))

    def sample(
        self, rng: np.random.Generator, scale: str = "proposal", min_sigma: float = 1e-6
    ) -> float:
        """Thompson draw θ̃ ~ N(μ, σ²) or N(μ, σ²/max(n, 1))."""
        return float(rng.normal(self.mu, self.std(scale, min_sigma)))

    def ucb(
        self,
        t_total: float,
        c: float = 2.0,
        *,
        count: float | None = None,
        scale: float | None = 1.0,
    ) -> float:
        """``mu + c·scale·sqrt(2 ln(t_total)/count)``; ``count`` defaults to ``n``.

        ``scale=None`` uses the arm's own σ; ``count <= 0`` returns ``+inf``.
        """
        s = math.sqrt(self.sigma2) if scale is None else float(scale)
        cnt = self.n if count is None else float(count)
        return float(ucb_index(self.mu, cnt, t_total, c, s))

    # ----- serialisation -----------------------------------------------------------------
    def to_dict(self) -> dict[str, float]:
        return {
            "mu": self.mu,
            "sigma2": self.sigma2,
            "n": self.n,
            "m2": self.m2,
            "prior_mu": float(self.prior_mu),  # type: ignore[arg-type]
            "prior_sigma2": float(self.prior_sigma2),  # type: ignore[arg-type]
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GaussianArmPosterior:
        return cls(
            mu=d["mu"],
            sigma2=d["sigma2"],
            n=d.get("n", 0.0),
            m2=d.get("m2", 0.0),
            prior_mu=d.get("prior_mu"),
            prior_sigma2=d.get("prior_sigma2"),
        )


class GaussianArmSet:
    """θ_k ~ N(μ_k, σ_k²) for every arm k of 𝒜 (§3.3), stored column-wise (numpy arrays).

    This is the per-group arm set of the contextual bandit. Columns: ``mu``, ``sigma2``,
    ``n`` (observation weight), ``m2``, ``prior_mu``, ``prior_sigma2`` and ``selections``
    (how often the arm was chosen in this group); ``n_updates`` counts applied reward
    observations. Updates go through :func:`welford_update`, so results are bit-identical
    to the scalar value object.
    """

    __slots__ = ("mu", "sigma2", "n", "m2", "prior_mu", "prior_sigma2", "selections", "n_updates")

    def __init__(self, prior_mu: np.ndarray, prior_sigma2: np.ndarray):
        pm = np.array(prior_mu, dtype=float).reshape(-1)
        ps = np.array(prior_sigma2, dtype=float).reshape(-1)
        if pm.shape != ps.shape or pm.size == 0:
            raise InvalidPosteriorError("prior_mu and prior_sigma2 must be non-empty and equally long")
        if not (np.all(np.isfinite(pm)) and np.all(np.isfinite(ps))) or np.any(ps < 0):
            raise InvalidPosteriorError("arm priors must be finite with non-negative variance")
        self.prior_mu = pm
        self.prior_sigma2 = ps
        self.mu = pm.copy()
        self.sigma2 = ps.copy()
        self.n = np.zeros(pm.size, dtype=float)
        self.m2 = np.zeros(pm.size, dtype=float)
        self.selections = np.zeros(pm.size, dtype=np.int64)
        self.n_updates = 0

    @classmethod
    def from_prior(
        cls, mu: np.ndarray, sigma: np.ndarray, min_sigma: float = 0.0
    ) -> GaussianArmSet:
        s = np.maximum(np.asarray(sigma, dtype=float), float(min_sigma))
        return cls(np.asarray(mu, dtype=float), s**2)

    def __len__(self) -> int:
        return int(self.mu.size)

    # ----- learning --------------------------------------------------------------------
    def update(self, arm: int, r: float, weight: float = 1.0) -> None:
        mu, m2, n = welford_update(self.mu[arm], self.m2[arm], self.n[arm], r, weight)
        if weight == 0:
            return
        self.mu[arm], self.m2[arm], self.n[arm] = mu, m2, n
        self.sigma2[arm] = posterior_variance(m2, n, self.prior_sigma2[arm])
        self.n_updates += 1

    # ----- vectorised sampling / indices -------------------------------------------------
    def std(
        self, idx: np.ndarray | None = None, scale: str = "proposal", min_sigma: float = 1e-6
    ) -> np.ndarray:
        _check_scale(scale)
        sel = slice(None) if idx is None else idx
        var = self.sigma2[sel]
        if scale == "standard_error":
            var = var / np.maximum(self.n[sel], 1.0)
        return np.maximum(np.sqrt(np.maximum(var, 0.0)), float(min_sigma))

    def sample(
        self,
        rng: np.random.Generator,
        idx: np.ndarray,
        scale: str = "proposal",
        min_sigma: float = 1e-6,
    ) -> np.ndarray:
        """Thompson draws θ̃_k for the arms in ``idx`` (one ``rng.normal`` call)."""
        return rng.normal(self.mu[idx], self.std(idx, scale, min_sigma))

    def ucb(
        self,
        t_total: float,
        c: float,
        idx: np.ndarray,
        counts: np.ndarray | None = None,
        scale_by_std: bool = True,
        min_sigma: float = 1e-6,
    ) -> np.ndarray:
        """UCB indices for ``idx``; ``counts`` default to the observation weights ``n``."""
        cnt = self.n[idx] if counts is None else np.asarray(counts, dtype=float)
        scale = self.std(idx, "proposal", min_sigma) if scale_by_std else 1.0
        return ucb_index(self.mu[idx], cnt, t_total, c, scale)

    # ----- views -------------------------------------------------------------------------
    def posterior(self, arm: int) -> GaussianArmPosterior:
        return GaussianArmPosterior(
            mu=self.mu[arm],
            sigma2=self.sigma2[arm],
            n=self.n[arm],
            m2=self.m2[arm],
            prior_mu=self.prior_mu[arm],
            prior_sigma2=self.prior_sigma2[arm],
        )

    def table(self, group: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": pd.Series([group] * len(self), dtype="string"),
                "arm": np.arange(len(self), dtype=np.int64),
                "mu": self.mu.copy(),
                "sigma": np.sqrt(self.sigma2),
                "n": self.n.copy(),
                "selections": self.selections.copy(),
            }
        )

    def copy(self) -> GaussianArmSet:
        return GaussianArmSet.from_dict(self.to_dict())

    # ----- serialisation (JSON-able; σ² is recomputed so it cannot drift from M2) --------
    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": self.mu.tolist(),
            "n": self.n.tolist(),
            "m2": self.m2.tolist(),
            "prior_mu": self.prior_mu.tolist(),
            "prior_sigma2": self.prior_sigma2.tolist(),
            "selections": self.selections.tolist(),
            "n_updates": int(self.n_updates),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GaussianArmSet:
        out = cls(np.asarray(d["prior_mu"], dtype=float), np.asarray(d["prior_sigma2"], dtype=float))
        k = len(out)
        for name in ("mu", "n", "m2", "selections"):
            if len(d[name]) != k:
                raise InvalidPosteriorError(f"arm-set column {name!r} has length {len(d[name])}, expected {k}")
        out.mu = np.asarray(d["mu"], dtype=float)
        out.n = np.asarray(d["n"], dtype=float)
        out.m2 = np.asarray(d["m2"], dtype=float)
        out.selections = np.asarray(d["selections"], dtype=np.int64)
        out.n_updates = int(d.get("n_updates", 0))
        if not (np.all(np.isfinite(out.mu)) and np.all(np.isfinite(out.m2))) or np.any(out.n < 0):
            raise InvalidPosteriorError("arm-set state has non-finite means/moments or negative counts")
        safe_n = np.where(out.n > 1.0, out.n - 1.0, 1.0)
        out.sigma2 = np.where(out.n > 1.0, out.m2 / safe_n, out.prior_sigma2)
        return out
