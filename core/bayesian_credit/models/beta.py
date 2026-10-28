"""Beta distribution value object used by both stages of the Two-Stage Beta-Beta model.

Implements the two closed forms in §3.3:

* conjugate update       Beta(α + d, β + n), posterior mean (α + d) / (α + β + d + n)
* method of moments      α = μ(μ(1-μ)/σ² − 1),  β = (1-μ)(μ(1-μ)/σ² − 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, stats

from ..exceptions import InvalidPosteriorError, NumericalInstabilityError


@dataclass(frozen=True)
class BetaPosterior:
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        a, b = float(self.alpha), float(self.beta)
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            raise InvalidPosteriorError(f"Beta parameters must be finite and > 0, got ({a}, {b})")
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)

    # ----- moments -------------------------------------------------------------------
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s * s * (s + 1.0))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    @property
    def concentration(self) -> float:
        """α + β — the prior/posterior 'pseudo-sample size'."""
        return self.alpha + self.beta

    @property
    def second_moment(self) -> float:
        """E[p²] (needed for closed-form reward variance)."""
        return self.var + self.mean**2

    @property
    def mode(self) -> float | None:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None

    # ----- updates -------------------------------------------------------------------
    def update(self, successes: float, failures: float) -> BetaPosterior:
        """Conjugate Beta-Bernoulli update: Beta(α + d, β + n)."""
        if successes < 0 or failures < 0:
            raise InvalidPosteriorError("counts must be non-negative")
        return BetaPosterior(self.alpha + successes, self.beta + failures)

    # ----- sampling / quantiles ------------------------------------------------------
    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size=size)

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        return stats.beta.ppf(q, self.alpha, self.beta)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return stats.beta.cdf(x, self.alpha, self.beta)

    def pdf_grid(self, n: int = 101) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 1.0, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = stats.beta.pdf(x, self.alpha, self.beta)
        y = np.where(np.isfinite(y), y, 0.0)
        return x, y

    def hdi(self, mass: float = 0.95) -> tuple[float, float]:
        """Highest-density interval (narrowest interval containing ``mass``)."""
        if not 0 < mass < 1:
            raise InvalidPosteriorError("mass must be in (0, 1)")
        dist = stats.beta(self.alpha, self.beta)

        def width(lo_q: float) -> float:
            return float(dist.ppf(lo_q + mass) - dist.ppf(lo_q))

        res = optimize.minimize_scalar(width, bounds=(0.0, 1.0 - mass), method="bounded")
        lo_q = float(res.x)
        return float(dist.ppf(lo_q)), float(dist.ppf(lo_q + mass))

    # ----- constructors --------------------------------------------------------------
    @classmethod
    def from_mean_concentration(cls, mean: float, kappa: float) -> BetaPosterior:
        """Beta with the given mean and pseudo-count κ (used to build child priors)."""
        mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
        if kappa <= 0:
            raise InvalidPosteriorError("kappa must be > 0")
        return cls(kappa * mean, kappa * (1.0 - mean))

    @classmethod
    def from_moments(cls, mean: float, var: float, eps: float = 1e-4, max_concentration: float = 1000.0) -> BetaPosterior:
        """Method-of-moments fit (§3.3, stage 2).

        α = μ(μ(1−μ)/σ² − 1), β = (1−μ)(μ(1−μ)/σ² − 1).
        Raises :class:`NumericalInstabilityError` when σ² ≥ μ(1−μ) (no Beta has that much
        variance) or σ² is not positive-finite; a near-zero variance is capped at
        ``max_concentration`` instead of exploding.
        """
        mu = float(np.clip(mean, eps, 1.0 - eps))
        if not np.isfinite(var) or var < 0:
            raise NumericalInstabilityError(f"invalid variance {var!r} for method of moments")
        bound = mu * (1.0 - mu)
        if var >= bound:
            raise NumericalInstabilityError(
                f"variance {var:.4g} >= mu(1-mu) = {bound:.4g}; Beta method of moments undefined"
            )
        if var <= 1e-12:
            kappa = max_concentration
        else:
            kappa = bound / var - 1.0
            kappa = min(max(kappa, 1e-3), max_concentration)
        return cls(mu * kappa, (1.0 - mu) * kappa)

    # ----- misc ----------------------------------------------------------------------
    def to_dict(self, hdi_mass: float = 0.95) -> dict[str, Any]:
        lo, hi = self.hdi(hdi_mass)
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "std": self.std,
            "hdi_low": lo,
            "hdi_high": hi,
            "concentration": self.concentration,
        }

    def __repr__(self) -> str:
        return f"Beta(α={self.alpha:.3f}, β={self.beta:.3f}, mean={self.mean:.4f})"
