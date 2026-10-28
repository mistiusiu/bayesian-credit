"""Stage 1 of the Two-Stage Beta-Beta model: hierarchical Beta-Bernoulli default model.

Proposal §3.3 (p. 26)::

    p_{g,a} = P(D = 1 | X_g, A_a)
    p_0     ~ Beta(α_0, β_0)                       portfolio (global) level
    p_g     ~ Beta(α_g, β_g)                       borrower group g
    p_{g,a} ~ Beta(α_{g,a}, β_{g,a})               group g under lending policy (arm) a
    p_{g,a} | D ~ Beta(α_{g,a} + d_{g,a}, β_{g,a} + n_{g,a})
    E[p_{g,a}]  = (α_{g,a} + d_{g,a}) / (α_{g,a} + β_{g,a} + d_{g,a} + n_{g,a})

where ``d`` counts observed defaults and ``n`` observed non-defaults.

**[decision]** (SPEC §2.2) priors are built top-down by empirical-Bayes shrinkage: a child's
prior is its parent's posterior mean ``m`` with pseudo-count (concentration) ``κ``::

    α_g     = κ_group · m_0,   β_g     = κ_group · (1 − m_0),    m_0 = E[p_0 | data]
    α_{g,a} = κ_leaf  · m_g,   β_{g,a} = κ_leaf  · (1 − m_g),    m_g = E[p_g | data]

and every level's posterior is the conjugate update of its prior with that level's own
counts, so groups / policies with few observations shrink toward the portfolio.

**[decision] Structured leaf prior** (docs/MATH.md "Structured leaf prior"). An optional
*leaf-prior tilt* ``t(g, a)`` (a logit shift; installed by ``TwoStageBetaBeta`` from the
policy-effect model, :mod:`.policy_effect`) moves the leaf prior mean to::

    m_{g,a} = σ(logit(m_g) + t(g, a)),   α_{g,a} = κ_leaf · m_{g,a},   β_{g,a} = κ_leaf · (1 − m_{g,a})

so that an arm never observed for a group inherits the group's risk *adjusted for its
rate, term and size*. κ_leaf and the conjugate update are unchanged, so the leaf's own data
dominate as they accumulate. An unseen group's arms get ``σ(logit(m_0) + t(g, a))`` with
κ_group. Without a tilt (or ``t = 0``) everything is exactly the untilted model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy import special

from ..config import HierarchyConfig
from ..exceptions import InvalidPosteriorError, NumericalInstabilityError
from .beta import BetaPosterior

__all__ = [
    "AUTO_KAPPA_FALLBACK",
    "AUTO_KAPPA_MIN_OBS",
    "AUTO_KAPPA_VAR_FLOOR",
    "LEAF_KEY_SEPARATOR",
    "HierarchicalBetaBernoulli",
    "LeafPriorTilt",
    "beta_binomial_kappa",
    "beta_hdi",
    "tilted_mean",
]

#: ``(group, arm) -> logit shift`` of the leaf prior mean (see the module docstring).
LeafPriorTilt = Callable[[str, int], float]

#: Children with fewer observations than this are ignored by the ``"auto"`` κ estimator.
AUTO_KAPPA_MIN_OBS = 5
#: Floor on the excess (between-children) variance used by the ``"auto"`` κ estimator.
AUTO_KAPPA_VAR_FLOOR = 1e-6

_HIERARCHY_DEFAULTS = {f.name: f.default for f in dataclasses.fields(HierarchyConfig)}
#: κ used when ``"auto"`` cannot be estimated (fewer than two children with ``n ≥ 5``):
#: the numeric defaults of :class:`~bayesian_credit.config.HierarchyConfig`.
AUTO_KAPPA_FALLBACK: dict[str, float] = {
    "group": float(_HIERARCHY_DEFAULTS["kappa_group"]),
    "leaf": float(_HIERARCHY_DEFAULTS["kappa_leaf"]),
}
#: ``summary()["key"]`` of a leaf is ``f"{group}{LEAF_KEY_SEPARATOR}{arm}"``.
LEAF_KEY_SEPARATOR = "#"

_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0  # golden-section ratio 0.618...


# ----------------------------------------------------------------------------------------
# helpers shared by the risk models
# ----------------------------------------------------------------------------------------
def beta_hdi(
    alpha: float | np.ndarray, beta: float | np.ndarray, mass: float = 0.95, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised highest-density interval of many Beta distributions at once.

    Same definition as :meth:`BetaPosterior.hdi` — the narrowest interval
    ``[F⁻¹(q), F⁻¹(q + mass)]`` over ``q ∈ [0, 1 − mass]`` — solved by a golden-section
    search on all rows simultaneously (``tol`` is the bracket width in ``q``). Roughly 30x
    faster than calling ``BetaPosterior.hdi`` per row, which matters for ``summary()`` /
    ``predict_table()`` over thousands of leaves. For J-shaped Betas (α < 1 or β < 1) the
    interval touches the boundary, as it should.

    Returns ``(low, high)`` arrays with the broadcast shape of ``alpha`` and ``beta``
    (at least 1-D).
    """
    if not 0.0 < mass < 1.0:
        raise InvalidPosteriorError("mass must be in (0, 1)")
    a, b = np.broadcast_arrays(
        np.atleast_1d(np.asarray(alpha, dtype=float)), np.atleast_1d(np.asarray(beta, dtype=float))
    )
    if a.size == 0:
        return np.empty(a.shape), np.empty(a.shape)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a > 0) and np.all(b > 0)):
        raise InvalidPosteriorError("Beta parameters must be finite and > 0")

    def width(q: np.ndarray) -> np.ndarray:
        return special.betaincinv(a, b, q + mass) - special.betaincinv(a, b, q)

    span = 1.0 - mass
    lo = np.zeros(a.shape)
    hi = np.full(a.shape, span)
    c = hi - _INV_PHI * (hi - lo)
    d = lo + _INV_PHI * (hi - lo)
    fc, fd = width(c), width(d)
    n_iter = max(1, math.ceil(math.log(tol / span) / math.log(_INV_PHI))) if tol < span else 1
    for _ in range(n_iter):
        left = fc < fd  # minimum lies in [lo, d]
        lo, hi = np.where(left, lo, c), np.where(left, d, hi)
        keep_x, keep_f = np.where(left, c, d), np.where(left, fc, fd)
        x_new = np.where(left, hi - _INV_PHI * (hi - lo), lo + _INV_PHI * (hi - lo))
        f_new = width(x_new)
        c, d = np.where(left, x_new, keep_x), np.where(left, keep_x, x_new)
        fc, fd = np.where(left, f_new, keep_f), np.where(left, keep_f, f_new)
    q = 0.5 * (lo + hi)
    return special.betaincinv(a, b, q), special.betaincinv(a, b, q + mass)


def beta_binomial_kappa(
    defaults: Sequence[float] | np.ndarray,
    n_obs: Sequence[float] | np.ndarray,
    kappa_min: float = 1.0,
    kappa_max: float = 1000.0,
    min_obs: int = AUTO_KAPPA_MIN_OBS,
) -> float | None:
    """Beta-binomial method-of-moments concentration across the children of one parent.

    With child default rates ``r_c = d_c / n_c`` (only children with ``n_c ≥ min_obs``),
    ``n``-weighted mean ``m`` and ``n``-weighted variance ``v``, the expected binomial
    sampling variance ``m(1 − m) / mean(n_c)`` is subtracted from ``v`` (floored at
    ``1e-6``) to leave the between-children variance. Since a Beta(κm, κ(1−m)) has variance
    ``m(1−m)/(κ+1)``::

        κ = m(1 − m) / v − 1,   clipped to [kappa_min, kappa_max].

    Returns ``None`` when fewer than two children qualify (κ not identifiable). When
    ``m ∈ {0, 1}`` no heterogeneity can be measured and ``kappa_max`` (complete pooling, the
    same answer as ``v`` at its floor) is returned.
    """
    d = np.asarray(defaults, dtype=float)
    n = np.asarray(n_obs, dtype=float)
    keep = n >= max(min_obs, 1)
    lo, hi = sorted((float(kappa_min), float(kappa_max)))
    if int(keep.sum()) < 2:
        return None
    d, n = d[keep], n[keep]
    rates = d / n
    w = n / n.sum()
    m = float(w @ rates)
    if m <= 0.0 or m >= 1.0:
        return hi
    v = float(w @ (rates - m) ** 2) - m * (1.0 - m) / float(n.mean())
    v = max(v, AUTO_KAPPA_VAR_FLOOR)
    return float(np.clip(m * (1.0 - m) / v - 1.0, lo, hi))


def _as_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    if isinstance(values, (pd.Index, pd.api.extensions.ExtensionArray, np.ndarray)):
        return pd.Series(values)
    return pd.Series(list(values), dtype=object)


def _as_float_array(series: pd.Series) -> np.ndarray:
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        series = pd.to_numeric(series, errors="coerce")
    return series.to_numpy(dtype=float, na_value=np.nan)


def _clean_observations(
    groups: Any, arms: Any, values: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Align ``(group, arm, value)`` triplets and drop unusable rows.

    A row is skipped when the group is missing, the arm is missing / negative (``-1`` =
    historical policy not mapped onto the arm grid) or the value is missing / non-finite.
    Returns ``(groups[object str], arms[int64], values[float], n_skipped)``.
    """
    g, a, v = _as_series(groups), _as_series(arms), _as_series(values)
    if not len(g) == len(a) == len(v):
        raise ValueError(f"length mismatch: groups={len(g)}, arms={len(a)}, values={len(v)}")
    gs = g.astype("string")
    a_arr = _as_float_array(a)
    v_arr = _as_float_array(v)
    with np.errstate(invalid="ignore"):
        keep = ~gs.isna().to_numpy() & np.isfinite(a_arr) & (a_arr >= 0) & np.isfinite(v_arr)
    if np.any(a_arr[keep] != np.floor(a_arr[keep])):
        raise ValueError("arm indices must be integers")
    return (
        gs.to_numpy(dtype=object)[keep],
        a_arr[keep].astype(np.int64),
        v_arr[keep],
        int((~keep).sum()),
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def tilted_mean(mean: float, tilt: float) -> float:
    """``σ(logit(mean) + tilt)``; exactly ``mean`` when ``tilt == 0``."""
    if tilt == 0.0:
        return mean
    m = min(max(float(mean), 1e-12), 1.0 - 1e-12)
    z = math.log(m) - math.log1p(-m) + float(tilt)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ----------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------
class HierarchicalBetaBernoulli:
    """Three-level hierarchical Beta-Bernoulli model of default (proposal §3.3, Stage 1).

    Levels: global → group ``g`` (segmenter key, ``str``) → leaf ``(g, a)`` (``a`` = arm
    index, ``int``). Prior construction is top-down:

    * global prior ``Beta(α0, β0)``; global posterior ``Beta(α0 + D, β0 + N − D)``;
    * group prior ``BetaPosterior.from_mean_concentration(m_0, κ_group)`` with ``m_0`` the
      global posterior mean;
    * leaf prior ``from_mean_concentration(m_g, κ_leaf)`` with ``m_g`` the group posterior
      mean;
    * the posterior at every level is the conjugate update with that level's own counts —
      exactly the proposal's ``Beta(α_{g,a} + d, β_{g,a} + n)`` with mean
      ``(α + d)/(α + β + d + n)``.

    ``κ = "auto"`` (``HierarchyConfig.kappa_group`` / ``kappa_leaf``) estimates the
    concentration by the Beta-binomial method of moments across the children of each parent
    (:func:`beta_binomial_kappa`): one ``κ_group`` from the groups, and one ``κ_leaf`` per
    group from that group's leaves. Children need ``n ≥ 5``; κ is clipped to
    ``[kappa_min, kappa_max]``. A group whose leaves cannot identify κ uses the median of
    the identified per-group values; if none is identified (or for ``κ_group``) the numeric
    config default (20 / 10) is used, recorded in :attr:`diagnostics`.

    **Online updates.** :meth:`update` increments the counts at all three levels, so every
    level's posterior always reflects all data seen. The *priors* of child levels (the
    parent posterior means ``m_0, m_g`` they are anchored on, and the κ's) are frozen at
    :meth:`fit` time and only re-derived by :meth:`refresh_priors`. Re-deriving them on every
    update would make each arm's posterior drift with outcomes of *other* arms of the group
    (and change ``κ = "auto"``) on every single observation, which destabilises the arm
    posteriors that Thompson Sampling and the shadow engine read during a run; refreshing at
    controlled points (e.g. a re-fit or a breaker "resegment") keeps the run reproducible.

    Unseen nodes: :meth:`posterior` of an unseen leaf of a known group is the leaf prior
    (shrunk group posterior); for an unseen group it is the group prior (shrunk global).

    **Leaf-prior tilt.** ``leaf_prior_tilt`` (constructor keyword or
    :meth:`set_leaf_prior_tilt`) is a callable ``(group, arm) -> float`` whose value shifts
    the logit of the leaf prior mean (structured leaf prior; module docstring). It survives
    :meth:`fit` / :meth:`refresh_priors` (it is configuration, not data) and is evaluated
    lazily, so :meth:`refresh_priors` re-applies it to the new anchors. A serialisable tilt
    (one with ``to_dict``, e.g. :class:`~.policy_effect.PolicyEffectTilt`) round-trips
    through :meth:`to_dict` / :meth:`from_dict`; the tilt table actually used for the seen
    groups is stored as well (``leaf_tilt_table``) and restores a non-serialisable tilt.
    """

    def __init__(self, cfg: HierarchyConfig | None = None, *, leaf_prior_tilt: LeafPriorTilt | None = None) -> None:
        self.cfg = cfg if cfg is not None else HierarchyConfig()
        self.cfg.validate()
        self._tilt: LeafPriorTilt | None = None
        self._reset()
        self.set_leaf_prior_tilt(leaf_prior_tilt)

    # ----- state ----------------------------------------------------------------------
    def _reset(self) -> None:
        cfg = self.cfg
        self._global: list[int] = [0, 0]  # [defaults d, non-defaults n]
        self._groups: dict[str, list[int]] = {}
        self._leaves: dict[tuple[str, int], list[int]] = {}
        self._global_prior = BetaPosterior(cfg.alpha0, cfg.beta0)
        self._global_anchor: float = self._global_prior.mean
        self._group_anchor: dict[str, float] = {}
        self._kappa_group: float = self._default_kappa("group")
        self._kappa_leaf: dict[str, float] = {}
        self._kappa_leaf_default: float = self._default_kappa("leaf")
        self._kappa_info: dict[str, Any] = {}
        self._skipped = 0
        self._fitted = False

    def _default_kappa(self, level: str) -> float:
        value = getattr(self.cfg, f"kappa_{level}")
        if isinstance(value, str):  # "auto" before any data: use the numeric default
            lo, hi = sorted((float(self.cfg.kappa_min), float(self.cfg.kappa_max)))
            return float(np.clip(AUTO_KAPPA_FALLBACK[level], lo, hi))
        return float(value)

    # ----- estimation -----------------------------------------------------------------
    def fit(self, groups: Sequence[str], arms: Sequence[int], defaults: Sequence[int]) -> Self:
        """Batch fit from aligned sequences (replaces any previous state).

        ``defaults`` must be 0/1; rows with a missing group, ``arm < 0`` / missing arm, or a
        missing default flag are skipped (counted in ``diagnostics["skipped"]``).
        """
        g, a, d, skipped = _clean_observations(groups, arms, defaults)
        if d.size and not np.all((d == 0.0) | (d == 1.0)):
            raise ValueError("defaults must be 0/1 (missing values are skipped)")
        self._reset()
        self._skipped = skipped
        if d.size:
            frame = pd.DataFrame({"g": g, "a": a, "d": d})
            agg = frame.groupby(["g", "a"], sort=True)["d"].agg(["sum", "count"])
            sums = agg["sum"].to_numpy()
            cnts = agg["count"].to_numpy()
            for (gk, ak), s, c in zip(agg.index, sums, cnts, strict=True):
                dflt, n = int(round(float(s))), int(c)
                group = str(gk)
                self._leaves[(group, int(ak))] = [dflt, n - dflt]
                gc = self._groups.setdefault(group, [0, 0])
                gc[0] += dflt
                gc[1] += n - dflt
                self._global[0] += dflt
                self._global[1] += n - dflt
        self._fitted = True
        self.refresh_priors()
        return self

    def update(self, group: str, arm: int, defaulted: bool) -> None:
        """Online conjugate update with one resolved loan at all three levels.

        Priors stay frozen (see class docstring). A missing ``defaulted`` or ``arm < 0`` is
        ignored, mirroring :meth:`fit`.
        """
        if _is_missing(defaulted) or _is_missing(arm):
            return
        arm_i = int(arm)
        if arm_i < 0:
            return
        value = float(defaulted)
        if value not in (0.0, 1.0):
            raise ValueError(f"defaulted must be 0/1 or bool, got {defaulted!r}")
        idx = 0 if value == 1.0 else 1
        g = str(group)
        self._global[idx] += 1
        self._groups.setdefault(g, [0, 0])[idx] += 1
        self._leaves.setdefault((g, arm_i), [0, 0])[idx] += 1

    def refresh_priors(self) -> None:
        """Re-derive child priors (anchors and κ) from the current counts, top-down."""
        cfg = self.cfg
        info: dict[str, Any] = {}
        lo, hi = sorted((float(cfg.kappa_min), float(cfg.kappa_max)))
        self._global_anchor = self.global_posterior().mean

        if isinstance(cfg.kappa_group, str):
            counts = list(self._groups.values())
            est = beta_binomial_kappa(
                [c[0] for c in counts], [c[0] + c[1] for c in counts], lo, hi
            )
            self._kappa_group = est if est is not None else self._default_kappa("group")
            info["kappa_group_source"] = "auto" if est is not None else "auto_fallback"
        else:
            self._kappa_group = float(cfg.kappa_group)
            info["kappa_group_source"] = "config"
        self._group_anchor = {g: self.group_posterior(g).mean for g in self._groups}

        if isinstance(cfg.kappa_leaf, str):
            by_group: dict[str, list[list[int]]] = {}
            for (g, _arm), c in self._leaves.items():
                by_group.setdefault(g, []).append(c)
            estimates: dict[str, float] = {}
            for g, children in by_group.items():
                est = beta_binomial_kappa(
                    [c[0] for c in children], [c[0] + c[1] for c in children], lo, hi
                )
                if est is not None:
                    estimates[g] = est
            self._kappa_leaf = estimates
            if estimates:
                self._kappa_leaf_default = float(np.median(list(estimates.values())))
                info["kappa_leaf_default_source"] = "median_of_groups"
            else:
                self._kappa_leaf_default = self._default_kappa("leaf")
                info["kappa_leaf_default_source"] = "auto_fallback"
            info["kappa_leaf_source"] = "auto"
            info["kappa_leaf_estimated_groups"] = len(estimates)
        else:
            self._kappa_leaf = {}
            self._kappa_leaf_default = float(cfg.kappa_leaf)
            info["kappa_leaf_source"] = "config"
        self._kappa_info = info

    # ----- leaf-prior tilt ---------------------------------------------------------------
    def set_leaf_prior_tilt(self, tilt: LeafPriorTilt | None) -> None:
        """Install (or remove with ``None``) the leaf-prior tilt ``(group, arm) -> logit shift``."""
        if tilt is not None and not callable(tilt):
            raise TypeError("leaf_prior_tilt must be a callable (group, arm) -> float or None")
        self._tilt = tilt

    @property
    def leaf_prior_tilt(self) -> LeafPriorTilt | None:
        return self._tilt

    def prior_tilt(self, group: str, arm: int | None) -> float:
        """Logit shift ``t(g, a)`` of the leaf prior mean (0 without a tilt or arm)."""
        if self._tilt is None or arm is None or _is_missing(arm):
            return 0.0
        a = int(arm)
        if a < 0:
            return 0.0
        value = float(self._tilt(str(group), a))
        if not math.isfinite(value):
            raise NumericalInstabilityError(f"leaf-prior tilt of ({group!r}, {a}) is not finite: {value!r}")
        return value

    def _tilt_arms(self) -> list[int]:
        n = getattr(self._tilt, "n_arms", None)
        if isinstance(n, (int, np.integer)) and n > 0:
            return list(range(int(n)))
        return sorted({a for _g, a in self._leaves})

    def tilt_table(self) -> pd.DataFrame:
        """``group, arm, tilt`` for every seen group × arm (the arms the tilt knows, else the
        observed arms); empty without a tilt."""
        rows = [(g, a, self.prior_tilt(g, a)) for g in sorted(self._groups) for a in self._tilt_arms()] if self._tilt else []
        return pd.DataFrame(
            {
                "group": pd.array([r[0] for r in rows], dtype="string"),
                "arm": np.asarray([r[1] for r in rows], dtype=np.int64),
                "tilt": np.asarray([r[2] for r in rows], dtype=float),
            }
        )

    # ----- priors / posteriors --------------------------------------------------------
    @property
    def global_prior(self) -> BetaPosterior:
        """``p_0 ~ Beta(α0, β0)``."""
        return self._global_prior

    def global_posterior(self) -> BetaPosterior:
        d, n = self._global
        return self._global_prior.update(d, n)

    def group_prior(self, group: str | None = None) -> BetaPosterior:
        """``Beta(κ_group·m_0, κ_group·(1 − m_0))`` — identical for every group."""
        return BetaPosterior.from_mean_concentration(self._global_anchor, self._kappa_group)

    def group_posterior(self, group: str) -> BetaPosterior:
        prior = self.group_prior(group)
        counts = self._groups.get(str(group))
        return prior if counts is None else prior.update(counts[0], counts[1])

    def leaf_prior(self, group: str, arm: int | None = None) -> BetaPosterior:
        """``Beta(κ_leaf·m_{g,a}, κ_leaf·(1 − m_{g,a}))``, ``m_{g,a} = σ(logit(m_g) + t(g, a))``
        with ``m_g`` frozen at the last refresh (``t = 0`` without a tilt or ``arm``).

        A group created online after the last refresh is anchored on ``m_0`` (its prior mean).
        """
        g = str(group)
        anchor = self._group_anchor.get(g, self._global_anchor)
        return BetaPosterior.from_mean_concentration(tilted_mean(anchor, self.prior_tilt(g, arm)), self.kappa_leaf(g))

    def unseen_group_prior(self, group: str, arm: int | None = None) -> BetaPosterior:
        """Prior of arm ``arm`` of a group without data: the group prior, its mean tilted
        to ``σ(logit(m_0) + t(g, a))`` (exactly :meth:`group_prior` when ``t = 0``)."""
        t = self.prior_tilt(group, arm)
        if t == 0.0:
            return self.group_prior(group)
        return BetaPosterior.from_mean_concentration(tilted_mean(self._global_anchor, t), self._kappa_group)

    def posterior(self, group: str, arm: int) -> BetaPosterior:
        """Leaf posterior ``Beta(α_{g,a} + d_{g,a}, β_{g,a} + n_{g,a})``.

        Unseen leaf of a known group → its (tilted) leaf prior; unseen group → the group
        prior (mean tilted by ``t(g, a)``, see :meth:`unseen_group_prior`).
        """
        g = str(group)
        if g not in self._groups:
            return self.unseen_group_prior(g, arm)
        prior = self.leaf_prior(g, arm)
        counts = self._leaves.get((g, int(arm)))
        return prior if counts is None else prior.update(counts[0], counts[1])

    def p_default(self, group: str, arm: int) -> float:
        """Posterior mean ``E[p_{g,a}] = (α + d)/(α + β + d + n)``."""
        return self.posterior(group, arm).mean

    def sample(self, group: str, arm: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.posterior(group, arm).sample(rng, n)

    # ----- counts / concentrations ------------------------------------------------------
    def counts(self, group: str, arm: int) -> tuple[int, int]:
        """``(d_{g,a}, n_{g,a})`` = (defaults, non-defaults) observed at the leaf."""
        c = self._leaves.get((str(group), int(arm)), (0, 0))
        return int(c[0]), int(c[1])

    def group_counts(self, group: str) -> tuple[int, int]:
        c = self._groups.get(str(group), (0, 0))
        return int(c[0]), int(c[1])

    def global_counts(self) -> tuple[int, int]:
        return int(self._global[0]), int(self._global[1])

    def n_obs(self, group: str, arm: int) -> int:
        d, n = self.counts(group, arm)
        return d + n

    @property
    def kappa_group(self) -> float:
        return self._kappa_group

    def kappa_leaf(self, group: str | None = None) -> float:
        if group is None:
            return self._kappa_leaf_default
        return self._kappa_leaf.get(str(group), self._kappa_leaf_default)

    def shrinkage_weight(self, group: str, arm: int | None = None) -> float:
        """``n/(n + κ)``: weight of the node's own empirical rate in its posterior mean."""
        if arm is None:
            n, kappa = sum(self.group_counts(group)), self._kappa_group
        else:
            n, kappa = self.n_obs(group, arm), self.kappa_leaf(group)
        return n / (n + kappa)

    @property
    def groups(self) -> list[str]:
        return sorted(self._groups)

    @property
    def leaves(self) -> list[tuple[str, int]]:
        return sorted(self._leaves)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def diagnostics(self) -> dict[str, Any]:
        d, n = self._global
        return {
            "n_obs": d + n,
            "n_defaults": d,
            "n_groups": len(self._groups),
            "n_leaves": len(self._leaves),
            "skipped": self._skipped,
            "kappa_group": self._kappa_group,
            "kappa_leaf_default": self._kappa_leaf_default,
            **self._kappa_info,
            "policy_effect": self._tilt_diagnostics(),
        }

    def _tilt_diagnostics(self) -> dict[str, Any]:
        if self._tilt is None:
            return {"installed": False}
        fn = getattr(self._tilt, "diagnostics", None)
        extra = fn() if callable(fn) else {}
        return {"installed": True, "type": type(self._tilt).__name__, **extra}

    # ----- reporting --------------------------------------------------------------------
    def summary(self, hdi_mass: float = 0.95) -> pd.DataFrame:
        """One row per node: global, every group, every observed leaf.

        Columns ``level, key, group, arm, alpha, beta, n_obs, defaults, mean, hdi_low,
        hdi_high, shrinkage_weight, prior_mean, empirical_rate, kappa, prior_tilt_logit``.
        ``group`` is NA for the global row and ``arm`` is NA for global/group rows.
        ``shrinkage_weight = n/(n + κ)`` (0 for the global level, whose prior is not
        data-derived). ``prior_mean`` of a leaf is the (tilted) leaf prior mean and
        ``prior_tilt_logit`` its logit shift ``t(g, a)`` (0 for global/group rows).
        """
        level: list[str] = ["global"]
        key: list[str] = ["global"]
        grp: list[Any] = [pd.NA]
        arm: list[Any] = [pd.NA]
        gd, gn = self._global
        post = self.global_posterior()
        alpha, beta = [post.alpha], [post.beta]
        n_obs, defaults = [gd + gn], [gd]
        prior_mean, kappa = [self._global_prior.mean], [self._global_prior.concentration]
        weight = [0.0]
        tilt = [0.0]

        gprior = self.group_prior()
        for g in sorted(self._groups):
            d, n = self._groups[g]
            p = gprior.update(d, n)
            level.append("group")
            key.append(g)
            grp.append(g)
            arm.append(pd.NA)
            alpha.append(p.alpha)
            beta.append(p.beta)
            n_obs.append(d + n)
            defaults.append(d)
            prior_mean.append(gprior.mean)
            kappa.append(self._kappa_group)
            weight.append((d + n) / (d + n + self._kappa_group))
            tilt.append(0.0)

        for g, a in sorted(self._leaves):
            d, n = self._leaves[(g, a)]
            lp = self.leaf_prior(g, a)
            p = lp.update(d, n)
            k = self.kappa_leaf(g)
            level.append("leaf")
            key.append(f"{g}{LEAF_KEY_SEPARATOR}{a}")
            grp.append(g)
            arm.append(a)
            alpha.append(p.alpha)
            beta.append(p.beta)
            n_obs.append(d + n)
            defaults.append(d)
            prior_mean.append(lp.mean)
            kappa.append(k)
            weight.append((d + n) / (d + n + k))
            tilt.append(self.prior_tilt(g, a))

        al, be = np.asarray(alpha), np.asarray(beta)
        lo, hi = beta_hdi(al, be, hdi_mass)
        nobs = np.asarray(n_obs, dtype=np.int64)
        ndef = np.asarray(defaults, dtype=np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            emp = np.where(nobs > 0, ndef / np.maximum(nobs, 1), np.nan)
        return pd.DataFrame(
            {
                "level": pd.array(level, dtype="string"),
                "key": pd.array(key, dtype="string"),
                "group": pd.array(grp, dtype="string"),
                "arm": pd.array(arm, dtype="Int64"),
                "alpha": al,
                "beta": be,
                "n_obs": nobs,
                "defaults": ndef,
                "mean": al / (al + be),
                "hdi_low": lo,
                "hdi_high": hi,
                "shrinkage_weight": np.asarray(weight, dtype=float),
                "prior_mean": np.asarray(prior_mean, dtype=float),
                "empirical_rate": emp,
                "kappa": np.asarray(kappa, dtype=float),
                "prior_tilt_logit": np.asarray(tilt, dtype=float),
            }
        )

    # ----- serialisation ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state; :meth:`from_dict` restores it exactly.

        The tilt is stored twice: ``leaf_prior_tilt`` (the serialised tilt function when it
        has ``to_dict``, else ``None``) and ``leaf_tilt_table`` (``[group, arm, tilt]`` for
        every seen group × arm, the values the priors actually used).
        """
        tilt_fn = getattr(self._tilt, "to_dict", None) if self._tilt is not None else None
        table = self.tilt_table()
        return {
            "type": "HierarchicalBetaBernoulli",
            "version": 2,
            "cfg": self.cfg.to_dict(),
            "fitted": self._fitted,
            "skipped": self._skipped,
            "global": [int(self._global[0]), int(self._global[1])],
            "groups": {g: [int(c[0]), int(c[1])] for g, c in self._groups.items()},
            "leaves": [[g, int(a), int(c[0]), int(c[1])] for (g, a), c in self._leaves.items()],
            "global_anchor": float(self._global_anchor),
            "group_anchor": {g: float(v) for g, v in self._group_anchor.items()},
            "kappa_group": float(self._kappa_group),
            "kappa_leaf": {g: float(v) for g, v in self._kappa_leaf.items()},
            "kappa_leaf_default": float(self._kappa_leaf_default),
            "kappa_info": dict(self._kappa_info),
            "leaf_prior_tilt": tilt_fn() if callable(tilt_fn) else None,
            "leaf_tilt_table": [
                [str(g), int(a), float(t)]
                for g, a, t in zip(table["group"].tolist(), table["arm"].tolist(), table["tilt"].tolist(), strict=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HierarchicalBetaBernoulli:
        from .policy_effect import TiltTable, tilt_from_dict

        obj = cls(HierarchyConfig.from_dict(data.get("cfg")))
        if data.get("leaf_prior_tilt"):
            obj._tilt = tilt_from_dict(data["leaf_prior_tilt"])
        elif data.get("leaf_tilt_table"):
            obj._tilt = TiltTable(data["leaf_tilt_table"])
        obj._fitted = bool(data.get("fitted", True))
        obj._skipped = int(data.get("skipped", 0))
        obj._global = [int(v) for v in data["global"]]
        obj._groups = {str(g): [int(c[0]), int(c[1])] for g, c in data["groups"].items()}
        obj._leaves = {(str(g), int(a)): [int(d), int(n)] for g, a, d, n in data["leaves"]}
        obj._global_anchor = float(data["global_anchor"])
        obj._group_anchor = {str(g): float(v) for g, v in data["group_anchor"].items()}
        obj._kappa_group = float(data["kappa_group"])
        obj._kappa_leaf = {str(g): float(v) for g, v in data["kappa_leaf"].items()}
        obj._kappa_leaf_default = float(data["kappa_leaf_default"])
        obj._kappa_info = dict(data.get("kappa_info", {}))
        return obj

    def __repr__(self) -> str:
        d, n = self._global
        return (
            f"HierarchicalBetaBernoulli(n_obs={d + n}, groups={len(self._groups)}, "
            f"leaves={len(self._leaves)}, κ_group={self._kappa_group:.3g}, "
            f"κ_leaf={self._kappa_leaf_default:.3g}, tilt={'yes' if self._tilt is not None else 'no'})"
        )
