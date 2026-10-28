"""Policy protocol of the decision layer and the shared per-group arm-set machinery.

SPEC §2.8 / proposal §3.3: a *policy* maps a borrower context ``X_t`` (pre-decision
variables only, Appendix F.3) and an eligibility mask over the arm space ``𝒜`` to a
lending policy ``A_t``; it later receives interim and terminal reward observations.
Every policy also supports the governance operations required by the PSI circuit
breaker (G.5): ``freeze``/``unfreeze`` (no learning while the breaker is open) and
``snapshot``/``restore`` (champion rollback), and is JSON-serialisable through
``state``/``from_state`` for checkpoint/resume (G.6).

:class:`ArmSetPolicy` implements all of that once for every bandit in this package: one
:class:`~bayesian_credit.bandits.arm_state.GaussianArmSet` per borrower group ``g``
(``"*"`` when the policy is flat), lazily created with a configurable warm start, plus
an optional pooled parent level that receives every reward (the hierarchy's parent).
Subclasses only implement the selection rule :meth:`ArmSetPolicy._choose`.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from ..config import BanditConfig
from ..data.arms import ArmSpace
from ..data.schema import BorrowerContext
from ..data.segmentation import DIMENSIONS, Segmenter, dti_band, emp_band, fico_band, income_band
from ..exceptions import ConfigurationError, NumericalInstabilityError
from ..types import FeedbackKind
from .arm_state import GaussianArmPosterior, GaussianArmSet

#: Group key of a flat policy and of the pooled (all-borrowers) parent level.
POOLED_GROUP = "*"

#: Warm-start provider: group -> (mu[n_arms], sigma[n_arms]) prior per arm, typically
#: ``reward_fn.expected(arm, two_stage.risk(g, k))`` mean/std (SPEC §2.8 "risk_model").
ArmPriorProvider = Callable[[str], tuple[np.ndarray, np.ndarray]]

ARM_TABLE_COLUMNS = ("group", "arm", "mu", "sigma", "n", "selections")


# ======================================================================================
# Decision
# ======================================================================================
@dataclass(eq=False)
class Decision:
    """A_t = argmax_k θ̃_k chosen for one borrower (§3.3): the arm, the borrower group it
    was chosen in, and diagnostics.

    ``scores`` holds the per-arm quantity the rule maximised (Thompson draws θ̃_k, UCB
    indices or q-values; NaN for ineligible arms). ``fallback`` marks decisions made by
    the circuit-breaker fallback rather than by the policy itself.
    """

    arm: int
    group: str
    policy_name: str
    scores: np.ndarray | None = None
    fallback: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.arm = int(self.arm)
        self.group = str(self.group)
        self.fallback = bool(self.fallback)
        if self.meta is None:
            self.meta = {}
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=float)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Decision):
            return NotImplemented
        if (self.arm, self.group, self.policy_name, self.fallback, self.meta) != (
            other.arm,
            other.group,
            other.policy_name,
            other.fallback,
            other.meta,
        ):
            return False
        if self.scores is None or other.scores is None:
            return self.scores is None and other.scores is None
        return bool(np.array_equal(self.scores, other.scores, equal_nan=True))

    __hash__ = None  # type: ignore[assignment]  # mutable value object

    def to_dict(self, include_scores: bool = True) -> dict[str, Any]:
        """JSON-able dict (NaN/inf scores become ``None``)."""
        scores = None
        if include_scores and self.scores is not None:
            scores = [float(s) if math.isfinite(s) else None for s in self.scores.tolist()]
        return {
            "arm": self.arm,
            "group": self.group,
            "policy_name": self.policy_name,
            "scores": scores,
            "fallback": self.fallback,
            "meta": jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Decision:
        scores = d.get("scores")
        arr = None
        if scores is not None:
            arr = np.array([np.nan if s is None else float(s) for s in scores], dtype=float)
        return cls(
            arm=int(d["arm"]),
            group=str(d["group"]),
            policy_name=str(d["policy_name"]),
            scores=arr,
            fallback=bool(d.get("fallback", False)),
            meta=dict(d.get("meta") or {}),
        )


# ======================================================================================
# Policy protocol
# ======================================================================================
class Policy(ABC):
    """Protocol every decision policy implements (§3.3 policy selection, G.3 feedback,
    G.5 governance; SPEC §2.8).

    * ``select(context, eligible, rng, *, historical_arm=None) -> Decision``
    * ``update(decision, reward, kind, weight)`` — ``weight`` defaults to 1 for terminal
      and ``BanditConfig.interim_weight`` for interim feedback; ``0`` means no update.
    * ``freeze()`` / ``unfreeze()``: while frozen ``update`` is a silent no-op counted in
      ``dropped_updates`` (the breaker's "no adaptation" guarantee).
    * ``snapshot()`` / ``restore()``: deep, JSON-able learning state (champion rollback).
    * ``state()`` / ``from_state()``: snapshot plus configuration (checkpoint/resume).
    """

    name: str = "policy"

    def __init__(self) -> None:
        self._frozen = False
        self._dropped_updates = 0

    # ----- decisions & learning ------------------------------------------------------------
    @abstractmethod
    def select(
        self,
        context: BorrowerContext,
        eligible: np.ndarray,
        rng: np.random.Generator,
        *,
        historical_arm: int | None = None,
    ) -> Decision: ...

    @abstractmethod
    def update(
        self,
        decision: Decision,
        reward: float,
        kind: FeedbackKind | str = FeedbackKind.TERMINAL,
        weight: float | None = None,
    ) -> None: ...

    # ----- governance ----------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def dropped_updates(self) -> int:
        return self._dropped_updates

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    @abstractmethod
    def snapshot(self) -> dict[str, Any]: ...

    @abstractmethod
    def restore(self, snap: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def state(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        prior_provider: ArmPriorProvider | None = None,
    ) -> Policy: ...

    # ----- inspection ------------------------------------------------------------------------
    @abstractmethod
    def arm_table(self, group: str | None = None) -> pd.DataFrame: ...

    @abstractmethod
    def group_keys(self) -> list[str]: ...

    def group_for(self, context: BorrowerContext | Mapping[str, Any]) -> str:
        """Group key this policy would use for ``context`` (``"*"`` for flat policies)."""
        return POOLED_GROUP

    def means_from_snapshot(self, snap: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Posterior mean per arm for every group stored in ``snap`` (champion fallback)."""
        raise NotImplementedError(f"{type(self).__name__} does not expose posterior means")


# ======================================================================================
# Helpers
# ======================================================================================
def jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and enums to JSON-friendly Python values."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Enum):
        return value.value
    return value


def coerce_kind(kind: FeedbackKind | str) -> FeedbackKind:
    try:
        return FeedbackKind(kind)
    except ValueError as exc:
        raise ConfigurationError(f"unknown feedback kind {kind!r}") from exc


def resolve_weight(kind: FeedbackKind, weight: float | None, interim_weight: float) -> float:
    """Default weights: 1 for terminal, ``interim_weight`` for interim observations."""
    if weight is None:
        return 1.0 if kind == FeedbackKind.TERMINAL else float(interim_weight)
    w = float(weight)
    if not math.isfinite(w) or w < 0:
        raise NumericalInstabilityError(f"observation weight must be finite and >= 0, got {weight!r}")
    return w


def normalise_eligible(eligible: np.ndarray | None, arm_space: ArmSpace) -> np.ndarray:
    """Boolean mask over arms with the null arm forced eligible; ``None`` ⇒ all arms."""
    n = arm_space.n_arms
    if eligible is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.array(eligible, dtype=bool).reshape(-1)
        if mask.size != n:
            raise ValueError(f"eligible mask has {mask.size} entries, arm space has {n} arms")
    if arm_space.null_arm is not None:
        mask[arm_space.null_arm] = True
    if not mask.any():
        raise ValueError("no eligible arm")
    return mask


def argmax_eligible(values: np.ndarray, idx: np.ndarray) -> int:
    """Arm in ``idx`` (ascending) maximising ``values``; ties go to the lowest index."""
    return int(idx[int(np.argmax(values[idx]))])


def masked_scores(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.size, np.nan)
    out[mask] = values[mask]
    return out


def decline_arm(arm_space: ArmSpace, mask: np.ndarray) -> int:
    """The null arm; without one, the eligible arm with the smallest exposure."""
    if arm_space.null_arm is not None:
        return int(arm_space.null_arm)
    idx = np.flatnonzero(mask)
    return int(idx[int(np.argmin(arm_space.amounts[idx]))])


def valid_arm(arm: Any, n_arms: int) -> int | None:
    """``arm`` as an int in ``[0, n_arms)``, else ``None`` (None/NaN/-1/out of range)."""
    if arm is None:
        return None
    try:
        f = float(arm)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f != int(f):
        return None
    a = int(f)
    return a if 0 <= a < n_arms else None


_ABSENT = object()

#: Dimensions whose label depends only on the numeric value (``pd.to_numeric`` + bands).
_NUMERIC_BANDS = (fico_band, dti_band, income_band, emp_band)


def _cache_token(v: Any, numeric_band: bool) -> Any:
    """Hashable token of a feature value that determines its segment label.

    numpy and Python scalars of the same kind share a token (so primed entries match
    lookups however the context was built); numbers collapse to ``float`` only for the
    numeric band dimensions, where the label depends on the value alone.
    """
    if v is _ABSENT:
        return _ABSENT
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return ("bool", bool(v))
    if isinstance(v, str):
        return ("str", str(v))
    if isinstance(v, (int, np.integer)):
        return ("num", float(v)) if numeric_band else ("int", int(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return None
        return ("num", float(v)) if numeric_band else ("float", float(v))
    return (type(v), v)


class GroupKeyResolver:
    """Borrower group g of the context X_t (§3.3 hierarchy; SPEC §1.4): a memoised
    ``Segmenter.key_for`` (one segmenter call per *distinct* input).

    ``Segmenter.key_for`` builds a one-row DataFrame per call (milliseconds), far too slow
    per decision. The key is a pure function of the dimensions' source columns, so the
    result is cached on the tuple of those values (absent vs missing vs value are
    distinguished); every miss delegates to the segmenter, so keys are identical.
    :meth:`prime` fills the cache for a whole batch of contexts with one vectorised
    ``Segmenter.assign`` call (≈ µs per row), which matters for continuous dimensions
    such as ``dti_band`` / ``income_band`` where most borrowers are distinct.
    """

    def __init__(self, segmenter: Segmenter, max_size: int = 200_000):
        self.segmenter = segmenter
        self.max_size = int(max_size)
        self._dims = tuple(
            (DIMENSIONS[d][0], DIMENSIONS[d][1] in _NUMERIC_BANDS) for d in segmenter.dimensions
        )
        self._cache: dict[tuple[Any, ...], str] = {}

    def _token_tuple(self, feats: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(_cache_token(feats.get(src, _ABSENT), band) for src, band in self._dims)

    def __call__(self, context: BorrowerContext | Mapping[str, Any]) -> str:
        feats = getattr(context, "features", context)
        key = self._token_tuple(feats)
        try:
            hit = self._cache.get(key)
        except TypeError:  # unhashable feature value: no caching
            return self.segmenter.key_for(context)
        if hit is None:
            hit = self.segmenter.key_for(context)
            if len(self._cache) >= self.max_size:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def prime(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised keys for the rows of ``df`` (also cached for later lookups)."""
        keys = self.segmenter.assign(df)
        n = len(df)
        if n == 0:
            return keys
        columns = [
            [_cache_token(v, band) for v in df[src].tolist()] if src in df.columns else [_ABSENT] * n
            for src, band in self._dims
        ]
        if len(self._cache) + n > self.max_size:
            self._cache.clear()
        for token, key in zip(zip(*columns, strict=True), keys.tolist(), strict=True):
            try:
                self._cache[token] = str(key)
            except TypeError:
                continue
        return keys


# ======================================================================================
# Shared per-group arm-set machinery
# ======================================================================================
class ArmSetPolicy(Policy):
    """Per-group Gaussian arm sets with warm start, pooled parent, freeze and snapshots.

    Contextualisation (§3.3 / SPEC §2.8): with a :class:`Segmenter` the policy keeps one
    arm set per borrower group ``g = segmenter.key_for(X_t)``, created on first visit;
    without one it is flat (single group ``"*"``). Cold start of a new group follows
    ``cfg.warm_start``:

    * ``"risk_model"`` — ``prior_provider(g)`` (μ_k⁰, σ_k⁰ from the Two-Stage Beta-Beta
      reward distribution). Without a provider it falls back to ``"global"`` and every
      decision carries ``meta["warm_start_fallback"] = True``.
    * ``"global"`` — copy the pooled parent's current posteriors (μ_k, σ_k²) as the group
      prior with ``n = 0`` **[decision]**: the parent informs exploration but carries no
      pseudo-count, so the group's own data take over from its first observations.
    * ``"flat"`` — N(``cfg.prior_mu``, ``cfg.prior_sigma``²) for every arm.

    The pooled parent (contextual policies only) is updated with every reward; it is
    reported as group ``"*"`` by :meth:`arm_table` and used by :meth:`simpsons_check`.
    **[decision]** ``pin_null_arm``: the decline arm's prior is N(0, min_sigma²) because
    its reward is 0 by construction (SPEC §2.7), so "lend" must beat a sure zero.

    Selection counts are kept per (group, arm). ``snapshot`` captures the learning state
    (arm sets, pooled parent, segmenter, counters) but not the governance flags
    (``frozen``, ``dropped_updates``): restoring a champion never rewrites the audit trail.
    ``state`` adds class, configuration and the governance flags.
    """

    default_name: ClassVar[str] = "arm_set"
    requires_segmenter: ClassVar[bool] = False
    supports_segmenter: ClassVar[bool] = True
    snapshot_format: ClassVar[int] = 1

    def __init__(
        self,
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        *,
        cfg: BanditConfig | None = None,
        prior_provider: ArmPriorProvider | None = None,
        warm_start: str | None = None,
        name: str | None = None,
        pin_null_arm: bool = True,
    ):
        super().__init__()
        cfg = cfg if cfg is not None else BanditConfig()
        if warm_start is not None:
            cfg = dataclasses.replace(cfg, warm_start=warm_start)
        cfg.validate()
        if segmenter is None and self.requires_segmenter:
            raise ConfigurationError(f"{type(self).__name__} requires a Segmenter")
        if segmenter is not None and not self.supports_segmenter:
            raise ConfigurationError(
                f"{type(self).__name__} is flat; use ContextualThompsonSampling for groups"
            )
        self.arm_space = arm_space
        self.cfg = cfg
        self.name = name or self.default_name
        self.pin_null_arm = bool(pin_null_arm)
        self.prior_provider = prior_provider
        self.segmenter: Segmenter | None = None
        self._resolver: GroupKeyResolver | None = None
        if segmenter is not None:
            self._set_segmenter(segmenter)
        self._groups: dict[str, GaussianArmSet] = {}
        self._n_decisions = 0
        self._n_updates = 0
        self._seg_version = 0
        self._pooled: ArmSetPolicy | None = None  # created lazily (subclass params set first)

    # ----- construction hooks --------------------------------------------------------------
    def _params(self) -> dict[str, Any]:
        """Rule-specific constructor parameters persisted in :meth:`state`."""
        return {}

    @classmethod
    def _construct(
        cls,
        arm_space: ArmSpace,
        segmenter: Segmenter | None,
        cfg: BanditConfig,
        prior_provider: ArmPriorProvider | None,
        name: str | None,
        pin_null_arm: bool,
        params: dict[str, Any],
    ) -> Self:
        return cls(
            arm_space,
            segmenter,
            cfg=cfg,
            prior_provider=prior_provider,
            name=name,
            pin_null_arm=pin_null_arm,
            **params,
        )

    def _make_pooled(self) -> ArmSetPolicy:
        """Flat parent level updated with every reward (default: same rule, flat prior)."""
        return type(self)._construct(
            self.arm_space,
            None,
            dataclasses.replace(self.cfg, warm_start="flat"),
            None,
            f"{self.name}:pooled",
            self.pin_null_arm,
            self._params(),
        )

    def _set_segmenter(self, segmenter: Segmenter) -> None:
        self.segmenter = segmenter
        self._resolver = GroupKeyResolver(segmenter)

    # ----- properties ------------------------------------------------------------------------
    @property
    def contextual(self) -> bool:
        return self.segmenter is not None

    @property
    def pooled(self) -> ArmSetPolicy | None:
        """Pooled parent level (contextual policies only; ``None`` when flat)."""
        if self.segmenter is None:
            return None
        if self._pooled is None:
            self._pooled = self._make_pooled()
        return self._pooled

    @property
    def warm_start_fallback(self) -> bool:
        """True when ``warm_start="risk_model"`` was requested without a prior provider."""
        return self.cfg.warm_start == "risk_model" and self.prior_provider is None

    @property
    def segmentation_version(self) -> int:
        """Incremented by :meth:`resegment`; stamped on contextual decisions."""
        return self._seg_version

    @property
    def n_decisions(self) -> int:
        return self._n_decisions

    @property
    def n_updates(self) -> int:
        return self._n_updates

    # ----- groups ------------------------------------------------------------------------------
    def group_for(self, context: BorrowerContext | Mapping[str, Any]) -> str:
        if self._resolver is None:
            return POOLED_GROUP
        return self._resolver(context)

    def prime_groups(self, contexts: pd.DataFrame) -> pd.Series:
        """Optional speed-up: compute (and cache) the group keys of a batch of contexts,
        e.g. an epoch's arrivals, with one vectorised segmenter call. Returns the keys."""
        if self._resolver is None:
            return pd.Series(POOLED_GROUP, index=contexts.index, dtype="string", name="y_group")
        return self._resolver.prime(contexts)

    def _initial_prior(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        """Warm-start prior (mu, sigma²) per arm for a new group."""
        k = self.arm_space.n_arms
        cfg = self.cfg
        if cfg.warm_start == "risk_model" and self.prior_provider is not None:
            mu_raw, sigma_raw = self.prior_provider(group)
            mu = np.array(mu_raw, dtype=float).reshape(-1)
            sigma = np.array(sigma_raw, dtype=float).reshape(-1)
            if mu.size != k or sigma.size != k:
                raise ConfigurationError(
                    f"prior_provider({group!r}) returned {mu.size}/{sigma.size} values for {k} arms"
                )
            # Arms the risk model cannot evaluate fall back to the flat prior.
            mu = np.where(np.isfinite(mu), mu, cfg.prior_mu)
            sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, cfg.prior_sigma)
            sigma2 = np.maximum(sigma, cfg.min_sigma) ** 2
        elif cfg.warm_start in ("risk_model", "global") and self.pooled is not None:
            parent = self.pooled._ensure_group(POOLED_GROUP)[0]
            mu = parent.mu.copy()
            sigma2 = np.maximum(parent.sigma2, cfg.min_sigma**2)
        else:
            mu = np.full(k, float(cfg.prior_mu))
            sigma2 = np.full(k, max(float(cfg.prior_sigma), cfg.min_sigma) ** 2)
        if self.pin_null_arm and self.arm_space.null_arm is not None:
            mu[self.arm_space.null_arm] = 0.0
            sigma2[self.arm_space.null_arm] = cfg.min_sigma**2
        return mu, sigma2

    def _ensure_group(self, group: str) -> tuple[GaussianArmSet, bool]:
        arms = self._groups.get(group)
        if arms is not None:
            return arms, False
        mu, sigma2 = self._initial_prior(group)
        arms = GaussianArmSet(mu, sigma2)
        self._groups[group] = arms
        return arms, True

    def resegment(
        self, segmenter: Segmenter, prior_provider: ArmPriorProvider | None = None
    ) -> None:
        """Switch to a new segmentation (breaker action ``resegment``).

        Group-level arm sets are discarded and re-warm-started lazily (using the new
        ``prior_provider`` if given); the pooled parent is segmentation-free and kept.
        Feedback for decisions made under the previous segmentation updates the pooled
        level only.
        """
        if not self.contextual:
            raise ConfigurationError("resegment() needs a contextual (segmented) policy")
        self._set_segmenter(segmenter)
        if prior_provider is not None:
            self.prior_provider = prior_provider
        self._groups = {}
        self._seg_version += 1

    # ----- decisions -----------------------------------------------------------------------------
    @abstractmethod
    def _choose(
        self,
        arms: GaussianArmSet,
        idx: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        historical_arm: int | None,
    ) -> tuple[int, np.ndarray | None, dict[str, Any]]:
        """Selection rule: return (arm, per-arm scores or None, extra meta)."""

    def select(
        self,
        context: BorrowerContext,
        eligible: np.ndarray,
        rng: np.random.Generator,
        *,
        historical_arm: int | None = None,
    ) -> Decision:
        mask = normalise_eligible(eligible, self.arm_space)
        group = self.group_for(context)
        arms, created = self._ensure_group(group)
        n_obs = arms.n_updates
        idx = np.flatnonzero(mask)
        arm, scores, extra = self._choose(arms, idx, mask, rng, historical_arm)
        arm = int(arm)
        if not mask[arm]:  # pragma: no cover - guards subclass bugs
            raise RuntimeError(f"{type(self).__name__} chose ineligible arm {arm}")
        arms.selections[arm] += 1
        if self.pooled is not None:
            self.pooled._ensure_group(POOLED_GROUP)[0].selections[arm] += 1
        self._n_decisions += 1
        meta: dict[str, Any] = {"group_new": created, "n_group_obs": int(n_obs)}
        if self.contextual:
            meta["seg_version"] = self._seg_version
        if self.warm_start_fallback:
            meta["warm_start_fallback"] = True
        meta.update(extra)
        return Decision(arm=arm, group=group, policy_name=self.name, scores=scores, meta=meta)

    # ----- learning --------------------------------------------------------------------------------
    def update(
        self,
        decision: Decision,
        reward: float,
        kind: FeedbackKind | str = FeedbackKind.TERMINAL,
        weight: float | None = None,
    ) -> None:
        kind = coerce_kind(kind)
        w = resolve_weight(kind, weight, self.cfg.interim_weight)
        r = float(reward)
        if not math.isfinite(r):
            raise NumericalInstabilityError(f"non-finite reward {reward!r} for policy {self.name!r}")
        arm = valid_arm(decision.arm, self.arm_space.n_arms)
        if arm is None:
            raise ValueError(f"decision arm {decision.arm!r} is outside the arm space")
        if w == 0.0:
            return
        if self._frozen:
            self._dropped_updates += 1
            return
        if self.contextual:
            same_segmentation = decision.meta.get("seg_version", self._seg_version) == self._seg_version
            if same_segmentation:
                self._ensure_group(decision.group)[0].update(arm, r, w)
            pooled = self.pooled
            if pooled is not None:
                pooled._ensure_group(POOLED_GROUP)[0].update(arm, r, w)
                pooled._n_updates += 1
        else:
            self._ensure_group(POOLED_GROUP)[0].update(arm, r, w)
        self._n_updates += 1

    # ----- snapshots & state -------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "format": self.snapshot_format,
            "groups": {g: a.to_dict() for g, a in self._groups.items()},
            "pooled": self.pooled.snapshot() if self.pooled is not None else None,
            "segmenter": self.segmenter.to_dict() if self.segmenter is not None else None,
            "seg_version": self._seg_version,
            "n_decisions": self._n_decisions,
            "n_updates": self._n_updates,
        }

    def restore(self, snap: Mapping[str, Any]) -> None:
        groups = {str(g): GaussianArmSet.from_dict(d) for g, d in snap["groups"].items()}
        for g, arms in groups.items():
            if len(arms) != self.arm_space.n_arms:
                raise ConfigurationError(
                    f"snapshot group {g!r} has {len(arms)} arms, arm space has {self.arm_space.n_arms}"
                )
        seg_d = snap.get("segmenter")
        if self.contextual and seg_d:
            seg = Segmenter.from_dict(seg_d)
            if seg != self.segmenter:
                self._set_segmenter(seg)
        if self.contextual:
            self._pooled = self._make_pooled()
            if snap.get("pooled") is not None:
                self._pooled.restore(snap["pooled"])
        self._groups = groups
        self._seg_version = int(snap.get("seg_version", 0))
        self._n_decisions = int(snap.get("n_decisions", 0))
        self._n_updates = int(snap.get("n_updates", 0))

    def state(self) -> dict[str, Any]:
        st = self.snapshot()
        st.update(
            {
                "class": type(self).__name__,
                "name": self.name,
                "config": self.cfg.to_dict(),
                "params": jsonable(self._params()),
                "arm_space": self.arm_space.to_dict(),
                "pin_null_arm": self.pin_null_arm,
                "frozen": self.frozen,
                "dropped_updates": self.dropped_updates,
            }
        )
        return st

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        prior_provider: ArmPriorProvider | None = None,
    ) -> Self:
        """Rebuild a policy from :meth:`state`.

        A segmenter stored in the state wins over the ``segmenter`` argument (the stored
        groups are keyed by it — it may have been refined mid-run); the argument is only
        used for states that carry no ``"segmenter"`` entry.
        """
        stored_class = state.get("class")
        if stored_class is not None and stored_class != cls.__name__:
            raise ConfigurationError(f"state is for {stored_class}, not {cls.__name__}")
        stored_arms = state.get("arm_space")
        if stored_arms is not None and ArmSpace.from_dict(stored_arms) != arm_space:
            raise ConfigurationError("arm space differs from the one the policy state was built on")
        if "segmenter" in state:
            seg = Segmenter.from_dict(state["segmenter"]) if state["segmenter"] else None
        else:
            seg = segmenter
        if not cls.supports_segmenter:
            seg = None
        cfg = BanditConfig.from_dict(state.get("config"))
        obj = cls._construct(
            arm_space,
            seg,
            cfg,
            prior_provider,
            state.get("name"),
            bool(state.get("pin_null_arm", True)),
            dict(state.get("params") or {}),
        )
        obj.restore(state)
        obj._frozen = bool(state.get("frozen", False))
        obj._dropped_updates = int(state.get("dropped_updates", 0))
        return obj

    # ----- inspection ----------------------------------------------------------------------------
    def group_keys(self) -> list[str]:
        return sorted(self._groups)

    def arm_table(self, group: str | None = None) -> pd.DataFrame:
        """Posterior table ``group, arm, mu, sigma, n, selections``.

        ``group=None`` concatenates all decision groups; ``"*"`` on a contextual policy
        returns the pooled parent level; an unknown group gives an empty frame.
        """
        if group is None:
            frames = [self._groups[g].table(g) for g in sorted(self._groups)]
        elif group == POOLED_GROUP and self.pooled is not None:
            return self.pooled.arm_table(POOLED_GROUP)
        elif group in self._groups:
            frames = [self._groups[group].table(group)]
        else:
            frames = []
        if not frames:
            return empty_arm_table()
        return pd.concat(frames, ignore_index=True)

    def posterior(self, group: str, arm: int) -> GaussianArmPosterior:
        """The :class:`GaussianArmPosterior` of ``arm`` in ``group`` (a copy)."""
        if group == POOLED_GROUP and self.pooled is not None:
            return self.pooled.posterior(POOLED_GROUP, arm)
        if group not in self._groups:
            raise KeyError(group)
        return self._groups[group].posterior(int(arm))

    def means_from_snapshot(self, snap: Mapping[str, Any]) -> dict[str, np.ndarray]:
        out = {str(g): np.asarray(d["mu"], dtype=float) for g, d in snap.get("groups", {}).items()}
        pooled = snap.get("pooled")
        if pooled is not None and POOLED_GROUP in pooled.get("groups", {}):
            out[POOLED_GROUP] = np.asarray(pooled["groups"][POOLED_GROUP]["mu"], dtype=float)
        return out

    def simpsons_check(self, observed_only: bool = True) -> pd.DataFrame:
        """Simpson's-paradox diagnostic: pooled-best vs group-best arm per group (§3.1/3.2).

        For each group compares the arm with the highest *pooled* posterior mean with the
        arm with the highest *group* posterior mean, and reports the group-level posterior
        mean gap ``mu_g[group_best] − mu_g[pooled_best]`` (≥ 0). With ``observed_only``
        the "best" arm at each level is chosen among arms observed at that level (falling
        back to all arms when nothing has been observed). A reversal means aggregating
        groups would recommend a different policy than the group's own evidence.
        """
        cols = [
            "group",
            "n_group_obs",
            "pooled_best_arm",
            "group_best_arm",
            "pooled_best_mu",
            "group_best_mu",
            "group_mu_of_pooled_best",
            "group_n_of_pooled_best",
            "gap",
            "reversal",
        ]
        pooled = self.pooled
        if pooled is None or not self._groups or POOLED_GROUP not in pooled._groups:
            return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
        parent = pooled._groups[POOLED_GROUP]
        p_best = _best_arm(parent, observed_only)
        rows = []
        for g in sorted(self._groups):
            arms = self._groups[g]
            g_best = _best_arm(arms, observed_only)
            rows.append(
                {
                    "group": g,
                    "n_group_obs": int(arms.n_updates),
                    "pooled_best_arm": p_best,
                    "group_best_arm": g_best,
                    "pooled_best_mu": float(parent.mu[p_best]),
                    "group_best_mu": float(arms.mu[g_best]),
                    "group_mu_of_pooled_best": float(arms.mu[p_best]),
                    "group_n_of_pooled_best": float(arms.n[p_best]),
                    "gap": float(arms.mu[g_best] - arms.mu[p_best]),
                    "reversal": bool(g_best != p_best),
                }
            )
        return pd.DataFrame(rows, columns=cols)


def _best_arm(arms: GaussianArmSet, observed_only: bool) -> int:
    if observed_only and np.any(arms.n > 0):
        idx = np.flatnonzero(arms.n > 0)
        return argmax_eligible(arms.mu, idx)
    return int(np.argmax(arms.mu))


def empty_arm_table() -> pd.DataFrame:
    dtypes = {"group": "string", "arm": np.int64, "selections": np.int64}
    return pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, float)) for c in ARM_TABLE_COLUMNS})
