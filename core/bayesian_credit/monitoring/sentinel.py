"""PSI Sentinel, Bayesian circuit breaker and governed policies (§3.3; G.5; SPEC §2.9).

§3.3 (last paragraph): a PSI Sentinel monitors live data for feature-space deviations
from the training distribution and "acts as a Bayesian circuit breaker, preventing a
'Champion' model from persisting when the underlying economic ground truth has shifted",
prompting the human in the loop to re-evaluate the grouping criteria. G.5 adds that the
monitoring mechanism must not independently authorise changes — human review is part of
the loop. Table 9 bands: PSI < 0.10 normal, 0.10–0.25 watch, > 0.25 breaker.

* :class:`CircuitBreaker` — ``CLOSED → OPEN`` on a BREAKER reading; ``OPEN → HALF_OPEN``
  only on human :meth:`~CircuitBreaker.acknowledge`; ``HALF_OPEN → CLOSED`` after
  ``cfg.recovery_readings`` consecutive NORMAL readings; a BREAKER reading while
  HALF_OPEN re-opens; WATCH readings reset the recovery counter (WATCH is a flag, not a
  state). Every transition is recorded as a :class:`BreakerTransition`.
* :class:`PSISentinel` — evaluates windows with a :class:`PSICalculator`, steps the
  breaker, and opens a :class:`ReviewRecord` (the human-in-the-loop queue) on each trip.
* :class:`GovernedPolicy` — wraps any :class:`~bayesian_credit.bandits.base.Policy`:
  while the breaker is OPEN the policy is frozen and decisions come from the fallback
  (``cfg.fallback``); on acknowledgement it unfreezes, and ``rollback`` restores the
  champion snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..bandits.base import (
    POOLED_GROUP,
    ArmPriorProvider,
    Decision,
    Policy,
    argmax_eligible,
    decline_arm,
    masked_scores,
    normalise_eligible,
    valid_arm,
)
from ..config import PSIConfig
from ..data.arms import ArmSpace
from ..data.schema import BorrowerContext
from ..data.segmentation import Segmenter
from ..exceptions import ConfigurationError, GovernanceError
from ..types import BreakerState, FeedbackKind, ReviewAction, SentinelLevel
from .psi import PSICalculator, PSIResult

FALLBACK_MODES = ("champion_snapshot", "decline", "historical")



def _num_or_none(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _nan_if_none(x: Any) -> float:
    return float("nan") if x is None else float(x)


# ======================================================================================
# Records
# ======================================================================================
@dataclass
class BreakerTransition:
    """One state change of the Bayesian circuit breaker (§3.3; G.5 audit trail)."""

    epoch: str
    from_state: BreakerState
    to_state: BreakerState
    reason: str
    psi: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "from_state": BreakerState(self.from_state).value,
            "to_state": BreakerState(self.to_state).value,
            "reason": self.reason,
            "psi": _num_or_none(self.psi),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BreakerTransition:
        return cls(
            epoch=str(d["epoch"]),
            from_state=BreakerState(d["from_state"]),
            to_state=BreakerState(d["to_state"]),
            reason=str(d.get("reason", "")),
            psi=d.get("psi"),
        )


@dataclass
class ReviewRecord:
    """Human-in-the-loop review opened when the breaker trips (G.5)."""

    id: str
    epoch: str
    psi_aggregate: float
    per_feature: dict[str, float]
    status: str = "pending"  # "pending" | "acknowledged"
    action: ReviewAction | None = None
    note: str | None = None
    acknowledged_epoch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "epoch": self.epoch,
            "psi_aggregate": _num_or_none(self.psi_aggregate),
            "per_feature": {k: _num_or_none(v) for k, v in self.per_feature.items()},
            "status": self.status,
            "action": None if self.action is None else ReviewAction(self.action).value,
            "note": self.note,
            "acknowledged_epoch": self.acknowledged_epoch,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReviewRecord:
        return cls(
            id=str(d["id"]),
            epoch=str(d["epoch"]),
            psi_aggregate=_nan_if_none(d.get("psi_aggregate")),
            per_feature={k: _nan_if_none(v) for k, v in (d.get("per_feature") or {}).items()},
            status=str(d.get("status", "pending")),
            action=None if d.get("action") is None else ReviewAction(d["action"]),
            note=d.get("note"),
            acknowledged_epoch=d.get("acknowledged_epoch"),
        )


@dataclass
class SentinelReading:
    """Result of one :meth:`PSISentinel.evaluate` call (a Table 9 reading, G.5).

    ``skipped`` marks windows too small to score (``n_current < min_window_size``, e.g.
    empty): they are recorded with NaN PSI and leave the breaker untouched.
    """

    epoch: str
    level: SentinelLevel
    psi_aggregate: float
    per_feature: dict[str, float]
    breaker_before: BreakerState
    breaker_after: BreakerState
    tripped: bool
    review_id: str | None = None
    n_current: int = 0
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "level": SentinelLevel(self.level).value,
            "psi_aggregate": _num_or_none(self.psi_aggregate),
            "per_feature": {k: _num_or_none(v) for k, v in self.per_feature.items()},
            "breaker_before": BreakerState(self.breaker_before).value,
            "breaker_after": BreakerState(self.breaker_after).value,
            "tripped": bool(self.tripped),
            "review_id": self.review_id,
            "n_current": int(self.n_current),
            "skipped": bool(self.skipped),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SentinelReading:
        return cls(
            epoch=str(d["epoch"]),
            level=SentinelLevel(d["level"]),
            psi_aggregate=_nan_if_none(d.get("psi_aggregate")),
            per_feature={k: _nan_if_none(v) for k, v in (d.get("per_feature") or {}).items()},
            breaker_before=BreakerState(d["breaker_before"]),
            breaker_after=BreakerState(d["breaker_after"]),
            tripped=bool(d.get("tripped", False)),
            review_id=d.get("review_id"),
            n_current=int(d.get("n_current", 0)),
            skipped=bool(d.get("skipped", False)),
        )


# ======================================================================================
# Circuit breaker
# ======================================================================================
class CircuitBreaker:
    """Bayesian circuit breaker state machine (§3.3; G.5 Table 9 "trigger Bayesian
    review/circuit breaker").

    ``observe(level, epoch, psi)`` steps the machine with a sentinel reading;
    ``acknowledge(epoch)`` is the human action that moves ``OPEN → HALF_OPEN``. Listeners
    in ``on_transition`` are called after every transition (state already updated).
    """

    def __init__(self, cfg: PSIConfig | None = None):
        self.cfg = cfg if cfg is not None else PSIConfig()
        self.state: BreakerState = BreakerState.CLOSED
        self.recovery_count = 0
        self.transitions: list[BreakerTransition] = []
        self.on_transition: list[Callable[[BreakerTransition], None]] = []

    @property
    def is_open(self) -> bool:
        return self.state == BreakerState.OPEN

    def _transition(
        self, to_state: BreakerState, epoch: str, reason: str, psi: float | None
    ) -> BreakerTransition:
        tr = BreakerTransition(str(epoch), self.state, to_state, reason, _num_or_none(psi))
        self.state = to_state
        self.transitions.append(tr)
        for listener in list(self.on_transition):
            listener(tr)
        return tr

    def observe(
        self, level: SentinelLevel | str, epoch: str, psi: float | None = None
    ) -> BreakerTransition | None:
        """Apply one reading; returns the transition it caused, if any."""
        level = SentinelLevel(level)
        if self.state == BreakerState.CLOSED:
            if level == SentinelLevel.BREAKER:
                return self._transition(BreakerState.OPEN, epoch, "psi_breaker", psi)
            return None
        if self.state == BreakerState.OPEN:
            return None  # waits for human acknowledgement
        # HALF_OPEN
        if level == SentinelLevel.BREAKER:
            self.recovery_count = 0
            return self._transition(BreakerState.OPEN, epoch, "psi_breaker_half_open", psi)
        if level == SentinelLevel.WATCH:
            self.recovery_count = 0
            return None
        self.recovery_count += 1
        if self.recovery_count >= self.cfg.recovery_readings:
            self.recovery_count = 0
            return self._transition(BreakerState.CLOSED, epoch, "recovered", psi)
        return None

    def acknowledge(
        self, epoch: str, reason: str = "acknowledged", psi: float | None = None
    ) -> BreakerTransition:
        """Human acknowledgement: ``OPEN → HALF_OPEN`` (then CLOSED if no recovery needed)."""
        if self.state != BreakerState.OPEN:
            raise GovernanceError(f"cannot acknowledge: breaker is {self.state.value}, not open")
        self.recovery_count = 0
        tr = self._transition(BreakerState.HALF_OPEN, epoch, reason, psi)
        if self.cfg.recovery_readings <= 0:
            self._transition(BreakerState.CLOSED, epoch, "recovered", psi)
        return tr

    def to_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "state": self.state.value,
            "recovery_count": self.recovery_count,
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], cfg: PSIConfig | None = None) -> CircuitBreaker:
        if cfg is None:
            cfg = PSIConfig.from_dict(d["cfg"]) if d.get("cfg") is not None else PSIConfig()
        br = cls(cfg)
        br.state = BreakerState(d.get("state", BreakerState.CLOSED))
        br.recovery_count = int(d.get("recovery_count", 0))
        br.transitions = [BreakerTransition.from_dict(t) for t in d.get("transitions", [])]
        return br


# ======================================================================================
# Sentinel
# ======================================================================================
class PSISentinel:
    """PSI Sentinel (G.5): scores windows, classifies them (Table 9), drives the breaker
    and keeps the human-in-the-loop review queue.

    Levels: ``BREAKER`` if PSI > ``cfg.breaker_threshold``, ``WATCH`` if PSI ≥
    ``cfg.watch_threshold``, else ``NORMAL`` (PSI aggregated over features by
    ``cfg.aggregate``). ``monitor_only=True`` records readings without ever stepping the
    breaker (simulation ``breaker_mode="off"``). Windows with fewer than
    ``min_window_size`` rows are recorded as skipped and do not step the breaker.
    Listener lists: ``on_reading`` (every reading), ``on_acknowledge`` (each acknowledged
    :class:`ReviewRecord`) and ``breaker.on_transition``.
    """

    def __init__(
        self,
        calculator: PSICalculator,
        cfg: PSIConfig | None = None,
        *,
        breaker: CircuitBreaker | None = None,
        monitor_only: bool = False,
        min_window_size: int = 1,
    ):
        self.cfg = cfg if cfg is not None else PSIConfig()
        self.cfg.validate()
        if not calculator.fitted:
            raise ConfigurationError("PSISentinel needs a fitted PSICalculator")
        self.calculator = calculator
        self.breaker = breaker if breaker is not None else CircuitBreaker(self.cfg)
        self.monitor_only = bool(monitor_only)
        self.min_window_size = max(int(min_window_size), 1)
        self.history: list[SentinelReading] = []
        self.reviews: list[ReviewRecord] = []
        self.last_result: PSIResult | None = None
        self.on_reading: list[Callable[[SentinelReading], None]] = []
        self.on_acknowledge: list[Callable[[ReviewRecord], None]] = []

    @classmethod
    def from_reference(
        cls, reference_df: pd.DataFrame, cfg: PSIConfig | None = None, **kwargs: Any
    ) -> PSISentinel:
        """Fit a calculator on ``reference_df`` with ``cfg.features / n_bins / eps``."""
        cfg = cfg if cfg is not None else PSIConfig()
        return cls(PSICalculator.from_config(cfg).fit(reference_df), cfg, **kwargs)

    # ----- views -------------------------------------------------------------------------
    @property
    def state(self) -> BreakerState:
        return self.breaker.state

    @property
    def transitions(self) -> list[BreakerTransition]:
        return self.breaker.transitions

    @property
    def pending_reviews(self) -> list[ReviewRecord]:
        return [r for r in self.reviews if r.status == "pending"]

    def review(self, review_id: str) -> ReviewRecord:
        for r in self.reviews:
            if r.id == review_id:
                return r
        raise GovernanceError(f"unknown review {review_id!r}")

    def classify(self, psi: float) -> SentinelLevel:
        """Table 9 band of an aggregate PSI value (NaN ⇒ NORMAL)."""
        if psi is None or not math.isfinite(psi):
            return SentinelLevel.NORMAL
        if psi > self.cfg.breaker_threshold:
            return SentinelLevel.BREAKER
        if psi >= self.cfg.watch_threshold:
            return SentinelLevel.WATCH
        return SentinelLevel.NORMAL

    # ----- operation -----------------------------------------------------------------------
    def evaluate(self, window_df: pd.DataFrame, epoch: str) -> SentinelReading:
        """Score ``window_df`` against the reference and step the breaker."""
        epoch = str(epoch)
        result = self.calculator.psi(window_df)
        agg = result.aggregate(self.cfg.aggregate)
        skipped = result.n_current < self.min_window_size or not math.isfinite(agg)
        level = SentinelLevel.NORMAL if skipped else self.classify(agg)
        before = self.breaker.state
        tripped = False
        review_id = None
        if not skipped and not self.monitor_only:
            tr = self.breaker.observe(level, epoch, agg)
            if tr is not None and tr.to_state == BreakerState.OPEN:
                tripped = True
                review = ReviewRecord(
                    id=f"review-{len(self.reviews) + 1:04d}",
                    epoch=epoch,
                    psi_aggregate=agg,
                    per_feature=dict(result.per_feature),
                )
                self.reviews.append(review)
                review_id = review.id
        reading = SentinelReading(
            epoch=epoch,
            level=level,
            psi_aggregate=agg,
            per_feature=dict(result.per_feature),
            breaker_before=before,
            breaker_after=self.breaker.state,
            tripped=tripped,
            review_id=review_id,
            n_current=result.n_current,
            skipped=skipped,
        )
        self.history.append(reading)
        self.last_result = result
        for listener in list(self.on_reading):
            listener(reading)
        return reading

    def acknowledge(
        self,
        review_id: str,
        action: ReviewAction | str,
        note: str = "",
        epoch: str | None = None,
    ) -> ReviewRecord:
        """Human acknowledgement of a pending review: ``OPEN → HALF_OPEN``.

        ``epoch`` defaults to the latest evaluated epoch. Listeners in ``on_acknowledge``
        (e.g. :class:`GovernedPolicy`, which unfreezes and performs ``rollback``) are
        called afterwards; ``resegment`` is the caller's job.
        """
        review = self.review(review_id)
        if review.status != "pending":
            raise GovernanceError(f"review {review_id!r} is already {review.status}")
        try:
            act = ReviewAction(action)
        except ValueError as exc:
            raise GovernanceError(f"unknown review action {action!r}") from exc
        ack_epoch = str(epoch) if epoch is not None else (
            self.history[-1].epoch if self.history else review.epoch
        )
        review.status = "acknowledged"
        review.action = act
        review.note = note
        review.acknowledged_epoch = ack_epoch
        if self.breaker.state == BreakerState.OPEN:
            self.breaker.acknowledge(ack_epoch, f"acknowledged:{act.value}", review.psi_aggregate)
        for listener in list(self.on_acknowledge):
            listener(review)
        return review

    def history_frame(self) -> pd.DataFrame:
        """One row per reading: epoch, level, aggregate, breaker states, ``psi_<feature>``."""
        base = [
            "epoch",
            "level",
            "psi_aggregate",
            "breaker_before",
            "breaker_after",
            "tripped",
            "review_id",
            "n_current",
            "skipped",
        ]
        rows = []
        for r in self.history:
            row: dict[str, Any] = {
                "epoch": r.epoch,
                "level": SentinelLevel(r.level).value,
                "psi_aggregate": r.psi_aggregate,
                "breaker_before": BreakerState(r.breaker_before).value,
                "breaker_after": BreakerState(r.breaker_after).value,
                "tripped": r.tripped,
                "review_id": r.review_id,
                "n_current": r.n_current,
                "skipped": r.skipped,
            }
            for f, v in r.per_feature.items():
                row[f"psi_{f}"] = v
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=base)
        return pd.DataFrame(rows)

    # ----- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "calculator": self.calculator.to_dict(),
            "breaker": self.breaker.to_dict(),
            "history": [r.to_dict() for r in self.history],
            "reviews": [r.to_dict() for r in self.reviews],
            "monitor_only": self.monitor_only,
            "min_window_size": self.min_window_size,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PSISentinel:
        cfg = PSIConfig.from_dict(d.get("cfg"))
        calc = PSICalculator.from_dict(d["calculator"])
        breaker = CircuitBreaker.from_dict(d["breaker"], cfg) if d.get("breaker") else None
        s = cls(
            calc,
            cfg,
            breaker=breaker,
            monitor_only=bool(d.get("monitor_only", False)),
            min_window_size=int(d.get("min_window_size", 1)),
        )
        s.history = [SentinelReading.from_dict(r) for r in d.get("history", [])]
        s.reviews = [ReviewRecord.from_dict(r) for r in d.get("reviews", [])]
        return s


# ======================================================================================
# Governed policy
# ======================================================================================
class GovernedPolicy(Policy):
    """A policy under PSI-Sentinel governance (§3.3 champion/circuit breaker; G.5).

    Implements the :class:`Policy` protocol by delegation and listens to the sentinel:

    * breaker trips ``CLOSED → OPEN``: a **champion snapshot** of the policy is taken and
      the policy is frozen (``update`` becomes a counted no-op). A re-trip from
      ``HALF_OPEN`` keeps the champion of the unresolved incident, so ``rollback``
      discards everything learned since the incident began.
    * while OPEN, ``select`` returns ``fallback=True`` decisions from ``fallback``:
      ``"champion_snapshot"`` — greedy argmax of the champion's posterior means for the
      borrower's group (pooled means for groups unknown to the champion; decline if the
      champion has no estimate): no exploration, no learning; ``"decline"`` — the null
      arm; ``"historical"`` — ``historical_arm`` if valid and eligible, else decline.
    * breaker back to ``HALF_OPEN``/``CLOSED``: unfreeze. On acknowledgement
      (:meth:`handle_acknowledge`) ``rollback`` restores the champion snapshot, then
      every hook in ``on_acknowledge`` is called with the :class:`ReviewAction`
      (the simulation runs ``resegment`` there).

    ``freeze()``/``unfreeze()`` on the wrapper are a manual operator freeze, combined with
    the breaker (the inner policy is frozen if either asks for it).
    """

    def __init__(
        self,
        policy: Policy,
        sentinel: PSISentinel,
        arm_space: ArmSpace,
        fallback: str | None = None,
    ):
        super().__init__()
        self.policy = policy
        self.sentinel = sentinel
        self.arm_space = arm_space
        self.fallback = fallback if fallback is not None else sentinel.cfg.fallback
        if self.fallback not in FALLBACK_MODES:
            raise ConfigurationError(f"fallback must be one of {FALLBACK_MODES}, got {self.fallback!r}")
        self.on_acknowledge: list[Callable[[ReviewAction], None]] = []
        self._champion: dict[str, Any] | None = None
        self._champion_epoch: str | None = None
        self._champion_means: dict[str, np.ndarray] | None = None
        self._manual_frozen = False
        self._n_fallback = 0
        self.sentinel.breaker.on_transition.append(self._on_transition)
        self.sentinel.on_acknowledge.append(self._on_review_acknowledged)
        self._sync()

    # ----- identity / governance views -------------------------------------------------------
    @property
    def name(self) -> str:  # type: ignore[override]
        return self.policy.name

    @property
    def breaker_state(self) -> BreakerState:
        return self.sentinel.breaker.state

    @property
    def champion_snapshot(self) -> dict[str, Any] | None:
        return self._champion

    @property
    def champion_epoch(self) -> str | None:
        return self._champion_epoch

    @property
    def n_fallback(self) -> int:
        """Number of decisions made by the fallback."""
        return self._n_fallback

    @property
    def frozen(self) -> bool:
        return self.policy.frozen

    @property
    def dropped_updates(self) -> int:
        return self.policy.dropped_updates

    def freeze(self) -> None:
        self._manual_frozen = True
        self._apply_freeze()

    def unfreeze(self) -> None:
        self._manual_frozen = False
        self._apply_freeze()

    def detach(self) -> None:
        """Stop listening to the sentinel (e.g. before discarding this wrapper)."""
        if self._on_transition in self.sentinel.breaker.on_transition:
            self.sentinel.breaker.on_transition.remove(self._on_transition)
        if self._on_review_acknowledged in self.sentinel.on_acknowledge:
            self.sentinel.on_acknowledge.remove(self._on_review_acknowledged)

    # ----- breaker plumbing --------------------------------------------------------------------
    def _apply_freeze(self) -> None:
        want = self._manual_frozen or self.breaker_state == BreakerState.OPEN
        if want and not self.policy.frozen:
            self.policy.freeze()
        elif not want and self.policy.frozen:
            self.policy.unfreeze()

    def _take_champion(self, epoch: str | None) -> None:
        self._champion = self.policy.snapshot()
        self._champion_epoch = epoch
        self._champion_means = None

    def _sync(self) -> None:
        if self.breaker_state == BreakerState.OPEN and self._champion is None:
            last = self.sentinel.breaker.transitions
            self._take_champion(last[-1].epoch if last else None)
        self._apply_freeze()

    def _on_transition(self, tr: BreakerTransition) -> None:
        if tr.to_state == BreakerState.OPEN and (
            tr.from_state == BreakerState.CLOSED or self._champion is None
        ):
            self._take_champion(tr.epoch)
        self._apply_freeze()

    def _on_review_acknowledged(self, review: ReviewRecord) -> None:
        if review.action is not None:
            self.handle_acknowledge(review.action)

    def handle_acknowledge(self, action: ReviewAction | str) -> None:
        """Apply an acknowledged review action, then fire the ``on_acknowledge`` hooks."""
        act = ReviewAction(action)
        if act == ReviewAction.ROLLBACK and self._champion is not None:
            self.policy.restore(self._champion)
        self._apply_freeze()
        for hook in list(self.on_acknowledge):
            hook(act)

    def _champion_lookup(self, group: str) -> np.ndarray | None:
        if self._champion is None:
            return None
        if self._champion_means is None:
            try:
                self._champion_means = self.policy.means_from_snapshot(self._champion)
            except NotImplementedError:
                self._champion_means = {}
        means = self._champion_means.get(group)
        if means is None:
            means = self._champion_means.get(POOLED_GROUP)
        return means

    def _fallback_decision(
        self,
        context: BorrowerContext,
        eligible: np.ndarray,
        historical_arm: int | None,
    ) -> Decision:
        mask = normalise_eligible(eligible, self.arm_space)
        group = self.policy.group_for(context)
        scores = None
        meta: dict[str, Any] = {
            "fallback_mode": self.fallback,
            "breaker_state": BreakerState.OPEN.value,
            "champion_epoch": self._champion_epoch,
        }
        if self.fallback == "champion_snapshot":
            means = self._champion_lookup(group)
            if means is None or means.size != self.arm_space.n_arms:
                arm = decline_arm(self.arm_space, mask)
                meta["champion_missing"] = True
            else:
                arm = argmax_eligible(means, np.flatnonzero(mask))
                scores = masked_scores(means, mask)
        elif self.fallback == "decline":
            arm = decline_arm(self.arm_space, mask)
        else:
            h = valid_arm(historical_arm, self.arm_space.n_arms)
            arm = h if h is not None and mask[h] else decline_arm(self.arm_space, mask)
            meta["historical_arm"] = h
        if getattr(self.policy, "contextual", False):
            meta["seg_version"] = int(getattr(self.policy, "segmentation_version", 0))
        self._n_fallback += 1
        return Decision(
            arm=arm,
            group=group,
            policy_name=self.policy.name,
            scores=scores,
            fallback=True,
            meta=meta,
        )

    # ----- Policy protocol ---------------------------------------------------------------------
    def select(
        self,
        context: BorrowerContext,
        eligible: np.ndarray,
        rng: np.random.Generator,
        *,
        historical_arm: int | None = None,
    ) -> Decision:
        self._sync()
        if self.breaker_state == BreakerState.OPEN:
            return self._fallback_decision(context, eligible, historical_arm)
        decision = self.policy.select(context, eligible, rng, historical_arm=historical_arm)
        decision.meta["breaker_state"] = self.breaker_state.value
        return decision

    def update(
        self,
        decision: Decision,
        reward: float,
        kind: FeedbackKind | str = FeedbackKind.TERMINAL,
        weight: float | None = None,
    ) -> None:
        self._sync()
        self.policy.update(decision, reward, kind, weight)

    def snapshot(self) -> dict[str, Any]:
        return self.policy.snapshot()

    def restore(self, snap: Mapping[str, Any]) -> None:
        self.policy.restore(snap)

    def state(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "fallback": self.fallback,
            "policy": self.policy.state(),
            "champion": self._champion,
            "champion_epoch": self._champion_epoch,
            "manual_frozen": self._manual_frozen,
            "n_fallback": self._n_fallback,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        arm_space: ArmSpace,
        segmenter: Segmenter | None = None,
        prior_provider: ArmPriorProvider | None = None,
        *,
        sentinel: PSISentinel | None = None,
    ) -> GovernedPolicy:
        """Rebuild from :meth:`state`; ``sentinel`` (restored separately) is required."""
        if sentinel is None:
            raise ConfigurationError("GovernedPolicy.from_state needs the (restored) sentinel")
        from ..bandits import policy_from_state  # bandits/__init__ imports every policy class

        inner = policy_from_state(state["policy"], arm_space, segmenter, prior_provider)
        obj = cls(inner, sentinel, arm_space, fallback=state.get("fallback"))
        obj._champion = state.get("champion")
        obj._champion_epoch = state.get("champion_epoch")
        obj._champion_means = None
        obj._manual_frozen = bool(state.get("manual_frozen", False))
        obj._n_fallback = int(state.get("n_fallback", 0))
        obj._sync()
        return obj

    def arm_table(self, group: str | None = None) -> pd.DataFrame:
        return self.policy.arm_table(group)

    def group_keys(self) -> list[str]:
        return self.policy.group_keys()

    def group_for(self, context: BorrowerContext | Mapping[str, Any]) -> str:
        return self.policy.group_for(context)

    def prime_groups(self, contexts: pd.DataFrame) -> pd.Series | None:
        """Delegates :meth:`ArmSetPolicy.prime_groups` when the wrapped policy has it."""
        prime = getattr(self.policy, "prime_groups", None)
        return prime(contexts) if prime is not None else None

    def means_from_snapshot(self, snap: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self.policy.means_from_snapshot(snap)


__all__ = [
    "FALLBACK_MODES",
    "BreakerTransition",
    "CircuitBreaker",
    "GovernanceError",
    "GovernedPolicy",
    "PSISentinel",
    "ReviewRecord",
    "SentinelReading",
]
