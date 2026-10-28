"""Simulation events and their JSON codec (SPEC §2.11.5-6; proposal §3.6, G.6).

The replay simulation is event-sourced at a *coarse* granularity: one JSON object per
line in ``<run_dir>/events.jsonl``::

    {"ts": "2026-10-16T12:00:00+00:00", "type": "EpochCompleted", "epoch": "2015-03",
     "payload": {...}}

Per-borrower events (BorrowerArrived / PolicySelected) are deliberately *not* emitted:
the per-decision record lives in ``decisions/epoch=YYYY-MM.parquet`` (columnar, cheap),
so the event log stays small enough for the API to tail. Every event class below fixes
its ``type`` and documents the payload keys the runner writes.

:class:`EventSink` is the output interface; :class:`JSONLSink` (append-only file) is the
default and :class:`KafkaSink` an optional adapter (``confluent-kafka`` import guarded).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "EVENT_TYPES",
    "BreakerTripped",
    "CheckpointWritten",
    "EpochCompleted",
    "EpochStarted",
    "EventSink",
    "FeedbackReleased",
    "JSONLSink",
    "KafkaSink",
    "ResegmentationApplied",
    "ReviewAcknowledged",
    "RunCancelled",
    "RunCompleted",
    "RunFailed",
    "RunPaused",
    "RunResumed",
    "RunStarted",
    "SentinelEvaluated",
    "SimulationEvent",
    "decode_event",
    "encode_event",
    "jsonable",
    "utc_now",
]


def utc_now() -> str:
    """ISO-8601 UTC timestamp with second precision (used for every ``ts`` field)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars, arrays, enums and paths to plain JSON
    values. Non-finite floats become ``None`` so the output is strict JSON."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):  # pragma: no cover - NaT and friends
            return None
    try:
        if value != value:  # NaN-like (pd.NA raises TypeError on bool())
            return None
    except (TypeError, ValueError):
        return None
    return str(value)


# ======================================================================================
# Event classes
# ======================================================================================
@dataclass
class SimulationEvent:
    """Base event: ``type`` (class constant), ``epoch`` ('YYYY-MM' or None), ``payload``
    (JSON-able dict) and ``ts`` (UTC ISO timestamp, set at construction)."""

    type: ClassVar[str] = "SimulationEvent"

    epoch: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "type": self.type, "epoch": self.epoch, "payload": jsonable(self.payload)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SimulationEvent:
        """Decode into the registered subclass for ``d["type"]`` (unknown types keep the
        base class, with the original type preserved in ``payload["_type"]``)."""
        etype = str(d.get("type", "SimulationEvent"))
        klass = _REGISTRY.get(etype)
        payload = dict(d.get("payload") or {})
        if klass is None:
            klass = SimulationEvent
            payload.setdefault("_type", etype)
        epoch = d.get("epoch")
        return klass(epoch=None if epoch is None else str(epoch), payload=payload, ts=str(d.get("ts") or utc_now()))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.type] = cls


_REGISTRY: dict[str, type[SimulationEvent]] = {}


@dataclass
class RunStarted(SimulationEvent):
    """Fresh run. Payload: ``n_epochs, first_epoch, last_epoch, n_loans, policies,
    feedback_mode, breaker_mode, pid``."""

    type: ClassVar[str] = "RunStarted"


@dataclass
class RunResumed(SimulationEvent):
    """Restart from a checkpoint. Payload: ``from_epoch_index, from_epoch, checkpoint,
    n_epochs, pid`` (``from_epoch`` is the last epoch contained in the checkpoint)."""

    type: ClassVar[str] = "RunResumed"


@dataclass
class EpochStarted(SimulationEvent):
    """Payload: ``epoch_index, n_arrivals``."""

    type: ClassVar[str] = "EpochStarted"


@dataclass
class SentinelEvaluated(SimulationEvent):
    """PSI reading of the epoch's window. Payload: ``SentinelReading.to_dict()``
    (``level, psi_aggregate, per_feature, breaker_before, breaker_after, tripped,
    review_id, n_current, skipped``)."""

    type: ClassVar[str] = "SentinelEvaluated"


@dataclass
class BreakerTripped(SimulationEvent):
    """Breaker opened. Payload: ``review_id, psi_aggregate, per_feature, breaker_mode,
    auto_ack_epoch`` (the epoch at which ``auto_ack`` will acknowledge, else None)."""

    type: ClassVar[str] = "BreakerTripped"


@dataclass
class ReviewAcknowledged(SimulationEvent):
    """Human (or ``auto_ack``) acknowledgement. Payload: ``review_id, action, note, auto,
    rebased, breaker_state``."""

    type: ClassVar[str] = "ReviewAcknowledged"


@dataclass
class ResegmentationApplied(SimulationEvent):
    """Breaker action ``resegment``. Payload: ``review_id, dimensions_before,
    dimensions_after, n_refit_loans, n_groups``."""

    type: ClassVar[str] = "ResegmentationApplied"


@dataclass
class FeedbackReleased(SimulationEvent):
    """Per-epoch feedback summary. Payload: ``n_interim, n_terminal, n_skipped_replay,
    n_dropped_frozen, n_observed_outcomes, pending, by_policy``."""

    type: ClassVar[str] = "FeedbackReleased"


@dataclass
class CheckpointWritten(SimulationEvent):
    """Payload: ``epoch_index, path, bytes, seconds``."""

    type: ClassVar[str] = "CheckpointWritten"


@dataclass
class EpochCompleted(SimulationEvent):
    """Payload: ``epoch_index, n_arrivals, n_decisions, breaker_state, psi_aggregate,
    pending_feedback, per_policy`` (per policy: ``n, reward, regret, decline_rate,
    fallback_rate``)."""

    type: ClassVar[str] = "EpochCompleted"


@dataclass
class RunPaused(SimulationEvent):
    """Payload: ``reason`` ("control" or "awaiting_review"), ``review_id``."""

    type: ClassVar[str] = "RunPaused"


@dataclass
class RunCancelled(SimulationEvent):
    """Payload: ``reason`` ("control" | "signal"), ``checkpoint`` (path or None)."""

    type: ClassVar[str] = "RunCancelled"


@dataclass
class RunCompleted(SimulationEvent):
    """Payload: ``n_epochs, n_decisions, wall_seconds, report``."""

    type: ClassVar[str] = "RunCompleted"


@dataclass
class RunFailed(SimulationEvent):
    """Payload: ``error, error_type, epoch_index``."""

    type: ClassVar[str] = "RunFailed"


#: Every event type the runner can emit, in rough life-cycle order.
EVENT_TYPES: tuple[str, ...] = (
    "RunStarted",
    "RunResumed",
    "EpochStarted",
    "SentinelEvaluated",
    "BreakerTripped",
    "ReviewAcknowledged",
    "ResegmentationApplied",
    "FeedbackReleased",
    "CheckpointWritten",
    "EpochCompleted",
    "RunPaused",
    "RunCancelled",
    "RunCompleted",
    "RunFailed",
)


# ======================================================================================
# Codec
# ======================================================================================
def encode_event(event: SimulationEvent | Mapping[str, Any]) -> str:
    """One strict-JSON line (no trailing newline) for an event or an event dict."""
    d = event.to_dict() if isinstance(event, SimulationEvent) else jsonable(dict(event))
    return json.dumps(d, allow_nan=False, separators=(",", ":"), ensure_ascii=False)


def decode_event(line: str | bytes | Mapping[str, Any]) -> SimulationEvent:
    """Inverse of :func:`encode_event` (accepts a JSON line or an already parsed dict)."""
    if isinstance(line, Mapping):
        return SimulationEvent.from_dict(line)
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    return SimulationEvent.from_dict(json.loads(text))


# ======================================================================================
# Sinks
# ======================================================================================
@runtime_checkable
class EventSink(Protocol):
    """Where the runner sends events (SPEC §2.11.6)."""

    def emit(self, event: SimulationEvent) -> None: ...

    def close(self) -> None: ...


class JSONLSink:
    """Append-only JSON-lines file sink (the default event store backend).

    Each event is written as one line and flushed immediately, so a SIGKILL never leaves
    a partially written *earlier* event; at worst the very last line is truncated, which
    :meth:`bayesian_credit.simulation.store.EventStore.read` skips.
    """

    def __init__(self, path: str | os.PathLike[str], fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = bool(fsync)

    def emit(self, event: SimulationEvent | Mapping[str, Any]) -> None:
        line = encode_event(event) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())

    def close(self) -> None:  # nothing kept open between events
        return None


class KafkaSink:
    """Optional Kafka adapter: publishes every event (JSON) to ``topic``.

    Requires ``confluent-kafka`` (``pip install bayesian-credit[kafka]``); the import is
    guarded so the core package never depends on it. ``config`` is passed to
    ``confluent_kafka.Producer`` (e.g. ``{"bootstrap.servers": "localhost:9092"}``).
    """

    def __init__(self, topic: str, config: Mapping[str, Any] | None = None, key: str | None = None):
        try:
            from confluent_kafka import Producer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("KafkaSink needs confluent-kafka: pip install 'bayesian-credit[kafka]'") from exc
        self.topic = topic
        self.key = key
        self._producer = Producer(dict(config or {"bootstrap.servers": "localhost:9092"}))

    def emit(self, event: SimulationEvent | Mapping[str, Any]) -> None:  # pragma: no cover - needs a broker
        self._producer.produce(self.topic, value=encode_event(event).encode("utf-8"), key=self.key)
        self._producer.poll(0)

    def close(self) -> None:  # pragma: no cover - needs a broker
        self._producer.flush()
