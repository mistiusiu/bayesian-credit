"""Event-sourced, resumable replay simulation (SPEC §2.11; proposal §3.6, G.3, G.6).

* :mod:`.environment` — :class:`BayesianEnvironment` (posterior-predictive rewards, online
  risk updates from realized outcomes; also the hindsight oracle).
* :mod:`.feedback` — :class:`DelayedFeedbackQueue` (interim / terminal feedback by release epoch).
* :mod:`.events` / :mod:`.store` — event dataclasses + JSONL codec, the run-directory file
  protocol, the event store and atomic checkpoints.
* :mod:`.policies` — policy-spec parsing and warm-start prior providers.
* :mod:`.runner` — :class:`SimulationRunner` (run / resume / pause / cancel / review).
* :mod:`.cli` — ``python -m bayesian_credit.simulation.cli {fit,run,resume,shadow,report}``.
"""

from __future__ import annotations

from .environment import BayesianEnvironment
from .events import (
    EVENT_TYPES,
    EventSink,
    JSONLSink,
    KafkaSink,
    SimulationEvent,
    decode_event,
    encode_event,
)
from .feedback import DelayedFeedbackQueue, FeedbackBatch, FeedbackItem
from .policies import (
    PolicySpec,
    build_policies,
    parse_policy_spec,
    parse_policy_specs,
    risk_model_prior_provider,
)
from .runner import (
    ARM_SNAPSHOT_COLUMNS,
    DECISION_COLUMNS,
    SENTINEL_COLUMNS,
    TIMESERIES_COLUMNS,
    SimulationResult,
    SimulationRunner,
)
from .store import Checkpoint, EventStore, RunDirectory, read_decisions

__all__ = [
    "ARM_SNAPSHOT_COLUMNS",
    "DECISION_COLUMNS",
    "EVENT_TYPES",
    "SENTINEL_COLUMNS",
    "TIMESERIES_COLUMNS",
    "BayesianEnvironment",
    "Checkpoint",
    "DelayedFeedbackQueue",
    "EventSink",
    "EventStore",
    "FeedbackBatch",
    "FeedbackItem",
    "JSONLSink",
    "KafkaSink",
    "PolicySpec",
    "RunDirectory",
    "SimulationEvent",
    "SimulationResult",
    "SimulationRunner",
    "build_policies",
    "decode_event",
    "encode_event",
    "parse_policy_spec",
    "parse_policy_specs",
    "read_decisions",
    "risk_model_prior_provider",
]
