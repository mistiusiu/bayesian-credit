"""Delayed feedback queue of the replay simulation (SPEC §2.11; proposal G.3 "real delayed
feedback").

A lending decision is not rewarded when it is made: the borrower's repayment behaviour is
observed month by month (*interim* signals, §1.7.2) and the loan's terminal outcome only
at resolution (*terminal* feedback, possibly years later; never for loans still open at
the end of the stream - real censoring). The queue holds every pending observation keyed
by its **release epoch** ``'YYYY-MM'`` and hands back everything due at an epoch.

Storage. The simulation schedules millions of interim observations (decisions x
``interim_months``), so items are stored **column-wise**: per release epoch a list of
chunks, each a dict of numpy arrays (:data:`QUEUE_COLUMNS`). A min-heap over the
release-epoch keys gives O(log E) access to the earliest bucket; ``'YYYY-MM'`` strings
sort chronologically. Within a bucket, items keep their insertion order, so
:meth:`DelayedFeedbackQueue.pop_due` is deterministic - a prerequisite of bit-for-bit
resume. :meth:`schedule` / :meth:`pop_due` speak :class:`FeedbackItem`; the runner uses
the columnar twins :meth:`schedule_batch` / :meth:`pop_due_batch`.

The decision attached to an item is stored in its *replayable* form only (``arm``,
``group``, ``policy_name``, ``fallback`` and ``meta["seg_version"]``): exactly what
``Policy.update`` reads, without the per-arm score vector.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import FeedbackKind

__all__ = [
    "KIND_CODES",
    "QUEUE_COLUMNS",
    "DelayedFeedbackQueue",
    "FeedbackBatch",
    "FeedbackItem",
]

#: FeedbackKind <-> int8 code used in the columnar store.
KIND_CODES: dict[FeedbackKind, int] = {FeedbackKind.INTERIM: 0, FeedbackKind.TERMINAL: 1}
_KIND_FROM_CODE = {v: k for k, v in KIND_CODES.items()}

#: Column -> numpy dtype of the columnar store. ``row`` is an optional index into the
#: caller's loan table (the runner's stream), -1 when unknown; ``seg_version`` -1 means
#: "decision carried no segmentation version"; ``state`` -1 for terminal items.
QUEUE_COLUMNS: dict[str, Any] = {
    "kind": np.int8,
    "policy": object,
    "loan_id": np.int64,
    "row": np.int64,
    "arm": np.int32,
    "group": object,
    "seg_version": np.int32,
    "fallback": np.bool_,
    "month": np.int16,
    "months_paid": np.int16,
    "state": np.int8,
    "match": np.bool_,
    "decision_epoch": object,
}
_DEFAULTS: dict[str, Any] = {
    "row": -1,
    "seg_version": -1,
    "fallback": False,
    "month": 0,
    "months_paid": 0,
    "state": -1,
    "match": False,
    "decision_epoch": None,
}


def _coerce_kind(kind: FeedbackKind | str | int) -> FeedbackKind:
    if isinstance(kind, (int, np.integer)) and not isinstance(kind, bool):
        return _KIND_FROM_CODE[int(kind)]
    return FeedbackKind(kind)


def decision_payload(
    arm: int, group: str, policy_name: str, fallback: bool = False, seg_version: int | None = None
) -> dict[str, Any]:
    """The replayable decision dict stored with a feedback item."""
    meta: dict[str, Any] = {} if seg_version is None or int(seg_version) < 0 else {"seg_version": int(seg_version)}
    return {"arm": int(arm), "group": str(group), "policy_name": str(policy_name), "fallback": bool(fallback), "meta": meta}


# ======================================================================================
# Item
# ======================================================================================
@dataclass(slots=True)
class FeedbackItem:
    """One pending reward observation for one (policy, loan) decision.

    ``kind`` interim → ``month`` (loan age, 1-based), ``months_paid`` and ``state``
    (:class:`~bayesian_credit.types.State` value) describe the borrower's position;
    terminal → the loan's resolution. ``is_historical_match`` is true when the chosen arm
    equals the loan's historical arm (``y_historical_arm``) - only then is the realized
    outcome a valid reward for the chosen arm (replay / hybrid feedback modes).
    """

    release_epoch: str
    kind: FeedbackKind
    policy_name: str
    loan_id: int
    decision: dict[str, Any]
    group: str
    arm: int
    month: int = 0
    months_paid: int = 0
    state: int | None = None
    is_historical_match: bool = False
    row: int = -1
    decision_epoch: str | None = None

    def __post_init__(self) -> None:
        self.release_epoch = str(self.release_epoch)
        self.kind = _coerce_kind(self.kind)
        self.loan_id = int(self.loan_id)
        self.arm = int(self.arm)
        self.group = str(self.group)
        self.policy_name = str(self.policy_name)
        if self.state is not None and int(self.state) < 0:
            self.state = None
        if self.decision is None:
            self.decision = decision_payload(self.arm, self.group, self.policy_name)

    @property
    def seg_version(self) -> int:
        meta = (self.decision or {}).get("meta") or {}
        v = meta.get("seg_version")
        return -1 if v is None else int(v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_epoch": self.release_epoch,
            "kind": self.kind.value,
            "policy_name": self.policy_name,
            "loan_id": self.loan_id,
            "decision": dict(self.decision),
            "group": self.group,
            "arm": self.arm,
            "month": int(self.month),
            "months_paid": int(self.months_paid),
            "state": None if self.state is None else int(self.state),
            "is_historical_match": bool(self.is_historical_match),
            "row": int(self.row),
            "decision_epoch": self.decision_epoch,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FeedbackItem:
        return cls(
            release_epoch=str(d["release_epoch"]),
            kind=_coerce_kind(d["kind"]),
            policy_name=str(d["policy_name"]),
            loan_id=int(d["loan_id"]),
            decision=dict(d.get("decision") or {}),
            group=str(d["group"]),
            arm=int(d["arm"]),
            month=int(d.get("month", 0)),
            months_paid=int(d.get("months_paid", 0)),
            state=d.get("state"),
            is_historical_match=bool(d.get("is_historical_match", False)),
            row=int(d.get("row", -1)),
            decision_epoch=d.get("decision_epoch"),
        )

    def _row(self) -> dict[str, Any]:
        dec = self.decision or {}
        return {
            "kind": KIND_CODES[self.kind],
            "policy": self.policy_name,
            "loan_id": self.loan_id,
            "row": self.row,
            "arm": self.arm,
            "group": self.group,
            "seg_version": self.seg_version,
            "fallback": bool(dec.get("fallback", False)),
            "month": int(self.month),
            "months_paid": int(self.months_paid),
            "state": -1 if self.state is None else int(self.state),
            "match": bool(self.is_historical_match),
            "decision_epoch": self.decision_epoch,
        }


# ======================================================================================
# Columnar batch
# ======================================================================================
@dataclass
class FeedbackBatch:
    """Column-wise view of many feedback items (what :meth:`pop_due_batch` returns)."""

    columns: dict[str, np.ndarray] = field(default_factory=dict)
    release_epoch: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    def __len__(self) -> int:
        return int(self.release_epoch.size)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def kind(self) -> np.ndarray:
        return self.columns["kind"]

    def item(self, i: int) -> FeedbackItem:
        c = self.columns
        kind = _KIND_FROM_CODE[int(c["kind"][i])]
        state = int(c["state"][i])
        return FeedbackItem(
            release_epoch=str(self.release_epoch[i]),
            kind=kind,
            policy_name=str(c["policy"][i]),
            loan_id=int(c["loan_id"][i]),
            decision=decision_payload(
                int(c["arm"][i]), str(c["group"][i]), str(c["policy"][i]), bool(c["fallback"][i]), int(c["seg_version"][i])
            ),
            group=str(c["group"][i]),
            arm=int(c["arm"][i]),
            month=int(c["month"][i]),
            months_paid=int(c["months_paid"][i]),
            state=None if state < 0 else state,
            is_historical_match=bool(c["match"][i]),
            row=int(c["row"][i]),
            decision_epoch=c["decision_epoch"][i],
        )

    def items(self) -> list[FeedbackItem]:
        return [self.item(i) for i in range(len(self))]

    def __iter__(self) -> Iterator[FeedbackItem]:
        return (self.item(i) for i in range(len(self)))


def _empty_columns() -> dict[str, np.ndarray]:
    return {c: np.empty(0, dtype=dt) for c, dt in QUEUE_COLUMNS.items()}


def _as_column(name: str, value: Any, n: int) -> np.ndarray:
    dtype = QUEUE_COLUMNS[name]
    if value is None:
        value = _DEFAULTS.get(name)
    arr = np.asarray(value, dtype=dtype) if dtype is not object else np.asarray(value, dtype=object)
    if arr.ndim == 0:
        out = np.empty(n, dtype=dtype)
        out[...] = arr.item() if dtype is not object else value
        return out
    if arr.shape != (n,):
        raise ValueError(f"column {name!r} has shape {arr.shape}, expected ({n},)")
    return arr


def _concat(chunks: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        return _empty_columns()
    if len(chunks) == 1:
        return dict(chunks[0])
    return {c: np.concatenate([ch[c] for ch in chunks]) for c in QUEUE_COLUMNS}


def _chunk_len(chunk: dict[str, np.ndarray]) -> int:
    return int(chunk["kind"].size)


# ======================================================================================
# Queue
# ======================================================================================
class DelayedFeedbackQueue:
    """Min-heap of release epochs -> FIFO buckets of pending feedback (columnar storage).

    ``schedule(item)`` / ``schedule_batch(release_epochs, **columns)`` add items;
    ``pop_due(epoch)`` / ``pop_due_batch(epoch)`` remove and return everything with
    ``release_epoch <= epoch`` in (release epoch, insertion) order. ``len(queue)``,
    :meth:`pending_by_kind` and :meth:`pending_by_policy` describe what is pending;
    :meth:`to_dict` / :meth:`from_dict` round-trip the queue for checkpoints (numpy arrays
    inside, so the checkpoint is a plain pickle-able dict).
    """

    def __init__(self) -> None:
        self._buckets: dict[str, list[tuple[dict[str, np.ndarray], dict[str, int]]]] = {}
        self._heap: list[str] = []
        self._n = 0
        self._by_kind: dict[str, int] = {k.value: 0 for k in FeedbackKind}
        self._by_policy: dict[str, int] = {}

    # ----- bookkeeping ---------------------------------------------------------------------
    @staticmethod
    def _policy_counts(chunk: dict[str, np.ndarray]) -> dict[str, int]:
        pol = chunk["policy"]
        if pol.size and bool(np.all(pol == pol[0])):  # the runner schedules one policy per chunk
            return {str(pol[0]): int(pol.size)}
        names, counts = np.unique(pol.astype(str), return_counts=True)
        return {str(n): int(c) for n, c in zip(names.tolist(), counts.tolist(), strict=True)}

    def _add_chunk(self, epoch: str, chunk: dict[str, np.ndarray]) -> None:
        n = _chunk_len(chunk)
        if n == 0:
            return
        bucket = self._buckets.get(epoch)
        if bucket is None:
            self._buckets[epoch] = bucket = []
            heapq.heappush(self._heap, epoch)
        counts = self._policy_counts(chunk)
        bucket.append((chunk, counts))
        self._count(chunk, counts, +1)

    def _count(self, chunk: dict[str, np.ndarray], policy_counts: Mapping[str, int], sign: int) -> None:
        n = _chunk_len(chunk)
        self._n += sign * n
        n_terminal = int(np.count_nonzero(chunk["kind"] == KIND_CODES[FeedbackKind.TERMINAL]))
        self._by_kind[FeedbackKind.TERMINAL.value] += sign * n_terminal
        self._by_kind[FeedbackKind.INTERIM.value] += sign * (n - n_terminal)
        for name, cnt in policy_counts.items():
            v = self._by_policy.get(name, 0) + sign * int(cnt)
            if v:
                self._by_policy[name] = v
            else:
                self._by_policy.pop(name, None)

    # ----- scheduling ------------------------------------------------------------------------
    def schedule(self, item: FeedbackItem) -> None:
        """Add one item (convenience API; the runner uses :meth:`schedule_batch`)."""
        row = item._row()
        chunk = {c: np.array([row[c]], dtype=QUEUE_COLUMNS[c]) for c in QUEUE_COLUMNS}
        self._add_chunk(item.release_epoch, chunk)

    def schedule_many(self, items: Iterable[FeedbackItem]) -> None:
        for it in items:
            self.schedule(it)

    def schedule_batch(self, release_epochs: Sequence[str] | np.ndarray | str, **columns: Any) -> int:
        """Add many items at once. ``release_epochs`` is one epoch for all rows or one per
        row; ``columns`` are :data:`QUEUE_COLUMNS` arrays (``kind``, ``policy``,
        ``loan_id``, ``arm``, ``group`` required; the rest default). Items sharing a release
        epoch keep their relative order. Returns the number of rows scheduled."""
        unknown = set(columns) - set(QUEUE_COLUMNS)
        if unknown:
            raise ValueError(f"unknown feedback columns {sorted(unknown)}")
        for req in ("kind", "policy", "loan_id", "arm", "group"):
            if req not in columns:
                raise ValueError(f"schedule_batch needs column {req!r}")
        kind = np.asarray(columns["kind"])
        if kind.ndim == 0:
            raise ValueError("'kind' must be an array (one code per row)")
        n = int(kind.size)
        if n == 0:
            return 0
        if kind.dtype.kind not in "iub":
            kind = np.array([KIND_CODES[_coerce_kind(k)] for k in kind.tolist()], dtype=np.int8)
        cols = {c: _as_column(c, columns.get(c), n) for c in QUEUE_COLUMNS if c != "kind"}
        cols["kind"] = kind.astype(np.int8)
        if isinstance(release_epochs, str):
            self._add_chunk(release_epochs, cols)
            return n
        rel = np.asarray(release_epochs, dtype=object)
        if rel.shape != (n,):
            raise ValueError(f"release_epochs has shape {rel.shape}, expected ({n},)")
        keys, inverse = np.unique(rel.astype(str), return_inverse=True)
        if keys.size == 1:
            self._add_chunk(str(keys[0]), cols)
            return n
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(keys.size + 1))
        for j, key in enumerate(keys.tolist()):
            sel = order[bounds[j] : bounds[j + 1]]
            self._add_chunk(str(key), {c: v[sel] for c, v in cols.items()})
        return n

    # ----- release ------------------------------------------------------------------------------
    def pop_due_batch(self, epoch: str) -> FeedbackBatch:
        """Remove and return (columnar) every item with ``release_epoch <= epoch``."""
        epoch = str(epoch)
        chunks: list[dict[str, np.ndarray]] = []
        releases: list[np.ndarray] = []
        while self._heap and self._heap[0] <= epoch:
            key = heapq.heappop(self._heap)
            for chunk, counts in self._buckets.pop(key, []):
                self._count(chunk, counts, -1)
                chunks.append(chunk)
                releases.append(np.full(_chunk_len(chunk), key, dtype=object))
        if not chunks:
            return FeedbackBatch(_empty_columns(), np.empty(0, dtype=object))
        return FeedbackBatch(_concat(chunks), np.concatenate(releases))

    def pop_due(self, epoch: str) -> list[FeedbackItem]:
        """Remove and return every item with ``release_epoch <= epoch`` (FIFO per epoch)."""
        return self.pop_due_batch(epoch).items()

    # ----- inspection -----------------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0

    def pending_by_kind(self) -> dict[str, int]:
        return dict(self._by_kind)

    def pending_by_policy(self) -> dict[str, int]:
        return dict(sorted(self._by_policy.items()))

    def release_epochs(self) -> list[str]:
        """Release epochs with pending items, earliest first."""
        return sorted(self._buckets)

    def next_release(self) -> str | None:
        return self._heap[0] if self._heap else None

    def peek(self, epoch: str | None = None) -> list[FeedbackItem]:
        """Items of one bucket (default: the earliest) without removing them."""
        key = epoch if epoch is not None else self.next_release()
        if key is None or key not in self._buckets:
            return []
        cols = _concat([ch for ch, _ in self._buckets[key]])
        return FeedbackBatch(cols, np.full(_chunk_len(cols), key, dtype=object)).items()

    def clear(self) -> None:
        self.__init__()

    # ----- serialisation ---------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain dict for checkpoints: ``{"version", "buckets": {epoch: {column: ndarray}}}``.

        Chunks of a bucket are concatenated (order preserved). Arrays are copies, so the
        snapshot is independent of later queue mutations.
        """
        return {
            "version": 1,
            "n": self._n,
            "buckets": {
                k: {c: v.copy() for c, v in _concat([ch for ch, _ in self._buckets[k]]).items()}
                for k in sorted(self._buckets)
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DelayedFeedbackQueue:
        q = cls()
        for key in sorted(d.get("buckets", {})):
            raw = d["buckets"][key]
            n = int(np.asarray(raw["kind"]).size)
            chunk = {c: _as_column(c, raw.get(c), n) for c in QUEUE_COLUMNS if c != "kind"}
            chunk["kind"] = np.asarray(raw["kind"], dtype=np.int8)
            q._add_chunk(str(key), chunk)
        return q

    def to_records(self) -> list[dict[str, Any]]:
        """Every pending item as a JSON-able dict (debugging / small queues)."""
        out = []
        for key in sorted(self._buckets):
            cols = _concat([ch for ch, _ in self._buckets[key]])
            out += [it.to_dict() for it in FeedbackBatch(cols, np.full(_chunk_len(cols), key, dtype=object))]
        return out

    def __repr__(self) -> str:
        return f"DelayedFeedbackQueue(n={self._n}, epochs={len(self._buckets)}, by_kind={self._by_kind})"

