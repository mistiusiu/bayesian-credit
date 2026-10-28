"""Run-directory file protocol, append-only event store and atomic checkpoints (SPEC §2.11.5,
proposal G.6 "state recovery").

The Django API and the simulation worker communicate **only through files** in a run
directory, so the worker can be a subprocess or a Kubernetes Job and can be SIGKILLed at
any moment (the "chaos" button). Layout (see ``docs/SIMULATION.md``)::

    <run_dir>/config.json                     framework + simulation config, dataset, fit_dir
    <run_dir>/progress.json                   status / epoch / pct / breaker state / pid ...
    <run_dir>/control.json                    API -> worker: {"command": "pause"|"resume"|"cancel", "at"}
    <run_dir>/reviews/<review_id>.json        worker -> API: breaker review record
    <run_dir>/reviews/<review_id>.ack.json    API -> worker: {"action", "note", "at"}
    <run_dir>/events.jsonl                    coarse event log (one JSON object per line)
    <run_dir>/decisions/epoch=YYYY-MM.parquet per-epoch decision logs
    <run_dir>/feedback/epoch=YYYY-MM.parquet  terminal feedback released per epoch (fills simulated_reward)
    <run_dir>/arms/epoch=YYYY-MM.parquet      bandit posterior snapshots
    <run_dir>/timeseries.parquet              per-epoch, per-policy metrics (rewritten every epoch)
    <run_dir>/sentinel.parquet                PSI per feature per epoch (rewritten every epoch)
    <run_dir>/checkpoints/latest.pkl          + epoch-stamped copies (last 3 kept)
    <run_dir>/runtime.json                    wall/CPU seconds, peak RSS, sessions
    <run_dir>/report.json                     EvaluationReport at the end

Every file the API may read concurrently is written to a temporary file in the same
directory, ``fsync``-ed and moved into place with :func:`os.replace` (atomic on POSIX), so
readers never observe a partial file and a SIGKILL never corrupts a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import CheckpointError
from .events import JSONLSink, SimulationEvent, jsonable, utc_now

__all__ = [
    "RUN_STATUSES",
    "Checkpoint",
    "EventStore",
    "RunDirectory",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_parquet",
    "config_hash",
    "epoch_file_name",
    "read_decisions",
    "read_json",
]

#: Values of ``progress.json["status"]``.
RUN_STATUSES = ("queued", "running", "paused", "awaiting_review", "completed", "failed", "cancelled")
#: Keys always present in ``progress.json``.
PROGRESS_KEYS = (
    "status",
    "epoch_index",
    "n_epochs",
    "current_epoch",
    "pct",
    "message",
    "updated_at",
    "breaker_state",
    "pid",
    "started_at",
    "finished_at",
    "error",
)
CONTROL_COMMANDS = ("pause", "resume", "cancel")
REVIEW_ACTIONS = ("resume", "resegment", "rollback")

_EPOCH_FILE = re.compile(r"^epoch=(\d{4}-\d{2})\.parquet$")
_CKPT_FILE = re.compile(r"^epoch=(\d{4}-\d{2})\.pkl$")


# ======================================================================================
# atomic file helpers
# ======================================================================================
def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without directory fds
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes, fsync: bool = True) -> Path:
    """Write ``data`` to ``path`` atomically (temp file + fsync + ``os.replace``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    if fsync:
        _fsync_dir(p.parent)
    return p


def atomic_write_json(path: str | os.PathLike[str], obj: Any, fsync: bool = True) -> Path:
    """Strict JSON (NaN -> null) written atomically."""
    text = json.dumps(jsonable(obj), indent=2, allow_nan=False, ensure_ascii=False)
    return atomic_write_bytes(path, text.encode("utf-8"), fsync=fsync)


def atomic_write_parquet(path: str | os.PathLike[str], df: pd.DataFrame, fsync: bool = True) -> Path:
    """Parquet (zstd) written atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(p)
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
        if fsync:
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Parse a JSON file; ``default`` if it is missing or (transiently) unreadable."""
    p = Path(path)
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError):
        return default


def config_hash(framework: Mapping[str, Any], simulation: Mapping[str, Any]) -> str:
    """Stable hash of the (framework, simulation) configuration of a run."""
    blob = json.dumps({"framework": jsonable(framework), "simulation": jsonable(simulation)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def epoch_file_name(epoch: str, suffix: str = "parquet") -> str:
    return f"epoch={epoch}.{suffix}"


def _epoch_files(directory: Path) -> list[tuple[str, Path]]:
    if not directory.is_dir():
        return []
    out = []
    for p in directory.iterdir():
        m = _EPOCH_FILE.match(p.name)
        if m:
            out.append((m.group(1), p))
    return sorted(out)


def _read_epoch_frames(directory: Path) -> pd.DataFrame | None:
    files = _epoch_files(directory)
    if not files:
        return None
    frames = [pd.read_parquet(p, engine="pyarrow") for _, p in files]
    frames = [f for f in frames if len(f.columns)]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ======================================================================================
# Event store
# ======================================================================================
class EventStore(JSONLSink):
    """Append-only JSONL event store (``events.jsonl``) with simple filtered reads."""

    def append(self, event: SimulationEvent | Mapping[str, Any]) -> None:
        self.emit(event)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # a line truncated by SIGKILL
                    continue

    def read(
        self,
        types: str | Iterable[str] | None = None,
        after: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Events as dicts with a 1-based ``seq`` (line order); filter by ``types``, keep
        ``seq > after``, at most ``limit`` rows."""
        wanted = {types} if isinstance(types, str) else (set(types) if types is not None else None)
        out: list[dict[str, Any]] = []
        for seq, ev in enumerate(self, start=1):
            if after is not None and seq <= after:
                continue
            if wanted is not None and ev.get("type") not in wanted:
                continue
            out.append({"seq": seq, **ev})
            if limit is not None and len(out) >= limit:
                break
        return out

    def count(self, types: str | Iterable[str] | None = None) -> int:
        return len(self.read(types))

    def truncate(self) -> None:
        """Start a fresh log (used by a fresh ``run`` in an existing directory)."""
        atomic_write_bytes(self.path, b"", fsync=False)


# ======================================================================================
# Checkpoints
# ======================================================================================
class Checkpoint:
    """Atomic pickle checkpoints: ``latest.pkl`` plus epoch-stamped copies (last ``keep``).

    The payload is a **plain dict** of JSON-like values and numpy arrays (never live
    objects), so a checkpoint stays loadable across code changes of the classes it
    describes and never needs the RNG bit-generator state (per-epoch seeding, see the
    runner).
    """

    LATEST = "latest.pkl"

    def __init__(self, directory: str | os.PathLike[str], keep: int = 3):
        self.directory = Path(directory)
        self.keep = max(int(keep), 1)

    @property
    def latest_path(self) -> Path:
        return self.directory / self.LATEST

    def exists(self) -> bool:
        return self.latest_path.is_file()

    def write(self, state: Mapping[str, Any], epoch: str) -> tuple[Path, int]:
        """Write ``state`` as ``latest.pkl`` and ``epoch=<epoch>.pkl``; returns (path, bytes)."""
        data = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
        self.directory.mkdir(parents=True, exist_ok=True)
        stamped = self.directory / epoch_file_name(epoch, "pkl")
        atomic_write_bytes(stamped, data)
        atomic_write_bytes(self.latest_path, data)
        self._prune()
        return self.latest_path, len(data)

    def _prune(self) -> None:
        stamped = self.list()
        for _, p in stamped[: max(len(stamped) - self.keep, 0)]:
            try:
                p.unlink()
            except FileNotFoundError:  # pragma: no cover
                pass

    def list(self) -> list[tuple[str, Path]]:
        if not self.directory.is_dir():
            return []
        out = []
        for p in self.directory.iterdir():
            m = _CKPT_FILE.match(p.name)
            if m:
                out.append((m.group(1), p))
        return sorted(out)

    def load(self, path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
        """The latest (or given) checkpoint dict; ``None`` if there is none.

        A corrupt ``latest.pkl`` (impossible with atomic writes, but cheap to guard)
        falls back to the newest readable epoch-stamped copy.
        """
        candidates: list[Path] = [Path(path)] if path is not None else [self.latest_path]
        if path is None:
            candidates += [p for _, p in reversed(self.list())]
        for p in candidates:
            if not p.is_file():
                continue
            try:
                with open(p, "rb") as fh:
                    state = pickle.load(fh)
            except (pickle.UnpicklingError, EOFError, OSError, AttributeError, ValueError) as exc:
                if path is not None:
                    raise CheckpointError(f"checkpoint {p} is unreadable: {exc}") from exc
                continue
            if not isinstance(state, dict) or "epoch_index" not in state:
                raise CheckpointError(f"checkpoint {p} has no epoch_index")
            return state
        return None

    def clear(self) -> None:
        if self.directory.is_dir():
            shutil.rmtree(self.directory)


# ======================================================================================
# Run directory
# ======================================================================================
class RunDirectory:
    """The run-directory protocol shared by the worker, the CLI and the Django API."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    # ----- paths ---------------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def progress_path(self) -> Path:
        return self.path / "progress.json"

    @property
    def control_path(self) -> Path:
        return self.path / "control.json"

    @property
    def reviews_dir(self) -> Path:
        return self.path / "reviews"

    @property
    def events_path(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def decisions_dir(self) -> Path:
        return self.path / "decisions"

    @property
    def feedback_dir(self) -> Path:
        return self.path / "feedback"

    @property
    def arms_dir(self) -> Path:
        return self.path / "arms"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def timeseries_path(self) -> Path:
        return self.path / "timeseries.parquet"

    @property
    def sentinel_path(self) -> Path:
        return self.path / "sentinel.parquet"

    @property
    def runtime_path(self) -> Path:
        return self.path / "runtime.json"

    @property
    def report_path(self) -> Path:
        return self.path / "report.json"

    @property
    def events(self) -> EventStore:
        return EventStore(self.events_path)

    @property
    def checkpoints(self) -> Checkpoint:
        return Checkpoint(self.checkpoints_dir)

    def ensure(self) -> RunDirectory:
        for d in (self.path, self.reviews_dir, self.decisions_dir, self.feedback_dir, self.arms_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ----- config ----------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        framework: Any,
        simulation: Any,
        dataset: Mapping[str, Any] | None = None,
        fit_dir: str | os.PathLike[str] | None = None,
    ) -> RunDirectory:
        """Prepare a run directory (what the API does before launching ``cli run``)."""
        rd = cls(path).ensure()
        rd.write_config(framework, simulation, dataset, fit_dir)
        rd.write_progress(status="queued", epoch_index=-1, message="queued")
        return rd

    def write_config(
        self,
        framework: Any,
        simulation: Any,
        dataset: Mapping[str, Any] | None = None,
        fit_dir: str | os.PathLike[str] | None = None,
    ) -> Path:
        fw = framework.to_dict() if hasattr(framework, "to_dict") else dict(framework)
        sim = simulation.to_dict() if hasattr(simulation, "to_dict") else dict(simulation)
        doc = {
            "framework": fw,
            "simulation": sim,
            "dataset": dict(dataset or {}),
            "fit_dir": None if fit_dir is None else str(fit_dir),
            "created_at": utc_now(),
        }
        return atomic_write_json(self.config_path, doc)

    def read_config(self) -> dict[str, Any]:
        cfg = read_json(self.config_path)
        if cfg is None:
            raise CheckpointError(f"{self.config_path} is missing or unreadable")
        return cfg

    def config_hash(self) -> str:
        cfg = self.read_config()
        return config_hash(cfg.get("framework", {}), cfg.get("simulation", {}))

    # ----- progress / control -------------------------------------------------------------------
    def read_progress(self) -> dict[str, Any]:
        return read_json(self.progress_path, {}) or {}

    def write_progress(self, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into ``progress.json`` (all :data:`PROGRESS_KEYS` present)."""
        cur = self.read_progress()
        base = {k: None for k in PROGRESS_KEYS}
        base.update(cur)
        base.update(fields)
        base["updated_at"] = utc_now()
        atomic_write_json(self.progress_path, base, fsync=False)
        return base

    def read_control(self) -> dict[str, Any] | None:
        ctl = read_json(self.control_path)
        if not isinstance(ctl, dict) or ctl.get("command") not in CONTROL_COMMANDS:
            return None
        return ctl

    def write_control(self, command: str) -> Path:
        if command not in CONTROL_COMMANDS:
            raise ValueError(f"control command must be one of {CONTROL_COMMANDS}")
        return atomic_write_json(self.control_path, {"command": command, "at": utc_now()}, fsync=False)

    # ----- reviews ---------------------------------------------------------------------------------
    def review_path(self, review_id: str) -> Path:
        return self.reviews_dir / f"{review_id}.json"

    def ack_path(self, review_id: str) -> Path:
        return self.reviews_dir / f"{review_id}.ack.json"

    def write_review(self, review: Mapping[str, Any]) -> Path:
        return atomic_write_json(self.review_path(str(review["id"])), dict(review), fsync=False)

    def read_review(self, review_id: str) -> dict[str, Any] | None:
        return read_json(self.review_path(review_id))

    def list_reviews(self) -> list[dict[str, Any]]:
        if not self.reviews_dir.is_dir():
            return []
        out = []
        for p in sorted(self.reviews_dir.glob("*.json")):
            if p.name.endswith(".ack.json"):
                continue
            rec = read_json(p)
            if isinstance(rec, dict):
                ack = self.read_ack(p.stem)
                out.append({**rec, "ack": ack})
        return out

    def write_ack(self, review_id: str, action: str, note: str = "", **extra: Any) -> Path:
        if action not in REVIEW_ACTIONS:
            raise ValueError(f"review action must be one of {REVIEW_ACTIONS}")
        return atomic_write_json(
            self.ack_path(review_id), {"action": action, "note": note, "at": utc_now(), **extra}, fsync=False
        )

    def read_ack(self, review_id: str) -> dict[str, Any] | None:
        ack = read_json(self.ack_path(review_id))
        return ack if isinstance(ack, dict) and ack.get("action") else None

    # ----- per-epoch tables ------------------------------------------------------------------------
    def write_decisions(self, epoch: str, df: pd.DataFrame) -> Path:
        return atomic_write_parquet(self.decisions_dir / epoch_file_name(epoch), df)

    def write_feedback(self, epoch: str, df: pd.DataFrame) -> Path:
        return atomic_write_parquet(self.feedback_dir / epoch_file_name(epoch), df)

    def write_arms(self, epoch: str, df: pd.DataFrame) -> Path:
        return atomic_write_parquet(self.arms_dir / epoch_file_name(epoch), df)

    def decision_epochs(self) -> list[str]:
        return [e for e, _ in _epoch_files(self.decisions_dir)]

    def read_decisions(self, fill_feedback: bool = True) -> pd.DataFrame:
        """All decision logs, epoch order; ``simulated_reward`` filled from ``feedback/``."""
        dec = _read_epoch_frames(self.decisions_dir)
        if dec is None:
            return pd.DataFrame()
        if fill_feedback:
            fb = self.read_feedback()
            if fb is not None and len(fb) and "simulated_reward" in dec.columns:
                sim = fb.loc[fb["source"] == "simulated", ["policy", "loan_id", "reward"]]
                sim = sim.drop_duplicates(["policy", "loan_id"], keep="last")
                merged = dec.loc[:, ["policy", "loan_id"]].merge(sim, on=["policy", "loan_id"], how="left")
                fill = merged["reward"].to_numpy(dtype=float)
                cur = dec["simulated_reward"].to_numpy(dtype=float)
                dec = dec.assign(simulated_reward=pd.Series(cur, index=dec.index).where(pd.notna(cur), fill))
        return dec

    def read_feedback(self) -> pd.DataFrame | None:
        return _read_epoch_frames(self.feedback_dir)

    def read_arms(self) -> pd.DataFrame | None:
        return _read_epoch_frames(self.arms_dir)

    # ----- whole-run tables ---------------------------------------------------------------------------
    def write_timeseries(self, df: pd.DataFrame) -> Path:
        return atomic_write_parquet(self.timeseries_path, df, fsync=False)

    def read_timeseries(self) -> pd.DataFrame:
        if not self.timeseries_path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(self.timeseries_path, engine="pyarrow")

    def write_sentinel(self, df: pd.DataFrame) -> Path:
        return atomic_write_parquet(self.sentinel_path, df, fsync=False)

    def read_sentinel(self) -> pd.DataFrame:
        if not self.sentinel_path.is_file():
            return pd.DataFrame()
        return pd.read_parquet(self.sentinel_path, engine="pyarrow")

    def write_runtime(self, runtime: Mapping[str, Any]) -> Path:
        return atomic_write_json(self.runtime_path, dict(runtime), fsync=False)

    def read_runtime(self) -> dict[str, Any] | None:
        return read_json(self.runtime_path)

    def write_report(self, report: Mapping[str, Any]) -> Path:
        return atomic_write_json(self.report_path, dict(report))

    def read_report(self) -> dict[str, Any] | None:
        return read_json(self.report_path)

    # ----- housekeeping ----------------------------------------------------------------------------
    def truncate_after(self, epoch: str | None) -> int:
        """Delete per-epoch outputs (decisions / feedback / arms) for epochs after
        ``epoch`` (all of them when ``epoch`` is None). Returns the number of files removed."""
        removed = 0
        for d in (self.decisions_dir, self.feedback_dir, self.arms_dir):
            for e, p in _epoch_files(d):
                if epoch is None or e > epoch:
                    p.unlink()
                    removed += 1
        return removed

    def clear_outputs(self) -> None:
        """Remove every worker output (fresh run); keeps config.json and control.json."""
        self.truncate_after(None)
        for p in (self.timeseries_path, self.sentinel_path, self.report_path, self.runtime_path):
            if p.exists():
                p.unlink()
        self.checkpoints.clear()
        if self.reviews_dir.is_dir():
            shutil.rmtree(self.reviews_dir)
        self.events.truncate()
        self.ensure()


def read_decisions(run_dir: str | os.PathLike[str] | RunDirectory) -> pd.DataFrame:
    """Concatenate ``decisions/epoch=*.parquet`` of a run (``simulated_reward`` filled)."""
    rd = run_dir if isinstance(run_dir, RunDirectory) else RunDirectory(run_dir)
    return rd.read_decisions()
