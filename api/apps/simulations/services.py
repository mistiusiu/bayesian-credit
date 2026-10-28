"""Simulation service layer on top of the core run-directory protocol
(:class:`bayesian_credit.simulation.store.RunDirectory`).

The API never talks to a running worker directly: it reads ``progress.json``,
``timeseries.parquet``, ``arms/``, ``decisions/``, ``sentinel.parquet``, ``events.jsonl``,
``reviews/`` and ``report.json``, and writes ``control.json`` (pause / resume / cancel) and
``reviews/<id>.ack.json``. Worker liveness comes from the job's pid, so a SIGKILLed worker
(chaos) is reported as ``interrupted`` even though its ``progress.json`` still says running.
"""

from __future__ import annotations

import json
import logging
import math
import signal
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from bayesian_credit.config import FrameworkConfig, PSIConfig, SimulationConfig
from bayesian_credit.data.arms import ArmSpace
from bayesian_credit.simulation.store import RunDirectory
from django.db import transaction

from apps.common import cache
from apps.common.exceptions import Conflict
from apps.common.jobs import JobRunner, kill_worker, mark_failed, reap_lost_workers, worker_alive
from apps.common.models import Job
from apps.common.utils import (
    clean,
    epoch_add,
    epoch_end,
    epoch_index,
    epoch_range,
    epoch_start,
    iso,
    iso_from_any,
    num,
    parse_iso,
    to_epoch,
    utcnow,
)
from apps.fits.models import Fit

from .models import Simulation

logger = logging.getLogger("apps.simulations")

TERMINAL = ("completed", "failed", "cancelled")
LIVE = ("queued", "running", "paused", "awaiting_review")
#: Not part of the default event listing (per-borrower / chatty types); request them by ``type=``.
QUIET_EVENT_TYPES = ("BorrowerArrived", "PolicySelected", "EpochStarted")
ADAPTIVE_EXCLUDED = ("historical", "random")
#: Arms need this many effective observations to be a group's best arm in the Simpson panel.
MIN_ARM_OBS = 3


def run_directory(sim: Simulation) -> RunDirectory:
    return RunDirectory(sim.directory)


# ============================================================================ create
def default_psi_settings(framework: FrameworkConfig) -> dict[str, Any]:
    psi = framework.psi
    return {
        "features": list(psi.features),
        "window_months": int(psi.window_months),
        "thresholds": {"watch": float(psi.watch_threshold), "breaker": float(psi.breaker_threshold)},
        "aggregate": psi.aggregate,
        "recovery_readings": int(psi.recovery_readings),
        "fallback": psi.fallback,
        "auto_action": psi.auto_action,
        "auto_ack_epochs": int(psi.auto_ack_epochs),
    }


def psi_config_from_settings(base: PSIConfig, settings: dict[str, Any]) -> PSIConfig:
    data = base.to_dict()
    data.update(
        features=list(settings["features"]),
        window_months=int(settings["window_months"]),
        watch_threshold=float(settings["thresholds"]["watch"]),
        breaker_threshold=float(settings["thresholds"]["breaker"]),
        aggregate=settings["aggregate"],
        recovery_readings=int(settings["recovery_readings"]),
        fallback=settings["fallback"],
        auto_action=settings["auto_action"],
        auto_ack_epochs=int(settings["auto_ack_epochs"]),
    )
    if settings.get("n_bins") is not None:
        data["n_bins"] = int(settings["n_bins"])
    return PSIConfig.from_dict(data)


def core_configs(fit: Fit, config: dict[str, Any]) -> tuple[FrameworkConfig, SimulationConfig]:
    """(FrameworkConfig, SimulationConfig) for the runner from the contract-shaped config."""
    framework = FrameworkConfig.from_dict(fit.framework or {})
    framework.psi = psi_config_from_settings(framework.psi, config["psi"])
    framework.bandit.interim_weight = float(config["interim_weight"])
    framework.validate()
    simulation = SimulationConfig.from_dict(
        {
            "start": epoch_start(config["start"]),
            "end": epoch_end(config["end"]),
            "max_loans": config["max_loans"],
            "seed": int(config["seed"]),
            "feedback_mode": config["feedback_mode"],
            "policies": list(config["policies"]),
            "interim_months": int(config["interim_months"]),
            "breaker_mode": config["breaker_mode"],
            "checkpoint_every": int(config["checkpoint_every"]),
            "psi_window_months": int(config["psi"]["window_months"]),
            "hindsight_sample": config.get("hindsight_sample", "elapsed"),
        }
    )
    return framework, simulation


def create_simulation(fit: Fit, name: str, config: dict[str, Any]) -> Simulation:
    """Create the row + run directory (``config.json``, ``progress.json``) and launch the worker."""
    framework, simulation = core_configs(fit, config)
    ds = fit.dataset
    with transaction.atomic():
        sim = Simulation.objects.create(name=name, fit=fit, config=config, policies=list(config["policies"]))
        run_dir = Path(sim.directory)
        sim.run_dir = str(run_dir)
        sim.save(update_fields=["run_dir"])
    from apps.common import artifacts

    artifacts.reset_dir(artifacts.simulation_dir(sim.pk))
    RunDirectory.create(
        run_dir,
        framework,
        simulation,
        dataset={
            "id": ds.pk,
            "name": ds.name,
            "source": ds.source,
            "path": str(ds.parquet),
            "parquet": str(ds.parquet),
            "simulation_id": sim.pk,
            "fit_id": fit.pk,
        },
        fit_dir=str(fit.directory),
    )
    job = JobRunner.create(Job.Kind.SIMULATION, target_type="simulation", target_id=sim.pk, params={"mode": "run"})
    sim.job = job
    sim.save(update_fields=["job"])
    JobRunner.launch(job)
    sim.refresh_from_db()
    return refresh_status(sim)


# ============================================================================ status
def _job_killed(job: Job | None) -> bool:
    """The job died without raising (SIGKILL / OOM / chaos) rather than failing with an error."""
    return bool(job is not None and job.status == Job.Status.FAILED and not job.error)


def refresh_status(sim: Simulation) -> Simulation:
    """Reconcile the stored status with ``progress.json`` and the worker's liveness."""
    job = sim.job
    if job is not None:
        job.refresh_from_db()
        if job.is_active:
            reap_lost_workers([job])
            job.refresh_from_db()
    rd = run_directory(sim)
    prog = rd.read_progress() if sim.run_dir else {}
    p_status = str(prog.get("status") or "queued")
    status = p_status
    if p_status in TERMINAL:
        status = p_status
    elif job is None:
        status = p_status
    elif job.status in Job.ACTIVE:
        if job.status == Job.Status.RUNNING and p_status == "queued":
            status = "running"
    elif job.status == Job.Status.SUCCEEDED:
        # the worker returned without a terminal progress status (should not happen)
        status = "completed" if p_status == "running" and rd.report_path.is_file() else p_status
        if status in LIVE:
            status = "interrupted"
    elif job.status == Job.Status.CANCELLED:
        status = "cancelled"
    elif _job_killed(job):
        status = "interrupted"
    else:
        status = "failed"

    changed = []
    if status != sim.status:
        if status == "interrupted" and sim.status in LIVE:
            _record_kill(sim, prog, detected=True)
            changed.append("recovery")
        sim.status = status
        changed.append("status")
    if prog and prog != sim.progress:
        sim.progress = prog
        changed.append("progress")
    started = parse_iso(prog.get("started_at")) if prog else None
    if started and sim.started_at is None:
        sim.started_at = started
        changed.append("started_at")
    if status in TERMINAL:
        finished = parse_iso(prog.get("finished_at")) or (job.finished_at if job else None) or utcnow()
        if sim.finished_at is None:
            sim.finished_at = finished
            changed.append("finished_at")
    elif sim.finished_at is not None:
        sim.finished_at = None
        changed.append("finished_at")
    if changed:
        sim.save(update_fields=sorted(set(changed)))
    return sim


def _record_kill(sim: Simulation, prog: dict[str, Any], detected: bool = False) -> None:
    rec = dict(sim.recovery or {})
    if rec.get("killed_at") and not rec.get("resumed_at"):
        return  # this kill is already on record
    current = to_epoch(prog.get("current_epoch")) if prog else None
    sim.recovery = {
        "killed_at": iso(utcnow()),
        "killed_at_epoch": current,
        "resumed_at": None,
        "resumed_from_epoch": None,
        "replayed_epochs": 0,
        "deterministic_match": None,
        "recovery_seconds": None,
        "detected": bool(detected),
    }


# ============================================================================ readers
def read_timeseries(sim: Simulation) -> pd.DataFrame:
    rd = run_directory(sim)
    return cache.memo("timeseries", cache.file_key(rd.timeseries_path), rd.read_timeseries)


def read_sentinel(sim: Simulation) -> pd.DataFrame:
    rd = run_directory(sim)
    return cache.memo("sentinel", cache.file_key(rd.sentinel_path), rd.read_sentinel)


def read_events(sim: Simulation) -> list[dict[str, Any]]:
    rd = run_directory(sim)
    return cache.memo("events", cache.file_key(rd.events_path), lambda: rd.events.read())


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _query(sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(sql, list(params)).df()
    finally:
        con.close()


def decision_counts(sim: Simulation) -> pd.DataFrame:
    """Decisions per (epoch, policy, group, arm) from ``decisions/epoch=*.parquet``."""
    rd = run_directory(sim)

    def compute() -> pd.DataFrame:
        files = sorted(rd.decisions_dir.glob("epoch=*.parquet")) if rd.decisions_dir.is_dir() else []
        if not files:
            return pd.DataFrame(columns=["epoch", "policy", "group", "arm", "n"])
        glob = _sql_path(rd.decisions_dir / "epoch=*.parquet")
        df = _query(
            f"SELECT CAST(epoch AS VARCHAR) AS epoch, CAST(policy AS VARCHAR) AS policy, "
            f'CAST("group" AS VARCHAR) AS "group", CAST(arm AS BIGINT) AS arm, count(*) AS n '
            f"FROM read_parquet({glob}, union_by_name=true) GROUP BY ALL"
        )
        df["epoch"] = df["epoch"].map(to_epoch)
        return df

    return cache.memo("decisions", cache.dir_key(rd.decisions_dir, "epoch=*.parquet"), compute)


def arm_snapshots(sim: Simulation) -> pd.DataFrame:
    """All posterior snapshots ``(epoch, policy, group, arm, mu, sigma, n, selections)``."""
    rd = run_directory(sim)

    def compute() -> pd.DataFrame:
        files = sorted(rd.arms_dir.glob("epoch=*.parquet")) if rd.arms_dir.is_dir() else []
        if not files:
            return pd.DataFrame(columns=["epoch", "policy", "group", "arm", "mu", "sigma", "n", "selections"])
        df = _query(f"SELECT * FROM read_parquet({_sql_path(rd.arms_dir / 'epoch=*.parquet')}, union_by_name=true)")
        df["epoch"] = df["epoch"].map(to_epoch)
        df["policy"] = df["policy"].astype(str)
        df["group"] = df["group"].astype(str)
        df["arm"] = df["arm"].astype(int)
        return df

    return cache.memo("arms", cache.dir_key(rd.arms_dir, "epoch=*.parquet"), compute)


def completed_epochs(sim: Simulation) -> list[str]:
    ts = read_timeseries(sim)
    if ts.empty or "epoch" not in ts.columns:
        return []
    return sorted({to_epoch(e) for e in ts["epoch"].tolist() if to_epoch(e)})


def arm_space_of(sim: Simulation) -> ArmSpace:
    from apps.fits.services import fit_arm_space

    return fit_arm_space(sim.fit)


def arm_row(arm_space: ArmSpace, arm: int) -> dict[str, Any]:
    pol = arm_space[int(arm)]
    return {
        "arm": int(arm),
        "amount": float(pol.amount),
        "rate": float(pol.rate),
        "term": int(pol.term),
        "is_null": bool(pol.is_null),
        "installment": round(float(pol.installment), 2),
        "label": pol.label(),
    }


def group_dims(sim: Simulation, group: str) -> dict[str, str]:
    dims = ((sim.fit.config or {}).get("segmentation") or {}).get("dimensions") or []
    parts = str(group).split("|")
    if len(parts) != len(dims):  # e.g. after a resegment the key has more parts
        dims = list(dims) + [f"dim{i}" for i in range(len(dims), len(parts))]
    return {d: parts[i] if i < len(parts) else "" for i, d in enumerate(dims)}


# ============================================================================ headline / detail
def headline(sim: Simulation, ts: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """types.ts ``PolicyHeadline`` per configured policy (empty before the first epoch)."""
    ts = read_timeseries(sim) if ts is None else ts
    if ts.empty:
        return []
    out = []
    order = list(sim.policies or sim.config.get("policies") or [])
    for policy in order + sorted(set(ts["policy"].astype(str)) - set(order)):
        rows = ts.loc[ts["policy"].astype(str) == policy]
        if rows.empty:
            continue
        rows = rows.assign(_e=rows["epoch"].map(to_epoch)).sort_values("_e")
        last = rows.iloc[-1]
        n = float(pd.to_numeric(rows["n_decisions"], errors="coerce").fillna(0).sum())
        w = pd.to_numeric(rows["n_decisions"], errors="coerce").fillna(0).to_numpy()

        def wmean(col: str, weights: np.ndarray = w, frame: pd.DataFrame = rows) -> float | None:
            if col not in frame.columns:
                return None
            x = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & (weights > 0)
            return float(np.sum(x[ok] * weights[ok]) / np.sum(weights[ok])) if ok.any() else None

        decline = wmean("decline_rate")
        lend_w = w * (1.0 - pd.to_numeric(rows.get("decline_rate", 0), errors="coerce").fillna(0).to_numpy())
        cum_reward = num(last.get("cum_reward")) or 0.0
        out.append(
            {
                "policy": policy,
                "n_decisions": int(n),
                "cum_reward": round(cum_reward, 2),
                "cum_realized": round(num(last.get("cum_realized")) or 0.0, 2),
                "cum_regret": round(num(last.get("cum_regret")) or 0.0, 2),
                "mean_reward": round(cum_reward / n, 2) if n else 0.0,
                "decline_rate": round(decline, 4) if decline is not None else 0.0,
                "mean_amount": num(wmean("mean_amount", lend_w), 2),
                "mean_rate": num(wmean("mean_rate", lend_w), 6),
            }
        )
    return out


def uplift_vs_historical(rows: list[dict[str, Any]]) -> float | None:
    hist = next((r for r in rows if r["policy"] == "historical"), None)
    adaptive = [r for r in rows if r["policy"] not in ADAPTIVE_EXCLUDED]
    if hist is None or not adaptive:
        return None
    best = max(adaptive, key=lambda r: r["cum_reward"])
    return round(best["cum_reward"] - hist["cum_reward"], 2)


def _level(value: float | None, thresholds: dict[str, float]) -> str:
    if value is None:
        return "normal"
    if value > thresholds.get("breaker", 0.25):
        return "breaker"
    if value >= thresholds.get("watch", 0.10):
        return "watch"
    return "normal"


def _thresholds(sim: Simulation) -> dict[str, float]:
    t = ((sim.config or {}).get("psi") or {}).get("thresholds") or {}
    return {"watch": float(t.get("watch", 0.10)), "breaker": float(t.get("breaker", 0.25))}


def recovery_info(sim: Simulation, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """types.ts ``RecoveryInfo`` for the latest kill, completed from the event log."""
    rec = dict(sim.recovery or {})
    if not rec.get("killed_at"):
        return None
    killed_at = parse_iso(rec["killed_at"])
    resumed = None
    for ev in events:
        if ev.get("type") == "RunResumed" and killed_at is not None:
            ts = parse_iso(ev.get("ts"))
            if ts is not None and ts >= killed_at.replace(microsecond=0):
                resumed = ev
                break
    if resumed is not None:
        payload = resumed.get("payload") or {}
        from_epoch = to_epoch(payload.get("from_epoch"))
        rec["resumed_at"] = rec.get("resumed_at") or iso_from_any(resumed.get("ts"))
        rec["resumed_from_epoch"] = from_epoch
        killed_epoch = rec.get("killed_at_epoch")
        if killed_epoch and from_epoch:
            rec["replayed_epochs"] = max(epoch_index(killed_epoch) - epoch_index(from_epoch), 0)
        elif killed_epoch:
            rec["replayed_epochs"] = epoch_index(killed_epoch) - epoch_index(to_epoch(sim.config["start"])) + 1
        # last EpochCompleted per epoch before the restart vs the first one after it
        before: dict[str, Any] = {}
        after: dict[str, Any] = {}
        after_ts: dict[str, Any] = {}
        for ev in events:
            if ev.get("type") != "EpochCompleted":
                continue
            e = to_epoch(ev.get("epoch"))
            per = (ev.get("payload") or {}).get("per_policy")
            if ev["seq"] < resumed["seq"]:
                before[e] = per
            elif e not in after:
                after[e] = per
                after_ts[e] = parse_iso(ev.get("ts"))
        replayed = [
            e for e in before if e and (not from_epoch or e > from_epoch) and (not killed_epoch or e <= killed_epoch)
        ]
        # "passed the kill point": the killed epoch was recomputed (or, with nothing to replay,
        # the new worker completed its first epoch)
        target = killed_epoch if (killed_epoch and killed_epoch in before) else None
        passed = target in after if target else bool(after)
        if passed:
            rec["deterministic_match"] = all(_same_metrics(before[e], after.get(e)) for e in replayed)
            ts = after_ts.get(target) if target else min((t for t in after_ts.values() if t), default=None)
            if ts is not None and killed_at is not None:
                rec["recovery_seconds"] = round(max((ts - killed_at).total_seconds(), 0.0), 1)
    return {
        "killed_at": rec["killed_at"],
        "killed_at_epoch": rec.get("killed_at_epoch"),
        "resumed_at": rec.get("resumed_at"),
        "resumed_from_epoch": rec.get("resumed_from_epoch"),
        "replayed_epochs": int(rec.get("replayed_epochs") or 0),
        "deterministic_match": rec.get("deterministic_match"),
        "recovery_seconds": rec.get("recovery_seconds"),
    }


def _same_metrics(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_metrics(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_metrics(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def simulation_payload(sim: Simulation, *, refresh: bool = True) -> dict[str, Any]:
    """types.ts ``Simulation``."""
    from apps.common.serializers import JobSerializer
    from apps.sentinel.models import Review
    from apps.sentinel.services import sync_reviews

    if refresh:
        refresh_status(sim)
        try:
            sync_reviews(sim)
        except Exception:  # noqa: BLE001 - a malformed review file must not break the detail view
            logger.exception("review sync failed for simulation %s", sim.pk)
    cfg = sim.config or {}
    prog = sim.progress or {}
    ts = read_timeseries(sim)
    epochs = completed_epochs(sim)
    total = int(prog.get("n_epochs") or 0) or len(epoch_range(to_epoch(cfg.get("start")), to_epoch(cfg.get("end"))))
    done = len(epochs)
    rows = headline(sim, ts)
    current = epochs[-1] if epochs else None
    sentinel = read_sentinel(sim)
    thresholds = _thresholds(sim)
    psi_agg = None
    breaker = str(prog.get("breaker_state") or "closed")
    if not sentinel.empty and current is not None:
        last = sentinel.loc[sentinel["epoch"].map(to_epoch) == current]
        if not last.empty:
            psi_agg = num(last["aggregate"].iloc[0]) if "aggregate" in last.columns else None
            if "breaker_state" in last.columns and not prog.get("breaker_state"):
                breaker = str(last["breaker_state"].iloc[-1])
    if not ts.empty and current is not None and "psi_aggregate" in ts.columns and psi_agg is None:
        psi_agg = num(ts.loc[ts["epoch"].map(to_epoch) == current, "psi_aggregate"].iloc[0])
    events = read_events(sim)
    # the latest reading may belong to an epoch still in progress (e.g. awaiting review)
    latest = next(
        (
            e
            for e in reversed(events)
            if e.get("type") == "SentinelEvaluated" and not (e.get("payload") or {}).get("skipped")
        ),
        None,
    )
    if latest is not None and (current is None or (to_epoch(latest.get("epoch")) or "") >= current):
        psi_agg = num((latest.get("payload") or {}).get("psi_aggregate"))
    checkpoints = [e for e in events if e.get("type") == "CheckpointWritten"]
    alive = sim.worker_alive()
    job = sim.job
    status = sim.status
    progress = 1.0 if status == "completed" else (round(done / total, 4) if total else 0.0)
    return clean(
        {
            "id": sim.pk,
            "name": sim.name,
            "fit": sim.fit_id,
            "fit_name": sim.fit.name,
            "dataset": sim.fit.dataset_id,
            "status": status,
            "progress": progress,
            "pct": round(progress * 100, 1),
            "config": cfg,
            "epochs_total": total,
            "epochs_done": done,
            "current_epoch": current,
            "breaker_state": breaker if breaker in ("closed", "open", "half_open") else "closed",
            "sentinel_level": _level(psi_agg, thresholds),
            "psi_aggregate": num(psi_agg, 6),
            "pending_reviews": Review.objects.filter(simulation=sim, status=Review.Status.PENDING).count(),
            "last_checkpoint_epoch": to_epoch(checkpoints[-1].get("epoch")) if checkpoints else None,
            "restarts": int(sim.resume_count),
            "worker": {
                "pid": job.pid if (alive and job is not None) else None,
                "alive": alive,
                "last_heartbeat": iso_from_any(prog.get("updated_at")),
            },
            "last_recovery": recovery_info(sim, events),
            "headline": rows,
            "uplift_vs_historical": uplift_vs_historical(rows),
            "message": prog.get("message") or (job.message if job else ""),
            "created_at": iso(sim.created_at),
            "started_at": iso(sim.started_at) or (iso(job.started_at) if job else None),
            "finished_at": iso(sim.finished_at) if status in TERMINAL else None,
            "job": JobSerializer(job).data if job is not None else None,
        }
    )


# ============================================================================ timeseries
TIMESERIES_FIELDS = (
    "n_decisions",
    "reward",
    "cum_reward",
    "realized",
    "cum_realized",
    "regret",
    "cum_regret",
    "rolling_reward",
    "decline_rate",
    "mean_amount",
    "mean_rate",
    "breaker_state",
    "psi_aggregate",
)
EXTRA_TIMESERIES_FIELDS = (
    "fallback_rate",
    "n_feedback_interim",
    "n_feedback_terminal",
    "pending_feedback",
    "n_groups",
    "sentinel_level",
    "rolling_reward_runner",
)
#: Trailing window of ``rolling_reward`` (docs/API.md §8.4; the runner's own column uses 3).
ROLLING_EPOCHS = 6


def policy_param(sim: Simulation, policy: str | None) -> str:
    from rest_framework.exceptions import ValidationError

    policies = list(sim.policies or sim.config.get("policies") or [])
    if policy:
        if policy not in policies:
            raise ValidationError({"policy": [f"policy must be one of {', '.join(policies)}"]})
        return policy
    if not policies:
        raise ValidationError({"policy": ["the simulation has no policies"]})
    return policies[0]


def epoch_param(sim: Simulation, epoch: str | None, available: Sequence[str]) -> str:
    from rest_framework.exceptions import ValidationError

    if not available:
        raise Conflict("no completed epochs yet")
    if not epoch:
        return available[-1]
    e = to_epoch(epoch)
    if e not in available:
        raise ValidationError({"epoch": ["epoch must be a completed epoch (YYYY-MM)"]})
    return e


def timeseries_payload(sim: Simulation, policy: str | None = None) -> dict[str, Any]:
    ts = read_timeseries(sim)
    policies = list(sim.policies or [])
    if policy:
        policy = policy_param(sim, policy)
        policies = [policy]
    rows = []
    if not ts.empty:
        frame = ts.assign(_epoch=ts["epoch"].map(to_epoch), _policy=ts["policy"].astype(str))
        frame = frame.loc[frame["_policy"].isin(policies)] if policies else frame
        order = {p: i for i, p in enumerate(policies)}
        frame = frame.assign(_order=frame["_policy"].map(lambda p: order.get(p, len(order))))
        frame = frame.sort_values(["_epoch", "_order"], kind="stable")
        # rolling_reward = Σ reward / Σ n_decisions over the trailing ROLLING_EPOCHS epochs
        rew = pd.to_numeric(frame["reward"], errors="coerce").fillna(0.0)
        dec = pd.to_numeric(frame["n_decisions"], errors="coerce").fillna(0.0)
        by = frame["_policy"]
        roll_r = rew.groupby(by).transform(lambda x: x.rolling(ROLLING_EPOCHS, min_periods=1).sum())
        roll_n = dec.groupby(by).transform(lambda x: x.rolling(ROLLING_EPOCHS, min_periods=1).sum())
        frame = frame.assign(
            rolling_reward_runner=frame["rolling_reward"] if "rolling_reward" in frame.columns else np.nan,
            rolling_reward=(roll_r / roll_n.where(roll_n > 0)).to_numpy(),
        )
        for rec in frame.to_dict("records"):
            row = {"epoch": rec["_epoch"], "policy": rec["_policy"]}
            for f in TIMESERIES_FIELDS + EXTRA_TIMESERIES_FIELDS:
                if f in rec:
                    row[f] = rec[f]
            row["n_decisions"] = int(num(row.get("n_decisions")) or 0)
            row["rolling_reward_runner"] = num(row.get("rolling_reward_runner"), 2)
            for f in ("reward", "cum_reward", "realized", "cum_realized", "regret", "cum_regret", "rolling_reward"):
                row[f] = round(num(row.get(f)) or 0.0, 2)
            row["decline_rate"] = num(row.get("decline_rate"), 6) or 0.0
            row["mean_amount"] = num(row.get("mean_amount"), 2)
            row["mean_rate"] = num(row.get("mean_rate"), 6)
            bs = str(row.get("breaker_state") or "closed")
            row["breaker_state"] = bs if bs in ("closed", "open", "half_open") else "closed"
            row["psi_aggregate"] = num(row.get("psi_aggregate"), 6)
            rows.append(row)
    return clean(
        {
            "simulation": sim.pk,
            "policies": policies,
            "epochs": sorted({r["epoch"] for r in rows}),
            "rows": rows,
        }
    )


# ============================================================================ arms / posteriors / simpson
def _snapshot_upto(snaps: pd.DataFrame, policy: str, epoch: str) -> tuple[str | None, pd.DataFrame]:
    """Latest snapshot epoch ≤ ``epoch`` for ``policy`` and its rows."""
    if snaps.empty:
        return None, snaps
    p = snaps.loc[snaps["policy"] == policy]
    eligible = sorted(e for e in p["epoch"].unique() if e and e <= epoch)
    if not eligible:
        return None, p.iloc[0:0]
    snap_epoch = eligible[-1]
    return snap_epoch, p.loc[p["epoch"] == snap_epoch]


def _pooled(rows: pd.DataFrame, weights: dict[str, float], n_arms: int) -> pd.DataFrame:
    """Decision-weighted pooled mu/sigma per arm (``n`` summed)."""
    out = {
        "arm": np.arange(n_arms),
        "mu": np.full(n_arms, np.nan),
        "sigma": np.full(n_arms, np.nan),
        "n": np.zeros(n_arms),
    }
    if rows.empty:
        return pd.DataFrame(out)
    work = rows.assign(_w=rows["group"].map(lambda g: weights.get(g, 0.0)))
    if work["_w"].sum() <= 0:
        work = work.assign(_w=1.0)
    for arm, grp in work.groupby("arm"):
        if not 0 <= int(arm) < n_arms:
            continue
        mu = pd.to_numeric(grp["mu"], errors="coerce").to_numpy(dtype=float)
        sigma = pd.to_numeric(grp["sigma"], errors="coerce").to_numpy(dtype=float)
        w = grp["_w"].to_numpy(dtype=float)
        ok = np.isfinite(mu) & (w > 0)
        if ok.any():
            out["mu"][int(arm)] = float(np.sum(mu[ok] * w[ok]) / np.sum(w[ok]))
            oks = ok & np.isfinite(sigma)
            if oks.any():
                out["sigma"][int(arm)] = float(np.sum(sigma[oks] * w[oks]) / np.sum(w[oks]))
        out["n"][int(arm)] = float(pd.to_numeric(grp["n"], errors="coerce").fillna(0).sum())
    return pd.DataFrame(out)


def _posterior_rows(
    rows: pd.DataFrame, group: str | None, counts: pd.DataFrame, policy: str, epoch: str, k: int
) -> pd.DataFrame:
    """Arm posteriors indexed by arm: the group's own table, else the policy's pooled level
    (``group == "*"``), else a decision-weighted average over the snapshotted groups."""
    if rows.empty:
        return pd.DataFrame(columns=["mu", "sigma", "n"])
    if group:
        return rows.loc[rows["group"] == group].drop_duplicates("arm", keep="last").set_index("arm")
    pooled = rows.loc[rows["group"] == "*"]
    if not pooled.empty:
        return pooled.drop_duplicates("arm", keep="last").set_index("arm")
    return _pooled(rows, group_weights(counts, policy, epoch), k).set_index("arm")


def group_weights(counts: pd.DataFrame, policy: str, epoch: str) -> dict[str, float]:
    c = counts.loc[(counts["policy"] == policy) & (counts["epoch"] <= epoch)]
    return {str(g): float(n) for g, n in c.groupby("group")["n"].sum().items()}


def arms_payload(sim: Simulation, policy: str | None, epoch: str | None, group: str | None) -> dict[str, Any]:
    """types.ts ``SimArms``."""
    epochs = completed_epochs(sim)
    policy = policy_param(sim, policy)
    epoch = epoch_param(sim, epoch, epochs)
    arm_space = arm_space_of(sim)
    k = arm_space.n_arms
    counts = decision_counts(sim)
    pc = counts.loc[counts["policy"] == policy]
    if group:
        pc = pc.loc[pc["group"] == group]
    this = pc.loc[pc["epoch"] == epoch].groupby("arm")["n"].sum()
    cum = pc.loc[pc["epoch"] <= epoch].groupby("arm")["n"].sum()
    n_epoch = float(this.sum())
    n_cum = float(cum.sum())
    snaps = arm_snapshots(sim)
    snap_epoch, rows = _snapshot_upto(snaps, policy, epoch)
    post = _posterior_rows(rows, group, counts, policy, epoch, k)
    arms = []
    for a in range(k):
        base = arm_row(arm_space, a)
        mu = num(post["mu"].get(a)) if "mu" in post.columns and a in post.index else None
        sigma = num(post["sigma"].get(a)) if "sigma" in post.columns and a in post.index else None
        n = num(post["n"].get(a)) if "n" in post.columns and a in post.index else None
        sel = float(this.get(a, 0))
        csel = float(cum.get(a, 0))
        arms.append(
            {
                **base,
                "mu": num(mu, 2),
                "sigma": num(sigma, 2),
                "n": round(n or 0.0, 2),
                "selections": int(sel),
                "cum_selections": int(csel),
                "share": round(sel / n_epoch, 6) if n_epoch else 0.0,
                "cum_share": round(csel / n_cum, 6) if n_cum else 0.0,
            }
        )
    by_group = counts.loc[(counts["policy"] == policy) & (counts["epoch"] <= epoch)].groupby("group")["n"].sum()
    groups = [{"group": str(g), "n": int(n)} for g, n in by_group.sort_values(ascending=False, kind="stable").items()]
    return clean(
        {
            "simulation": sim.pk,
            "policy": policy,
            "epoch": epoch,
            "group": group or None,
            "snapshot_epoch": snap_epoch,
            "n_decisions": int(n_epoch),
            "amounts": list(arm_space.amount_grid),
            "rates": list(arm_space.rate_grid),
            "terms": list(arm_space.term_grid),
            "arms": arms,
            "groups": groups,
        }
    )


def posteriors_payload(
    sim: Simulation, policy: str | None, group: str | None, arms: Sequence[int] = ()
) -> dict[str, Any]:
    """types.ts ``PosteriorHistory`` (one row per snapshot epoch × arm)."""
    epochs = completed_epochs(sim)
    if not epochs:
        raise Conflict("no completed epochs yet")
    policy = policy_param(sim, policy)
    arm_space = arm_space_of(sim)
    k = arm_space.n_arms
    counts = decision_counts(sim)
    pc = counts.loc[counts["policy"] == policy]
    snaps_all = arm_snapshots(sim)
    if not group:  # docs/API.md §8.6: default group = the largest one
        sizes = pc.groupby("group")["n"].sum().sort_values(ascending=False, kind="stable")
        snapped = set(snaps_all.loc[snaps_all["policy"] == policy, "group"]) if not snaps_all.empty else set()
        group = next((str(g) for g in sizes.index if str(g) in snapped and str(g) != "*"), None)
    if group:
        pc = pc.loc[pc["group"] == group]
    wanted = [int(a) for a in arms if 0 <= int(a) < k and not arm_space[int(a)].is_null]
    if not wanted:
        top = pc.groupby("arm")["n"].sum().sort_values(ascending=False, kind="stable")
        wanted = [int(a) for a in top.index if 0 <= int(a) < k and not arm_space[int(a)].is_null][:4]
        if not wanted:
            wanted = [int(a) for a in arm_space.lending_arms[:4]]
    snaps = snaps_all.loc[snaps_all["policy"] == policy] if not snaps_all.empty else snaps_all
    snap_epochs = sorted(e for e in snaps["epoch"].unique() if e) if not snaps.empty else []
    rows = []
    for e in snap_epochs:
        at = snaps.loc[snaps["epoch"] == e]
        post = _posterior_rows(at, group, counts, policy, e, k)
        for a in wanted:
            if a not in post.index:
                continue
            mu, sigma = num(post["mu"].get(a)), num(post["sigma"].get(a))
            if mu is None:
                continue
            rows.append(
                {
                    "epoch": e,
                    "arm": a,
                    "mu": round(mu, 2),
                    "sigma": round(sigma or 0.0, 2),
                    "n": round(num(post["n"].get(a)) or 0.0, 2),
                }
            )
    every = int(((sim.config or {}).get("posterior_snapshot_every")) or _snapshot_every(sim))
    return clean(
        {
            "simulation": sim.pk,
            "policy": policy,
            "group": group or None,
            "snapshot_every": every,
            "arms": [arm_row(arm_space, a) for a in wanted],
            "rows": rows,
        }
    )


def _snapshot_every(sim: Simulation) -> int:
    try:
        cfg = run_directory(sim).read_config()
        return int((cfg.get("simulation") or {}).get("posterior_snapshot_every") or 3)
    except Exception:  # noqa: BLE001
        return 3


def simpson_payload(sim: Simulation, policy: str | None, epoch: str | None) -> dict[str, Any]:
    """types.ts ``SimpsonPanel``: pooled-best arm vs per-group best arm (aggregation bias)."""
    epochs = completed_epochs(sim)
    policy = policy_param(sim, policy)
    epoch = epoch_param(sim, epoch, epochs)
    arm_space = arm_space_of(sim)
    lending = [int(a) for a in arm_space.lending_arms]
    counts = decision_counts(sim)
    weights = group_weights(counts, policy, epoch)
    _, snap = _snapshot_upto(arm_snapshots(sim), policy, epoch)
    if snap.empty:
        raise Conflict(f"no posterior snapshot for policy {policy} at or before {epoch}")
    snap = snap.loc[snap["arm"].isin(lending)].assign(mu=lambda f: pd.to_numeric(f["mu"], errors="coerce"))
    snap = snap.loc[np.isfinite(snap["mu"].to_numpy(dtype=float))]
    rows = snap.loc[snap["group"] != "*"]
    if rows.empty:
        raise Conflict(f"policy {policy} keeps no per-group reward posteriors (nothing to compare)")
    table = rows.pivot_table(index="group", columns="arm", values="mu", aggfunc="last")
    n_eff = rows.assign(_n=pd.to_numeric(rows["n"], errors="coerce").fillna(0.0) if "n" in rows.columns else 0.0)
    qualified = n_eff.pivot_table(index="group", columns="arm", values="_n", aggfunc="last") >= MIN_ARM_OBS
    w = np.array([weights.get(str(g), 0.0) for g in table.index], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(table))
    star = snap.loc[snap["group"] == "*"].drop_duplicates("arm", keep="last")
    pooled: dict[int, float] = {}
    if not star.empty:
        pooled = {int(a): float(m) for a, m in zip(star["arm"], star["mu"], strict=True)}
    else:
        for arm in table.columns:
            col = table[arm].to_numpy(dtype=float)
            ok = np.isfinite(col)
            pooled[int(arm)] = float(np.sum(col[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else -np.inf
    best_arm = max(pooled, key=lambda a: pooled[a])
    out_rows = []
    forfeited = 0.0
    for gi, g in enumerate(table.index):
        all_mus = table.loc[g]
        all_mus = all_mus[np.isfinite(all_mus.to_numpy(dtype=float))]
        if all_mus.empty:
            continue
        ok = qualified.loc[g].reindex(all_mus.index).fillna(False).to_numpy(dtype=bool)
        mus = all_mus[ok] if ok.any() else all_mus
        g_best = int(mus.idxmax())
        g_best_mu = float(mus.max())
        pooled_mu = float(all_mus.get(best_arm, np.nan))
        if not math.isfinite(pooled_mu):
            pooled_mu = g_best_mu
        gap = max(g_best_mu - pooled_mu, 0.0)
        share = w[gi] / w.sum()
        forfeited += share * gap
        out_rows.append(
            {
                "group": str(g),
                "dims": group_dims(sim, str(g)),
                "n": int(weights.get(str(g), 0)),
                "group_best_arm": g_best,
                "group_best_label": arm_space[g_best].label(),
                "group_best_mu": round(g_best_mu, 2),
                "pooled_arm_mu": round(pooled_mu, 2),
                "gap": round(gap, 2),
                "agrees": g_best == best_arm,
            }
        )
    out_rows.sort(key=lambda r: -r["n"])
    return clean(
        {
            "simulation": sim.pk,
            "policy": policy,
            "epoch": epoch,
            "pooled_best_arm": int(best_arm),
            "pooled_best_label": arm_space[int(best_arm)].label(),
            "pooled_best_mu": round(pooled[best_arm], 2),
            "n_groups": len(out_rows),
            "n_agree": sum(1 for r in out_rows if r["agrees"]),
            "forfeited_per_decision": round(forfeited, 2),
            "groups": out_rows,
        }
    )


# ============================================================================ events
def _fmt_money(x: Any) -> str:
    v = num(x)
    if v is None:
        return "n/a"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}${v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:.1f}k"
    return f"{sign}${v:.0f}"


def event_level(etype: str, payload: dict[str, Any]) -> str:
    if etype in ("BreakerTripped", "RunFailed", "WorkerKilled"):
        return "critical"
    if etype == "SentinelEvaluated":
        level = payload.get("level")
        return {"breaker": "critical", "watch": "warning"}.get(level, "info")
    if etype in ("RunCancelled", "ReviewAcknowledged", "ResegmentationApplied", "RunPaused", "RunResumed"):
        return "warning"
    return "info"


def event_message(etype: str, epoch: str | None, p: dict[str, Any]) -> str:
    try:
        if etype == "RunStarted":
            pols = p.get("policies") or []
            return (
                f"run started: {len(pols)} policies, {p.get('n_epochs', '?')} epochs "
                f"({p.get('first_epoch', '?')}..{p.get('last_epoch', '?')}), {p.get('n_loans', '?')} loans, "
                f"{p.get('feedback_mode', '?')} feedback"
            )
        if etype == "RunResumed":
            frm = p.get("from_epoch") or "start"
            return f"resumed on pid {p.get('pid', '?')} from checkpoint {frm}"
        if etype == "EpochStarted":
            return f"epoch {epoch} started: {p.get('n_arrivals', '?')} arrivals"
        if etype == "SentinelEvaluated":
            if p.get("skipped"):
                return f"PSI skipped (window of {p.get('n_current', 0)} loans)"
            per = p.get("per_feature") or {}
            top = max(per, key=lambda f: per[f] if per[f] is not None else -1) if per else None
            agg = num(p.get("psi_aggregate"))
            text = f"PSI {agg:.3f} ({p.get('level', 'normal')})" if agg is not None else "PSI n/a"
            if top:
                text += f", top feature {top}"
            if p.get("breaker_before") != p.get("breaker_after"):
                text += f"; breaker {p.get('breaker_before')} → {p.get('breaker_after')}"
            return text
        if etype == "BreakerTripped":
            psi = num(p.get("psi_aggregate"))
            return (
                f"breaker open: PSI {psi:.3f}; review {p.get('review_id')} opened; policies frozen"
                if psi is not None
                else f"breaker open; review {p.get('review_id')} opened; policies frozen"
            ) + (f" ({p.get('breaker_mode')})" if p.get("breaker_mode") else "")
        if etype == "ReviewAcknowledged":
            who = "auto-acknowledged" if p.get("auto") else "acknowledged"
            return (
                f"review {p.get('review_id')} {who}: {p.get('action')}; breaker {p.get('breaker_state', 'half_open')}"
            )
        if etype == "ResegmentationApplied":
            before = "×".join(p.get("dimensions_before") or [])
            after = "×".join(p.get("dimensions_after") or [])
            return f"resegmented {before} → {after} ({p.get('n_groups', '?')} groups, {p.get('n_refit_loans', '?')} loans refit)"
        if etype == "FeedbackReleased":
            return (
                f"released {p.get('n_interim', 0)} interim and {p.get('n_terminal', 0)} terminal observations"
                f" ({p.get('pending', 0)} pending)"
            )
        if etype == "CheckpointWritten":
            size = num(p.get("bytes"))
            return f"checkpoint written at {epoch}" + (f" ({size / 1e6:.1f} MB)" if size else "")
        if etype == "EpochCompleted":
            per = p.get("per_policy") or {}
            parts = [f"{name} {_fmt_money((v or {}).get('reward'))}" for name, v in list(per.items())[:4]]
            return f"epoch {epoch}: {p.get('n_decisions', '?')} decisions" + (f"; {', '.join(parts)}" if parts else "")
        if etype == "RunPaused":
            return f"paused ({p.get('reason', 'control')})" + (
                f", review {p['review_id']}" if p.get("review_id") else ""
            )
        if etype == "RunCancelled":
            return f"cancelled ({p.get('reason', 'control')})"
        if etype == "RunCompleted":
            wall = num(p.get("wall_seconds"))
            return f"run completed: {p.get('n_epochs', '?')} epochs, {p.get('n_decisions', '?')} decisions" + (
                f" in {wall:.0f}s" if wall is not None else ""
            )
        if etype == "RunFailed":
            return f"run failed: {p.get('error_type', 'Error')}: {p.get('error', '')}"
        if etype == "WorkerKilled":
            return f"worker pid {p.get('pid', '?')} killed ({p.get('signal', 'SIGKILL')}, {p.get('reason', 'chaos')})"
    except Exception:  # noqa: BLE001 - a message is cosmetic
        pass
    return etype


def payload_aliases(
    etype: str, epoch: str | None, payload: dict[str, Any], review_ids: dict[str, int]
) -> dict[str, Any]:
    """Add the payload keys documented in docs/API.md §8.8 next to the runner's own keys."""
    p = dict(payload)
    if etype == "SentinelEvaluated":
        per = p.get("per_feature") or {}
        finite = {f: v for f, v in per.items() if v is not None}
        p.setdefault("aggregate", p.get("psi_aggregate"))
        p.setdefault("top_feature", max(finite, key=finite.get) if finite else None)
    elif etype == "BreakerTripped":
        p.setdefault("review", review_ids.get(str(p.get("review_id"))))
        p.setdefault("psi", p.get("psi_aggregate"))
    elif etype == "ReviewAcknowledged":
        p.setdefault("review", review_ids.get(str(p.get("review_id"))))
    elif etype == "FeedbackReleased":
        p.setdefault("interim", p.get("n_interim"))
        p.setdefault("terminal", p.get("n_terminal"))
    elif etype == "CheckpointWritten":
        p.setdefault("epoch", epoch)
    elif etype == "EpochCompleted":
        per = p.get("per_policy") or {}
        p.setdefault("reward", {name: (v or {}).get("reward") for name, v in per.items()})
    elif etype == "RunResumed":
        p.setdefault("from_checkpoint", p.get("from_epoch"))
    return p


def to_contract_event(ev: dict[str, Any], review_ids: dict[str, int] | None = None) -> dict[str, Any]:
    etype = str(ev.get("type") or "")
    epoch = to_epoch(ev.get("epoch"))
    payload = payload_aliases(etype, epoch, ev.get("payload") or {}, review_ids or {})
    return {
        "seq": int(ev["seq"]),
        "type": etype,
        "epoch": epoch,
        "ts": iso_from_any(ev.get("ts")),
        "level": str(ev.get("level") or event_level(etype, payload)),
        "message": str(ev.get("message") or event_message(etype, epoch, payload)),
        "payload": payload,
    }


def events_payload(
    sim: Simulation, types: Sequence[str] = (), after: int = 0, limit: int = 100, order: str = "asc"
) -> dict[str, Any]:
    """types.ts ``EventPage``."""
    events = read_events(sim)
    last_seq = int(events[-1]["seq"]) if events else 0
    wanted = set(types)
    rows = [
        ev
        for ev in events
        if ev["seq"] > after and ((ev.get("type") in wanted) if wanted else ev.get("type") not in QUIET_EVENT_TYPES)
    ]
    if order == "desc":
        rows = list(reversed(rows))
    page = rows[:limit]
    from apps.sentinel.models import Review

    review_ids = dict(Review.objects.filter(simulation=sim).values_list("review_id", "pk"))
    results = [to_contract_event(ev, review_ids) for ev in page]
    for item in results:
        if item["type"] == "RunStarted":
            item["payload"].setdefault("seed", (sim.config or {}).get("seed"))
    return clean(
        {
            "simulation": sim.pk,
            "results": results,
            "last_seq": last_seq,
            "has_more": len(rows) > limit,
        }
    )


def append_api_event(sim: Simulation, etype: str, epoch: str | None, payload: dict[str, Any]) -> None:
    """Append an API-side event (e.g. ``WorkerKilled``) to ``events.jsonl``.

    Only called when no worker is alive, so the append cannot interleave with the runner's
    writes. A line truncated by the SIGKILL is terminated first so the new event parses.
    """
    path = run_directory(sim).events_path
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"ts": iso(utcnow()), "type": etype, "epoch": epoch, "payload": clean(payload)},
        separators=(",", ":"),
        allow_nan=False,
    )
    prefix = ""
    if path.is_file() and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(prefix + line + "\n")


# ============================================================================ sentinel
def _reference_window(sim: Simulation) -> dict[str, Any]:
    ds = sim.fit.dataset
    start = to_epoch(sim.config.get("start"))
    ref_to = epoch_add(start, -1) if start else None

    def count() -> int:
        try:
            df = _query(
                f"SELECT count(*) AS n FROM read_parquet({_sql_path(ds.parquet)}) WHERE issue_d < CAST(? AS TIMESTAMP)",
                [epoch_start(start)],
            )
            return int(df["n"].iloc[0])
        except Exception:  # noqa: BLE001
            return 0

    n = None
    for ev in read_events(sim):
        if ev.get("type") == "RunStarted":
            n = (ev.get("payload") or {}).get("n_reference")
            break
    if n is None:
        n = cache.memo("refcount", (cache.file_key(ds.parquet), start), count)
    return {"from": ds.issue_min or ref_to, "to": ref_to, "n": int(n or 0)}


def sentinel_payload(sim: Simulation) -> dict[str, Any]:
    """types.ts ``SentinelTimeline``."""
    from apps.sentinel.models import Review
    from apps.sentinel.services import sync_reviews

    cfg = sim.config or {}
    psi_cfg = cfg.get("psi") or {}
    thresholds = _thresholds(sim)
    df = read_sentinel(sim)
    epochs = set(completed_epochs(sim))
    features = list(psi_cfg.get("features") or [])
    features_psi: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    if not df.empty:
        df = df.assign(_epoch=df["epoch"].map(to_epoch), _psi=pd.to_numeric(df["psi"], errors="coerce"))
        if epochs:
            df = df.loc[df["_epoch"].isin(epochs)]
        seen = list(dict.fromkeys(df["feature"].astype(str)))
        features = [f for f in features if f in seen] + [f for f in seen if f not in features]
        for rec in df.sort_values(["_epoch"], kind="stable").to_dict("records"):
            psi = num(rec["_psi"])
            if psi is None:
                continue
            features_psi.append(
                {
                    "epoch": rec["_epoch"],
                    "feature": str(rec["feature"]),
                    "psi": round(psi, 6),
                    "level": _level(psi, thresholds),
                }
            )
        for epoch, grp in df.groupby("_epoch", sort=True):
            vals = grp["_psi"].to_numpy(dtype=float)
            ok = np.isfinite(vals)
            agg = num(grp["aggregate"].iloc[0]) if "aggregate" in grp.columns else None
            psi_max = float(vals[ok].max()) if ok.any() else None
            psi_mean = float(vals[ok].mean()) if ok.any() else None
            if agg is None:
                agg = psi_max if psi_cfg.get("aggregate", "max") == "max" else psi_mean
            top = str(grp["feature"].iloc[int(np.nanargmax(np.where(ok, vals, -np.inf)))]) if ok.any() else None
            bs = str(grp["breaker_state"].iloc[-1]) if "breaker_state" in grp.columns else "closed"
            aggregate.append(
                {
                    "epoch": epoch,
                    "psi_max": num(psi_max, 6),
                    "psi_mean": num(psi_mean, 6),
                    "aggregate": num(agg, 6),
                    "level": _level(agg, thresholds),
                    "breaker_state": bs if bs in ("closed", "open", "half_open") else "closed",
                    "top_feature": top,
                }
            )
    try:
        sync_reviews(sim)
    except Exception:  # noqa: BLE001
        logger.exception("review sync failed for simulation %s", sim.pk)
    review_ids = dict(Review.objects.filter(simulation=sim).values_list("review_id", "pk"))
    transitions: list[dict[str, Any]] = []
    rebased: list[str] = []
    last_review: str | None = None
    for ev in read_events(sim):
        etype = ev.get("type")
        p = ev.get("payload") or {}
        epoch = to_epoch(ev.get("epoch"))
        if (
            etype == "SentinelEvaluated"
            and p.get("breaker_before")
            and p.get("breaker_before") != p.get("breaker_after")
        ):
            before, after = str(p["breaker_before"]), str(p["breaker_after"])
            psi_val = num(p.get("psi_aggregate"))
            if after == "open":
                reason = (
                    f"aggregate PSI {psi_val:.3f} > {thresholds['breaker']}"
                    if psi_val is not None
                    else "PSI above the breaker threshold"
                )
                if before == "half_open":
                    reason += " while half-open"
            elif after == "closed":
                reason = f"{int(psi_cfg.get('recovery_readings') or 3)} consecutive normal readings"
            else:
                reason = f"{before} → {after}"
            if p.get("review_id"):
                last_review = str(p["review_id"])
            transitions.append(
                {
                    "epoch": epoch,
                    "ts": iso_from_any(ev.get("ts")),
                    "from_state": before,
                    "to_state": after,
                    "reason": reason,
                    "psi_aggregate": num(p.get("psi_aggregate"), 6),
                    "review": review_ids.get(str(p.get("review_id"))) if p.get("review_id") else None,
                    "action": None,
                }
            )
        elif etype == "BreakerTripped" and p.get("review_id"):
            last_review = str(p["review_id"])
            if transitions and transitions[-1]["to_state"] == "open" and transitions[-1]["review"] is None:
                transitions[-1]["review"] = review_ids.get(last_review)
        elif etype == "ReviewAcknowledged":
            rid = str(p.get("review_id") or last_review or "")
            action = p.get("action")
            transitions.append(
                {
                    "epoch": epoch,
                    "ts": iso_from_any(ev.get("ts")),
                    "from_state": "open",
                    "to_state": str(p.get("breaker_state") or "half_open"),
                    "reason": f"review acknowledged: {action}" + (" (auto_ack)" if p.get("auto") else ""),
                    "psi_aggregate": num(p.get("psi_aggregate"), 6),
                    "review": review_ids.get(rid),
                    "action": action if action in ("resume", "resegment", "rollback") else None,
                }
            )
            if p.get("rebased") and epoch:
                rebased.append(epoch)
    by_epoch = {a["epoch"]: a["aggregate"] for a in aggregate}
    for t in transitions:
        if t["psi_aggregate"] is None:
            t["psi_aggregate"] = by_epoch.get(t["epoch"])
    current = aggregate[-1] if aggregate else None
    reference = _reference_window(sim)
    reference["rebased_at"] = rebased
    return clean(
        {
            "simulation": sim.pk,
            "features": features,
            "thresholds": thresholds,
            "aggregate_mode": psi_cfg.get("aggregate", "max"),
            "window_months": int(psi_cfg.get("window_months") or 3),
            "reference": reference,
            "features_psi": features_psi,
            "aggregate": aggregate,
            "transitions": transitions,
            "current": {
                "epoch": current["epoch"] if current else None,
                "breaker_state": current["breaker_state"] if current else "closed",
                "level": current["level"] if current else "normal",
                "psi_aggregate": current["aggregate"] if current else None,
                "top_feature": current["top_feature"] if current else None,
            },
            "shift_counts": {
                "watch": sum(1 for a in aggregate if a["level"] == "watch"),
                "breaker": sum(1 for a in aggregate if a["level"] == "breaker"),
            },
        }
    )


def _bucket_bounds(calc, feature: str) -> list[tuple[float | None, float | None]]:
    fb = calc.bins[feature]
    if fb.kind != "numeric":
        return [(None, None)] * len(fb.labels)
    bounds = [-math.inf, *fb.edges, math.inf]
    out = [
        (None if lo == -math.inf else float(lo), None if hi == math.inf else float(hi))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    return out + [(None, None)]  # __missing__


def decision_epochs(sim: Simulation) -> list[str]:
    return run_directory(sim).decision_epochs()


def window_loan_ids(sim: Simulation, epoch: str, window_months: int) -> list[int]:
    """Loan ids of the PSI window ending at ``epoch``: the arrivals of the last
    ``window_months`` epochs *with arrivals* (the runner's ``_window``)."""
    rd = run_directory(sim)
    epochs = [e for e in decision_epochs(sim) if e <= epoch]
    files = [rd.decisions_dir / f"epoch={e}.parquet" for e in epochs[-max(int(window_months), 1) :]]
    files = [f for f in files if f.is_file()]
    if not files:
        return []
    lst = ", ".join(_sql_path(f) for f in files)
    df = _query(f"SELECT DISTINCT CAST(loan_id AS BIGINT) AS loan_id FROM read_parquet([{lst}], union_by_name=true)")
    return [int(x) for x in df["loan_id"].tolist()]


def psi_reference(sim: Simulation, epoch: str) -> str | None:
    """Window-end epoch of the PSI reference used for the reading at ``epoch``.

    ``None`` = the original reference (``issue_d < start``). A review acknowledged with
    ``rebased=true`` re-fits the reference on the most recently evaluated window (the
    runner's ``_rebase_reference``); a resume restores the checkpoint's reference.
    """
    ref: str | None = None
    rebases: list[tuple[str, str]] = []
    seen: set[str] = set()
    result: str | None = None
    for ev in read_events(sim):
        etype = ev.get("type")
        e = to_epoch(ev.get("epoch"))
        p = ev.get("payload") or {}
        if etype == "RunResumed":
            c = to_epoch(p.get("from_epoch"))
            rebases = [r for r in rebases if c is not None and r[0] <= c]
            ref = rebases[-1][1] if rebases else None
            seen = {x for x in seen if c is not None and x <= c}
        elif etype == "SentinelEvaluated" and e:
            seen.add(e)
            if e == epoch:
                result = ref
        elif etype == "ReviewAcknowledged" and p.get("rebased") and e:
            window_end = e if e in seen else _previous_epoch(sim, e)
            if window_end:
                rebases.append((e, window_end))
                ref = window_end
    return result


def _previous_epoch(sim: Simulation, epoch: str) -> str | None:
    earlier = [e for e in decision_epochs(sim) if e < epoch]
    return earlier[-1] if earlier else None


def buckets_payload(sim: Simulation, feature: str | None, epoch: str | None) -> dict[str, Any]:
    """types.ts ``PsiBucketTable``: recomputed with ``PSICalculator`` from the reference
    window (``issue_d < start``) and the epoch's PSI window of arrivals (``decisions/``)."""
    from bayesian_credit.monitoring.psi import PSICalculator
    from rest_framework.exceptions import ValidationError

    from apps.datasets.services import read_loans_by_id

    epochs = completed_epochs(sim)
    epoch = epoch_param(sim, epoch, epochs)
    psi_cfg = sim.config.get("psi") or {}
    features = list(psi_cfg.get("features") or [])
    feature = feature or (features[0] if features else None)
    if feature not in features:
        raise ValidationError({"feature": [f"not a monitored feature; one of {', '.join(features)}"]})
    window = int(psi_cfg.get("window_months") or 3)
    ds = sim.fit.dataset
    start = to_epoch(sim.config.get("start"))
    framework = FrameworkConfig.from_dict(sim.fit.framework or {})

    rebased_on = psi_reference(sim, epoch)

    def reference_calc() -> PSICalculator:
        calc = PSICalculator(features, n_bins=framework.psi.n_bins, eps=framework.psi.eps)
        if rebased_on is None:
            ref = pd.read_parquet(
                ds.parquet, columns=["issue_d", *features], filters=[("issue_d", "<", pd.Timestamp(epoch_start(start)))]
            )
        else:
            ref_ids = window_loan_ids(sim, rebased_on, window)
            ref = read_loans_by_id(ds, ref_ids, columns=["loan_id", *features])
        return calc.fit(ref)

    calc = cache.memo(
        "psi_ref", (cache.file_key(ds.parquet), sim.pk, start, rebased_on, tuple(features)), reference_calc
    )
    ids = window_loan_ids(sim, epoch, window)
    win = read_loans_by_id(ds, ids, columns=["loan_id", *features]) if ids else pd.DataFrame(columns=features)
    result = calc.psi(win)
    table = result.tables.get(feature)
    psi = num(result.per_feature.get(feature))
    bounds = _bucket_bounds(calc, feature)
    buckets = []
    if table is not None:
        for i, rec in enumerate(table.to_dict("records")):
            lo, hi = bounds[i] if i < len(bounds) else (None, None)
            buckets.append(
                {
                    "bucket": str(rec["bucket"]),
                    "lo": lo,
                    "hi": hi,
                    "expected": num(rec["expected_share"], 6),
                    "actual": num(rec["actual_share"], 6),
                    "contribution": num(rec["contribution"], 6),
                }
            )
    kind = calc.bins[feature].kind
    stored = None
    sdf = read_sentinel(sim)
    if not sdf.empty:
        hit = sdf.loc[(sdf["epoch"].map(to_epoch) == epoch) & (sdf["feature"].astype(str) == feature)]
        if not hit.empty:
            stored = num(hit["psi"].iloc[0])
    return clean(
        {
            "simulation": sim.pk,
            "feature": feature,
            "epoch": epoch,
            "kind": "numeric" if kind == "numeric" else "categorical",
            "psi": psi,
            "psi_runner": stored,
            "level": _level(psi, _thresholds(sim)),
            "n_reference": int(result.n_reference),
            "n_window": int(result.n_current),
            "reference_rebased_at": rebased_on,
            "buckets": buckets,
        }
    )


# ============================================================================ report
def uplift_pct_to_fraction(node: Any, path: str = "details") -> list[str]:
    """Convert, in place, every ``uplift_pct`` under ``node`` from the core's percent to the
    contract's fraction of ``|expected_profit_historical|``; returns the converted paths."""
    converted: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if key == "uplift_pct":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    node[key] = round(float(value) / 100.0, 8)
                    converted.append(child)
            else:
                converted += uplift_pct_to_fraction(value, child)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            converted += uplift_pct_to_fraction(value, f"{path}[{i}]")
    return converted


def _core_report(sim: Simulation) -> dict[str, Any] | None:
    """The core ``EvaluationReport`` of the run with the API's post-processing (``None`` before the
    first completed epoch).

    Built with :func:`bayesian_credit.pipeline.build_report` from the run directory and the fit's
    latest succeeded shadow-portfolio summary (the runner writes ``report.json`` without one).
    Memoised on the identity of the run files *and* of that portfolio, so a later portfolio run
    or another epoch rebuilds it; ``report.json`` is the fallback if a rebuild fails.
    """
    from apps.common import artifacts
    from apps.fits.services import latest_portfolio, load_fitted, portfolio_loan_row

    rd = run_directory(sim)
    if not completed_epochs(sim):
        return None
    job, shadow = latest_portfolio(sim.fit)
    summary_file = artifacts.portfolio_dir(sim.fit_id, job.pk) / "summary.json" if job is not None else None
    key = (
        cache.file_key(rd.report_path, rd.timeseries_path, rd.events_path, rd.runtime_path, rd.sentinel_path),
        job.pk if job is not None else None,
        cache.file_key(summary_file) if summary_file is not None else None,
    )

    def build() -> dict[str, Any]:
        from bayesian_credit.pipeline import build_report

        try:
            core = build_report(rd.path, fitted=load_fitted(sim.fit), shadow_summary=shadow, write=False)
        except Exception:  # noqa: BLE001 - serve the runner's report rather than nothing
            logger.exception("could not rebuild the report of simulation %s", sim.pk)
            core = rd.read_report()
            if not core:
                raise
        details = core.get("dimensions") or {}
        cf = details.get("counterfactual_evaluation")
        if isinstance(cf, dict) and cf.get("available") and job is not None:
            cf["portfolio_job_id"] = job.pk
            cf["portfolio_request"] = (job.params or {}).get("request") or {}
            cf["top_loans"] = [portfolio_loan_row(r) for r in cf.get("top_loans") or []]
        feasibility = details.get("computational_feasibility")
        if isinstance(feasibility, dict):  # CPU of the API process, not of the worker
            feasibility.pop("process_cpu_seconds", None)
        uplift_pct_to_fraction(details)
        return core

    return cache.memo("report", key, build)


def runtime_stats(sim: Simulation, core: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """types.ts ``RuntimeStats`` from ``runtime.json`` and the event log."""
    rt = dict(core.get("runtime") or run_directory(sim).read_runtime() or {})
    wall = num(rt.get("wall_seconds")) or 0.0
    cpu = num(rt.get("cpu_seconds")) or 0.0
    decisions = int(rt.get("n_decisions") or 0)
    types = [e.get("type") for e in events]
    rec = recovery_info(sim, events)
    return {
        "cpu_seconds": round(cpu, 3),
        "cpu_utilization": round(cpu / wall, 4) if wall > 0 else 0.0,
        "peak_memory_mb": num(rt.get("peak_rss_mb"), 1) or 0.0,
        "wall_seconds": round(wall, 3),
        "epochs": len(completed_epochs(sim)),
        "decisions": decisions,
        "decisions_per_second": num(rt.get("decisions_per_second"), 1)
        or (round(decisions / wall, 1) if wall > 0 else 0.0),
        "restarts": int(max(sim.resume_count, types.count("RunResumed"))),
        "checkpoints_written": int(rt.get("checkpoints_written") or types.count("CheckpointWritten")),
        "failed_workloads": int(types.count("RunFailed") + types.count("WorkerKilled")),
        "recovery_seconds": rec.get("recovery_seconds") if rec else None,
        "sessions": rt.get("sessions") or [],
    }


def report_payload(sim: Simulation) -> dict[str, Any] | None:
    """types.ts ``EvaluationReport`` (``status: partial`` until the run completed)."""
    core = _core_report(sim)
    if core is None:
        return None
    events = read_events(sim)
    dimensions = []
    for dim in core.get("table11") or []:
        metrics = []
        for m in dim.get("metrics") or []:
            by_policy = m.get("by_policy")
            metrics.append(
                {
                    "key": m.get("key"),
                    "label": m.get("label"),
                    "value": num(m.get("value")),
                    "unit": m.get("unit"),
                    "better": m.get("better"),
                    "baseline": num(m.get("baseline")),
                    "baseline_label": m.get("baseline_label"),
                    "by_policy": None if by_policy is None else {k: num(v) for k, v in by_policy.items()},
                    "note": m.get("note"),
                }
            )
        dimensions.append(
            {"key": dim.get("key"), "label": dim.get("label"), "axis": dim.get("axis"), "metrics": metrics}
        )
    # the runner writes report.json just before it flips progress.json to "completed", so the
    # file itself says "partial"; the run's status is the source of truth
    final = sim.status == "completed"
    return clean(
        {
            "simulation": sim.pk,
            "fit": sim.fit_id,
            "dataset": sim.fit.dataset_id,
            "generated_at": iso_from_any(core.get("generated_at")) or iso(utcnow()),
            "status": "final" if final else "partial",
            "dimensions": dimensions,
            "runtime": runtime_stats(sim, core, events),
            "policy_summary": headline(sim),
            "config": {"fit": sim.fit.config, "simulation": sim.config},
            "details": core.get("dimensions") or {},
            "config_hashes": core.get("config_hashes") or {},
        }
    )


# ============================================================================ control
def _require_run_dir(sim: Simulation) -> RunDirectory:
    rd = run_directory(sim)
    if not rd.config_path.is_file():
        raise Conflict(f"simulation {sim.pk} has no run directory")
    return rd


def pause(sim: Simulation) -> Simulation:
    refresh_status(sim)
    if sim.status not in ("running", "queued"):
        raise Conflict(f"cannot pause a {sim.status} simulation")
    _require_run_dir(sim).write_control("pause")
    return sim


def cancel(sim: Simulation) -> Simulation:
    refresh_status(sim)
    if sim.status in TERMINAL:
        raise Conflict(f"simulation already {sim.status}")
    rd = _require_run_dir(sim)
    rd.write_control("cancel")
    if sim.status == "interrupted" or not sim.worker_alive():
        # no worker will consume the command: record the cancellation ourselves
        if sim.job is not None and sim.job.is_active:
            Job.objects.filter(pk=sim.job.pk).update(
                status=Job.Status.CANCELLED, finished_at=utcnow(), message="cancelled by operator"
            )
        rd.write_progress(status="cancelled", finished_at=iso(utcnow()), message="cancelled by operator")
        append_api_event(
            sim,
            "RunCancelled",
            to_epoch((sim.progress or {}).get("current_epoch")),
            {"reason": "operator", "worker": "dead"},
        )
        refresh_status(sim)
    return sim


def resume(sim: Simulation) -> Simulation:
    refresh_status(sim)
    rd = _require_run_dir(sim)
    if sim.status == "paused":
        rd.write_control("resume")
        return sim
    if sim.status == "awaiting_review":
        raise Conflict("acknowledge the pending sentinel review to resume")
    if sim.status not in Simulation.RESUMABLE_STATUSES:
        raise Conflict(f"cannot resume a {sim.status} simulation")
    return relaunch(sim)


def relaunch(sim: Simulation) -> Simulation:
    """Start a new worker in resume mode (continues from ``checkpoints/latest.pkl``)."""
    rd = run_directory(sim)
    rd.control_path.unlink(missing_ok=True)  # a stale pause/cancel must not stop the new worker
    job = JobRunner.create(
        Job.Kind.SIMULATION,
        target_type="simulation",
        target_id=sim.pk,
        params={"mode": "resume", "previous_job": sim.job_id},
    )
    rec = dict(sim.recovery or {})
    if rec.get("killed_at") and not rec.get("resumed_at"):
        rec["resumed_at"] = iso(utcnow())
        sim.recovery = rec
    sim.job = job
    sim.resume_count += 1
    sim.status = Simulation.Status.QUEUED
    sim.finished_at = None
    sim.save(update_fields=["job", "resume_count", "status", "recovery", "finished_at"])
    rd.write_progress(status="queued", message="resume requested", finished_at=None, error=None)
    JobRunner.launch(job)
    sim.refresh_from_db()
    return refresh_status(sim)


def chaos(sim: Simulation) -> Simulation:
    """SIGKILL the worker mid-run (G.6 workload-termination experiment)."""
    refresh_status(sim)
    job = sim.job
    if sim.status not in ("running", "paused", "awaiting_review") or job is None or not worker_alive(job):
        raise Conflict("chaos needs a running worker")
    pid = job.pid
    prog = run_directory(sim).read_progress()
    killed = kill_worker(job, signal.SIGKILL)
    if not killed:
        raise Conflict("the worker exited before it could be killed")
    mark_failed(job, "chaos: worker killed")
    sim.recovery = None
    _record_kill(sim, prog, detected=False)
    sim.status = Simulation.Status.INTERRUPTED
    sim.save(update_fields=["recovery", "status"])
    append_api_event(
        sim,
        "WorkerKilled",
        to_epoch(prog.get("current_epoch")),
        {"pid": pid, "signal": "SIGKILL", "reason": "chaos", "epoch_index": prog.get("epoch_index")},
    )
    return refresh_status(sim)


def bulk_refresh(sims: Iterable[Simulation]) -> None:
    """Refresh status and review queue of the live runs of a list page."""
    from apps.sentinel.services import sync_reviews

    for sim in sims:
        if sim.status in LIVE or sim.status == "interrupted":
            try:
                refresh_status(sim)
                sync_reviews(sim)
            except Exception:  # noqa: BLE001 - one broken run directory must not break the list
                logger.exception("could not refresh simulation %s", sim.pk)
