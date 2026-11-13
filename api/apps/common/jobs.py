"""Background jobs: create, launch, execute, reap.

* :class:`JobRunner.launch` — hands the job to the backend (:mod:`apps.common.backends`): a
  Temporal ``BackgroundJob`` workflow by default, inline under ``JOBS_SYNC``.
* :func:`execute` — runs in the worker (``manage.py run_job <id>``, supervised by the Temporal
  activity): marks the job running, dispatches on ``job.kind`` to the app handler
  (``HANDLERS``), and records success / failure (traceback in ``error``, last lines in
  ``log_tail``). The process's pid is stored on the row; a simulation job's process *is* the
  simulation worker, so the chaos kill (``POST /simulations/{id}/chaos/``) SIGKILLs the run
  mid-epoch.
* :func:`reap_lost_workers` — called lazily by list/detail views (never at import time): an
  active job its backend reports dead is marked failed with ``message="worker lost"``.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections.abc import Iterable
from typing import Any

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils.module_loading import import_string

from .backends import get_backend
from .models import Job
from .temporal.workflows import WORKER_LOST
from .utils import utcnow

logger = logging.getLogger("apps.jobs")

#: kind -> dotted path of ``handler(job, reporter) -> None``.
HANDLERS: dict[str, str] = {
    Job.Kind.INGEST: "apps.datasets.jobs.run_ingest",
    Job.Kind.FIT: "apps.fits.jobs.run_fit",
    Job.Kind.SHADOW_PORTFOLIO: "apps.fits.jobs.run_portfolio",
    Job.Kind.SIMULATION: "apps.simulations.jobs.run_simulation",
}

#: kind -> dotted path of ``on_failure(job, exc) -> None`` (optional clean-up hooks).
FAILURE_HOOKS: dict[str, str] = {
    Job.Kind.INGEST: "apps.datasets.jobs.on_failure",
    Job.Kind.FIT: "apps.fits.jobs.on_failure",
    Job.Kind.SIMULATION: "apps.simulations.jobs.on_failure",
}


class JobCancelled(Exception):
    """Raised by a handler (via :meth:`ProgressReporter.check_cancelled`) to stop early."""


# ============================================================================ progress
class ProgressReporter:
    """Writes progress / message / log tail to the Job row, at most every
    ``settings.JOB_PROGRESS_INTERVAL`` seconds for :meth:`progress` (log lines are always
    written immediately — they are milestones)."""

    def __init__(self, job: Job, interval: float | None = None) -> None:
        self.job = job
        self.interval = float(settings.JOB_PROGRESS_INTERVAL if interval is None else interval)
        self.lines: list[str] = list(job.log_tail or [])
        self._last_write = 0.0
        self._dirty = False

    def _stamp(self, message: str) -> str:
        return f"[{utcnow().strftime('%H:%M:%S')}] {message}"

    def log(self, message: str, progress: float | None = None) -> None:
        message = str(message).strip()
        logger.info("job %s: %s", self.job.pk, message)
        for part in message.splitlines() or [""]:
            self.lines.append(self._stamp(part))
        self.lines = self.lines[-Job.LOG_TAIL_LINES :]
        self.job.message = message.splitlines()[-1] if message else ""
        if progress is not None:
            self.job.progress = min(max(float(progress), 0.0), 1.0)
        self._dirty = True
        self.flush(force=True)

    def progress(self, fraction: float, message: str | None = None) -> None:
        self.job.progress = min(max(float(fraction), 0.0), 1.0)
        if message is not None and message != self.job.message:
            self.job.message = str(message)
        self._dirty = True
        self.flush()

    def flush(self, force: bool = False) -> None:
        if not self._dirty:
            return
        now = time.monotonic()
        if not force and now - self._last_write < self.interval:
            return
        self.job.log_tail = list(self.lines)
        Job.objects.filter(pk=self.job.pk).update(
            progress=self.job.progress, message=self.job.message, log_tail=self.job.log_tail
        )
        self._last_write = now
        self._dirty = False

    def check_cancelled(self) -> None:
        status = Job.objects.filter(pk=self.job.pk).values_list("status", flat=True).first()
        if status == Job.Status.CANCELLED:
            raise JobCancelled()


# ============================================================================ liveness
def worker_alive(job: Job) -> bool:
    """A worker is executing ``job`` right now (per the job backend, e.g. Temporal)."""
    return bool(job.is_active and get_backend().probe(job).alive)


def kill_worker(job: Job) -> bool:
    """SIGKILL the job's worker process mid-run (chaos); ``True`` once it is gone."""
    if not job.is_active:
        return False
    return get_backend().kill(job)


def mark_failed(job: Job, message: str, error: str = "") -> Job:
    """Mark an active job failed (no-op if it already finished)."""
    with transaction.atomic():
        fresh = Job.objects.select_for_update().get(pk=job.pk)
        if fresh.status not in Job.ACTIVE:
            return fresh
        fresh.status = Job.Status.FAILED
        fresh.message = message
        if error:
            fresh.error = error
        fresh.finished_at = utcnow()
        tail = list(fresh.log_tail or []) + [f"[{fresh.finished_at.strftime('%H:%M:%S')}] {message}"]
        fresh.log_tail = tail[-Job.LOG_TAIL_LINES :]
        fresh.save(update_fields=["status", "message", "error", "finished_at", "log_tail"])
    return fresh


def reap_lost_workers(jobs: Iterable[Job] | None = None) -> int:
    """Mark active jobs that no worker will ever finish as failed ("worker lost").

    With Temporal that means the job's workflow closed (or never existed) while the row is
    still queued / running; a workflow that is merely waiting for a worker or retrying is
    left alone. Inline (``JOBS_SYNC``) jobs are never reaped.
    """
    if jobs is None:
        jobs = Job.objects.filter(status__in=Job.ACTIVE)
    backend = get_backend()
    lost = 0
    for job in jobs:
        if job.status not in Job.ACTIVE:
            continue
        if backend.probe(job).state != "dead":
            continue
        fresh = Job.objects.filter(pk=job.pk).first()
        if fresh is not None and fresh.status in Job.ACTIVE:
            mark_failed(fresh, WORKER_LOST)
            lost += 1
    return lost


# ============================================================================ runner
class JobRunner:
    """Creates jobs and hands them to the configured backend (see :mod:`apps.common.backends`)."""

    @staticmethod
    def create(kind: str, *, target_type: str = "", target_id: int | None = None, params: dict | None = None) -> Job:
        return Job.objects.create(kind=kind, target_type=target_type, target_id=target_id, params=params or {})

    @staticmethod
    def launch(job: Job) -> Job:
        return get_backend().launch(job)


def execute(job: Job | int, *, inline: bool = False) -> Job:
    """Run a job to completion in this process (the worker side of :class:`JobRunner`)."""
    if not isinstance(job, Job):
        job = Job.objects.get(pk=job)
    if job.status not in Job.ACTIVE:
        logger.warning("job %s is %s; nothing to do", job.pk, job.status)
        return job
    job.status = Job.Status.RUNNING
    job.started_at = utcnow()
    job.finished_at = None
    job.pid = None if inline else os.getpid()
    job.save(update_fields=["status", "started_at", "finished_at", "pid"])
    reporter = ProgressReporter(job)
    reporter.log(f"{job.kind} job started" + ("" if inline else f" (pid {os.getpid()})"))
    try:
        handler = import_string(HANDLERS[job.kind])
        handler(job, reporter)
    except JobCancelled:
        job.status = Job.Status.CANCELLED
        reporter.log("cancelled")
    except BaseException as exc:  # noqa: BLE001 - every failure must be recorded on the row
        tb = traceback.format_exc()
        logger.error("job %s failed: %s\n%s", job.pk, exc, tb)
        job.status = Job.Status.FAILED
        job.error = tb
        for line in tb.strip().splitlines()[-6:]:
            reporter.lines.append(reporter._stamp(line))
        reporter.log(f"failed: {type(exc).__name__}: {exc}")
        hook = FAILURE_HOOKS.get(job.kind)
        if hook:
            try:
                import_string(hook)(job, exc)
            except Exception:  # noqa: BLE001
                logger.exception("failure hook for job %s raised", job.pk)
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise  # recorded above; never swallow an interpreter exit
    else:
        current = Job.objects.filter(pk=job.pk).values_list("status", flat=True).first()
        if current == Job.Status.CANCELLED:
            job.status = Job.Status.CANCELLED
        else:
            job.status = Job.Status.SUCCEEDED
            job.progress = 1.0
            reporter.log(f"{job.kind} job succeeded", progress=1.0)
    finally:
        job.finished_at = utcnow()
        reporter.flush(force=True)
        job.log_tail = reporter.lines
        job.save(update_fields=["status", "progress", "message", "error", "finished_at", "log_tail"])
        if not inline:
            close_old_connections()
    return job


def job_target(job: Job) -> dict[str, Any] | None:
    if not job.target_type or job.target_id is None:
        return None
    return {"type": job.target_type, "id": int(job.target_id)}
