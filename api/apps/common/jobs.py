"""Job runner: launch background work inline (``JOBS_SYNC``) or as a detached subprocess.

* :class:`JobRunner.launch` — ``settings.JOBS_SYNC`` → :func:`execute` inline; otherwise
  ``python manage.py run_job <id>`` in a new session (``start_new_session=True``), stdout and
  stderr appended to ``<ARTIFACT_ROOT>/jobs/<id>.log``; the child's pid is stored on the row.
  A daemon thread ``wait()``s on the child so finished workers never linger as zombies.
* :func:`execute` — runs in the worker: marks the job running, dispatches on ``job.kind`` to the
  app handler (``HANDLERS``), and records success / failure (traceback in ``error``, last lines
  in ``log_tail``). A simulation job's process *is* the simulation worker, so SIGKILLing its pid
  (``POST /simulations/{id}/chaos/``) kills the run mid-epoch.
* :func:`reap_lost_workers` — called lazily by list/detail views (never at import time): an
  active job whose pid is gone is marked failed with ``message="worker lost"``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils.module_loading import import_string

from . import artifacts
from .models import Job
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

WORKER_LOST = "worker lost"
NEVER_STARTED_AFTER = timedelta(minutes=10)


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


# ============================================================================ process helpers
def pid_alive(pid: int | None) -> bool:
    """True if ``pid`` is a live (non-zombie) process."""
    if not pid or pid <= 0:
        return False
    try:  # reap our own exited child first (otherwise it lingers as a zombie)
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
        if state in {"Z", "X"}:
            return False
    except (OSError, IndexError):
        pass
    return True


def worker_alive(job: Job) -> bool:
    """A job's worker is alive when its pid is a live process running ``run_job <id>``."""
    if not job.pid or job.pid == os.getpid():
        return False
    if not pid_alive(job.pid):
        return False
    try:
        with open(f"/proc/{job.pid}/cmdline", "rb") as fh:
            argv = fh.read().split(b"\0")
        if b"run_job" in argv:
            return str(job.pk).encode() in argv
        return False  # pid reused by an unrelated process
    except OSError:
        return True  # no /proc (non-Linux): trust kill(pid, 0)


def kill_worker(job: Job, sig: int = signal.SIGKILL) -> bool:
    """Send ``sig`` to the job's worker (its whole process group when it leads one)."""
    if not worker_alive(job):
        return False
    try:
        if os.getpgid(job.pid) == job.pid:
            os.killpg(job.pid, sig)
        else:
            os.kill(job.pid, sig)
    except ProcessLookupError:
        return False
    # give the kernel a moment and reap the child so the pid is not reported alive
    for _ in range(50):
        if not pid_alive(job.pid):
            break
        time.sleep(0.02)
    return True


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
    """Mark active jobs whose worker process is gone as failed ("worker lost").

    Inline (``JOBS_SYNC``) jobs have no pid and are never reaped. Queued jobs that never got
    a pid within ``NEVER_STARTED_AFTER`` (the launching process died) are failed as well.
    """
    if jobs is None:
        jobs = Job.objects.filter(status__in=Job.ACTIVE)
    lost = 0
    now = utcnow()
    for job in jobs:
        if job.status not in Job.ACTIVE:
            continue
        if job.pid is None:
            if not settings.JOBS_SYNC and job.created_at and now - job.created_at > NEVER_STARTED_AFTER:
                mark_failed(job, "worker never started")
                lost += 1
            continue
        if job.pid == os.getpid():
            continue
        if not worker_alive(job):
            fresh = Job.objects.filter(pk=job.pk).first()
            if fresh is not None and fresh.status in Job.ACTIVE:
                mark_failed(fresh, WORKER_LOST)
                lost += 1
    return lost


# ============================================================================ runner
class JobRunner:
    """Starts jobs (see module docstring)."""

    @staticmethod
    def create(kind: str, *, target_type: str = "", target_id: int | None = None, params: dict | None = None) -> Job:
        return Job.objects.create(kind=kind, target_type=target_type, target_id=target_id, params=params or {})

    @staticmethod
    def launch(job: Job) -> Job:
        if settings.JOBS_SYNC:
            execute(job, inline=True)
            job.refresh_from_db()
            return job
        log_path = artifacts.job_log_path(job.pk)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "DJANGO_SETTINGS_MODULE": os.environ.get("DJANGO_SETTINGS_MODULE", "config.settings"),
                "ARTIFACT_ROOT": str(settings.ARTIFACT_ROOT),
                "DATA_ROOT": str(settings.DATA_ROOT),
                "LENDING_CLUB_PATH": str(settings.LENDING_CLUB_PATH),
                "JOBS_SYNC": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        try:
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(
                    [sys.executable, "manage.py", "run_job", str(job.pk)],
                    cwd=str(settings.BASE_DIR),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
        except OSError as exc:
            return mark_failed(job, f"could not start worker: {exc}", traceback.format_exc())
        threading.Thread(target=proc.wait, name=f"reap-job-{job.pk}", daemon=True).start()
        Job.objects.filter(pk=job.pk).update(pid=proc.pid)
        job.pid = proc.pid
        return job


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
