"""Activities executed by ``manage.py temporal_worker`` (synchronous, on a thread pool).

``run_job`` is a supervisor: the job itself runs in a child ``manage.py run_job <id>
--supervised`` process (stdout/stderr appended to ``<ARTIFACT_ROOT>/jobs/<id>.log``), so a
multi-GB fit is returned to the OS when it ends, a simulation can be SIGKILLed mid-epoch
without taking the worker down, and the child's pid on the Job row is a real process. The
supervisor heartbeats every half second; a cancellation (the ``chaos`` update) SIGKILLs the child,
a worker shutdown sends SIGTERM first so a simulation checkpoints before it exits.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time

from django.conf import settings
from django.db import close_old_connections
from temporalio import activity
from temporalio.exceptions import ApplicationError

from apps.common import artifacts
from apps.common.models import Job

from .workflows import JOB_FAILED, GCInput, GCResult, JobInput, JobOutcome, MarkLostInput

logger = logging.getLogger("apps.jobs")

HEARTBEAT_EVERY = 0.5
#: SIGTERM → SIGKILL grace on worker shutdown (the simulation runner checkpoints on SIGTERM).
SHUTDOWN_GRACE = 45.0


def _child_env() -> dict[str, str]:
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
    return env


def _signal_child(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)  # the child leads its own session / process group
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.send_signal(sig)


def _stop_child(proc: subprocess.Popen, *, graceful: bool) -> None:
    if proc.poll() is not None:
        return
    if graceful:
        _signal_child(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=SHUTDOWN_GRACE)
            return
        except subprocess.TimeoutExpired:
            pass
    _signal_child(proc, signal.SIGKILL)
    proc.wait()


def _log(job_id: int, line: str) -> None:
    path = artifacts.job_log_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"[supervisor {socket.gethostname()}] {line}\n")


@activity.defn(name="run_job")
def run_job(inp: JobInput) -> JobOutcome:
    info = activity.info()
    close_old_connections()
    job = Job.objects.filter(pk=inp.job_id).first()
    if job is None:
        raise ApplicationError(f"job {inp.job_id} does not exist", type=JOB_FAILED, non_retryable=True)
    if job.status not in Job.ACTIVE:
        return JobOutcome(job.pk, job.status, job.message)  # settled before this attempt started
    _log(job.pk, f"attempt {info.attempt} of workflow {info.workflow_id}")
    log_path = artifacts.job_log_path(job.pk)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "manage.py", "run_job", str(job.pk), "--supervised"],
            cwd=str(settings.BASE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_child_env(),
        )
    # Cancellation is raised into this thread asynchronously; shield the supervision loop so
    # the child is always reaped first. A pending CancelledError is re-raised on exit.
    with activity.shield_thread_cancel_exception():
        try:
            while proc.poll() is None:
                if activity.is_cancelled():
                    details = activity.cancellation_details()
                    graceful = bool(details and details.worker_shutdown)
                    _log(job.pk, f"cancelled ({'worker shutdown' if graceful else 'kill'}): stopping pid {proc.pid}")
                    _stop_child(proc, graceful=graceful)
                    break
                if activity.is_worker_shutdown():
                    _log(job.pk, f"worker shutting down: SIGTERM pid {proc.pid}")
                    _stop_child(proc, graceful=True)
                    break
                activity.heartbeat({"pid": proc.pid, "host": socket.gethostname()})
                time.sleep(HEARTBEAT_EVERY)
        finally:
            _stop_child(proc, graceful=False)
    rc = proc.returncode
    close_old_connections()
    job.refresh_from_db()
    if job.status == Job.Status.FAILED:
        raise ApplicationError(f"job {job.pk} failed: {job.message}", type=JOB_FAILED, non_retryable=True)
    if job.status in Job.FINISHED:
        return JobOutcome(job.pk, job.status, job.message)
    if activity.is_worker_shutdown():
        raise ApplicationError(f"worker shut down while job {job.pk} was {job.status}", type="WorkerShutdown")
    # still queued / running: the child died without recording an outcome (signal, OOM)
    _log(job.pk, f"child exited with {rc} without settling the job")
    raise ApplicationError(f"job {job.pk} worker exited with code {rc} without settling", type="WorkerLost")


@activity.defn(name="mark_job_lost")
def mark_job_lost(inp: MarkLostInput) -> None:
    from apps.common.jobs import mark_failed

    close_old_connections()
    job = Job.objects.filter(pk=inp.job_id).first()
    if job is not None:
        mark_failed(job, inp.message)


@activity.defn(name="gc_artifacts")
def gc_artifacts(inp: GCInput) -> GCResult:
    close_old_connections()
    orphans = artifacts.find_orphans()
    for path in orphans:
        logger.info("%s orphan %s", "deleting" if inp.apply else "found", path)
        if inp.apply:
            artifacts.remove(path)
    return GCResult(removed=[str(p) for p in orphans])


ACTIVITIES = [run_job, mark_job_lost, gc_artifacts]
