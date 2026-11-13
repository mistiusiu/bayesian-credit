"""A job backend for tests: jobs run inline (``JOBS_SYNC``), and a job whose ``pid`` points at
a local stand-in process (``python -c 'sleep' run_job <id>``) is treated as a live worker that
the chaos kill SIGKILLs — so liveness, reaping and chaos can be exercised without a Temporal
server (``test_temporal.py`` covers the real one)."""

from __future__ import annotations

import os
import signal
import time

from apps.common.backends import InlineBackend, WorkerProbe
from apps.common.models import Job


def pid_alive(pid: int | None) -> bool:
    """True if ``pid`` is a live (non-zombie) process; reaps our own exited children."""
    if not pid or pid <= 0:
        return False
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
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
            if fh.read().rsplit(")", 1)[1].split()[0] in {"Z", "X"}:
                return False
    except (OSError, IndexError):
        pass
    return True


def _is_stand_in(job: Job) -> bool:
    try:
        with open(f"/proc/{job.pid}/cmdline", "rb") as fh:
            argv = fh.read().split(b"\0")
    except OSError:
        return False
    return b"run_job" in argv and str(job.pk).encode() in argv


class StandInBackend(InlineBackend):
    def probe(self, job: Job) -> WorkerProbe:
        if job.pid is None:
            return super().probe(job)
        return WorkerProbe("alive" if pid_alive(job.pid) and _is_stand_in(job) else "dead")

    def kill(self, job: Job) -> bool:
        if not self.probe(job).alive:
            return False
        os.kill(job.pid, signal.SIGKILL)
        for _ in range(100):
            if not pid_alive(job.pid):
                return True
            time.sleep(0.02)
        return False
