"""Where background jobs run (``settings.JOBS_BACKEND``).

* :class:`TemporalBackend` (default) — every job is a ``BackgroundJob`` workflow on the Temporal
  server, executed by ``manage.py temporal_worker`` processes. Worker liveness, lost-worker
  detection and the chaos kill all go through Temporal, so workers can live in other
  containers, pods or nodes than the API.
* :class:`InlineBackend` (``JOBS_SYNC``) — runs the job inside the request (tests,
  ``manage.py demo --inline``).
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from functools import cache
from typing import Any

from django.conf import settings
from django.utils.module_loading import import_string

from .models import Job

logger = logging.getLogger("apps.jobs")

#: Attempts per job kind: a lost ingest / fit / portfolio worker is retried on another worker;
#: a lost simulation is left ``interrupted`` for the operator to resume from its checkpoint.
MAX_ATTEMPTS: dict[str, int] = {
    Job.Kind.INGEST: 3,
    Job.Kind.FIT: 3,
    Job.Kind.SHADOW_PORTFOLIO: 3,
    Job.Kind.SIMULATION: 1,
}


@dataclass
class WorkerProbe:
    """``alive``: a worker is executing the job · ``waiting``: the job is owned by the
    orchestrator but no worker holds it right now · ``dead``: nobody will ever finish it ·
    ``unknown``: the orchestrator could not be asked."""

    state: str
    detail: str = ""

    @property
    def alive(self) -> bool:
        return self.state == "alive"


class JobBackend:
    name = "base"

    def launch(self, job: Job) -> Job:
        raise NotImplementedError

    def probe(self, job: Job) -> WorkerProbe:
        raise NotImplementedError

    def kill(self, job: Job) -> bool:
        """SIGKILL the job's worker process mid-run; ``True`` once it is gone."""
        return False

    def health(self) -> dict[str, Any]:
        return {"backend": self.name, "temporal": None}

    def links(self, job: Job) -> dict[str, Any]:
        return {"workflow_id": job.workflow_id or None, "temporal_url": None}


class InlineBackend(JobBackend):
    name = "inline"

    def launch(self, job: Job) -> Job:
        from .jobs import execute

        execute(job, inline=True)
        job.refresh_from_db()
        return job

    def probe(self, job: Job) -> WorkerProbe:
        return WorkerProbe("unknown", "inline jobs run inside the request")


class TemporalBackend(JobBackend):
    name = "temporal"

    def launch(self, job: Job) -> Job:
        from .jobs import mark_failed
        from .temporal import client
        from .temporal.workflows import JobInput

        wid = client.workflow_id(job.pk, job.kind)
        Job.objects.filter(pk=job.pk).update(workflow_id=wid)
        job.workflow_id = wid
        inp = JobInput(
            job_id=job.pk,
            kind=job.kind,
            max_attempts=MAX_ATTEMPTS.get(job.kind, 1),
            heartbeat_timeout=settings.TEMPORAL_HEARTBEAT_TIMEOUT,
        )
        try:
            client.start_job(wid, inp)
        except Exception as exc:  # noqa: BLE001 - the request must answer; the job records why
            logger.exception("could not start workflow %s", wid)
            return mark_failed(
                job, f"could not start Temporal workflow at {settings.TEMPORAL_ADDRESS}: {exc}", traceback.format_exc()
            )
        return job

    def probe(self, job: Job) -> WorkerProbe:
        from .temporal import client

        if not job.workflow_id:
            return WorkerProbe("dead", "no workflow was ever started")
        try:
            p = client.probe(job.workflow_id)
        except Exception as exc:  # noqa: BLE001 - an unreachable server must not break reads
            logger.warning("could not describe workflow %s: %s", job.workflow_id, exc)
            return WorkerProbe("unknown", str(exc))
        if p.state in ("closed", "missing", "lost"):
            return WorkerProbe("dead", f"workflow {p.state}")
        detail = f"attempt {p.attempt}" if p.attempt else ""
        if p.worker:
            detail = f"{detail} on {p.worker}".strip()
        return WorkerProbe(p.state, detail)

    def kill(self, job: Job) -> bool:
        from .temporal import client

        if not job.workflow_id:
            return False
        return client.chaos(job.workflow_id).killed

    def health(self) -> dict[str, Any]:
        from .temporal import client

        try:
            workers = client.task_queue_workers(timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            return {"backend": self.name, "temporal": client.summary(connected=False, workers=[], error=str(exc))}
        return {"backend": self.name, "temporal": client.summary(connected=True, workers=workers, error=None)}

    def links(self, job: Job) -> dict[str, Any]:
        from .temporal import client

        wid = job.workflow_id or None
        return {"workflow_id": wid, "temporal_url": client.ui_url(wid) if wid else None}


@cache
def _load(path: str) -> JobBackend:
    return import_string(path)()


def get_backend() -> JobBackend:
    path = settings.JOBS_BACKEND or (
        "apps.common.backends.InlineBackend" if settings.JOBS_SYNC else "apps.common.backends.TemporalBackend"
    )
    return _load(path)
