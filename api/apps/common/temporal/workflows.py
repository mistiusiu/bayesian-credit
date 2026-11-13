"""Workflow definitions. Deterministic and Django-free (they run in the workflow sandbox):
activities are referenced by name, inputs and outputs are plain dataclasses.

``BackgroundJob`` — one per :class:`~apps.common.models.Job` (ingest, fit, shadow portfolio,
simulation). It runs the ``run_job`` activity, which supervises ``manage.py run_job <id>`` in a
child process on whichever worker picked it up, heartbeating while the child lives:

* the child settles the Job row itself (succeeded / failed / cancelled), exactly as before;
* a child that dies without settling (SIGKILL, OOM) or a worker that stops heartbeating
  (pod gone) is a *lost worker*: Temporal retries the activity on another worker up to
  ``max_attempts`` times (the next attempt re-runs the job), then ``mark_job_lost`` fails the
  row with ``"worker lost"``. Simulations use one attempt: a lost simulation becomes
  ``interrupted`` and is resumed from its checkpoint by the operator (G.6);
* the ``chaos`` update cancels the activity, whose supervisor SIGKILLs the child mid-epoch,
  and returns once the process is gone.

``ArtifactGC`` — run by a Temporal Schedule (``TEMPORAL_GC_SCHEDULE``): removes artifact
directories and job logs no database row refers to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError

JOB_WORKFLOW = "BackgroundJob"
GC_WORKFLOW = "ArtifactGC"
#: ``ApplicationError.type`` of a job whose worker recorded a failure: retrying cannot help.
JOB_FAILED = "JobFailed"
WORKER_LOST = "worker lost"
CHAOS_KILLED = "chaos: worker killed"


@dataclass
class JobInput:
    job_id: int
    kind: str
    max_attempts: int = 1
    heartbeat_timeout: float = 60.0


@dataclass
class JobOutcome:
    job_id: int
    status: str
    message: str = ""


@dataclass
class MarkLostInput:
    job_id: int
    message: str


@dataclass
class ChaosResult:
    killed: bool
    message: str = ""


@dataclass
class GCInput:
    apply: bool = True


@dataclass
class GCResult:
    removed: list[str]


@workflow.defn(name=JOB_WORKFLOW)
class BackgroundJobWorkflow:
    def __init__(self) -> None:
        self._activity: workflow.ActivityHandle | None = None
        self._kill_requested = False
        self._killed = False
        self._done = False

    @workflow.run
    async def run(self, inp: JobInput) -> JobOutcome:
        self._activity = workflow.start_activity(
            "run_job",
            inp,
            result_type=JobOutcome,
            start_to_close_timeout=timedelta(days=7),
            heartbeat_timeout=timedelta(seconds=inp.heartbeat_timeout),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                maximum_attempts=max(int(inp.max_attempts), 1),
                non_retryable_error_types=[JOB_FAILED],
            ),
            cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
        )
        try:
            return await self._activity
        except ActivityError as err:
            if self._kill_requested and isinstance(err.cause, CancelledError):
                self._killed = True
                await self._mark_lost(inp.job_id, CHAOS_KILLED)
                return JobOutcome(inp.job_id, "killed", CHAOS_KILLED)
            if isinstance(err.cause, ApplicationError) and err.cause.type == JOB_FAILED:
                return JobOutcome(inp.job_id, "failed", err.cause.message)
            await self._mark_lost(inp.job_id, WORKER_LOST)
            return JobOutcome(inp.job_id, "lost", WORKER_LOST)
        finally:
            self._done = True

    async def _mark_lost(self, job_id: int, message: str) -> None:
        await workflow.execute_activity(
            "mark_job_lost",
            MarkLostInput(job_id, message),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=RetryPolicy(maximum_attempts=20, maximum_interval=timedelta(seconds=30)),
        )

    @workflow.update(name="chaos")
    async def chaos(self) -> ChaosResult:
        """SIGKILL the job's worker process mid-run and wait until it is gone."""
        if self._done or self._activity is None:
            return ChaosResult(False, "the job already finished")
        self._kill_requested = True
        self._activity.cancel()
        await workflow.wait_condition(lambda: self._done)
        if self._killed:
            return ChaosResult(True, CHAOS_KILLED)
        return ChaosResult(False, "the worker exited before it could be killed")


@workflow.defn(name=GC_WORKFLOW)
class ArtifactGCWorkflow:
    @workflow.run
    async def run(self, inp: GCInput) -> GCResult:
        return await workflow.execute_activity(
            "gc_artifacts",
            inp,
            result_type=GCResult,
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
