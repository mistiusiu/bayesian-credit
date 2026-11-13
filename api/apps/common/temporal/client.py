"""The API's Temporal client.

Django views are synchronous and the Temporal SDK is asyncio, so each process keeps one
:class:`temporalio.client.Client` on a private event loop in a daemon thread and submits
coroutines to it (:func:`call`). The loop is created lazily and re-created after a fork.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

from django.conf import settings
from temporalio.api.enums.v1 import EventType, PendingActivityState, TaskQueueType
from temporalio.api.taskqueue.v1 import TaskQueue
from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
    ScheduleUpdate,
    ScheduleUpdateInput,
    WorkflowExecutionStatus,
    WorkflowHandle,
)
from temporalio.service import RPCError, RPCStatusCode

from .workflows import GC_WORKFLOW, JOB_WORKFLOW, ChaosResult, GCInput, JobInput

GC_SCHEDULE_ID = "bayesian-credit-artifact-gc"


class _Bridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pid: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Client | None = None

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._pid != os.getpid():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="temporal-client", daemon=True).start()
                self._loop, self._pid, self._client = loop, os.getpid(), None
            return self._loop

    async def client(self) -> Client:
        if self._client is None:
            self._client = await connect()
        return self._client


_bridge = _Bridge()


async def connect(identity: str | None = None) -> Client:
    api_key = settings.TEMPORAL_API_KEY or None
    return await Client.connect(
        settings.TEMPORAL_ADDRESS,
        namespace=settings.TEMPORAL_NAMESPACE,
        api_key=api_key,
        tls=bool(api_key),
        identity=identity,
        connect_timeout=timedelta(seconds=settings.TEMPORAL_RPC_TIMEOUT),
    )


def call[T](fn: Callable[[Client], Awaitable[T]], timeout: float | None = None) -> T:
    """Run ``await fn(client)`` on the client loop and wait for the result."""

    async def run() -> T:
        return await fn(await _bridge.client())

    future = asyncio.run_coroutine_threadsafe(run(), _bridge.loop())
    try:
        return future.result(timeout=settings.TEMPORAL_RPC_TIMEOUT if timeout is None else timeout)
    except TimeoutError:
        future.cancel()
        raise


def workflow_id(job_id: int, kind: str) -> str:
    """Unique across database resets (primary keys restart, Temporal history does not)."""
    return f"bc-{kind.replace('_', '-')}-{job_id}-{uuid.uuid4().hex[:10]}"


def ui_url(wid: str) -> str:
    return f"{settings.TEMPORAL_UI_URL}/namespaces/{quote(settings.TEMPORAL_NAMESPACE)}/workflows/{quote(wid)}"


def start_job(wid: str, inp: JobInput) -> None:
    async def start(client: Client) -> None:
        await client.start_workflow(
            JOB_WORKFLOW,
            inp,
            id=wid,
            task_queue=settings.TEMPORAL_TASK_QUEUE,
            static_summary=f"{inp.kind} job #{inp.job_id}",
        )

    call(start)


@dataclass
class Probe:
    """What Temporal knows about a job's workflow.

    ``state``: ``alive`` (a worker holds the job), ``waiting`` (open, but no worker has the
    activity — queued, or between attempts), ``lost`` (the job will not run again; the workflow
    is recording that), ``closed`` (the workflow ended), ``missing`` (no such workflow).
    """

    state: str
    attempt: int | None = None
    worker: str | None = None
    last_heartbeat: datetime | None = None


_RUN_JOB_GAVE_UP = {
    EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED: "activity_task_failed_event_attributes",
    EventType.EVENT_TYPE_ACTIVITY_TASK_TIMED_OUT: "activity_task_timed_out_event_attributes",
    EventType.EVENT_TYPE_ACTIVITY_TASK_CANCELED: "activity_task_canceled_event_attributes",
}


async def _run_job_gave_up(handle: WorkflowHandle) -> bool:
    """``run_job`` failed, timed out or was cancelled for good (retries happen inside one
    scheduled activity, so this is final) — the open workflow is only waiting for a worker
    to record it."""
    scheduled: set[int] = set()
    async for event in handle.fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            if event.activity_task_scheduled_event_attributes.activity_type.name == "run_job":
                scheduled.add(event.event_id)
        elif event.event_type in _RUN_JOB_GAVE_UP:
            attrs = getattr(event, _RUN_JOB_GAVE_UP[event.event_type])
            if attrs.scheduled_event_id in scheduled:
                return True
    return False


def probe(wid: str, timeout: float | None = None) -> Probe:
    async def describe(client: Client) -> Probe:
        handle = client.get_workflow_handle(wid)
        try:
            desc = await handle.describe()
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return Probe("missing")
            raise
        if desc.status != WorkflowExecutionStatus.RUNNING:
            return Probe("closed")
        pending = list(desc.raw_description.pending_activities)
        if any(act.activity_type.name == "mark_job_lost" for act in pending):
            return Probe("lost")  # run_job gave up for good; only the bookkeeping is left
        for act in pending:
            if act.activity_type.name != "run_job":
                continue
            beat = act.last_heartbeat_time.ToDatetime(tzinfo=UTC) if act.HasField("last_heartbeat_time") else None
            started = act.state in (
                PendingActivityState.PENDING_ACTIVITY_STATE_STARTED,
                PendingActivityState.PENDING_ACTIVITY_STATE_CANCEL_REQUESTED,
            )
            return Probe(
                "alive" if started else "waiting",
                attempt=act.attempt,
                worker=act.last_worker_identity or None,
                last_heartbeat=beat,
            )
        if not pending and await _run_job_gave_up(handle):
            return Probe("lost")
        return Probe("waiting")

    return call(describe, timeout)


def chaos(wid: str, timeout: float = 60.0) -> ChaosResult:
    async def update(client: Client) -> ChaosResult:
        try:
            return await client.get_workflow_handle(wid).execute_update("chaos", result_type=ChaosResult)
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:  # closed or unknown workflow: nothing to kill
                return ChaosResult(False, exc.message)
            raise

    return call(update, timeout)


def task_queue_workers(timeout: float | None = None) -> list[str]:
    """Identities of the workers polling the job task queue (activity side)."""

    async def describe(client: Client) -> list[str]:
        resp = await client.workflow_service.describe_task_queue(
            DescribeTaskQueueRequest(
                namespace=settings.TEMPORAL_NAMESPACE,
                task_queue=TaskQueue(name=settings.TEMPORAL_TASK_QUEUE),
                task_queue_type=TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY,
            )
        )
        return sorted({p.identity for p in resp.pollers})

    return call(describe, timeout)


async def ensure_gc_schedule(client: Client, cron: str) -> str:
    """Create or update the artifact-GC schedule (``cron`` empty: delete it)."""
    handle = client.get_schedule_handle(GC_SCHEDULE_ID)
    if not cron.strip():
        try:
            await handle.delete()
            return "deleted"
        except RPCError as exc:
            if exc.status == RPCStatusCode.NOT_FOUND:
                return "disabled"
            raise
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            GC_WORKFLOW,
            GCInput(apply=True),
            id="bayesian-credit-artifact-gc-run",
            task_queue=settings.TEMPORAL_TASK_QUEUE,
        ),
        spec=ScheduleSpec(cron_expressions=[cron]),
        policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
    )
    try:
        await client.create_schedule(GC_SCHEDULE_ID, schedule)
        return "created"
    except ScheduleAlreadyRunningError:

        def replace(_: ScheduleUpdateInput) -> ScheduleUpdate:
            return ScheduleUpdate(schedule=schedule)

        await handle.update(replace)
        return "updated"


def summary(**extra: Any) -> dict[str, Any]:
    return {
        "address": settings.TEMPORAL_ADDRESS,
        "namespace": settings.TEMPORAL_NAMESPACE,
        "task_queue": settings.TEMPORAL_TASK_QUEUE,
        "ui_url": settings.TEMPORAL_UI_URL,
        **extra,
    }
