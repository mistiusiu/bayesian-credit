"""The Temporal backend against a real (local dev) Temporal server.

A worker runs the real ``BackgroundJob`` workflow with stand-in activities (the real
``run_job`` spawns ``manage.py run_job``, which cannot see the in-memory test database), and
the API side — :class:`~apps.common.backends.TemporalBackend` launch / probe / chaos kill /
reaping and the health payload — talks to that server through the real client bridge.

The dev server is the ``temporal`` CLI when installed, otherwise the SDK downloads it; the
module is skipped when neither works (offline) unless ``BC_REQUIRE_TEMPORAL`` is set (CI).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from apps.common.temporal.workflows import (
    CHAOS_KILLED,
    JOB_FAILED,
    WORKER_LOST,
    BackgroundJobWorkflow,
    JobInput,
    JobOutcome,
    MarkLostInput,
)

pytestmark = pytest.mark.django_db

TASK_QUEUE = "bc-test-jobs"
#: job id -> scripted behaviour of the stand-in ``run_job`` activity
BEHAVIOUR: dict[int, str] = {}
CALLS: dict[int, list[str]] = defaultdict(list)
LOST: dict[int, str] = {}


@activity.defn(name="run_job")
def fake_run_job(inp: JobInput) -> JobOutcome:
    attempt = activity.info().attempt
    CALLS[inp.job_id].append(f"attempt {attempt}")
    behaviour = BEHAVIOUR.get(inp.job_id, "succeed")
    if behaviour == "fail":
        raise ApplicationError("boom", type=JOB_FAILED, non_retryable=True)
    if behaviour == "lost" or (behaviour == "lost_once" and attempt == 1):
        raise ApplicationError("worker exited with code -9 without settling", type="WorkerLost")
    if behaviour == "hang":
        with activity.shield_thread_cancel_exception():
            while not activity.is_cancelled():
                activity.heartbeat()
                time.sleep(0.1)
            CALLS[inp.job_id].append("killed")
    return JobOutcome(inp.job_id, "succeeded", "done")


@activity.defn(name="mark_job_lost")
def fake_mark_job_lost(inp: MarkLostInput) -> None:
    LOST[inp.job_id] = inp.message


class _Server:
    """Dev server + worker on a private event loop thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.target: str | None = None
        self.error: BaseException | None = None
        self._stop: asyncio.Event | None = None
        self.thread = threading.Thread(target=self.loop.run_until_complete, args=(self._main(),), daemon=True)

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            existing = shutil.which("temporal")
            env = await WorkflowEnvironment.start_local(
                dev_server_existing_path=existing, ui=False, dev_server_log_level="error"
            )
        except BaseException as exc:  # noqa: BLE001 - reported to the fixture, which skips
            self.error = exc
            self.ready.set()
            return
        try:
            worker = Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[BackgroundJobWorkflow],
                activities=[fake_run_job, fake_mark_job_lost],
                activity_executor=ThreadPoolExecutor(8),
                max_heartbeat_throttle_interval=timedelta(seconds=0.5),
                default_heartbeat_throttle_interval=timedelta(seconds=0.5),
                identity="bc-test-worker",
            )
            async with worker:
                self.target = env.client.service_client.config.target_host
                self.ready.set()
                await self._stop.wait()
        finally:
            await env.shutdown()

    def start(self) -> None:
        self.thread.start()
        self.ready.wait(timeout=180)

    def stop(self) -> None:
        if self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set)
        self.thread.join(timeout=60)


@pytest.fixture(scope="module")
def temporal_server():
    server = _Server()
    server.start()
    if server.target is None:
        server.stop()
        if os.environ.get("BC_REQUIRE_TEMPORAL"):
            pytest.fail(f"no Temporal dev server available: {server.error!r}")
        pytest.skip(f"no Temporal dev server available: {server.error!r}")
    yield server
    server.stop()


@pytest.fixture
def backend(temporal_server, settings):
    from apps.common import backends
    from apps.common.temporal import client

    settings.JOBS_SYNC = False
    settings.JOBS_BACKEND = "apps.common.backends.TemporalBackend"
    settings.TEMPORAL_ADDRESS = temporal_server.target
    settings.TEMPORAL_NAMESPACE = "default"
    settings.TEMPORAL_TASK_QUEUE = TASK_QUEUE
    settings.TEMPORAL_HEARTBEAT_TIMEOUT = 10
    client._bridge = client._Bridge()  # connect to this server, not a cached one
    backends._load.cache_clear()
    yield backends.get_backend()
    backends._load.cache_clear()


def _job(kind: str, behaviour: str):
    from apps.common.models import Job

    job = Job.objects.create(kind=kind, status="running", target_type="", params={})
    BEHAVIOUR[job.pk] = behaviour  # (pks are reused after each test's rollback)
    CALLS.pop(job.pk, None)
    LOST.pop(job.pk, None)
    return job


def _result(job) -> JobOutcome:
    from apps.common.temporal import client

    async def wait(c):
        return await c.get_workflow_handle(job.workflow_id, result_type=JobOutcome).result()

    return client.call(wait, timeout=60)


def _wait_for(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def test_launch_runs_the_workflow_and_records_its_id(backend):
    from apps.common.jobs import JobRunner

    job = JobRunner.launch(_job("fit", "succeed"))
    assert job.workflow_id.startswith(f"bc-fit-{job.pk}-")
    job.refresh_from_db()
    assert job.workflow_id and job.status == "running"  # launching never settles the row itself
    assert _result(job) == JobOutcome(job.pk, "succeeded", "done")
    assert backend.links(job)["temporal_url"].endswith(f"/namespaces/default/workflows/{job.workflow_id}")


def test_closed_workflow_with_unsettled_row_is_reaped(backend):
    from apps.common.jobs import JobRunner, reap_lost_workers

    job = JobRunner.launch(_job("ingest", "succeed"))
    _result(job)
    assert backend.probe(job).state == "dead"
    assert reap_lost_workers([job]) == 1
    job.refresh_from_db()
    assert job.status == "failed" and job.message == WORKER_LOST


def test_failed_job_is_not_retried(backend):
    from apps.common.jobs import JobRunner

    job = JobRunner.launch(_job("fit", "fail"))
    outcome = _result(job)
    assert outcome.status == "failed" and CALLS[job.pk] == ["attempt 1"] and job.pk not in LOST


def test_lost_worker_is_retried_for_fits(backend):
    from apps.common.jobs import JobRunner

    job = JobRunner.launch(_job("fit", "lost_once"))
    assert _result(job).status == "succeeded"
    assert CALLS[job.pk] == ["attempt 1", "attempt 2"] and job.pk not in LOST


def test_lost_simulation_worker_is_marked_lost_without_retry(backend):
    from apps.common.jobs import JobRunner

    job = JobRunner.launch(_job("simulation", "lost"))
    assert _result(job) == JobOutcome(job.pk, "lost", WORKER_LOST)
    assert CALLS[job.pk] == ["attempt 1"] and LOST[job.pk] == WORKER_LOST


def test_chaos_update_kills_the_running_worker(backend):
    from apps.common.jobs import JobRunner, kill_worker, worker_alive

    job = JobRunner.launch(_job("simulation", "hang"))
    _wait_for(lambda: worker_alive(job))
    assert backend.probe(job).detail.startswith("attempt 1 on bc-test-worker")
    assert kill_worker(job) is True
    assert CALLS[job.pk][-1] == "killed" and LOST[job.pk] == CHAOS_KILLED
    assert _result(job) == JobOutcome(job.pk, "killed", CHAOS_KILLED)
    assert not worker_alive(job) and kill_worker(job) is False


def test_health_reports_server_and_workers(backend, api):
    body = api.get("/api/v1/health/").json()
    assert body["status"] == "ok"
    temporal = body["jobs"]["temporal"]
    assert body["jobs"]["backend"] == "temporal"
    assert temporal["connected"] and temporal["task_queue"] == TASK_QUEUE and "bc-test-worker" in temporal["workers"]


def test_unreachable_server_degrades_without_failing_reads(backend, settings, api):
    from apps.common import backends
    from apps.common.jobs import JobRunner, reap_lost_workers
    from apps.common.temporal import client

    settings.TEMPORAL_ADDRESS = "127.0.0.1:1"
    settings.TEMPORAL_RPC_TIMEOUT = 2
    client._bridge = client._Bridge()
    backends._load.cache_clear()
    body = api.get("/api/v1/health/").json()
    assert body["status"] == "degraded" and body["jobs"]["temporal"]["connected"] is False
    job = JobRunner.launch(_job("fit", "succeed"))
    assert job.status == "failed" and "could not start Temporal workflow" in job.message
    running = _job("fit", "succeed")
    running.workflow_id = "bc-fit-unknown"
    running.save(update_fields=["workflow_id"])
    assert backends.get_backend().probe(running).state == "unknown"
    assert reap_lost_workers([running]) == 0  # an unreachable server never fails jobs
