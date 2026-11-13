"""``python manage.py temporal_worker`` — execute background jobs from the Temporal task queue.

Runs the ``BackgroundJob`` and ``ArtifactGC`` workflows and their activities. Each job runs in
a supervised ``manage.py run_job <id>`` child process, at most ``--concurrency`` at a time.
Start as many workers as the hardware allows (any host that shares the database and
``ARTIFACT_ROOT``); Temporal hands each job to one of them and moves it to another when a
worker disappears. On start-up the worker also creates / updates the artifact-GC schedule
(``TEMPORAL_GC_SCHEDULE``).

SIGINT / SIGTERM shut the worker down gracefully: running children get SIGTERM (a simulation
checkpoints and stops) and the jobs are released back to Temporal.
"""

from __future__ import annotations

import asyncio
import signal
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from temporalio.worker import Worker

from apps.common.temporal import client as temporal_client
from apps.common.temporal.activities import ACTIVITIES
from apps.common.temporal.workflows import ArtifactGCWorkflow, BackgroundJobWorkflow


class Command(BaseCommand):
    help = "Run a Temporal worker that executes ingest, fit, shadow-portfolio and simulation jobs."

    def add_arguments(self, parser):
        parser.add_argument(
            "--concurrency",
            type=int,
            default=settings.TEMPORAL_WORKER_CONCURRENCY,
            help="jobs run at once by this worker (default TEMPORAL_WORKER_CONCURRENCY)",
        )
        parser.add_argument("--identity", default=None, help="worker identity shown in Temporal (default host@pid)")
        parser.add_argument("--no-schedules", action="store_true", help="do not create / update Temporal schedules")

    def handle(self, *args, **opts):
        asyncio.run(self._run(opts))

    async def _run(self, opts) -> None:
        concurrency = max(int(opts["concurrency"]), 1)
        identity = opts["identity"] or f"bc-worker@{socket.gethostname()}"
        client = await temporal_client.connect(identity=identity)
        if not opts["no_schedules"]:
            outcome = await temporal_client.ensure_gc_schedule(client, settings.TEMPORAL_GC_SCHEDULE)
            self.stdout.write(f"artifact GC schedule {temporal_client.GC_SCHEDULE_ID}: {outcome}")
        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="bc-activity")
        worker = Worker(
            client,
            task_queue=settings.TEMPORAL_TASK_QUEUE,
            workflows=[BackgroundJobWorkflow, ArtifactGCWorkflow],
            activities=ACTIVITIES,
            activity_executor=executor,
            max_concurrent_activities=concurrency,
            # cancellation (the chaos kill) reaches an activity on a heartbeat response: keep it prompt
            max_heartbeat_throttle_interval=timedelta(seconds=0.5),
            default_heartbeat_throttle_interval=timedelta(seconds=0.5),
            graceful_shutdown_timeout=timedelta(seconds=5),
        )
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        self.stdout.write(
            f"{identity} polling {settings.TEMPORAL_TASK_QUEUE!r} on {settings.TEMPORAL_ADDRESS} "
            f"(namespace {settings.TEMPORAL_NAMESPACE}, {concurrency} concurrent job(s))"
        )
        async with worker:
            await stop.wait()
            self.stdout.write("shutting down: releasing running jobs")
        executor.shutdown(wait=True)
        self.stdout.write("worker stopped")
