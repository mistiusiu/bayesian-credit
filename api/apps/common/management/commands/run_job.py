"""``python manage.py run_job <id>`` — the process that executes one background job.

Normally started by the ``run_job`` Temporal activity (``--supervised``); runnable by hand to
debug a job. A supervised child exits as soon as its supervisor disappears, so Temporal never
retries a job on another worker while an orphan still runs it here.
"""

from __future__ import annotations

import os
import sys
import threading
import time

from django.core.management.base import BaseCommand, CommandError

from apps.common.jobs import execute
from apps.common.models import Job

ORPHAN_EXIT_CODE = 137


def _exit_when_orphaned(parent: int) -> None:
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            sys.stderr.write(f"supervisor {parent} is gone; exiting\n")
            os._exit(ORPHAN_EXIT_CODE)


class Command(BaseCommand):
    help = "Execute one background job (ingest, fit, simulation, shadow_portfolio) and record its outcome."

    def add_arguments(self, parser):
        parser.add_argument("job_id", type=int)
        parser.add_argument("--supervised", action="store_true", help="exit when the parent process dies")

    def handle(self, *args, **opts):
        if opts["supervised"]:
            threading.Thread(target=_exit_when_orphaned, args=(os.getppid(),), daemon=True).start()
        job = Job.objects.filter(pk=opts["job_id"]).first()
        if job is None:
            raise CommandError(f"job {opts['job_id']} does not exist")
        job = execute(job, inline=False)
        self.stdout.write(f"job {job.pk} {job.kind}: {job.status} — {job.message}")
        if job.status == Job.Status.FAILED:
            sys.exit(1)
