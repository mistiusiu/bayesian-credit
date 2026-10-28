"""``python manage.py run_job <id>`` — the worker side of a background job (subprocess or k8s Job)."""

from __future__ import annotations

import sys

from django.core.management.base import BaseCommand, CommandError

from apps.common.jobs import execute
from apps.common.models import Job


class Command(BaseCommand):
    help = "Execute one background job (ingest, fit, simulation, shadow_portfolio) and record its outcome."

    def add_arguments(self, parser):
        parser.add_argument("job_id", type=int)

    def handle(self, *args, **opts):
        job = Job.objects.filter(pk=opts["job_id"]).first()
        if job is None:
            raise CommandError(f"job {opts['job_id']} does not exist")
        job = execute(job, inline=False)
        self.stdout.write(f"job {job.pk} {job.kind}: {job.status} — {job.message}")
        if job.status == Job.Status.FAILED:
            sys.exit(1)
