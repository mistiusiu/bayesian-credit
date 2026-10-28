"""Background job bookkeeping (SPEC §3: ``python manage.py run_job <job_id>`` subprocesses)."""

from __future__ import annotations

from pathlib import Path

from django.db import models

from . import artifacts


class Job(models.Model):
    """One unit of background work: ingest a dataset, fit models, run a simulation or a
    shadow portfolio. The worker process updates this row (status, progress, log tail)."""

    class Kind(models.TextChoices):
        INGEST = "ingest"
        FIT = "fit"
        SIMULATION = "simulation"
        SHADOW_PORTFOLIO = "shadow_portfolio"

    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        SUCCEEDED = "succeeded"
        FAILED = "failed"
        CANCELLED = "cancelled"

    ACTIVE = (Status.QUEUED, Status.RUNNING)
    FINISHED = (Status.SUCCEEDED, Status.FAILED, Status.CANCELLED)
    LOG_TAIL_LINES = 50

    kind = models.CharField(max_length=32, choices=Kind.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED, db_index=True)
    progress = models.FloatField(default=0.0)
    message = models.TextField(blank=True, default="")
    log_tail = models.JSONField(default=list, blank=True)
    pid = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    target_type = models.CharField(max_length=16, blank=True, default="")
    target_id = models.BigIntegerField(null=True, blank=True)
    params = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["target_type", "target_id"])]

    def __str__(self) -> str:
        return f"Job #{self.pk} {self.kind} ({self.status})"

    @property
    def is_active(self) -> bool:
        return self.status in self.ACTIVE

    @property
    def log_path(self) -> Path:
        return artifacts.job_log_path(self.pk)
