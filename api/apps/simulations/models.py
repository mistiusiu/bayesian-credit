"""Simulations: one replay run (run directory protocol in ``services.py``)."""

from __future__ import annotations

from pathlib import Path

from django.db import models

from apps.common import artifacts
from apps.common.jobs import worker_alive
from apps.common.models import Job
from apps.fits.models import Fit


class Simulation(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        PAUSED = "paused"
        AWAITING_REVIEW = "awaiting_review"
        INTERRUPTED = "interrupted"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    LIVE_STATUSES = (Status.QUEUED, Status.RUNNING, Status.PAUSED, Status.AWAITING_REVIEW)
    RESUMABLE_STATUSES = (Status.INTERRUPTED, Status.FAILED, Status.CANCELLED)

    name = models.CharField(max_length=200)
    fit = models.ForeignKey(Fit, on_delete=models.PROTECT, related_name="simulations")
    #: types.ts ``SimulationConfig`` (defaults applied, epochs as YYYY-MM).
    config = models.JSONField(default=dict, blank=True)
    policies = models.JSONField(default=list, blank=True)
    run_dir = models.CharField(max_length=500, blank=True, default="")
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.QUEUED, db_index=True)
    #: last ``progress.json`` seen (mirror, for lists and metrics)
    progress = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    job = models.ForeignKey(Job, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    resume_count = models.IntegerField(default=0)
    #: types.ts ``RecoveryInfo`` of the latest kill → resume cycle (plus bookkeeping keys)
    recovery = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"Simulation #{self.pk} {self.name} ({self.status})"

    @property
    def directory(self) -> Path:
        return Path(self.run_dir) if self.run_dir else artifacts.run_dir(self.pk)

    def worker_alive(self) -> bool:
        return bool(self.job is not None and self.job.is_active and worker_alive(self.job))
