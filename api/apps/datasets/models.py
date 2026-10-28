"""Datasets: a canonical LoanTable (+ ``y_*`` outcomes) materialised as parquet."""

from __future__ import annotations

from pathlib import Path

from django.db import models

from apps.common import artifacts
from apps.common.mixins import BuildStatusMixin
from apps.common.models import Job


class Dataset(BuildStatusMixin, models.Model):
    class Source(models.TextChoices):
        SYNTHETIC = "synthetic"
        LENDING_CLUB = "lending_club"

    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        READY = "ready"
        FAILED = "failed"
        CANCELLED = "cancelled"

    name = models.CharField(max_length=200)
    source = models.CharField(max_length=16, choices=Source.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    params = models.JSONField(default=dict, blank=True)
    parquet_path = models.CharField(max_length=500, blank=True, default="")
    n_loans = models.IntegerField(null=True, blank=True)
    n_resolved = models.IntegerField(null=True, blank=True)
    issue_min = models.CharField(max_length=7, null=True, blank=True)  # noqa: DJ001 - the contract distinguishes null from ''
    issue_max = models.CharField(max_length=7, null=True, blank=True)  # noqa: DJ001 - the contract distinguishes null from ''
    resolved_share = models.FloatField(null=True, blank=True)
    default_rate = models.FloatField(null=True, blank=True)
    status_counts = models.JSONField(default=dict, blank=True)
    summary = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    job = models.ForeignKey(Job, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")

    class Meta:
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"Dataset #{self.pk} {self.name} ({self.source})"

    @property
    def directory(self) -> Path:
        return artifacts.dataset_dir(self.pk)

    @property
    def parquet(self) -> Path:
        return Path(self.parquet_path) if self.parquet_path else self.directory / "loans.parquet"
