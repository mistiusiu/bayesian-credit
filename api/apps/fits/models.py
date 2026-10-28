"""Fits: a Two-Stage Beta-Beta + MCTM + reward + arm space bundle saved by ``FittedModels.save``."""

from __future__ import annotations

from pathlib import Path

from django.db import models

from apps.common import artifacts
from apps.common.mixins import BuildStatusMixin
from apps.common.models import Job
from apps.datasets.models import Dataset


class Fit(BuildStatusMixin, models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued"
        RUNNING = "running"
        READY = "ready"
        FAILED = "failed"
        CANCELLED = "cancelled"

    name = models.CharField(max_length=200)
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="fits")
    #: types.ts ``FitConfig`` (server defaults applied; ``train_end`` as YYYY-MM).
    config = models.JSONField(default=dict, blank=True)
    #: ``FrameworkConfig.to_dict()`` handed to the core.
    framework = models.JSONField(default=dict, blank=True)
    train_end = models.CharField(max_length=7)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    fit_dir = models.CharField(max_length=500, blank=True, default="")
    metrics = models.JSONField(null=True, blank=True)
    n_train = models.IntegerField(null=True, blank=True)
    n_test = models.IntegerField(null=True, blank=True)
    n_groups = models.IntegerField(null=True, blank=True)
    n_arms = models.IntegerField(null=True, blank=True)
    fit_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    job = models.ForeignKey(Job, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")

    class Meta:
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"Fit #{self.pk} {self.name}"

    @property
    def directory(self) -> Path:
        return Path(self.fit_dir) if self.fit_dir else artifacts.fit_dir(self.pk)
