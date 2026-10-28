"""Human-in-the-loop review queue, mirrored from ``<run_dir>/reviews/<review_id>.json``."""

from __future__ import annotations

from django.db import models

from apps.simulations.models import Simulation


class Review(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        ACKNOWLEDGED = "acknowledged"

    class Action(models.TextChoices):
        RESUME = "resume"
        RESEGMENT = "resegment"
        ROLLBACK = "rollback"

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE, related_name="reviews")
    #: the runner's id, e.g. ``review-0001`` (file name stem)
    review_id = models.CharField(max_length=100)
    epoch = models.CharField(max_length=7)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    created_at = models.DateTimeField()
    psi_aggregate = models.FloatField(null=True, blank=True)
    level = models.CharField(max_length=16, default="breaker")
    breaker_state = models.CharField(max_length=16, default="open")
    reason = models.TextField(blank=True, default="")
    top_features = models.JSONField(default=list, blank=True)
    action = models.CharField(max_length=16, choices=Action.choices, null=True, blank=True)  # noqa: DJ001 - the contract distinguishes null from ''
    note = models.TextField(blank=True, default="")
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.CharField(max_length=150, null=True, blank=True)  # noqa: DJ001 - the contract distinguishes null from ''
    auto = models.BooleanField(default=False)
    segmentation_before = models.JSONField(default=list, blank=True)
    segmentation_after = models.JSONField(null=True, blank=True)
    #: raw review record written by the runner
    record = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-id"]
        constraints = [models.UniqueConstraint(fields=["simulation", "review_id"], name="uniq_review_per_simulation")]

    def __str__(self) -> str:
        return f"Review #{self.pk} sim {self.simulation_id} {self.review_id} ({self.status})"
