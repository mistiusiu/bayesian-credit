"""Serializers shared across apps (types.ts ``Job``, ``JobTarget``, ``Health``)."""

from __future__ import annotations

from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers

from .jobs import job_target
from .models import Job


class JobTargetSerializer(serializers.Serializer):
    type = serializers.ChoiceField(choices=["dataset", "fit", "simulation"])
    id = serializers.IntegerField()


class JobSerializer(serializers.ModelSerializer):
    """types.ts ``Job``."""

    kind = serializers.ChoiceField(choices=list(Job.Kind.values), read_only=True)
    status = serializers.ChoiceField(choices=list(Job.Status.values), read_only=True)
    target = serializers.SerializerMethodField()
    log_tail = serializers.ListField(child=serializers.CharField(), read_only=True)

    class Meta:
        model = Job
        fields = [
            "id",
            "kind",
            "status",
            "progress",
            "message",
            "log_tail",
            "created_at",
            "started_at",
            "finished_at",
            "target",
            "pid",
        ]
        read_only_fields = fields

    @extend_schema_field(JobTargetSerializer(allow_null=True))
    def get_target(self, obj: Job):
        return job_target(obj)

    def to_representation(self, instance: Job):
        data = super().to_representation(instance)
        data["progress"] = round(float(data.get("progress") or 0.0), 4)
        data["log_tail"] = list(instance.log_tail or [])[-Job.LOG_TAIL_LINES :]
        if instance.status not in Job.ACTIVE:
            data["pid"] = None  # a finished job has no live worker
        return data


class HealthSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=["ok", "degraded"])
    version = serializers.CharField()
    python = serializers.CharField()
    core_version = serializers.CharField()
