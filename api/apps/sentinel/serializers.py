"""types.ts ``Review`` and ``AcknowledgeRequest``."""

from __future__ import annotations

from rest_framework import serializers

from apps.common.schemas import ReviewTopFeaturesItemSerializer

from .models import Review

REVIEW_ACTIONS = ["resume", "resegment", "rollback"]
SEGMENTATION_DIMENSIONS = [
    "grade",
    "sub_grade",
    "purpose_group",
    "home_ownership",
    "fico_band",
    "dti_band",
    "income_band",
    "emp_band",
    "verification_status",
    "application_type",
]


class ReviewSerializer(serializers.ModelSerializer):
    """types.ts ``Review``."""

    simulation = serializers.IntegerField(source="simulation_id", read_only=True)
    simulation_name = serializers.CharField(source="simulation.name", read_only=True)
    psi_aggregate = serializers.FloatField(allow_null=True, read_only=True)
    level = serializers.ChoiceField(choices=["normal", "watch", "breaker"], read_only=True)
    breaker_state = serializers.ChoiceField(choices=["closed", "open", "half_open"], read_only=True)
    top_features = ReviewTopFeaturesItemSerializer(many=True, read_only=True)
    segmentation_before = serializers.ListField(
        child=serializers.ChoiceField(choices=SEGMENTATION_DIMENSIONS), read_only=True
    )
    segmentation_after = serializers.ListField(
        child=serializers.ChoiceField(choices=SEGMENTATION_DIMENSIONS), allow_null=True, read_only=True
    )
    status = serializers.ChoiceField(choices=["pending", "acknowledged"], read_only=True)
    action = serializers.ChoiceField(choices=REVIEW_ACTIONS, allow_null=True, read_only=True)
    review_id = serializers.CharField(read_only=True, help_text="Runner-side id (file stem in <run_dir>/reviews/)")

    class Meta:
        model = Review
        fields = [
            "id",
            "simulation",
            "simulation_name",
            "epoch",
            "status",
            "created_at",
            "psi_aggregate",
            "level",
            "breaker_state",
            "reason",
            "top_features",
            "action",
            "note",
            "acknowledged_at",
            "acknowledged_by",
            "auto",
            "segmentation_before",
            "segmentation_after",
            "review_id",
        ]
        read_only_fields = fields


class AcknowledgeRequestSerializer(serializers.Serializer):
    """POST /sentinel/reviews/{id}/acknowledge/ body."""

    action = serializers.ChoiceField(choices=REVIEW_ACTIONS)
    note = serializers.CharField(required=False, allow_blank=True, default="", max_length=4000)
