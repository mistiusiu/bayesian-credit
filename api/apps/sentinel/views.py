"""``/sentinel/reviews/`` — the human-in-the-loop review queue."""

from __future__ import annotations

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import mixins, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.common.utils import parse_int
from apps.simulations.models import Simulation

from . import services
from .models import Review
from .serializers import AcknowledgeRequestSerializer, ReviewSerializer


@extend_schema_view(
    list=extend_schema(
        summary="Review queue (newest first)",
        tags=["sentinel"],
        parameters=[
            OpenApiParameter("simulation", OpenApiTypes.INT),
            OpenApiParameter("status", OpenApiTypes.STR, enum=["pending", "acknowledged"]),
        ],
    ),
    retrieve=extend_schema(summary="Review detail", tags=["sentinel"]),
)
class ReviewViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Review.objects.select_related("simulation").order_by("-id")
    serializer_class = ReviewSerializer
    filter_backends: list = []

    def get_queryset(self):
        qs = super().get_queryset()
        q = self.request.query_params
        sim_id = parse_int(q.get("simulation"), "simulation")
        if sim_id is not None:
            qs = qs.filter(simulation_id=sim_id)
        status = q.get("status")
        if status in (Review.Status.PENDING, Review.Status.ACKNOWLEDGED):
            qs = qs.filter(status=status)
        return qs

    def list(self, request, *args, **kwargs):
        sim_id = parse_int(request.query_params.get("simulation"), "simulation")
        sims = Simulation.objects.select_related("fit").exclude(run_dir="")
        services.sync_all(sims.filter(pk=sim_id) if sim_id is not None else sims)
        return super().list(request, *args, **kwargs)

    @extend_schema(
        summary="Acknowledge a breaker review (resume | resegment | rollback)",
        tags=["sentinel"],
        request=AcknowledgeRequestSerializer,
        responses={200: ReviewSerializer},
    )
    @action(detail=True, methods=["post"])
    def acknowledge(self, request, pk=None):
        review = self.get_object()
        body = AcknowledgeRequestSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        user = getattr(request, "user", None)
        by = user.get_username() if getattr(user, "is_authenticated", False) else "researcher"
        review = services.acknowledge(review, body.validated_data["action"], body.validated_data.get("note", ""), by=by)
        return Response(ReviewSerializer(review).data)
