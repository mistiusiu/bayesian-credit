"""``/simulations/`` — create (worker job), list, detail, read views and control actions."""

from __future__ import annotations

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view, inline_serializer
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.common import schemas as S
from apps.common.exceptions import Conflict
from apps.common.utils import csv_list, parse_int, parse_int_list
from apps.fits.services import require_ready

from . import services
from .models import Simulation
from .serializers import SimulationCreateSerializer, SimulationSerializer

TAG = ["simulations"]
POLICY = OpenApiParameter("policy", OpenApiTypes.STR, description="Policy label (default: first configured)")
EPOCH = OpenApiParameter("epoch", OpenApiTypes.STR, description="Completed epoch YYYY-MM (default: last)")
GROUP = OpenApiParameter("group", OpenApiTypes.STR, description="Group key (default: decision-weighted pool)")
EMPTY_BODY = inline_serializer("EmptyBody", fields={})


@extend_schema_view(
    list=extend_schema(
        summary="List simulations (newest first)",
        tags=TAG,
        parameters=[OpenApiParameter("fit", OpenApiTypes.INT)],
        responses=SimulationSerializer(many=True),
    ),
    retrieve=extend_schema(
        summary="Simulation detail (status, progress, breaker, headline metrics)",
        tags=TAG,
        responses=SimulationSerializer,
    ),
    create=extend_schema(
        summary="Start a replay simulation",
        tags=TAG,
        request=SimulationCreateSerializer,
        responses={201: SimulationSerializer},
    ),
)
class SimulationViewSet(
    mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet
):
    queryset = Simulation.objects.select_related("fit", "fit__dataset", "job").order_by("-id")
    serializer_class = SimulationSerializer
    filterset_fields = ["fit", "status"]

    def get_serializer_class(self):
        return SimulationCreateSerializer if self.action == "create" else SimulationSerializer

    def list(self, request, *args, **kwargs):
        qs = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(qs)
        sims = list(page if page is not None else qs)
        services.bulk_refresh(sims)
        data = [services.simulation_payload(s, refresh=False) for s in sims]
        return self.get_paginated_response(data) if page is not None else Response(data)

    def retrieve(self, request, *args, **kwargs):
        return Response(services.simulation_payload(self.get_object()))

    def create(self, request, *args, **kwargs):
        ser = SimulationCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        fit = require_ready(ser.validated_data["fit_obj"])
        sim = services.create_simulation(fit, ser.validated_data["name"], ser.validated_data["config"])
        return Response(services.simulation_payload(sim), status=status.HTTP_201_CREATED)

    def _sim(self) -> Simulation:
        return self.get_object()

    @extend_schema(
        summary="Per-epoch, per-policy metrics", tags=TAG, parameters=[POLICY], responses=S.TimeseriesSerializer
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def timeseries(self, request, pk=None):
        return Response(services.timeseries_payload(self._sim(), request.query_params.get("policy") or None))

    @extend_schema(
        summary="Arm selections and bandit posteriors at an epoch",
        tags=TAG,
        parameters=[POLICY, EPOCH, GROUP],
        responses=S.SimArmsSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def arms(self, request, pk=None):
        q = request.query_params
        return Response(
            services.arms_payload(self._sim(), q.get("policy") or None, q.get("epoch") or None, q.get("group") or None)
        )

    @extend_schema(
        summary="Bandit posterior history (μ, σ, n per snapshot epoch)",
        tags=TAG,
        parameters=[
            POLICY,
            GROUP,
            OpenApiParameter("arms", OpenApiTypes.STR, description="Comma list of arm ids (default: 4 most selected)"),
            OpenApiParameter("arm", OpenApiTypes.INT, description="Single arm (alias of arms)"),
        ],
        responses=S.PosteriorHistorySerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def posteriors(self, request, pk=None):
        q = request.query_params
        arms = parse_int_list(q.get("arms"), "arms") + parse_int_list(q.get("arm"), "arm")
        return Response(services.posteriors_payload(self._sim(), q.get("policy") or None, q.get("group") or None, arms))

    @extend_schema(
        summary="Aggregation-bias check: pooled-best vs group-best arm",
        tags=TAG,
        parameters=[POLICY, EPOCH],
        responses=S.SimpsonPanelSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def simpson(self, request, pk=None):
        q = request.query_params
        return Response(services.simpson_payload(self._sim(), q.get("policy") or None, q.get("epoch") or None))

    @extend_schema(
        summary="Event log (append-only events.jsonl)",
        tags=TAG,
        parameters=[
            OpenApiParameter("type", OpenApiTypes.STR, description="Comma list of event types"),
            OpenApiParameter("after", OpenApiTypes.INT, description="Exclusive lower bound on seq"),
            OpenApiParameter("limit", OpenApiTypes.INT, description="1..1000, default 100"),
            OpenApiParameter("order", OpenApiTypes.STR, enum=["asc", "desc"]),
        ],
        responses=S.EventPageSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def events(self, request, pk=None):
        q = request.query_params
        return Response(
            services.events_payload(
                self._sim(),
                types=csv_list(q.get("type")),
                after=parse_int(q.get("after"), "after", default=0, lo=0),
                limit=parse_int(q.get("limit"), "limit", default=100, lo=1, hi=1000),
                order="desc" if q.get("order") == "desc" else "asc",
            )
        )

    @extend_schema(
        summary="PSI per feature per epoch and breaker transitions",
        tags=["sentinel"],
        responses=S.SentinelTimelineSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def sentinel(self, request, pk=None):
        return Response(services.sentinel_payload(self._sim()))

    @extend_schema(
        summary="PSI bucket table of one feature at one epoch",
        tags=["sentinel"],
        parameters=[OpenApiParameter("feature", OpenApiTypes.STR), EPOCH],
        responses=S.PsiBucketTableSerializer,
    )
    @action(detail=True, methods=["get"], url_path="sentinel/buckets", pagination_class=None, filter_backends=[])
    def sentinel_buckets(self, request, pk=None):
        q = request.query_params
        return Response(services.buckets_payload(self._sim(), q.get("feature") or None, q.get("epoch") or None))

    @extend_schema(
        summary="Table 11 evaluation report (404 until report.json exists)",
        tags=TAG,
        responses=S.EvaluationReportSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def report(self, request, pk=None):
        sim = self._sim()
        services.refresh_status(sim)
        report = services.report_payload(sim)
        if report is None:
            raise Conflict(f"no completed epochs yet (simulation {sim.pk} is {sim.status})")
        return Response(report)

    def _control(self, fn):
        sim = fn(self._sim())
        return Response(services.simulation_payload(sim))

    @extend_schema(
        summary="Pause a running simulation (writes control.json)",
        tags=TAG,
        request=EMPTY_BODY,
        responses=SimulationSerializer,
    )
    @action(detail=True, methods=["post"], pagination_class=None, filter_backends=[])
    def pause(self, request, pk=None):
        return self._control(services.pause)

    @extend_schema(
        summary="Resume: a paused run continues; an interrupted / failed / cancelled run restarts from its last checkpoint",
        tags=TAG,
        request=EMPTY_BODY,
        responses=SimulationSerializer,
    )
    @action(detail=True, methods=["post"], pagination_class=None, filter_backends=[])
    def resume(self, request, pk=None):
        return self._control(services.resume)

    @extend_schema(
        summary="Cancel (writes control.json; marks a dead run cancelled)",
        tags=TAG,
        request=EMPTY_BODY,
        responses=SimulationSerializer,
    )
    @action(detail=True, methods=["post"], pagination_class=None, filter_backends=[])
    def cancel(self, request, pk=None):
        return self._control(services.cancel)

    @extend_schema(
        summary="Chaos: SIGKILL the worker process (G.6 workload termination)",
        tags=TAG,
        request=EMPTY_BODY,
        responses=SimulationSerializer,
    )
    @action(detail=True, methods=["post"], pagination_class=None, filter_backends=[])
    def chaos(self, request, pk=None):
        return self._control(services.chaos)
