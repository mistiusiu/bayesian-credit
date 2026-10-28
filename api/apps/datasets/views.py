"""``/datasets/`` — create (ingest job), list, detail, variables, summary, loans."""

from __future__ import annotations

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.common import artifacts
from apps.common import schemas as S
from apps.common.exceptions import Conflict
from apps.common.jobs import JobRunner, reap_lost_workers
from apps.common.models import Job
from apps.common.utils import csv_list, parse_bool, parse_int

from . import services
from .models import Dataset
from .serializers import DatasetCreateSerializer, DatasetSerializer, VariableRowSerializer


def ready_dataset(ds: Dataset) -> Dataset:
    if ds.job is not None and ds.job.is_active:
        reap_lost_workers([ds.job])
        ds.job.refresh_from_db()
    if not ds.is_ready:
        raise Conflict(f"dataset {ds.pk} is not ready (status {ds.current_status})")
    if not ds.parquet.is_file():
        raise Conflict(f"dataset {ds.pk} has no parquet file at {ds.parquet}")
    return ds


@extend_schema_view(
    list=extend_schema(summary="List datasets (newest first)", tags=["datasets"]),
    retrieve=extend_schema(summary="Dataset detail", tags=["datasets"]),
    create=extend_schema(
        summary="Create a dataset (starts an ingest job)",
        tags=["datasets"],
        request=DatasetCreateSerializer,
        responses={201: DatasetSerializer},
    ),
)
class DatasetViewSet(
    mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet
):
    queryset = Dataset.objects.select_related("job").order_by("-id")
    serializer_class = DatasetSerializer
    filterset_fields = ["source"]

    def get_serializer_class(self):
        return DatasetCreateSerializer if self.action == "create" else DatasetSerializer

    def list(self, request, *args, **kwargs):
        reap_lost_workers()
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        ds = self.get_object()
        if ds.job is not None and ds.job.is_active:
            reap_lost_workers([ds.job])
            ds.job.refresh_from_db()
        return Response(DatasetSerializer(ds).data)

    def create(self, request, *args, **kwargs):
        ser = DatasetCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        params = ser.validated_data
        ds = Dataset.objects.create(name=params["name"], source=params["source"], params=params)
        artifacts.reset_dir(ds.directory)
        job = JobRunner.create(Job.Kind.INGEST, target_type="dataset", target_id=ds.pk, params={"dataset": ds.pk})
        ds.job = job
        ds.save(update_fields=["job"])
        JobRunner.launch(job)
        ds.refresh_from_db()
        return Response(DatasetSerializer(ds).data, status=status.HTTP_201_CREATED)

    @extend_schema(
        summary="Table 3 variables with presence / dtype / null share",
        tags=["datasets"],
        responses=VariableRowSerializer(many=True),
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def variables(self, request, pk=None):
        ds = ready_dataset(self.get_object())
        rows = (ds.summary or {}).get("variables")
        if not rows:
            import pandas as pd

            rows = services.variable_rows(pd.read_parquet(ds.parquet))
        return Response(rows)

    @extend_schema(
        summary="Feature distributions, monthly arrivals, status / grade / term counts",
        tags=["datasets"],
        parameters=[OpenApiParameter("features", OpenApiTypes.STR, description="Comma list of Table 3 variables")],
        responses=S.DatasetSummarySerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def summary(self, request, pk=None):
        ds = ready_dataset(self.get_object())
        return Response(services.build_summary(ds, csv_list(request.query_params.get("features"))))

    @extend_schema(
        summary="Browse loans (context + historical policy + post-hoc outcome)",
        tags=["datasets"],
        parameters=[
            OpenApiParameter("limit", OpenApiTypes.INT, description="Page size (default 50, max 500)"),
            OpenApiParameter("offset", OpenApiTypes.INT),
            OpenApiParameter("grade", OpenApiTypes.STR, description="Comma list, e.g. A,B"),
            OpenApiParameter("purpose", OpenApiTypes.STR, description="Comma list"),
            OpenApiParameter("home_ownership", OpenApiTypes.STR, description="Comma list"),
            OpenApiParameter("group", OpenApiTypes.STR, description="Exact y_group key"),
            OpenApiParameter("resolved", OpenApiTypes.BOOL),
            OpenApiParameter("search", OpenApiTypes.STR, description="loan_id substring"),
            OpenApiParameter(
                "ordering", OpenApiTypes.STR, description="loan_id (default), issue_d, int_rate, … prefix - for desc"
            ),
        ],
        responses=S.PaginatedLoanRecordSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def loans(self, request, pk=None):
        ds = ready_dataset(self.get_object())
        q = request.query_params
        limit = parse_int(q.get("limit"), "limit", default=50, lo=1, hi=500)
        offset = parse_int(q.get("offset"), "offset", default=0, lo=0)
        count, page = services.query_loans(
            ds,
            grade=csv_list(q.get("grade")),
            purpose=csv_list(q.get("purpose")),
            home_ownership=csv_list(q.get("home_ownership")),
            group=q.get("group") or None,
            resolved=parse_bool(q.get("resolved"), "resolved"),
            search=q.get("search") or None,
            ordering=q.get("ordering") or "loan_id",
            limit=limit,
            offset=offset,
        )

        def link(new_offset: int) -> str:
            params = q.copy()
            params["limit"] = str(limit)
            params["offset"] = str(new_offset)
            return request.build_absolute_uri(f"{request.path}?{params.urlencode()}")

        return Response(
            {
                "count": count,
                "next": link(offset + limit) if offset + limit < count else None,
                "previous": link(max(0, offset - limit)) if offset > 0 else None,
                "results": services.loan_records(page),
            }
        )
