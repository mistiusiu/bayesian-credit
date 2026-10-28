"""``/fits/`` — create (fit job), list, detail, groups, risk, calibration, mctm, arms and the
shadow engine (``shadow/evaluate/``, ``shadow/portfolio/``)."""

from __future__ import annotations

from bayesian_credit.exceptions import LeakageError
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotFound
from rest_framework.response import Response

from apps.common import artifacts
from apps.common import schemas as S
from apps.common.exceptions import Conflict
from apps.common.jobs import JobRunner, reap_lost_workers
from apps.common.models import Job
from apps.common.serializers import JobSerializer
from apps.common.utils import parse_int

from . import services
from .models import Fit
from .serializers import FitCreateSerializer, FitSerializer, PortfolioRequestSerializer, ShadowEvaluateSerializer

TAG = ["fits"]
SHADOW = ["shadow"]


@extend_schema_view(
    list=extend_schema(
        summary="List fits (newest first)", tags=TAG, parameters=[OpenApiParameter("dataset", OpenApiTypes.INT)]
    ),
    retrieve=extend_schema(summary="Fit detail (config, status, holdout metrics)", tags=TAG),
    create=extend_schema(
        summary="Fit the risk models (starts a fit job)",
        tags=TAG,
        request=FitCreateSerializer,
        responses={201: FitSerializer},
    ),
)
class FitViewSet(mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    queryset = Fit.objects.select_related("dataset", "job").order_by("-id")
    serializer_class = FitSerializer
    filterset_fields = ["dataset"]

    def get_serializer_class(self):
        return FitCreateSerializer if self.action == "create" else FitSerializer

    def _ready(self) -> Fit:
        return services.require_ready(self.get_object())

    def list(self, request, *args, **kwargs):
        reap_lost_workers()
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        fit = self.get_object()
        if fit.job is not None and fit.job.is_active:
            reap_lost_workers([fit.job])
            fit.job.refresh_from_db()
        return Response(FitSerializer(fit).data)

    def create(self, request, *args, **kwargs):
        ser = FitCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        v = ser.validated_data
        ds = v["dataset_obj"]
        if not ds.is_ready:
            raise Conflict(f"dataset {ds.pk} is not ready (status {ds.current_status})")
        fit = Fit.objects.create(
            name=v["name"],
            dataset=ds,
            config=v["config"],
            framework=v["framework"].to_dict(),
            train_end=v["train_end_epoch"],
        )
        artifacts.reset_dir(artifacts.fit_dir(fit.pk))
        job = JobRunner.create(Job.Kind.FIT, target_type="fit", target_id=fit.pk, params={"fit": fit.pk})
        fit.job = job
        fit.save(update_fields=["job"])
        JobRunner.launch(job)
        fit.refresh_from_db()
        return Response(FitSerializer(fit).data, status=status.HTTP_201_CREATED)

    @extend_schema(
        summary="Hierarchy rows (global → group → leaf) with shrinkage",
        tags=TAG,
        parameters=[
            OpenApiParameter("level", OpenApiTypes.STR, enum=["global", "group", "leaf"]),
            OpenApiParameter("group", OpenApiTypes.STR, description="Group key; required for level=leaf"),
        ],
        responses=S.HierarchyRowSerializer(many=True),
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def groups(self, request, pk=None):
        q = request.query_params
        return Response(services.hierarchy_rows(self._ready(), q.get("level") or None, q.get("group") or None))

    @extend_schema(
        summary="Posterior risk of a group (× arm) with density grids and reward breakdown",
        tags=TAG,
        parameters=[
            OpenApiParameter("group", OpenApiTypes.STR, required=True),
            OpenApiParameter("arm", OpenApiTypes.INT),
        ],
        responses=S.RiskEstimateSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def risk(self, request, pk=None):
        from rest_framework.exceptions import ValidationError

        group = request.query_params.get("group")
        if not group:
            raise ValidationError({"group": ["This query parameter is required."]})
        arm = parse_int(request.query_params.get("arm"), "arm")
        return Response(services.risk_payload(self._ready(), group, arm))

    @extend_schema(summary="Holdout reliability, ROC and recovery fit", tags=TAG, responses=S.CalibrationSerializer)
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def calibration(self, request, pk=None):
        return Response(services.calibration_payload(self._ready()))

    @extend_schema(
        summary="Markov chain transition matrix, absorption, trajectory, P(default, t)",
        tags=TAG,
        parameters=[
            OpenApiParameter("structure", OpenApiTypes.STR, enum=["proposal", "extended"]),
            OpenApiParameter("horizon", OpenApiTypes.INT, description="1..240, default 60"),
            OpenApiParameter("start", OpenApiTypes.STR, enum=["current", "delinquent"]),
        ],
        responses=S.MctmSerializer,
    )
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def mctm(self, request, pk=None):
        q = request.query_params
        horizon = parse_int(q.get("horizon"), "horizon", default=60, lo=1, hi=240)
        return Response(
            services.mctm_payload(self._ready(), q.get("structure") or None, horizon, q.get("start") or "current")
        )

    @extend_schema(summary="Arm table with pooled expected reward per arm", tags=TAG, responses=S.FitArmsSerializer)
    @action(detail=True, methods=["get"], pagination_class=None, filter_backends=[])
    def arms(self, request, pk=None):
        return Response(services.arms_payload(self._ready()))

    @extend_schema(
        summary="Shadow-evaluate one borrower (by loan_id or by context)",
        tags=SHADOW,
        request=ShadowEvaluateSerializer,
        responses=S.ShadowReportSerializer,
    )
    @action(detail=True, methods=["post"], url_path="shadow/evaluate", pagination_class=None, filter_backends=[])
    def shadow_evaluate(self, request, pk=None):
        fit = self._ready()
        body = ShadowEvaluateSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        v = body.validated_data
        if v.get("loan_id") is not None:
            return Response(services.shadow_by_loan(fit, v["loan_id"]))
        try:
            return Response(services.shadow_by_context(fit, v["context"], v.get("historical_policy")))
        except LeakageError as exc:
            msg = f"LeakageError: {exc}"
            return Response(
                {"detail": msg, "code": "LeakageError", "context": [msg]}, status=status.HTTP_400_BAD_REQUEST
            )

    @extend_schema(
        methods=["GET"],
        summary="Shadow-portfolio runs of this fit (newest first)",
        tags=SHADOW,
        responses=S.PortfolioRunSerializer(many=True),
    )
    @extend_schema(
        methods=["POST"],
        summary="Start a shadow-portfolio job",
        tags=SHADOW,
        request=PortfolioRequestSerializer,
        responses={202: JobSerializer},
    )
    @action(
        detail=True, methods=["get", "post"], url_path="shadow/portfolio", pagination_class=None, filter_backends=[]
    )
    def shadow_portfolio(self, request, pk=None):
        fit = self._ready()
        if request.method == "POST":
            body = PortfolioRequestSerializer(data=request.data)
            body.is_valid(raise_exception=True)
            req = {
                "max_loans": body.validated_data["max_loans"],
                "seed": body.validated_data["seed"],
                "filters": body.validated_data.get("filters") or {},
            }
            job = JobRunner.create(
                Job.Kind.SHADOW_PORTFOLIO, target_type="fit", target_id=fit.pk, params={"request": req}
            )
            job = JobRunner.launch(job)
            return Response(JobSerializer(job).data, status=status.HTTP_202_ACCEPTED)
        jobs = list(
            Job.objects.filter(kind=Job.Kind.SHADOW_PORTFOLIO, target_type="fit", target_id=fit.pk).order_by("-id")
        )
        reap_lost_workers([j for j in jobs if j.is_active])
        out = []
        for job in jobs:
            job.refresh_from_db()
            res = services.portfolio_result(fit, job)
            out.append({"job": res["job"], "request": res["request"], "summary": res["summary"]})
        return Response(out)

    @extend_schema(
        summary="Shadow-portfolio result (summary is null until the job succeeds)",
        tags=SHADOW,
        parameters=[OpenApiParameter("job_id", OpenApiTypes.INT, OpenApiParameter.PATH)],
        responses=S.PortfolioResultSerializer,
    )
    @action(
        detail=True,
        methods=["get"],
        url_path=r"shadow/portfolio/(?P<job_id>[0-9]+)",
        pagination_class=None,
        filter_backends=[],
    )
    def shadow_portfolio_result(self, request, pk=None, job_id=None):
        fit = self.get_object()
        job = Job.objects.filter(
            pk=int(job_id), kind=Job.Kind.SHADOW_PORTFOLIO, target_type="fit", target_id=fit.pk
        ).first()
        if job is None:
            raise NotFound("portfolio job not found")
        if job.is_active:
            reap_lost_workers([job])
            job.refresh_from_db()
        return Response(services.portfolio_result(fit, job))
