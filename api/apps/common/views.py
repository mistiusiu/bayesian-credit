"""Platform endpoints: ``/health/``, ``/metrics/`` (Prometheus text) and ``/jobs/``."""

from __future__ import annotations

import os
import platform
from collections import Counter

import bayesian_credit
from django.conf import settings
from django.db import connection
from django.http import HttpRequest, HttpResponse
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import viewsets
from rest_framework.decorators import api_view, permission_classes, renderer_classes
from rest_framework.permissions import AllowAny
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

from . import artifacts
from .filters import JobFilter
from .jobs import reap_lost_workers
from .models import Job
from .serializers import HealthSerializer, JobSerializer


@extend_schema(responses=HealthSerializer, tags=["platform"], summary="Liveness / version")
@api_view(["GET"])
@permission_classes([AllowAny])
@renderer_classes([JSONRenderer])
def health(request):
    status = "ok"
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:  # noqa: BLE001 - report, never raise from a probe
        status = "degraded"
    try:
        root = artifacts.root()
        if not os.access(root, os.W_OK):
            status = "degraded"
    except OSError:
        status = "degraded"
    return Response(
        {
            "status": status,
            "version": settings.API_VERSION,
            "python": platform.python_version(),
            "core_version": getattr(bayesian_credit, "__version__", "unknown"),
        }
    )


def _metric(lines: list[str], name: str, help_text: str, kind: str, samples: list[tuple[dict, float]]) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {kind}")
    for labels, value in samples:
        if labels:
            lab = ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in labels.items())
            lines.append(f"{name}{{{lab}}} {value:g}")
        else:
            lines.append(f"{name} {value:g}")


def metrics(request: HttpRequest) -> HttpResponse:
    """Prometheus text exposition (no client library needed)."""
    from apps.datasets.models import Dataset
    from apps.fits.models import Fit
    from apps.sentinel.models import Review
    from apps.simulations.models import Simulation
    from apps.simulations.services import refresh_status

    reap_lost_workers()
    for sim in Simulation.objects.filter(status__in=Simulation.LIVE_STATUSES).select_related("job"):
        try:
            refresh_status(sim)
        except Exception:  # noqa: BLE001 - metrics must render
            pass

    lines: list[str] = []
    _metric(
        lines,
        "bc_build_info",
        "API build information.",
        "gauge",
        [
            (
                {
                    "version": settings.API_VERSION,
                    "core_version": getattr(bayesian_credit, "__version__", "unknown"),
                    "python": platform.python_version(),
                },
                1,
            )
        ],
    )
    job_counts = Counter(Job.objects.values_list("kind", "status"))
    _metric(
        lines,
        "bc_jobs",
        "Background jobs by kind and status.",
        "gauge",
        [({"kind": k, "status": s}, job_counts.get((k, s), 0)) for k in Job.Kind.values for s in Job.Status.values],
    )
    _metric(
        lines,
        "bc_jobs_active",
        "Queued or running jobs.",
        "gauge",
        [({}, sum(v for (k, s), v in job_counts.items() if s in Job.ACTIVE))],
    )
    sim_counts = Counter(Simulation.objects.values_list("status", flat=True))
    _metric(
        lines,
        "bc_simulations",
        "Simulations by status.",
        "gauge",
        [({"status": s}, sim_counts.get(s, 0)) for s in Simulation.Status.values],
    )
    running = sum(
        1
        for sim in Simulation.objects.filter(status__in=Simulation.LIVE_STATUSES).select_related("job")
        if sim.worker_alive()
    )
    _metric(lines, "bc_simulations_running", "Simulations with a live worker process.", "gauge", [({}, running)])
    restarts = sum(Simulation.objects.values_list("resume_count", flat=True))
    _metric(lines, "bc_simulation_restarts_total", "Resume-from-checkpoint restarts.", "counter", [({}, restarts)])
    ds_counts = Counter(d.status for d in Dataset.objects.select_related("job"))
    _metric(
        lines, "bc_datasets", "Datasets by status.", "gauge", [({"status": s}, n) for s, n in sorted(ds_counts.items())]
    )
    fit_counts = Counter(f.status for f in Fit.objects.select_related("job"))
    _metric(lines, "bc_fits", "Fits by status.", "gauge", [({"status": s}, n) for s, n in sorted(fit_counts.items())])
    rv_counts = Counter(Review.objects.values_list("status", flat=True))
    _metric(
        lines,
        "bc_reviews",
        "Sentinel reviews by status.",
        "gauge",
        [({"status": s}, rv_counts.get(s, 0)) for s in ("pending", "acknowledged")],
    )
    return HttpResponse("\n".join(lines) + "\n", content_type="text/plain; version=0.0.4; charset=utf-8")


@extend_schema_view(
    list=extend_schema(
        summary="List jobs (newest first)",
        tags=["jobs"],
        parameters=[
            OpenApiParameter("status", OpenApiTypes.STR, description="Comma list, e.g. queued,running"),
            OpenApiParameter(
                "kind", OpenApiTypes.STR, description="Comma list of ingest,fit,simulation,shadow_portfolio"
            ),
        ],
    ),
    retrieve=extend_schema(summary="Job status", tags=["jobs"]),
)
class JobViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Job.objects.all().order_by("-id")
    serializer_class = JobSerializer
    filterset_class = JobFilter

    def list(self, request, *args, **kwargs):
        reap_lost_workers()
        return super().list(request, *args, **kwargs)

    def retrieve(self, request, *args, **kwargs):
        job = self.get_object()
        if job.is_active:
            reap_lost_workers([job])
            job.refresh_from_db()
        return Response(self.get_serializer(job).data)


@extend_schema(exclude=True)
@api_view(["GET"])
@permission_classes([AllowAny])
@renderer_classes([JSONRenderer])
def api_root(request):
    base = request.build_absolute_uri("/api/v1/")
    names = ["health", "metrics", "jobs", "datasets", "fits", "simulations", "sentinel/reviews"]
    return Response(
        {
            **{name.replace("/", "_"): f"{base}{name}/" for name in names},
            "schema": request.build_absolute_uri("/api/schema/"),
            "docs": request.build_absolute_uri("/api/docs/"),
        }
    )


def not_found(request: HttpRequest, exception=None) -> HttpResponse:
    if request.path.startswith("/api/"):
        from django.http import JsonResponse

        return JsonResponse({"detail": "Not found.", "code": "not_found"}, status=404)
    from django.views.defaults import page_not_found

    return page_not_found(request, exception)


def server_error(request: HttpRequest) -> HttpResponse:
    if request.path.startswith("/api/"):
        from django.http import JsonResponse

        return JsonResponse({"detail": "Internal server error", "code": "server_error"}, status=500)
    from django.views.defaults import server_error as default_server_error

    return default_server_error(request)
