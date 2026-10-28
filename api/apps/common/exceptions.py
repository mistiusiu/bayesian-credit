"""Uniform error bodies (types.ts ``ApiErrorBody``): ``{"detail": "...", "code": "..."}`` or
DRF field errors ``{"field": ["..."]}``.

Framework exceptions from ``bayesian_credit.exceptions`` are mapped onto HTTP statuses with
the exception class name as ``code`` (e.g. a context with interim columns →
``400 {"detail": "columns not available at stage 'pre_decision': ...", "code": "LeakageError"}``).
"""

from __future__ import annotations

import logging

from bayesian_credit import exceptions as bc
from django.conf import settings
from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger("apps.api")

#: Most specific first.
FRAMEWORK_STATUS: tuple[tuple[type[Exception], int], ...] = (
    (bc.LeakageError, status.HTTP_400_BAD_REQUEST),
    (bc.ConfigurationError, status.HTTP_400_BAD_REQUEST),
    (bc.GovernanceError, status.HTTP_409_CONFLICT),
    (bc.CircuitBreakerOpenError, status.HTTP_409_CONFLICT),
    (bc.CheckpointError, status.HTTP_409_CONFLICT),
    (bc.DataSourceError, status.HTTP_422_UNPROCESSABLE_ENTITY),
    (bc.NumericalInstabilityError, status.HTTP_422_UNPROCESSABLE_ENTITY),
    (bc.InvalidPosteriorError, status.HTTP_422_UNPROCESSABLE_ENTITY),
    (bc.InvalidTransitionMatrixError, status.HTTP_422_UNPROCESSABLE_ENTITY),
    (bc.BayesianCreditError, status.HTTP_422_UNPROCESSABLE_ENTITY),
)


class Conflict(APIException):
    """409 — the resource is not in a state that allows the request (e.g. fit not ready)."""

    status_code = status.HTTP_409_CONFLICT
    default_detail = "The resource is not in a state that allows this request."
    default_code = "conflict"


class Unprocessable(APIException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = "The request could not be processed."
    default_code = "unprocessable"


class ServiceUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "A required component is not available."
    default_code = "unavailable"


def framework_status(exc: Exception) -> int | None:
    for cls, code in FRAMEWORK_STATUS:
        if isinstance(exc, cls):
            return code
    return None


def api_exception_handler(exc: Exception, context: dict) -> Response | None:
    code = framework_status(exc)
    if code is not None:
        return Response({"detail": str(exc), "code": type(exc).__name__}, status=code)

    response = drf_exception_handler(exc, context)
    if response is None:
        view = context.get("view")
        logger.exception("unhandled error in %s", type(view).__name__ if view else "view", exc_info=exc)
        detail = "Internal server error"
        if settings.DEBUG:
            detail = f"{detail}: {type(exc).__name__}: {exc}"
        return Response({"detail": detail, "code": "server_error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    if isinstance(response.data, dict) and "detail" in response.data and "code" not in response.data:
        codes = exc.get_codes() if isinstance(exc, APIException) else None
        if isinstance(codes, str):
            response.data["code"] = codes
    elif isinstance(response.data, list):  # ValidationError raised with a bare list
        response.data = {"non_field_errors": response.data}
    return response
