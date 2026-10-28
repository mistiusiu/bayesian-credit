"""Django settings for the Bayesian Credit API (shadow lending engine research console).

Everything deployment-specific comes from the environment (12-factor):

=======================  =================================================================
``DJANGO_SECRET_KEY``    secret key (an insecure development key is used when unset)
``DJANGO_DEBUG``         ``1``/``true`` enables debug (default ``1`` for local development)
``DJANGO_ALLOWED_HOSTS`` comma-separated host list (default ``*``)
``DATABASE_URL``         ``sqlite:///rel/path``, ``sqlite:////abs/path`` or
                         ``postgres://user:pass@host:5432/db?sslmode=require``;
                         default SQLite at ``api/db.sqlite3`` (``api/db/db.sqlite3`` when an
                         ``api/db/`` directory exists, e.g. the docker volume; ``SQLITE_PATH``
                         overrides the file)
``DATA_ROOT``            data directory (default ``<repo>/data``)
``ARTIFACT_ROOT``        datasets, fits, simulation run dirs, job logs
                         (default ``<DATA_ROOT>/artifacts``)
``LENDING_CLUB_PATH``    Kaggle CSV (default ``<DATA_ROOT>/raw/accepted_2007_to_2018Q4.csv.gz``)
``CORS_ALLOW_ALL_ORIGINS`` open CORS (default: on when debug); ``CORS_ALLOWED_ORIGINS`` list
``JOBS_SYNC``            run background jobs inline in the request (tests, ``manage.py demo``)
``JOB_PROGRESS_INTERVAL`` seconds between job progress writes (default 2)
=======================  =================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent  # api/
REPO_ROOT = BASE_DIR.parent

API_VERSION = "0.1.0"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _sqlite_default_path() -> Path:
    explicit = os.environ.get("SQLITE_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if (BASE_DIR / "db").is_dir():  # docker-compose / k8s volume mounted at api/db
        return BASE_DIR / "db" / "db.sqlite3"
    return BASE_DIR / "db.sqlite3"


SQLITE_OPTIONS = {
    "timeout": 30,
    "transaction_mode": "IMMEDIATE",
    "init_command": "PRAGMA journal_mode=WAL;PRAGMA synchronous=NORMAL;",
}


def parse_database_url(url: str | None) -> dict:
    """Minimal ``DATABASE_URL`` parser (sqlite + postgres), so no extra dependency is needed."""
    if not url:
        return {"ENGINE": "django.db.backends.sqlite3", "NAME": str(_sqlite_default_path()), "OPTIONS": SQLITE_OPTIONS}
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in {"sqlite", "sqlite3"}:
        path = unquote(parts.netloc + parts.path)
        if path in {"", "/", ":memory:", "/:memory:"}:
            name = ":memory:"
        elif path.startswith("//"):  # sqlite:////abs/path
            name = path[1:]
        else:  # sqlite:///relative/path -> relative to api/
            name = str((BASE_DIR / path.lstrip("/")).resolve())
        return {"ENGINE": "django.db.backends.sqlite3", "NAME": name, "OPTIONS": SQLITE_OPTIONS}
    if scheme in {"postgres", "postgresql", "pgsql", "psql"}:
        options = dict(parse_qsl(parts.query))
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": unquote(parts.path.lstrip("/")),
            "USER": unquote(parts.username or ""),
            "PASSWORD": unquote(parts.password or ""),
            "HOST": parts.hostname or "",
            "PORT": str(parts.port or ""),
            "OPTIONS": options,
            "CONN_MAX_AGE": int(os.environ.get("DB_CONN_MAX_AGE", "60")),
            "CONN_HEALTH_CHECKS": True,
        }
    raise ImproperlyConfigured(f"unsupported DATABASE_URL scheme {scheme!r} (use sqlite:// or postgres://)")


# --------------------------------------------------------------------------- core
DEBUG = env_bool("DJANGO_DEBUG", True)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or "django-insecure-bayesian-credit-development-key-change-me"
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", ["*"])
CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "django_filters",
    "drf_spectacular",
    "apps.common",
    "apps.datasets",
    "apps.fits",
    "apps.simulations",
    "apps.sentinel",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {"default": parse_database_url(os.environ.get("DATABASE_URL"))}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
try:  # WhiteNoise warns at start-up when the directory is missing (before collectstatic)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:  # read-only image: collectstatic must have created it
    pass
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

# --------------------------------------------------------------------------- CORS
CORS_ALLOW_ALL_ORIGINS = env_bool("CORS_ALLOW_ALL_ORIGINS", DEBUG)
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS")

# --------------------------------------------------------------------------- data & jobs
DATA_ROOT = env_path("DATA_ROOT", REPO_ROOT / "data")
ARTIFACT_ROOT = env_path("ARTIFACT_ROOT", DATA_ROOT / "artifacts")
LENDING_CLUB_PATH = env_path("LENDING_CLUB_PATH", DATA_ROOT / "raw" / "accepted_2007_to_2018Q4.csv.gz")
JOBS_SYNC = env_bool("JOBS_SYNC", False)
JOB_PROGRESS_INTERVAL = float(os.environ.get("JOB_PROGRESS_INTERVAL", "2"))
#: Only the in-pod subprocess executor is implemented by the API; see deploy/k8s/job-template.yaml
#: for running a run directory as a Kubernetes Job by hand.
SIMULATION_EXECUTOR = os.environ.get("SIMULATION_EXECUTOR", "subprocess")
#: Size of the per-process LRU cache of loaded fits (FittedModels + shadow engine).
FIT_CACHE_SIZE = int(os.environ.get("FIT_CACHE_SIZE", "4"))
#: Posterior-predictive draws per arm in POST /fits/{id}/shadow/evaluate/.
SHADOW_SAMPLES = int(os.environ.get("SHADOW_SAMPLES", "2000"))

# --------------------------------------------------------------------------- DRF
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
    "DEFAULT_PAGINATION_CLASS": "apps.common.pagination.DefaultPagination",
    "PAGE_SIZE": 50,
    "DEFAULT_FILTER_BACKENDS": ["django_filters.rest_framework.DjangoFilterBackend"],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "EXCEPTION_HANDLER": "apps.common.exceptions.api_exception_handler",
    "DATETIME_FORMAT": "%Y-%m-%dT%H:%M:%SZ",
    "COERCE_DECIMAL_TO_STRING": False,
    "TEST_REQUEST_DEFAULT_FORMAT": "json",
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Bayesian Credit API",
    "DESCRIPTION": (
        "REST API of the shadow lending engine (DSA4900): datasets (Lending Club / synthetic twin), "
        "hierarchical Two-Stage Beta-Beta + MCTM fits, the counterfactual shadow engine, replay "
        "simulations of contextual Thompson sampling vs baselines, and the PSI Sentinel review queue. "
        "The authoritative JSON contract is frontend/src/api/types.ts (snake_case keys, integer ids, "
        "RFC 3339 UTC timestamps, 'YYYY-MM' epochs, rates as fractions)."
    ),
    "VERSION": API_VERSION,
    "SERVE_INCLUDE_SCHEMA": False,
    "SCHEMA_PATH_PREFIX": r"/api/v1",
    "COMPONENT_SPLIT_REQUEST": True,
    "ENUM_NAME_OVERRIDES": {
        "JobStatusEnum": ["queued", "running", "succeeded", "failed", "cancelled"],
        "BuildStatusEnum": ["queued", "running", "ready", "failed", "cancelled"],
        "SimulationStatusEnum": [
            "queued",
            "running",
            "paused",
            "awaiting_review",
            "interrupted",
            "completed",
            "failed",
            "cancelled",
        ],
        "ReviewStatusEnum": ["pending", "acknowledged"],
        "SentinelLevelEnum": ["normal", "watch", "breaker"],
        "BreakerStateEnum": ["closed", "open", "half_open"],
        "ReviewActionEnum": ["resume", "resegment", "rollback"],
        "DataSourceEnum": ["synthetic", "lending_club"],
        "StageEnum": ["pre_decision", "policy", "disbursement", "interim", "terminal"],
        "HierarchyLevelEnum": ["global", "group", "leaf"],
        "TransitionStructureEnum": ["proposal", "extended"],
        "PsiAggregateEnum": ["max", "mean"],
        "SegmentationDimensionEnum": [
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
        ],
        "JobKindEnum": ["ingest", "fit", "simulation", "shadow_portfolio"],
    },
}

# --------------------------------------------------------------------------- logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "[{asctime}] {levelname} {name}: {message}", "style": "{"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "root": {"handlers": ["console"], "level": "WARNING"},
    "loggers": {
        "apps": {"handlers": ["console"], "level": os.environ.get("API_LOG_LEVEL", "INFO"), "propagate": False},
        "django.request": {"handlers": ["console"], "level": "ERROR", "propagate": False},
    },
}
