"""Settings for ``pytest`` (see ``pytest.ini``): in-memory SQLite, inline jobs, temp artifacts."""

import os
import tempfile
from pathlib import Path

from .settings import *  # noqa: F403

DEBUG = False
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
JOBS_SYNC = True
ARTIFACT_ROOT = Path(os.environ.get("TEST_ARTIFACT_ROOT") or tempfile.mkdtemp(prefix="bc-api-tests-"))
SHADOW_SAMPLES = 500
JOB_PROGRESS_INTERVAL = 0.0
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
LOGGING = {"version": 1, "disable_existing_loggers": False, "root": {"level": "WARNING"}}
