"""Small helpers shared by every app: JSON sanitising, epochs, timestamps, atomic files."""

from __future__ import annotations

import calendar
import datetime as dt
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.utils import timezone
from rest_framework.exceptions import ValidationError

EPOCH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])(?:-(\d{2}))?$")


# ----------------------------------------------------------------------------- JSON
def clean(value: Any) -> Any:
    """Recursively convert numpy / pandas / datetime values into JSON-safe Python.

    NaN, ±inf, ``pd.NA`` and ``NaT`` become ``None`` (the contract never uses NaN), numpy
    scalars become Python scalars, timestamps become RFC 3339 strings, enums their value.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [clean(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, Enum):
        return clean(value.value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else iso(value.to_pydatetime())
    if isinstance(value, dt.datetime):
        return iso(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def num(value: Any, digits: int | None = None) -> float | None:
    """Finite float or None (optionally rounded)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits) if digits is not None else f


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, data: Any) -> Path:
    """Write JSON via a temp file + ``os.replace`` so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(clean(data), fh, indent=2, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# ----------------------------------------------------------------------------- time
def utcnow() -> dt.datetime:
    return timezone.now()


def iso(value: dt.datetime | None) -> str | None:
    """RFC 3339 UTC string with second precision, e.g. ``2026-10-16T09:41:07Z``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if timezone.is_naive(value):
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: Any) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value if timezone.is_aware(value) else value.replace(tzinfo=dt.UTC)
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    out = ts.to_pydatetime()
    return out if timezone.is_aware(out) else out.replace(tzinfo=dt.UTC)


def iso_from_any(value: Any) -> str | None:
    return iso(parse_iso(value))


# ----------------------------------------------------------------------------- epochs
def to_epoch(value: Any) -> str | None:
    """``"YYYY-MM"`` from a date-like value (``None`` for missing)."""
    if value is None:
        return None
    if isinstance(value, str):
        m = EPOCH_RE.match(value.strip())
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return f"{ts.year:04d}-{ts.month:02d}"


def parse_epoch(value: Any, field: str = "epoch") -> str:
    """Validate ``YYYY-MM`` (``YYYY-MM-DD`` accepted) and return ``YYYY-MM``; 400 otherwise."""
    if isinstance(value, str):
        m = EPOCH_RE.match(value.strip())
        if m:
            if m.group(3) is not None:
                try:
                    dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    raise ValidationError({field: [f"invalid date {value!r}"]}) from None
            return f"{m.group(1)}-{m.group(2)}"
    raise ValidationError({field: ["Use YYYY-MM (or YYYY-MM-DD)."]})


def epoch_start(epoch: str) -> str:
    """``"2015-01"`` → ``"2015-01-01"``."""
    return f"{epoch}-01"


def epoch_end(epoch: str) -> str:
    """``"2016-06"`` → ``"2016-06-30"`` (last day of the month)."""
    y, m = (int(x) for x in epoch.split("-"))
    return f"{epoch}-{calendar.monthrange(y, m)[1]:02d}"


def epoch_index(epoch: str) -> int:
    y, m = (int(x) for x in epoch.split("-"))
    return y * 12 + (m - 1)


def index_epoch(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def epoch_add(epoch: str, months: int) -> str:
    return index_epoch(epoch_index(epoch) + months)


def epoch_range(start: str, end: str) -> list[str]:
    return [index_epoch(i) for i in range(epoch_index(start), epoch_index(end) + 1)]


# ----------------------------------------------------------------------------- query params
def csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_bool(value: Any, field: str) -> bool | None:
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise ValidationError({field: ["Must be true or false."]})


def parse_int(
    value: Any, field: str, *, default: int | None = None, lo: int | None = None, hi: int | None = None
) -> int | None:
    if value is None or value == "":
        return default
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError({field: ["A valid integer is required."]}) from None
    if lo is not None and out < lo:
        raise ValidationError({field: [f"Must be >= {lo}."]})
    if hi is not None and out > hi:
        raise ValidationError({field: [f"Must be <= {hi}."]})
    return out


def parse_int_list(value: str | None, field: str) -> list[int]:
    out = []
    for item in csv_list(value):
        try:
            out.append(int(item))
        except ValueError:
            raise ValidationError({field: [f"{item!r} is not an integer."]}) from None
    return out


def first(iterable: Iterable[Any], default: Any = None) -> Any:
    for item in iterable:
        return item
    return default
