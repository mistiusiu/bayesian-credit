"""Tiny in-process caches keyed by file identity (path, mtime, size) — run-directory files are
replaced atomically by the worker, so a changed file always has a new mtime/inode."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_MAX = 256


def file_key(*paths: Path) -> tuple:
    key = []
    for p in paths:
        try:
            st = os.stat(p)
            key.append((str(p), st.st_mtime_ns, st.st_size, st.st_ino))
        except OSError:
            key.append((str(p), None))
    return tuple(key)


def dir_key(directory: Path, pattern: str = "*") -> tuple:
    if not directory.is_dir():
        return (str(directory), None)
    return file_key(*sorted(directory.glob(pattern)))


def memo(namespace: str, key: tuple, compute: Callable[[], Any]) -> Any:
    full = (namespace, key)
    with _LOCK:
        if full in _CACHE:
            _CACHE.move_to_end(full)
            return _CACHE[full]
    value = compute()
    with _LOCK:
        _CACHE[full] = value
        _CACHE.move_to_end(full)
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return value


def clear() -> None:
    with _LOCK:
        _CACHE.clear()
