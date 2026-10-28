"""Filesystem layout under ``settings.ARTIFACT_ROOT`` (read at call time so tests can override it).

::

    <ARTIFACT_ROOT>/datasets/<id>/loans.parquet       canonical LoanTable + y_* columns
    <ARTIFACT_ROOT>/datasets/<id>/summary.json        cached GET /datasets/{id}/summary/
    <ARTIFACT_ROOT>/fits/<id>/                        FittedModels.save() + API caches (api_*.json)
    <ARTIFACT_ROOT>/fits/<id>/portfolio/<job_id>/     summary.json, per_loan.parquet, request.json
    <ARTIFACT_ROOT>/simulations/<id>/run/             simulation run directory (runner protocol)
    <ARTIFACT_ROOT>/jobs/<id>.log                     stdout/stderr of the worker subprocess
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings


def root() -> Path:
    path = Path(settings.ARTIFACT_ROOT)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(dataset_id: int) -> Path:
    return root() / "datasets" / str(dataset_id)


def fit_dir(fit_id: int) -> Path:
    return root() / "fits" / str(fit_id)


def portfolio_dir(fit_id: int, job_id: int) -> Path:
    return fit_dir(fit_id) / "portfolio" / str(job_id)


def simulation_dir(simulation_id: int) -> Path:
    return root() / "simulations" / str(simulation_id)


def run_dir(simulation_id: int) -> Path:
    return simulation_dir(simulation_id) / "run"


def job_log_path(job_id: int | None) -> Path:
    return root() / "jobs" / f"{job_id}.log"


def reset_dir(path: Path) -> Path:
    """Remove whatever a previous database left at ``path`` (primary keys can be reused after a
    database reset) and recreate it empty, so stale caches or run outputs are never served."""
    import shutil

    path = Path(path)
    resolved = path.resolve()
    if root().resolve() not in resolved.parents:
        raise ValueError(f"refusing to reset {path}: outside ARTIFACT_ROOT")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
