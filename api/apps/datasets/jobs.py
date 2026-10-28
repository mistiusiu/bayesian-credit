"""Ingest job: ``load_loans(..., with_outcomes=True)`` → ``loans.parquet`` + ``summarize``."""

from __future__ import annotations

import time
from typing import Any

from bayesian_credit.config import ArmSpaceConfig, SegmentationConfig
from bayesian_credit.data import ArmSpace, LendingClubLoader, Segmenter, load_loans, summarize
from django.conf import settings

from apps.common.jobs import ProgressReporter
from apps.common.models import Job
from apps.common.utils import clean, epoch_end, epoch_start, to_epoch

from .models import Dataset
from .services import variable_rows


def loader_kwargs(ds: Dataset) -> dict[str, Any]:
    """Map the (defaults-filled) DatasetCreate params onto ``load_loans`` keyword arguments."""
    p = ds.params or {}
    if ds.source == Dataset.Source.SYNTHETIC:
        kw: dict[str, Any] = {
            "n": int(p.get("n") or 200_000),
            "seed": int(p.get("seed") if p.get("seed") is not None else 42),
            "start": p.get("issue_from") or "2012-01",
            "end": p.get("issue_to") or "2018-12",
        }
        shift = p.get("shift")
        if shift:
            kw["shift_at"] = shift.get("at")
            kw["shift"] = dict(shift.get("deltas") or {}) or None
        if p.get("sample_fraction") is not None:
            kw["sample_fraction"] = float(p["sample_fraction"])
        return kw
    kw = {
        "path": p.get("path") or str(settings.LENDING_CLUB_PATH),
        "seed": int(p.get("seed") if p.get("seed") is not None else 42),
        "max_rows": p.get("max_rows"),
        "sample_fraction": p.get("sample_fraction"),
        "issue_from": epoch_start(p["issue_from"]) if p.get("issue_from") else None,
        "issue_to": epoch_end(p["issue_to"]) if p.get("issue_to") else None,
    }
    if p.get("max_rows"):
        kw["order"] = "random"  # a seeded subset spanning every issue year (DATA.md §2)
    return kw


def run_ingest(job: Job, reporter: ProgressReporter) -> None:
    ds = Dataset.objects.get(pk=job.target_id)
    Dataset.objects.filter(pk=ds.pk).update(status=Dataset.Status.RUNNING)
    kw = loader_kwargs(ds)
    reporter.log(f"loading {ds.source} loans ({', '.join(f'{k}={v}' for k, v in kw.items() if v is not None)})", 0.05)
    t0 = time.perf_counter()
    df = load_loans(
        ds.source,
        with_outcomes=True,
        arm_space=ArmSpace.from_config(ArmSpaceConfig()),
        segmenter=Segmenter.from_config(SegmentationConfig()),
        **kw,
    )
    reporter.log(f"loaded {len(df):,} loans and derived outcomes in {time.perf_counter() - t0:.1f}s", 0.6)
    meta = dict(df.attrs.get("synthetic") or {})
    df = df.sort_values("loan_id", kind="mergesort").reset_index(drop=True)
    df.attrs = {}
    path = ds.directory / "loans.parquet"
    LendingClubLoader.to_parquet(df, path)
    reporter.log(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)", 0.8)
    summary = summarize(df)
    summary["variables"] = variable_rows(df)
    if meta:
        summary["synthetic"] = meta
    summary = clean(summary)
    Dataset.objects.filter(pk=ds.pk).update(
        status=Dataset.Status.READY,
        parquet_path=str(path),
        n_loans=int(summary["n"]),
        n_resolved=int(summary.get("n_resolved") or 0),
        issue_min=to_epoch(summary.get("issue_min")),
        issue_max=to_epoch(summary.get("issue_max")),
        resolved_share=summary.get("resolved_share"),
        default_rate=summary.get("default_rate"),
        status_counts=summary.get("status_counts") or {},
        summary=summary,
    )
    reporter.log(
        f"dataset ready: {summary['n']:,} loans {to_epoch(summary.get('issue_min'))}..{to_epoch(summary.get('issue_max'))}, "
        f"resolved {summary.get('resolved_share') or 0:.1%}, default rate {summary.get('default_rate') or 0:.1%}",
        1.0,
    )


def on_failure(job: Job, exc: BaseException) -> None:
    Dataset.objects.filter(pk=job.target_id).update(status=Dataset.Status.FAILED)
