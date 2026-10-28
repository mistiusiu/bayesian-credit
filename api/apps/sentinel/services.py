"""Review queue sync: ``<run_dir>/reviews/<review_id>.json`` (runner) → ``Review`` rows (API).

The runner writes one record per breaker trip; the API mirrors them on read and writes
``<review_id>.ack.json`` when a human acknowledges (the runner consumes it: in
``breaker_mode="pause"`` the paused worker resumes; with a dead worker the ack waits for
the next resume).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from bayesian_credit.data.segmentation import Segmenter
from bayesian_credit.simulation.store import RunDirectory
from django.db import IntegrityError, transaction

from apps.common.exceptions import Conflict
from apps.common.utils import num, parse_iso, to_epoch, utcnow
from apps.simulations.models import Simulation

from .models import Review


def _top_features(per_feature: dict[str, Any] | None, k: int = 3) -> list[dict[str, Any]]:
    items = [(str(f), num(v)) for f, v in (per_feature or {}).items()]
    items = [(f, v) for f, v in items if v is not None]
    items.sort(key=lambda fv: -fv[1])
    return [{"feature": f, "psi": round(v, 6)} for f, v in items[:k]]


def _segmentation_before(sim: Simulation, rec: dict[str, Any]) -> list[str]:
    for key in ("segmentation_before", "segmentation", "dimensions_before", "dimensions"):
        dims = rec.get(key)
        if isinstance(dims, list) and dims:
            return [str(d) for d in dims]
    fit_cfg = (sim.fit.config or {}).get("segmentation") or {}
    return list(fit_cfg.get("dimensions") or [])


def _refined(dims: list[str]) -> list[str] | None:
    try:
        return list(Segmenter(dims).refine().dimensions)
    except Exception:  # noqa: BLE001 - informative only
        return None


def _reason(sim: Simulation, psi: float | None, tops: list[dict[str, Any]]) -> str:
    threshold = ((sim.config or {}).get("psi") or {}).get("thresholds", {}).get("breaker", 0.25)
    head = f"aggregate PSI {psi:.3f}" if psi is not None else "aggregate PSI"
    tail = f" ({tops[0]['feature']} {tops[0]['psi']:.3f})" if tops else ""
    return f"{head} exceeded the breaker threshold {threshold}{tail}"


def _record_created_at(rd: RunDirectory, rec: dict[str, Any]) -> datetime:
    for key in ("created_at", "ts", "opened_at"):
        ts = parse_iso(rec.get(key))
        if ts is not None:
            return ts
    try:
        return datetime.fromtimestamp(os.path.getmtime(rd.review_path(str(rec["id"]))), tz=UTC)
    except OSError:
        return utcnow()


def sync_reviews(sim: Simulation) -> list[Review]:
    """Upsert one ``Review`` per runner review record of ``sim``.

    Reads first and writes only what changed (this runs on every polled simulation detail, and
    a write transaction would contend with the worker for SQLite's lock); an acknowledgement
    made through the API is never downgraded.
    """
    if not sim.run_dir:
        return []
    rd = RunDirectory(sim.directory)
    records = rd.list_reviews()
    if not records:
        return []
    existing = {r.review_id: r for r in Review.objects.filter(simulation=sim)}
    out = []
    for rec in records:
        rid = str(rec.get("id") or "")
        if not rid:
            continue
        ack = rec.get("ack") or None  # the <id>.ack.json file (API or the runner's auto ack)
        tops = _top_features(rec.get("per_feature") or {})
        psi = num(rec.get("psi_aggregate"))
        acknowledged = rec.get("status") == "acknowledged" or ack is not None
        action = rec.get("action") or (ack or {}).get("action")
        auto = bool((ack or {}).get("auto", False))
        before = _segmentation_before(sim, rec)
        after = rec.get("segmentation_after") or rec.get("dimensions_after")
        if not after and action == "resegment":
            after = _refined(before)
        values = {
            "epoch": to_epoch(rec.get("epoch")) or str(rec.get("epoch") or ""),
            "created_at": _record_created_at(rd, rec),
            "psi_aggregate": psi,
            "level": str(rec.get("level") or "breaker"),
            "reason": str(rec.get("reason") or _reason(sim, psi, tops)),
            "top_features": tops,
            "segmentation_before": before,
            "record": {k: v for k, v in rec.items() if k != "ack"},
        }
        review = existing.get(rid)
        if review is None:
            try:
                with transaction.atomic():
                    review = Review.objects.create(simulation=sim, review_id=rid, breaker_state="open", **values)
            except IntegrityError:  # created concurrently by another request
                review = Review.objects.get(simulation=sim, review_id=rid)
        changed = [key for key, value in values.items() if getattr(review, key) != value]
        for key in changed:
            setattr(review, key, values[key])
        if acknowledged and review.status != Review.Status.ACKNOWLEDGED:
            review.status = Review.Status.ACKNOWLEDGED
            review.action = action
            review.note = str(rec.get("note") or (ack or {}).get("note") or "")
            review.auto = auto
            review.acknowledged_at = (
                parse_iso((ack or {}).get("at")) or parse_iso(rec.get("acknowledged_at")) or utcnow()
            )
            review.acknowledged_by = "auto_ack" if auto else str((ack or {}).get("by") or "researcher")
            review.breaker_state = "half_open"
            changed += ["status", "action", "note", "auto", "acknowledged_at", "acknowledged_by", "breaker_state"]
        if acknowledged and after and review.segmentation_after != after:
            review.segmentation_after = list(after)
            changed.append("segmentation_after")
        if changed:
            review.save(update_fields=sorted(set(changed)))
        out.append(review)
    return out


def sync_all(simulations: Iterable[Simulation] | None = None) -> None:
    sims = simulations if simulations is not None else Simulation.objects.select_related("fit").exclude(run_dir="")
    for sim in sims:
        reviews_dir = sim.directory / "reviews"
        if reviews_dir.is_dir() and any(reviews_dir.iterdir()):
            sync_reviews(sim)


def acknowledge(review: Review, action: str, note: str, by: str = "researcher") -> Review:
    """Write ``<review_id>.ack.json`` and mark the row acknowledged (409 if already done)."""
    sim = review.simulation
    sync_reviews(sim)
    review.refresh_from_db()
    rd = RunDirectory(sim.directory)
    if review.status == Review.Status.ACKNOWLEDGED or rd.read_ack(review.review_id) is not None:
        raise Conflict(f"review {review.pk} is already acknowledged")
    rd.write_ack(review.review_id, action, note, by=by, review=review.pk)
    review.status = Review.Status.ACKNOWLEDGED
    review.action = action
    review.note = note
    review.acknowledged_at = utcnow()
    review.acknowledged_by = by
    review.auto = False
    review.breaker_state = "half_open"
    if action == Review.Action.RESEGMENT:
        review.segmentation_after = _refined(list(review.segmentation_before or []))
    review.save(
        update_fields=[
            "status",
            "action",
            "note",
            "acknowledged_at",
            "acknowledged_by",
            "auto",
            "breaker_state",
            "segmentation_after",
        ]
    )
    return review
