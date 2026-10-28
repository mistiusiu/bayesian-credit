"""Shared pytest fixtures.

``e2e`` runs the whole research workflow once per session through the public API with inline
jobs (``JOBS_SYNC``): synthetic dataset (6k loans, 2016-06 shift) → fit → shadow evaluate →
shadow portfolio → replay simulation (Thompson / historical / random) → a short simulation
whose PSI breaker trips and leaves a pending review. Tests read the captured payloads and the
committed rows; their own writes are rolled back per test.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("JOBS_SYNC", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
TYPES_TS = REPO_ROOT / "frontend" / "src" / "api" / "types.ts"


def pytest_sessionfinish(session, exitstatus):
    from django.conf import settings

    root = Path(str(getattr(settings, "ARTIFACT_ROOT", "")))
    if root.name.startswith("bc-api-tests-") and root.is_dir():
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def api():
    from rest_framework.test import APIClient

    return APIClient()


def _ok(resp, status=200):
    assert resp.status_code == status, (resp.status_code, getattr(resp, "data", None) or resp.content[:500])
    return resp.json()


@pytest.fixture(scope="session")
def e2e(django_db_setup, django_db_blocker):
    from rest_framework.test import APIClient

    c = APIClient()
    out: dict = {}
    with django_db_blocker.unblock():
        ds = _ok(
            c.post(
                "/api/v1/datasets/",
                {
                    "name": "e2e synthetic",
                    "source": "synthetic",
                    "n": 6000,
                    "seed": 11,
                    "shift": {"at": "2016-06", "deltas": {}},
                },
                format="json",
            ),
            201,
        )
        out["dataset"] = ds
        dsid = ds["id"]
        out["variables"] = _ok(c.get(f"/api/v1/datasets/{dsid}/variables/"))
        out["summary"] = _ok(c.get(f"/api/v1/datasets/{dsid}/summary/"))
        out["loans"] = _ok(c.get(f"/api/v1/datasets/{dsid}/loans/", {"grade": "B", "limit": 5}))

        fit = _ok(
            c.post("/api/v1/fits/", {"dataset": dsid, "name": "e2e fit", "train_end": "2015-01-01"}, format="json"), 201
        )
        out["fit"] = fit
        fid = fit["id"]
        out["groups"] = _ok(c.get(f"/api/v1/fits/{fid}/groups/"))
        group = next(r["key"] for r in out["groups"] if r["level"] == "group" and r["n_children"] > 0)
        out["group"] = group
        out["leaves"] = _ok(c.get(f"/api/v1/fits/{fid}/groups/", {"level": "leaf", "group": group}))
        arm = out["leaves"][0]["arm"]
        out["arm"] = arm
        out["risk"] = _ok(c.get(f"/api/v1/fits/{fid}/risk/", {"group": group, "arm": arm}))
        out["risk_group"] = _ok(c.get(f"/api/v1/fits/{fid}/risk/", {"group": group}))
        out["calibration"] = _ok(c.get(f"/api/v1/fits/{fid}/calibration/"))
        out["mctm"] = _ok(c.get(f"/api/v1/fits/{fid}/mctm/", {"structure": "extended"}))
        out["mctm_default"] = _ok(c.get(f"/api/v1/fits/{fid}/mctm/", {"horizon": 24, "start": "delinquent"}))
        out["arms"] = _ok(c.get(f"/api/v1/fits/{fid}/arms/"))
        loan_id = out["loans"]["results"][0]["loan_id"]
        out["loan_id"] = loan_id
        out["shadow_loan"] = _ok(c.post(f"/api/v1/fits/{fid}/shadow/evaluate/", {"loan_id": loan_id}, format="json"))
        out["shadow_context"] = _ok(
            c.post(
                f"/api/v1/fits/{fid}/shadow/evaluate/",
                {
                    "context": {
                        "annual_inc": 72000,
                        "fico_range_low": 705,
                        "dti": 14.5,
                        "grade": "B",
                        "purpose": "credit_card",
                        "home_ownership": "RENT",
                    },
                    "historical_policy": {"amount": 12000, "rate": 0.11, "term": 36},
                },
                format="json",
            )
        )
        job = _ok(c.post(f"/api/v1/fits/{fid}/shadow/portfolio/", {"max_loans": 1000, "seed": 5}, format="json"), 202)
        out["portfolio_job"] = job
        out["portfolio"] = _ok(c.get(f"/api/v1/fits/{fid}/shadow/portfolio/{job['id']}/"))
        out["portfolio_runs"] = _ok(c.get(f"/api/v1/fits/{fid}/shadow/portfolio/"))

        sim = _ok(
            c.post(
                "/api/v1/simulations/",
                {
                    "fit": fid,
                    "name": "e2e replay",
                    "policies": ["thompson", "historical", "random"],
                    "start": "2015-01-01",
                    "end": "2016-06-30",
                    "max_loans": 1500,
                    "breaker_mode": "auto_ack",
                    "checkpoint_every": 2,
                    "psi": {"thresholds": {"watch": 0.08, "breaker": 0.15}},
                },
                format="json",
            ),
            201,
        )
        out["simulation"] = sim
        sid = sim["id"]
        out["timeseries"] = _ok(c.get(f"/api/v1/simulations/{sid}/timeseries/"))
        out["sim_arms"] = _ok(c.get(f"/api/v1/simulations/{sid}/arms/"))
        out["posteriors"] = _ok(c.get(f"/api/v1/simulations/{sid}/posteriors/"))
        out["simpson"] = _ok(c.get(f"/api/v1/simulations/{sid}/simpson/"))
        out["events"] = _ok(c.get(f"/api/v1/simulations/{sid}/events/", {"limit": 1000}))
        out["sentinel"] = _ok(c.get(f"/api/v1/simulations/{sid}/sentinel/"))
        last = out["sentinel"]["aggregate"][-1]["epoch"]
        out["buckets"] = _ok(
            c.get(f"/api/v1/simulations/{sid}/sentinel/buckets/", {"feature": "fico_range_low", "epoch": last})
        )
        out["report"] = _ok(c.get(f"/api/v1/simulations/{sid}/report/"))

        review_sim = _ok(
            c.post(
                "/api/v1/simulations/",
                {
                    "fit": fid,
                    "name": "e2e review",
                    "policies": ["thompson_se", "historical"],
                    "start": "2015-01",
                    "end": "2015-09",
                    "max_loans": 900,
                    "breaker_mode": "auto_ack",
                    "checkpoint_every": 3,
                    "psi": {"breaker_threshold": 0.03, "watch_threshold": 0.02, "auto_ack_epochs": 50},
                },
                format="json",
            ),
            201,
        )
        out["review_sim"] = review_sim
        out["reviews"] = _ok(c.get("/api/v1/sentinel/reviews/", {"simulation": review_sim["id"]}))
        out["jobs"] = _ok(c.get("/api/v1/jobs/"))
        out["health"] = _ok(c.get("/api/v1/health/"))
    return out
