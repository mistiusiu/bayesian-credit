"""Health, metrics, jobs, error bodies, request validation and the OpenAPI schema."""

from __future__ import annotations

import os

import pytest
from bayesian_credit import exceptions as bc

pytestmark = pytest.mark.django_db


def test_health(api):
    body = api.get("/api/v1/health/").json()
    assert body["status"] == "ok" and body["version"] and body["python"] and body["core_version"]


def test_metrics_prometheus_text(api, e2e):
    r = api.get("/api/v1/metrics/")
    assert r.status_code == 200 and r["Content-Type"].startswith("text/plain")
    text = r.content.decode()
    assert 'bc_jobs{kind="simulation",status="succeeded"}' in text
    assert "bc_simulations_running 0" in text and "bc_build_info{" in text


def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/v1/does-not-exist/")
    assert r.status_code == 404 and r.json() == {"detail": "Not found.", "code": "not_found"}


def test_root_redirects_to_docs(client):
    r = client.get("/")
    assert r.status_code == 302 and r["Location"] == "/api/docs/"


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (bc.LeakageError("x"), 400),
        (bc.ConfigurationError("x"), 400),
        (bc.GovernanceError("x"), 409),
        (bc.CheckpointError("x"), 409),
        (bc.DataSourceError("x"), 422),
        (bc.NumericalInstabilityError("x"), 422),
    ],
)
def test_framework_exceptions_mapping(exc, status):
    from apps.common.exceptions import api_exception_handler

    resp = api_exception_handler(exc, {"view": None})
    assert resp.status_code == status
    assert resp.data == {"detail": "x", "code": type(exc).__name__}


def test_unhandled_exception_is_json_500():
    from apps.common.exceptions import api_exception_handler

    resp = api_exception_handler(RuntimeError("boom"), {"view": None})
    assert resp.status_code == 500 and resp.data["code"] == "server_error"


def test_pid_helpers():
    import subprocess
    import sys

    from apps.common.jobs import pid_alive

    assert pid_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not pid_alive(proc.pid)
    assert not pid_alive(None)


def test_reap_marks_dead_workers(api):
    import subprocess
    import sys

    from apps.common.jobs import reap_lost_workers
    from apps.common.models import Job

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    job = Job.objects.create(kind="fit", status="running", pid=proc.pid)
    assert reap_lost_workers() >= 1
    job.refresh_from_db()
    assert job.status == "failed" and job.message == "worker lost" and job.finished_at


# ----------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"source": "synthetic"}, "name"),
        ({"name": "x", "source": "csv"}, "source"),
        ({"name": "x", "source": "synthetic", "n": 10}, "n"),
        ({"name": "x", "source": "synthetic", "issue_from": "2015-13"}, "issue_from"),
        ({"name": "x", "source": "synthetic", "shift": {"at": "2016-06", "deltas": {"loan_status": 1}}}, "shift"),
        ({"name": "x", "source": "synthetic", "sample_fraction": 1.5}, "sample_fraction"),
        ({"name": "x", "source": "lending_club", "path": "/etc/passwd"}, "path"),
    ],
)
def test_dataset_create_validation(api, body, field):
    r = api.post("/api/v1/datasets/", body, format="json")
    assert r.status_code == 400 and field in r.json(), r.json()


def test_fit_requires_ready_dataset(api):
    from apps.datasets.models import Dataset

    ds = Dataset.objects.create(name="pending", source="synthetic", status="queued", params={})
    r = api.post("/api/v1/fits/", {"dataset": ds.pk, "name": "f"}, format="json")
    assert r.status_code == 409
    assert api.get(f"/api/v1/datasets/{ds.pk}/variables/").status_code == 409


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ({"arm_space": {"rates": [13, 16]}}, "arm_space"),
        ({"hierarchy": {"kappa_group": "sometimes"}}, "hierarchy.kappa_group"),
        ({"recovery": {"min_obs": 0}}, "recovery.min_obs"),
        ({"mctm": {"structure": "circular"}}, "mctm.structure"),
        ({"reward": {"funding_rate": 0.9}}, "reward.funding_rate"),
        ({"mctm": {"structure": "proposal", "colour": "red"}}, "mctm"),
        ({"segmentation": {"dimensions": ["grade", "grade"]}}, "segmentation"),
        ({"segmentation": {"dimensions": ["zip_code"]}}, "segmentation"),
        ({"train_end": "2030-01"}, "train_end"),
        ({"dataset": 999999}, "dataset"),
        ({"arm_space": {"max_total_dti": -0.1}}, "arm_space.max_total_dti"),
        ({"arm_space": {"rate_corridor": "wide"}}, "arm_space.rate_corridor"),
        ({"hierarchy": {"policy_effect": "on"}}, "hierarchy.policy_effect"),
        ({"hierarchy": {"policy_effect_cfg": {"features": ["zip"]}}}, "non_field_errors"),
        ({"recovery": {"term_level": "yes"}}, "recovery.term_level"),
        ({"reward": {"prepayment_adjustment": "magic"}}, "reward.prepayment_adjustment"),
    ],
)
def test_fit_create_validation(api, e2e, extra, field):
    body = {"dataset": e2e["dataset"]["id"], "name": "bad", **extra}
    r = api.post("/api/v1/fits/", body, format="json")
    assert r.status_code == 400 and field in r.json(), r.json()


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ({"policies": []}, "policies"),
        ({"policies": ["thompson", "teleport"]}, "policies"),
        ({"policies": ["random", "random"]}, "policies"),
        ({"start": "2016-06", "end": "2015-01"}, "start"),
        ({"end": "2019-06"}, "end"),
        ({"psi": {"thresholds": {"watch": 0.3, "breaker": 0.2}}}, "psi"),
        ({"psi": {"features": ["loan_status"]}}, "psi"),
        ({"breaker_mode": "panic"}, "breaker_mode"),
        ({"interim_weight": 2}, "interim_weight"),
        ({"fit": 999999}, "fit"),
        ({"hindsight_sample": "everything"}, "hindsight_sample"),
    ],
)
def test_simulation_create_validation(api, e2e, extra, field):
    body = {"fit": e2e["fit"]["id"], "name": "bad", "policies": ["thompson"], **extra}
    r = api.post("/api/v1/simulations/", body, format="json")
    assert r.status_code == 400 and field in r.json(), r.json()


def test_policy_specs_are_normalised(api, e2e):
    from apps.simulations.serializers import SimulationCreateSerializer

    ser = SimulationCreateSerializer(
        data={
            "fit": e2e["fit"]["id"],
            "name": "x",
            "policies": ["Thompson", "epsilon_greedy(epsilon=0.05)", "thompson_se"],
        }
    )
    assert ser.is_valid(), ser.errors
    assert ser.validated_data["config"]["policies"] == ["thompson", "epsilon_greedy(epsilon=0.05)", "thompson_se"]


def test_openapi_schema_generates_without_warnings(tmp_path):
    from django.core.management import call_command

    out = tmp_path / "schema.yaml"
    call_command("spectacular", "--validate", "--fail-on-warn", "--file", str(out))
    text = out.read_text()
    assert "/api/v1/simulations/{id}/chaos/" in text and "/api/v1/fits/{id}/shadow/portfolio/{job_id}/" in text


def test_swagger_ui_served(client):
    assert client.get("/api/docs/").status_code == 200
    assert client.get("/api/schema/").status_code == 200
