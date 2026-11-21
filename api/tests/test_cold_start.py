"""Cold-start simulations through the public API: policies without an offline model, with or
without a trained fit (inline jobs; the dataset and fit come from ``conftest.e2e``)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.django_db

BODY = {
    "policies": ["greedy", "thompson", "historical"],
    "start": "2015-01",
    "end": "2015-06",
    "max_loans": 600,
    "breaker_mode": "off",
    "checkpoint_every": 3,
}


def _create(api, **extra):
    return api.post("/api/v1/simulations/", {**BODY, "name": "cold", **extra}, format="json")


@pytest.fixture
def fitless(api, e2e):
    r = _create(api, dataset=e2e["dataset"]["id"], cold_start=True)
    assert r.status_code == 201, r.json()
    return r.json()


def test_a_cold_start_runs_from_a_dataset_alone(fitless, e2e):
    assert fitless["status"] == "completed"
    assert fitless["fit"] is None and fitless["fit_name"] is None
    assert fitless["dataset"] == e2e["dataset"]["id"] and fitless["dataset_name"] == e2e["dataset"]["name"]
    assert fitless["cold_start"] is True and fitless["config"]["cold_start"] is True
    assert [h["policy"] for h in fitless["headline"]] == BODY["policies"]


def test_the_fitless_run_writes_its_own_world_reference(fitless):
    from apps.simulations.models import Simulation

    sim = Simulation.objects.get(pk=fitless["id"])
    assert (sim.directory / "fit" / "two_stage.json").is_file()
    assert sim.framework["train_end"].startswith("2015-01")
    assert sim.framework["data"]["snapshot_date"] == sim.source_dataset.snapshot_date


def test_cold_greedy_never_lends_while_thompson_explores(api, fitless):
    sid = fitless["id"]
    arms = api.get(f"/api/v1/simulations/{sid}/arms/", {"policy": "greedy"})
    assert arms.status_code == 200
    rows = {h["policy"]: h for h in fitless["headline"]}
    assert rows["greedy"]["cum_reward"] == pytest.approx(0.0, abs=1e-6)  # declines everyone
    assert rows["thompson"]["cum_reward"] != pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("view", ["timeseries", "events", "sentinel", "report", "posteriors", "simpson"])
def test_read_views_serve_a_fitless_run(api, fitless, view):
    r = api.get(f"/api/v1/simulations/{fitless['id']}/{view}/")
    assert r.status_code == 200, r.json()
    if view == "report":
        assert r.json()["fit"] is None and r.json()["config"]["fit"] is None


def test_a_cold_start_can_reuse_a_fits_world(api, e2e):
    r = _create(api, fit=e2e["fit"]["id"], cold_start=True)
    assert r.status_code == 201, r.json()
    sim = r.json()
    assert sim["fit"] == e2e["fit"]["id"] and sim["cold_start"] is True and sim["status"] == "completed"


def test_list_filters_by_dataset(api, fitless, e2e):
    r = api.get("/api/v1/simulations/", {"dataset": e2e["dataset"]["id"]})
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()["results"]]
    assert fitless["id"] in ids


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ({}, "fit"),  # warm runs still need a fit
        ({"cold_start": True}, "dataset"),  # a fit-less cold start needs a dataset
        ({"cold_start": True, "dataset": 999_999}, "dataset"),
    ],
)
def test_cold_start_validation(api, e2e, extra, field):
    r = _create(api, **extra)
    assert r.status_code == 400 and field in r.json(), r.json()


def test_a_fit_and_another_dataset_conflict(api, e2e):
    r = _create(api, fit=e2e["fit"]["id"], dataset=e2e["dataset"]["id"] + 10_000, cold_start=True)
    assert r.status_code == 400 and "dataset" in r.json()
