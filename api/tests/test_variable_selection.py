"""Variable selection for ablations: a dataset created without some Table 3 variables (e.g.
the FICO columns) never stores them, and fits / simulations built on it respect that."""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.django_db

NO_FICO = ["fico_range_low", "fico_range_high", "last_fico_range_low", "last_fico_range_high"]
NO_FICO_OR_GRADE = [*NO_FICO, "grade", "sub_grade"]
TABLE3_ORDER = ["fico_range_low", "fico_range_high", "grade", "sub_grade", "last_fico_range_low", "last_fico_range_high"]


def _create(api, **extra):
    body = {"name": "ablation", "source": "synthetic", "n": 4000, "seed": 9, **extra}
    return api.post("/api/v1/datasets/", body, format="json")


def test_catalogue_marks_required_columns_and_model_uses(api):
    rows = {r["name"]: r for r in api.get("/api/v1/datasets/catalogue/").json()}
    assert len(rows) == 60
    assert rows["fico_range_low"]["selectable"] and rows["fico_range_low"]["required_reason"] is None
    assert any(u.startswith("segmentation: fico_band") for u in rows["fico_range_low"]["uses"])
    assert "segmentation: grade (default)" in rows["grade"]["uses"]
    for name in ("loan_amnt", "issue_d", "loan_status", "installment", "last_pymnt_d"):
        assert not rows[name]["selectable"] and rows[name]["required_reason"], name
    assert rows["last_fico_range_low"]["uses"] == []  # interim: not read by any model component


def test_exclusion_is_validated(api):
    r = _create(api, exclude_variables=["loan_status", "zip_code"])
    assert r.status_code == 400
    msg = " ".join(r.json()["exclude_variables"])
    assert "zip_code" in msg and "loan_status" in msg


def test_ablated_dataset_fit_and_simulation(api):
    r = _create(api, exclude_variables=["sub_grade", *NO_FICO, "grade", "grade"])
    assert r.status_code == 201, r.json()
    ds = r.json()
    assert ds["status"] == "ready" and ds["params"]["exclude_variables"] == TABLE3_ORDER

    from apps.datasets.models import Dataset

    stored = pd.read_parquet(Dataset.objects.get(pk=ds["id"]).parquet)
    assert not set(NO_FICO_OR_GRADE) & set(stored.columns)
    assert stored["y_group"].astype(str).str.startswith("unknown|").all()

    variables = {v["name"]: v for v in api.get(f"/api/v1/datasets/{ds['id']}/variables/").json()}
    assert variables["fico_range_low"]["excluded"] and not variables["fico_range_low"]["present"]
    assert not variables["dti"]["excluded"] and variables["dti"]["present"]
    features = [f["feature"] for f in api.get(f"/api/v1/datasets/{ds['id']}/summary/").json()["features"]]
    assert "fico_range_low" not in features and "grade" not in features and "dti" in features

    # an explicit dimension built on an excluded variable is rejected ...
    bad = api.post(
        "/api/v1/fits/",
        {"dataset": ds["id"], "name": "bad", "segmentation": {"dimensions": ["grade", "fico_band"]}},
        format="json",
    )
    assert bad.status_code == 400 and "fico_band" in " ".join(bad.json()["segmentation"])
    # ... the defaults drop it
    fit = api.post("/api/v1/fits/", {"dataset": ds["id"], "name": "no fico", "train_end": "2016-01"}, format="json")
    assert fit.status_code == 201, fit.json()
    fit = fit.json()
    assert fit["status"] == "ready", fit["job"]
    assert fit["config"]["segmentation"]["dimensions"] == ["purpose_group", "home_ownership"]
    # a breaker resegment never refines on an excluded variable (it would be a constant "unknown")
    from apps.fits.models import Fit

    order = Fit.objects.get(pk=fit["id"]).framework["segmentation"]["refinement_order"]
    assert order[0] == "dti_band" and not {"fico_band", "sub_grade"} & set(order)
    cold = api.post(
        "/api/v1/simulations/",
        {"dataset": ds["id"], "cold_start": True, "name": "cold", "policies": ["thompson"],
         "segmentation": {"dimensions": ["fico_band", "purpose_group"]}},
        format="json",
    )
    assert cold.status_code == 400 and "fico_band" in " ".join(cold.json()["segmentation"])

    bad_psi = api.post(
        "/api/v1/simulations/",
        {"fit": fit["id"], "name": "bad psi", "policies": ["thompson"], "psi": {"features": ["fico_range_low", "dti"]}},
        format="json",
    )
    assert bad_psi.status_code == 400 and "fico_range_low" in " ".join(bad_psi.json()["psi"])
    sim = api.post(
        "/api/v1/simulations/",
        {
            "fit": fit["id"],
            "name": "every loan",
            "policies": ["thompson", "historical"],
            "start": "2016-01",
            "end": "2016-03",
            "max_loans": None,
            "breaker_mode": "off",
        },
        format="json",
    )
    assert sim.status_code == 201, sim.json()
    sim = sim.json()
    assert sim["config"]["max_loans"] is None
    assert sim["config"]["psi"]["features"] == ["dti", "annual_inc", "revol_util", "inq_last_6mths"]
    assert sim["status"] == "completed", sim["job"]


def test_without_income_the_pti_policy_effect_is_dropped(api):
    ds = _create(api, exclude_variables=["annual_inc"]).json()
    fit = api.post("/api/v1/fits/", {"dataset": ds["id"], "name": "no income"}, format="json")
    assert fit.status_code == 201, fit.json()
    assert "pti" not in fit.json()["config"]["hierarchy"]["policy_effect_cfg"]["features"]
    bad = api.post(
        "/api/v1/fits/",
        {"dataset": ds["id"], "name": "pti", "hierarchy": {"policy_effect_cfg": {"features": ["pti", "int_rate"]}}},
        format="json",
    )
    assert bad.status_code == 400 and "annual_inc" in str(bad.json())
