"""End-to-end workflow through the public API (inline jobs; see ``conftest.e2e``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.django_db

FIT_METRIC_KEYS = {
    "brier",
    "brier_baseline",
    "auc",
    "ece",
    "log_loss",
    "recovery_mae",
    "recovery_rmse",
    "transition_log_lik",
    "transition_log_lik_per_obs",
    "transition_prediction_error",
    "n_train",
    "n_test",
    "n_defaults_train",
    "base_rate_train",
    "base_rate_test",
    "fit_seconds",
}


# ----------------------------------------------------------------------------- datasets
def test_dataset_is_ready_with_statistics(e2e):
    ds = e2e["dataset"]
    assert ds["status"] == "ready"
    assert ds["n_loans"] == 6000
    assert ds["issue_min"] == "2012-01" and ds["issue_max"] == "2018-12"
    assert 0 < ds["resolved_share"] < 1 and 0 < ds["default_rate"] < 1
    assert sum(ds["status_counts"].values()) == 6000
    assert ds["params"]["shift"]["at"] == "2016-06" and ds["params"]["shift"]["deltas"]  # default shift filled in
    assert ds["job"]["kind"] == "ingest" and ds["job"]["status"] == "succeeded"
    assert ds["job"]["target"] == {"type": "dataset", "id": ds["id"]}


def test_variables_follow_table_3(e2e):
    rows = e2e["variables"]
    assert len(rows) == 60
    assert rows[0]["name"] == "annual_inc" and rows[0]["stage"] == "pre_decision"
    stages = {r["name"]: r["stage"] for r in rows}
    assert (
        stages["loan_amnt"] == "policy" and stages["issue_d"] == "disbursement" and stages["loan_status"] == "terminal"
    )
    assert {r["dtype"] for r in rows if r["present"]} >= {"float64", "string"}


def test_summary_features_and_arrivals(e2e):
    s = e2e["summary"]
    feats = {f["feature"]: f for f in s["features"]}
    assert feats["fico_range_low"]["kind"] == "numeric" and feats["fico_range_low"]["bins"]
    assert feats["grade"]["kind"] == "categorical" and len(feats["grade"]["levels"]) <= 12
    assert feats["term"]["kind"] == "categorical"
    assert len(s["monthly_arrivals"]) == 84  # 2012-01 .. 2018-12 zero-filled
    assert sum(m["n"] for m in s["monthly_arrivals"]) == 6000


def test_summary_custom_features(api, e2e):
    r = api.get(f"/api/v1/datasets/{e2e['dataset']['id']}/summary/", {"features": "dti,purpose"})
    assert r.status_code == 200
    assert [f["feature"] for f in r.json()["features"]] == ["dti", "purpose"]
    bad = api.get(f"/api/v1/datasets/{e2e['dataset']['id']}/summary/", {"features": "zip_code"})
    assert bad.status_code == 400 and "features" in bad.json()


def test_loans_page_and_filters(api, e2e):
    page = e2e["loans"]
    assert page["count"] > 0 and len(page["results"]) == 5 and page["previous"] is None and page["next"]
    rec = page["results"][0]
    assert rec["context"]["grade"] == "B"
    assert set(rec["context"]) == set(e2e["shadow_loan"]["context"])  # the 37 pre-decision columns
    assert "loan_status" not in rec["context"] and rec["outcome"]["loan_status"]
    dsid = e2e["dataset"]["id"]
    resolved = api.get(f"/api/v1/datasets/{dsid}/loans/", {"resolved": "true", "limit": 50}).json()
    assert all(r["outcome"]["resolved"] for r in resolved["results"])
    assert all(r["outcome"]["realized_reward"] is not None for r in resolved["results"])
    hit = api.get(f"/api/v1/datasets/{dsid}/loans/", {"search": str(rec["loan_id"])}).json()
    assert any(r["loan_id"] == rec["loan_id"] for r in hit["results"])
    ordered = api.get(f"/api/v1/datasets/{dsid}/loans/", {"ordering": "-int_rate", "limit": 3}).json()["results"]
    assert ordered[0]["policy"]["int_rate"] >= ordered[-1]["policy"]["int_rate"]
    assert api.get(f"/api/v1/datasets/{dsid}/loans/", {"ordering": "zip"}).status_code == 400


# ----------------------------------------------------------------------------- fits
def test_fit_ready_with_metrics(e2e):
    fit = e2e["fit"]
    assert fit["status"] == "ready"
    assert FIT_METRIC_KEYS <= set(fit["metrics"])
    assert 0.5 < fit["metrics"]["auc"] < 1
    assert fit["config"]["train_end"] == "2015-01"
    assert fit["n_arms"] == 129 and fit["n_groups"] > 10


def test_hierarchy_rows(e2e):
    rows = e2e["groups"]
    assert rows[0]["level"] == "global" and rows[0]["parent"] is None and rows[0]["kappa"] is None
    groups = [r for r in rows if r["level"] == "group"]
    assert rows[0]["n_children"] == len(groups)
    assert len(groups) == 7 * 3 * 4  # cartesian product of observed grade x purpose_group x home_ownership
    empty = [r for r in groups if r["n_obs"] == 0]
    assert all(
        r["shrinkage_weight"] == 0 and r["empirical_rate"] is None and r["mean"] == r["prior_mean"] for r in empty
    )
    assert all(
        r["parent"] == "global" and set(r["dims"]) == {"grade", "purpose_group", "home_ownership"} for r in groups
    )
    leaves = e2e["leaves"]
    assert leaves and all(r["level"] == "leaf" and r["parent"] == e2e["group"] and r["n_obs"] > 0 for r in leaves)
    assert all(0 <= r["shrinkage_weight"] <= 1 and r["hdi_low"] <= r["mean"] <= r["hdi_high"] for r in leaves)


def test_risk_estimate(api, e2e):
    risk = e2e["risk"]
    assert risk["arm"] == e2e["arm"] and risk["p_default"]["level"] == "leaf"
    assert risk["label"].startswith("$") and e2e["risk_group"]["label"] == "group posterior"
    assert len(risk["pdf"]["p_default"]) == 101 and len(risk["pdf"]["recovery"]) == 101
    assert risk["reward"] is not None and risk["reward"]["risk_adjusted"] == pytest.approx(risk["reward"]["mean"])
    assert risk["expected_loss_fraction"] == pytest.approx(
        risk["p_default"]["mean"] * (1 - risk["recovery"]["mean"]), rel=1e-4
    )
    g = e2e["risk_group"]
    assert g["arm"] is None and g["policy"] is None and g["reward"] is None and g["p_default"]["level"] == "group"
    fid = e2e["fit"]["id"]
    assert api.get(f"/api/v1/fits/{fid}/risk/").status_code == 400
    assert api.get(f"/api/v1/fits/{fid}/risk/", {"group": e2e["group"], "arm": 999}).status_code == 400


def test_calibration(e2e):
    cal = e2e["calibration"]
    assert len(cal["bins"]) == 10 and cal["n_test"] == sum(b["n"] for b in cal["bins"])
    sizes = [b["n"] for b in cal["bins"]]
    assert max(sizes) - min(sizes) <= 1  # prediction deciles (equal frequency)
    assert all(b["lo"] <= b["mean_predicted"] <= b["hi"] for b in cal["bins"])
    assert 0 < len(cal["roc"]) <= 200
    fprs = [p["fpr"] for p in cal["roc"]]
    assert fprs == sorted(fprs)
    assert len(cal["recovery"]["histogram"]) == 20 and len(cal["recovery"]["pdf"]) == 101


def test_mctm_views(e2e):
    ext = e2e["mctm"]
    assert ext["structure"] == "extended" and ext["fitted_structure"] == "proposal"
    assert ext["matrix"][0][2] > 0  # C -> R allowed (and observed) under EXTENDED
    assert ext["structural_zeros"][0][3] is True and ext["structural_zeros"][0][2] is False
    for row in ext["matrix"]:
        assert sum(row) == pytest.approx(1.0, abs=1e-6)
    d = e2e["mctm_default"]
    assert d["structure"] == "proposal" and d["matrix"][0][2] == 0
    assert d["start"] == "delinquent" and d["horizon"] == 24 and len(d["trajectory"]) == 25
    assert d["trajectory"][0]["delinquent"] == 1.0
    assert all(
        lo <= m <= hi
        for lr, mr, hr in zip(d["matrix_low"], d["matrix"], d["matrix_high"], strict=True)
        for lo, m, hi in zip(lr, mr, hr, strict=True)
    )


def test_fit_arms(e2e):
    arms = e2e["arms"]["arms"]
    assert len(arms) == 129 and arms[0]["is_null"] and arms[0]["label"] == "decline"
    assert sum(a["historical_count"] for a in arms) > 0
    assert sum(a["historical_share"] for a in arms) == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------------------- shadow engine
def test_shadow_by_loan(e2e):
    rep = e2e["shadow_loan"]
    assert rep["loan_id"] == e2e["loan_id"] and rep["leakage_safe"] is True
    assert all(rep["context"][c] is not None for c in rep["context_columns"])
    rec = rep["arms"][rep["recommended_arm"]]["reward_mean"]
    assert rep["uplift_expected"] == pytest.approx(rec - rep["historical_expected_reward"], abs=0.02)
    assert len(rep["arms"]) == 129 and len(rep["context"]) == 37
    assert rep["outcome"] is not None and rep["historical_arm"] is not None
    assert rep["arms"][rep["recommended_arm"]]["eligible"] or rep["recommended_arm"] == rep["historical_arm"]


def test_shadow_by_context(api, e2e):
    rep = e2e["shadow_context"]
    assert rep["loan_id"] is None and rep["outcome"] is None and rep["group"].startswith("B|credit_card|RENT")
    assert rep["historical_policy"] == {"amount": 12000.0, "rate": 0.11, "term": 36}
    fid = e2e["fit"]["id"]
    leak = api.post(
        f"/api/v1/fits/{fid}/shadow/evaluate/", {"context": {"annual_inc": 5e4, "total_pymnt": 1.0}}, format="json"
    )
    body = leak.json()
    assert leak.status_code == 400 and body["code"] == "LeakageError"
    assert body["context"][0].startswith("LeakageError:") and "total_pymnt" in body["context"][0]
    assert api.post(f"/api/v1/fits/{fid}/shadow/evaluate/", {}, format="json").status_code == 400
    assert api.post(f"/api/v1/fits/{fid}/shadow/evaluate/", {"loan_id": 1}, format="json").status_code == 404


def test_shadow_portfolio(e2e):
    assert e2e["portfolio_job"]["kind"] == "shadow_portfolio" and e2e["portfolio_job"]["status"] == "succeeded"
    res = e2e["portfolio"]
    s = res["summary"]
    assert s["n_loans"] == 1000
    assert {"rescued", "upsized", "repriced_down", "repriced_up", "declined", "unchanged"} <= set(s["counts"])
    assert sum(s["counts"].values()) == 1000
    assert res["by_grade"] and res["inclusion"] and 0 < len(res["top_loans"]) <= 50
    assert e2e["portfolio_runs"][0]["job"]["id"] == res["job"]["id"]


def test_model_extras_are_served(api, e2e):
    """Structured leaf prior, guardrails, prepayment, recovery by term, risk-appetite sweep."""
    fit = e2e["fit"]
    cfg = fit["config"]
    assert cfg["arm_space"]["max_payment_to_income"] == 0.2 and cfg["arm_space"]["max_total_dti"] == 0.43
    assert cfg["arm_space"]["max_amount_to_income"] == 0.5 and cfg["arm_space"]["rate_corridor"] == 0.05
    assert (
        cfg["hierarchy"]["policy_effect"] is True
        and cfg["hierarchy"]["policy_effect_cfg"]["max_abs_logit_shift"] == 3.0
    )
    assert cfg["recovery"]["term_level"] is True and cfg["reward"]["prepayment_adjustment"] == "empirical"
    m = fit["metrics"]
    ir = m["interest_realization"]
    assert (
        set(ir["by_term"]) == {"36", "60"} and 0.3 <= ir["global"] <= 1.0 and ir["prepayment_adjustment"] == "empirical"
    )
    assert {"status", "beta", "n_fit"} <= set(m["policy_effect"]) and m["guardrails"]["max_total_dti"] == 0.43
    for rep in (e2e["shadow_loan"], e2e["shadow_context"]):
        el = rep["eligibility"]
        assert {"reference_rate", "n_eligible", "n_eligible_lending", "historical_admitted", "guardrails"} <= set(el)
        assert el["n_eligible"] == sum(a["eligible"] for a in rep["arms"])
        assert all(0.3 <= a["realization"] <= 1.0 for a in rep["arms"])
    assert e2e["shadow_context"]["eligibility"]["reference_rate"] is not None  # grade B -> grade mean
    risk = e2e["risk"]
    assert risk["recovery"]["core_source"] in ("leaf", "group_term", "group", "global", "prior")
    assert risk["recovery"]["level"] in ("leaf", "group", "global")
    assert risk["reward"] is None or 0.3 <= risk["reward"]["realization"] <= 1.0
    s = e2e["portfolio"]["summary"]
    assert set(s["by_risk_aversion"]) == {"0.0", "0.5", "1.0"}
    assert s["by_risk_aversion"]["0.0"]["expected_profit_shadow"] == pytest.approx(
        s["expected_profit_shadow"], abs=0.05
    )
    # every uplift_pct of the payload is a fraction (contract), the core's percent is converted
    assert s["by_risk_aversion"]["0.0"]["uplift_pct"] == pytest.approx(s["uplift_pct"], rel=1e-3)
    assert s["uplift_decomposition"]["unrestricted"]["uplift_pct"] == pytest.approx(s["uplift_pct"], rel=1e-3)
    assert {"recommended", "historical", "reference", "mean_rate_recommended"} <= set(s["recommendation_mix"])
    dec = s["uplift_decomposition"]
    assert {"repricing_only", "repricing_and_resizing", "unrestricted"} <= set(dec)
    assert dec["unrestricted"]["uplift"] == pytest.approx(s["uplift"], abs=0.05)
    assert s["eligibility"]["guardrails"]["max_payment_to_income"] == 0.2
    assert e2e["simulation"]["config"]["hindsight_sample"] == "elapsed"


def test_fit_with_guardrails_off_and_no_prepayment(api, e2e):
    body = {
        "dataset": e2e["dataset"]["id"],
        "name": "no guardrails",
        "train_end": "2015-01",
        "arm_space": {
            "max_payment_to_income": None,
            "rate_corridor": None,
            "max_amount_to_income": None,
            "max_total_dti": None,
        },
        "reward": {"prepayment_adjustment": None},
        "recovery": {"term_level": False},
        "hierarchy": {"policy_effect": False},
    }
    r = api.post("/api/v1/fits/", body, format="json")
    assert r.status_code == 201, r.json()
    fit = r.json()
    assert (
        fit["config"]["arm_space"]["max_total_dti"] is None and fit["config"]["reward"]["prepayment_adjustment"] is None
    )
    rep = api.post(f"/api/v1/fits/{fit['id']}/shadow/evaluate/", {"loan_id": e2e["loan_id"]}, format="json").json()
    assert all(a["eligible"] for a in rep["arms"]) or rep["eligibility"]["guardrails"]["max_total_dti"] is None
    assert all(a["realization"] == 1.0 for a in rep["arms"])
    assert rep["eligibility"]["rate_corridor_applied"] is False


# ----------------------------------------------------------------------------- simulations
def test_simulation_completed(e2e):
    sim = e2e["simulation"]
    assert sim["status"] == "completed" and sim["progress"] == 1.0
    assert sim["epochs_done"] == sim["epochs_total"] == 18
    assert sim["current_epoch"] == "2016-06" and sim["last_checkpoint_epoch"] == "2016-06"
    assert [h["policy"] for h in sim["headline"]] == ["thompson", "historical", "random"]
    assert sim["uplift_vs_historical"] is not None
    assert sim["worker"]["alive"] is False and sim["job"]["status"] == "succeeded"
    assert sim["config"]["start"] == "2015-01" and sim["config"]["end"] == "2016-06"


def test_timeseries(e2e):
    ts = e2e["timeseries"]
    assert len(ts["epochs"]) == 18 and len(ts["rows"]) == 18 * 3
    assert [r["policy"] for r in ts["rows"][:3]] == ["thompson", "historical", "random"]
    th = [r for r in ts["rows"] if r["policy"] == "thompson"]
    assert th[-1]["cum_reward"] == pytest.approx(sum(r["reward"] for r in th), rel=1e-6, abs=0.1)


def test_simulation_arms_posteriors_simpson(api, e2e):
    arms = e2e["sim_arms"]
    assert arms["policy"] == "thompson" and arms["epoch"] == "2016-06" and len(arms["arms"]) == 129
    assert sum(a["selections"] for a in arms["arms"]) == arms["n_decisions"]
    assert arms["groups"] and arms["groups"][0]["n"] >= arms["groups"][-1]["n"]
    sid = e2e["simulation"]["id"]
    g = arms["groups"][0]["group"]
    by_group = api.get(f"/api/v1/simulations/{sid}/arms/", {"group": g, "epoch": "2015-06"}).json()
    assert by_group["group"] == g and by_group["epoch"] == "2015-06"
    post = e2e["posteriors"]
    assert post["arms"] and post["rows"] and post["snapshot_every"] >= 1
    simpson = e2e["simpson"]
    assert simpson["n_groups"] == len(simpson["groups"]) and simpson["n_agree"] <= simpson["n_groups"]
    assert all(r["gap"] >= 0 for r in simpson["groups"])
    assert api.get(f"/api/v1/simulations/{sid}/arms/", {"policy": "nope"}).status_code == 400
    assert api.get(f"/api/v1/simulations/{sid}/arms/", {"epoch": "2019-01"}).status_code == 400


def test_events(api, e2e):
    page = e2e["events"]
    types = [e["type"] for e in page["results"]]
    assert types[0] == "RunStarted" and types[-1] == "RunCompleted"
    assert "EpochStarted" not in types and types.count("EpochCompleted") == 18
    assert page["last_seq"] >= page["results"][-1]["seq"]
    sid = e2e["simulation"]["id"]
    only = api.get(f"/api/v1/simulations/{sid}/events/", {"type": "CheckpointWritten", "limit": 3}).json()
    assert [e["type"] for e in only["results"]] == ["CheckpointWritten"] * 3 and only["has_more"]
    after = api.get(
        f"/api/v1/simulations/{sid}/events/", {"after": only["results"][-1]["seq"], "type": "CheckpointWritten"}
    ).json()
    assert all(e["seq"] > only["results"][-1]["seq"] for e in after["results"])
    tail = api.get(f"/api/v1/simulations/{sid}/events/", {"order": "desc", "limit": 1}).json()
    assert tail["results"][0]["type"] == "RunCompleted"


def test_sentinel_timeline_and_buckets(e2e):
    tl = e2e["sentinel"]
    assert len(tl["aggregate"]) == 18 and len(tl["features_psi"]) == 18 * len(tl["features"])
    assert tl["thresholds"] == {"watch": 0.08, "breaker": 0.15}
    assert tl["reference"]["n"] > 0 and tl["reference"]["to"] == "2014-12"
    assert tl["current"]["epoch"] == "2016-06"
    b = e2e["buckets"]
    assert b["kind"] == "numeric" and b["n_window"] > 0
    assert b["psi"] == pytest.approx(b["psi_runner"], rel=1e-9)  # recomputed PSI matches the runner's reading
    assert sum(x["contribution"] for x in b["buckets"]) == pytest.approx(b["psi"], rel=1e-4)


def test_report(e2e):
    rep = e2e["report"]
    assert rep["status"] == "final" and rep["simulation"] == e2e["simulation"]["id"]
    assert [d["key"] for d in rep["dimensions"]][:2] == ["default_prediction", "recovery_prediction"]
    assert len(rep["dimensions"]) == 9
    assert rep["runtime"]["decisions"] > 0 and rep["runtime"]["checkpoints_written"] >= 9
    assert rep["config"]["simulation"]["policies"] == ["thompson", "historical", "random"]


# ----------------------------------------------------------------------------- sentinel reviews
def test_review_created_and_acknowledged(api, e2e):
    from apps.common.utils import read_json
    from apps.simulations.models import Simulation

    reviews = e2e["reviews"]["results"]
    assert e2e["review_sim"]["status"] == "completed"
    assert e2e["review_sim"]["policies"] if "policies" in e2e["review_sim"] else True
    pending = [r for r in reviews if r["status"] == "pending"]
    assert pending, "the low breaker threshold must trip the PSI breaker"
    review = pending[0]
    assert review["level"] == "breaker" and review["breaker_state"] == "open" and review["top_features"]
    assert review["segmentation_before"] == ["grade", "purpose_group", "home_ownership"]

    bad = api.post(f"/api/v1/sentinel/reviews/{review['id']}/acknowledge/", {"action": "explode"}, format="json")
    assert bad.status_code == 400
    ack = api.post(
        f"/api/v1/sentinel/reviews/{review['id']}/acknowledge/",
        {"action": "resegment", "note": "fico drift"},
        format="json",
    )
    assert ack.status_code == 200, ack.content
    body = ack.json()
    assert (
        body["status"] == "acknowledged" and body["action"] == "resegment" and body["acknowledged_by"] == "researcher"
    )
    assert body["segmentation_after"] == ["grade", "purpose_group", "home_ownership", "fico_band"]
    sim = Simulation.objects.get(pk=e2e["review_sim"]["id"])
    ack_file = read_json(Path(sim.run_dir) / "reviews" / f"{review['review_id']}.ack.json")
    assert ack_file["action"] == "resegment" and ack_file["note"] == "fico drift"
    again = api.post(f"/api/v1/sentinel/reviews/{review['id']}/acknowledge/", {"action": "resume"}, format="json")
    assert again.status_code == 409


def test_review_list_filters(api, e2e):
    sid = e2e["review_sim"]["id"]
    r = api.get("/api/v1/sentinel/reviews/", {"simulation": sid, "status": "pending"})
    assert r.status_code == 200 and all(x["simulation"] == sid for x in r.json()["results"])
    assert api.get("/api/v1/sentinel/reviews/", {"simulation": 999999}).json()["count"] == 0


def test_jobs_endpoint(api, e2e):
    kinds = {j["kind"] for j in e2e["jobs"]["results"]}
    assert kinds == {"ingest", "fit", "shadow_portfolio", "simulation"}
    one = e2e["jobs"]["results"][0]
    assert api.get(f"/api/v1/jobs/{one['id']}/").json()["id"] == one["id"]
    only = api.get("/api/v1/jobs/", {"kind": "fit,ingest"}).json()["results"]
    assert {j["kind"] for j in only} == {"fit", "ingest"}
    assert api.get("/api/v1/jobs/", {"status": "queued,running"}).json()["count"] == 0


def test_payloads_are_strict_json(e2e):
    json.dumps(e2e, allow_nan=False)  # raises on NaN / inf anywhere
