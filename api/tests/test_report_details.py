"""Structured-prior tilt on hierarchy rows, the shadow portfolio inside the evaluation report
and the units of ``uplift_pct`` in ``details`` (docs/API.md §13.1)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

pytestmark = pytest.mark.django_db

DETAIL_BLOCKS = (
    "recommendation_mix",
    "uplift_decomposition",
    "by_risk_aversion",
    "eligibility",
    "interest_realization",
)
LOAN_ROW_KEYS = {
    "loan_id",
    "grade",
    "group",
    "historical_arm",
    "recommended_arm",
    "historical_label",
    "recommended_label",
    "historical_expected_reward",
    "shadow_expected_reward",
    "realized_reward",
    "class",
}


def _report(api, sim_id: int) -> dict:
    r = api.get(f"/api/v1/simulations/{sim_id}/report/")
    assert r.status_code == 200, r.content
    return r.json()


# ----------------------------------------------------------------------------- hierarchy rows
def test_hierarchy_rows_serve_prior_tilt_and_core_source(api, e2e):
    from apps.fits.models import Fit
    from apps.fits.services import load_fitted

    fit = Fit.objects.get(pk=e2e["fit"]["id"])
    summary = pd.read_parquet(fit.directory / "hierarchy_summary.parquet")
    tilt = {str(k): v for k, v in zip(summary["key"], summary["prior_tilt_logit"], strict=True)}
    rows = e2e["groups"]
    assert all("prior_tilt_logit" in r for r in rows)
    assert all(r["prior_tilt_logit"] is None for r in rows)  # global and group rows carry no tilt

    leaves = e2e["leaves"]
    assert leaves
    for leaf in leaves:
        expected = tilt[leaf["key"]]
        if expected is None or not math.isfinite(expected):
            assert leaf["prior_tilt_logit"] is None
        else:
            assert leaf["prior_tilt_logit"] == pytest.approx(expected, abs=1e-6)

    recovery = load_fitted(fit).two_stage.recovery_model
    for leaf in leaves:
        core = recovery.resolve(leaf["group"], leaf["arm"])[1]
        assert leaf["recovery"]["core_source"] == core
        assert leaf["recovery"]["source"] == (
            "group" if core == "group_term" else core if core != "prior" else "global"
        )


# ----------------------------------------------------------------------------- report + portfolio
def test_report_counterfactual_uses_latest_portfolio(e2e):
    cf = e2e["report"]["details"]["counterfactual_evaluation"]
    portfolio = e2e["portfolio"]["summary"]
    assert cf["available"] is True and cf["portfolio_job_id"] == e2e["portfolio_job"]["id"]
    assert cf["portfolio_request"]["max_loans"] == 1000
    for block in DETAIL_BLOCKS:
        assert block in cf, block
    assert cf["n_loans"] == portfolio["n_loans"] == 1000
    assert cf["expected_profit_shadow"] == pytest.approx(portfolio["expected_profit_shadow"], abs=0.02)
    assert cf["top_loans"] and len(cf["top_loans"]) <= 10
    assert all(set(row) == LOAN_ROW_KEYS for row in cf["top_loans"])  # types.ts PortfolioLoanRow
    t11 = {
        m["key"]: m["value"]
        for d in e2e["report"]["dimensions"]
        if d["key"] == "counterfactual_evaluation"
        for m in d["metrics"]
    }
    assert t11["shadow_expected_profit"] == pytest.approx(portfolio["expected_profit_shadow"], abs=0.02)


def test_report_uplift_pct_are_fractions(e2e):
    details = e2e["report"]["details"]
    cf = details["counterfactual_evaluation"]
    portfolio = e2e["portfolio"]["summary"]
    # same unit as the portfolio endpoint: a fraction of |expected_profit_historical|
    assert cf["uplift_pct"] == pytest.approx(portfolio["uplift_pct"], abs=1e-5)
    for name, block in (cf["uplift_decomposition"] or {}).items():
        if isinstance(block, dict) and block.get("uplift_pct") is not None:
            assert block["uplift_pct"] == pytest.approx(portfolio["uplift_decomposition"][name]["uplift_pct"], abs=1e-5)
    for lam, point in (cf["by_risk_aversion"] or {}).items():
        if point.get("uplift_pct") is not None:
            assert point["uplift_pct"] == pytest.approx(portfolio["by_risk_aversion"][lam]["uplift_pct"], abs=1e-5)
    by_policy = details["policy_performance"]["by_policy"]
    hist = abs(by_policy["historical"]["total_expected_reward"])
    for row in by_policy.values():
        assert row["uplift_pct"] == pytest.approx(row["uplift_vs_historical"] / hist, rel=1e-6, abs=1e-9)


def test_report_follows_a_newer_portfolio_run(api, e2e):
    from apps.common.models import Job

    sim_id, fit_id = e2e["simulation"]["id"], e2e["fit"]["id"]
    first = _report(api, sim_id)["details"]["counterfactual_evaluation"]
    assert first["portfolio_job_id"] == e2e["portfolio_job"]["id"]

    job = api.post(f"/api/v1/fits/{fit_id}/shadow/portfolio/", {"max_loans": 300, "seed": 9}, format="json").json()
    assert job["status"] == "succeeded"
    newer = _report(api, sim_id)["details"]["counterfactual_evaluation"]
    assert newer["available"] is True and newer["portfolio_job_id"] == job["id"] and newer["n_loans"] == 300

    Job.objects.filter(pk=job["id"]).update(status="failed")  # only succeeded runs count
    again = _report(api, sim_id)["details"]["counterfactual_evaluation"]
    assert again["portfolio_job_id"] == e2e["portfolio_job"]["id"] and again["n_loans"] == 1000


def test_report_without_portfolio_run(api, e2e):
    from apps.common.models import Job

    Job.objects.filter(kind="shadow_portfolio", target_type="fit", target_id=e2e["fit"]["id"]).update(status="failed")
    report = _report(api, e2e["simulation"]["id"])
    cf = report["details"]["counterfactual_evaluation"]
    assert cf == {"available": False, "note": "shadow portfolio not run"}
    t11 = {
        m["key"]: m["value"]
        for d in report["dimensions"]
        if d["key"] == "counterfactual_evaluation"
        for m in d["metrics"]
    }
    assert t11["shadow_expected_profit"] is None


def test_uplift_pct_converter_reports_paths():
    from apps.simulations.services import uplift_pct_to_fraction

    details = {
        "policy_performance": {"by_policy": {"thompson": {"uplift_pct": 146.27, "uplift_vs_historical": 1.0}}},
        "counterfactual_evaluation": {
            "uplift_pct": 304.37,
            "uplift_decomposition": {"unrestricted": {"uplift_pct": 304.37}, "expected_profit_historical": 5.0},
            "by_risk_aversion": {"0.5": {"uplift_pct": None}},
            "top_loans": [{"uplift_pct": 10.0}],
        },
    }
    paths = uplift_pct_to_fraction(details)
    assert details["policy_performance"]["by_policy"]["thompson"]["uplift_pct"] == pytest.approx(1.4627)
    assert details["counterfactual_evaluation"]["uplift_pct"] == pytest.approx(3.0437)
    assert details["counterfactual_evaluation"]["by_risk_aversion"]["0.5"]["uplift_pct"] is None
    assert sorted(paths) == sorted(
        [
            "details.policy_performance.by_policy.thompson.uplift_pct",
            "details.counterfactual_evaluation.uplift_pct",
            "details.counterfactual_evaluation.uplift_decomposition.unrestricted.uplift_pct",
            "details.counterfactual_evaluation.top_loans[0].uplift_pct",
        ]
    )
