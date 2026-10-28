"""EvaluationReport (Table 11) built from run directories."""

import json
from functools import lru_cache

import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, SimulationConfig
from bayesian_credit.data import DEFAULT_SHIFT
from bayesian_credit.evaluation import DIMENSIONS, EvaluationReport
from bayesian_credit.pipeline import build_report, fit_models, prepare_dataset, run_shadow_portfolio
from bayesian_credit.simulation import RunDirectory, SimulationRunner

POLICIES = ("thompson", "epsilon_greedy", "historical", "random")


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, summary = prepare_dataset("synthetic", None, cfg=cfg, n=5000, seed=3, shift_at="2016-06", shift=dict(DEFAULT_SHIFT))
    fitted = fit_models(df, cfg, "2016-01-01")
    return cfg, df, fitted, summary


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg, df, fitted, _ = _data()
    path = tmp_path_factory.mktemp("report") / "run"
    sim = SimulationConfig(start="2016-01-01", end="2016-08-31", max_loans=1500, seed=2, policies=POLICIES,
                           interim_months=3, checkpoint_every=4)
    SimulationRunner(df, cfg, sim, path, fitted=fitted, install_signal_handlers=False).run()
    return path


def test_report_from_a_finished_run_is_complete_and_json_serialisable(run_dir):
    cfg, df, fitted, summary = _data()
    _, portfolio = run_shadow_portfolio(fitted, df, max_loans=500, seed=1, n_samples=200)
    rep = build_report(run_dir, fitted=fitted, shadow_summary=portfolio, timings={"fit": 0.5, "simulation": 3.0},
                       resilience={"passed": True, "recovery_seconds": 1.2}, dataset_summary=summary)
    text = json.dumps(rep, allow_nan=False)
    assert json.loads(text) == rep
    assert rep["status"] == "final" and rep["generated_at"]
    assert set(rep["dimensions"]) == {k for k, _, _ in DIMENSIONS}
    assert [row["key"] for row in rep["table11"]] == [k for k, _, _ in DIMENSIONS]
    for row in rep["table11"]:
        assert row["axis"] in {"predictive", "decision", "adaptability", "feasibility"} and row["metrics"]
        for m in row["metrics"]:
            assert set(m) == {"key", "label", "value", "unit", "better", "baseline", "baseline_label", "by_policy", "note"}
    d = rep["dimensions"]
    assert d["default_prediction"]["available"] and d["default_prediction"]["primary"]["n"] > 0
    assert d["default_prediction"]["primary"]["n"] <= d["default_prediction"]["secondary"]["n"]
    assert set(d["policy_performance"]["by_policy"]) == set(POLICIES)
    assert d["policy_performance"]["by_policy"]["historical"]["uplift_vs_historical"] == 0.0
    assert set(d["bandit_performance"]["by_policy"]) == set(POLICIES)
    for v in d["bandit_performance"]["by_policy"].values():
        assert 0.0 < v["selection_concentration_top3"] <= 1.0 and v["final_cumulative_regret"] >= 0.0
    assert d["counterfactual_evaluation"]["available"] and d["counterfactual_evaluation"]["n_loans"] == 500
    mix = d["counterfactual_evaluation"]["recommendation_mix"]
    assert mix["recommended"]["n"] == 500 and set(mix["recommended"]["by_term"]) == {"36", "60"}
    assert mix["mean_rate_recommended"] is not None and mix["mean_amount_historical"] is not None
    assert d["counterfactual_evaluation"]["eligibility"]["guardrails"]["rate_corridor"] == 0.05
    sweep = d["counterfactual_evaluation"]["by_risk_aversion"]
    assert set(sweep) == {"0.0", "0.5", "1.0"} and {"uplift", "mean_amount_recommended", "share_36m_recommended"} <= set(sweep["0.5"])
    assert d["counterfactual_evaluation"]["interest_realization"]["by_term"]
    for v in d["bandit_performance"]["by_policy"].values():
        sm = v["selection_mix"]
        assert {"by_term", "by_rate", "by_amount", "mean_rate", "mean_amount", "share_at_max_rate"} <= set(sm)
        assert len(v["top_arm_labels_recent"]) == len(v["top_arms_recent"])
    dm = d["distributional_monitoring"]
    assert dm["n_trips"] >= 1 and dm["epochs_breaker_open"] >= 1 and set(dm["per_feature"]) == set(cfg.psi.features)
    assert dm["actions"].get("resume", 0) >= 1 and dm["reviews"][0]["action"] == "resume"
    cf = d["computational_feasibility"]
    assert cf["n_decisions"] == 4 * cf["n_loans_streamed"] and cf["peak_rss_mb"] > 0 and cf["stage_seconds"]["fit"] == 0.5
    ir = d["infrastructure_resilience"]
    assert ir["checkpoints_written"] == 2 and ir["resume_test"]["passed"] is True
    assert {p["policy"] for p in rep["policy_summary"]} == set(POLICIES)
    assert rep["config_hashes"]["run"] and rep["dataset"]["n"] == 5000
    assert json.loads(RunDirectory(run_dir).report_path.read_text())["generated_at"] == rep["generated_at"]


def test_default_prediction_uses_the_elapsed_term_holdout(run_dir):
    _, df, fitted, _ = _data()
    rep = EvaluationReport.from_run_dir(run_dir, fitted=fitted).build()
    dp = rep["dimensions"]["default_prediction"]
    hold = df.loc[df["issue_d"] >= "2016-01-01"]
    end = hold["issue_d"].dt.to_period("M") + hold["term"].to_numpy()
    elapsed = (end.dt.to_timestamp() <= pd.Timestamp("2019-03-31")).to_numpy()
    resolved = hold["y_default"].notna().to_numpy()
    assert dp["primary"]["n"] == int((elapsed & resolved).sum()) > 0
    assert dp["secondary"]["n"] == int(resolved.sum())
    assert dp["primary"]["base_rate"] == pytest.approx(hold.loc[elapsed & resolved, "y_default"].mean())
    assert dp["n_elapsed_resolved"] == dp["primary"]["n"]
    t11 = {r["key"]: {m["key"]: m for m in r["metrics"]} for r in rep["table11"]}
    assert t11["default_prediction"]["brier"]["value"] == pytest.approx(dp["primary"]["brier"])
    assert t11["default_prediction"]["brier"]["baseline"] == pytest.approx(dp["primary"]["brier_climatology"])


def test_report_reads_fit_metrics_from_the_fit_dir(run_dir):
    rep = EvaluationReport.from_run_dir(run_dir).build()
    assert rep["dimensions"]["default_prediction"]["available"]
    assert rep["dimensions"]["counterfactual_evaluation"]["available"] is False


def test_partial_and_empty_reports_still_build(tmp_path, run_dir):
    empty = EvaluationReport().build()
    json.dumps(empty, allow_nan=False)
    assert empty["status"] == "partial" and not empty["dimensions"]["policy_performance"]["available"]
    rd = RunDirectory(run_dir)
    progress = rd.read_progress()
    rd.write_progress(status="running")
    try:
        assert EvaluationReport.from_run_dir(run_dir).build()["status"] == "partial"
    finally:
        rd.write_progress(**{k: progress[k] for k in ("status",)})
    out = tmp_path / "r.json"
    EvaluationReport.from_run_dir(run_dir).to_json(out)
    assert json.loads(out.read_text())["table11"]
