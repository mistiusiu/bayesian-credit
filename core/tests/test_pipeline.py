"""pipeline.py + CLI: dataset preparation, fit / save / load, predictive evaluation, commands."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, SegmentationConfig, SimulationConfig
from bayesian_credit.data import DEFAULT_SHIFT
from bayesian_credit.data.segmentation import Segmenter
from bayesian_credit.pipeline import (
    FittedModels,
    elapsed_term_mask,
    ensure_outcomes,
    evaluate_predictive,
    fit_models,
    load_dataset,
    prepare_dataset,
    run_shadow_portfolio,
)
from bayesian_credit.simulation import cli
from bayesian_credit.simulation.store import RunDirectory


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, summary = prepare_dataset("synthetic", None, cfg=cfg, n=5000, seed=3, shift_at="2016-06", shift=dict(DEFAULT_SHIFT))
    return cfg, df, summary


def test_prepare_dataset_writes_a_round_trippable_parquet(tmp_path):
    cfg = FrameworkConfig()
    df, summary = prepare_dataset("synthetic", tmp_path / "d.parquet", cfg=cfg, n=1500, seed=4)
    assert summary["path"] == str(tmp_path / "d.parquet") and summary["n"] == 1500
    assert {"issue_min", "issue_max", "resolved_share", "default_rate", "status_counts", "source", "params"} <= set(summary)
    back = load_dataset(tmp_path / "d.parquet")
    pd.testing.assert_frame_equal(back, df)
    assert back["y_group"].dtype == df["y_group"].dtype and back["issue_d"].dtype == np.dtype("datetime64[ns]")


def test_ensure_outcomes_follows_the_requested_segmentation():
    cfg, df, _ = _data()
    finer = FrameworkConfig(segmentation=SegmentationConfig(dimensions=("grade", "fico_band")))
    out = ensure_outcomes(df, finer)
    expected = Segmenter(("grade", "fico_band")).assign(df).astype(str).to_numpy()
    np.testing.assert_array_equal(out["y_group"].astype(str).to_numpy(), expected)
    raw = df.drop(columns=[c for c in df.columns if c.startswith("y_")])
    derived = ensure_outcomes(raw, cfg)
    pd.testing.assert_series_equal(derived["y_default"], df["y_default"].reset_index(drop=True), check_names=False)


def test_fit_save_load_round_trip_gives_identical_risk(tmp_path):
    cfg, df, _ = _data()
    fitted = fit_models(df, cfg, "2016-01-01")
    out = fitted.save(tmp_path / "fit")
    for name in FittedModels.FILES:
        assert (out / name).is_file(), name
    loaded = FittedModels.load(out)
    groups = sorted(df["y_group"].unique()) + ["Z|never|SEEN"]
    for g in groups:
        for a in range(fitted.arm_space.n_arms):
            assert loaded.two_stage.risk(g, a) == fitted.two_stage.risk(g, a)
    np.testing.assert_array_equal(loaded.mctm.matrix, fitted.mctm.matrix)
    assert loaded.segmenter == fitted.segmenter and loaded.arm_space == fitted.arm_space
    assert loaded.arm_space.guardrails == fitted.arm_space.guardrails
    assert loaded.reference_rates == fitted.reference_rates and loaded.reference_rates is not None
    assert loaded.policy_effect.coefficients == fitted.policy_effect.coefficients
    assert loaded.two_stage.policy_effect_info == fitted.two_stage.policy_effect_info
    assert {"policy_effect", "reference_rates", "guardrails"} <= set(fitted.metrics)
    hs0 = pd.read_parquet(out / "hierarchy_summary.parquet")
    assert "prior_tilt_logit" in hs0.columns
    assert loaded.train_end == "2016-01-01" and loaded.cfg.to_dict() == cfg.to_dict()
    assert json.loads(json.dumps(loaded.metrics)) == json.loads(json.dumps(fitted.metrics))
    assert loaded.reward_fn.expected(loaded.arm_space[40], loaded.two_stage.risk(groups[0], 40), loaded.mctm) == \
        fitted.reward_fn.expected(fitted.arm_space[40], fitted.two_stage.risk(groups[0], 40), fitted.mctm)
    m = fitted.metrics
    assert m["n_train"] == int((df["issue_d"] < "2016-01-01").sum())
    assert {"brier", "auc", "ece", "log_loss", "recovery_mae", "recovery_rmse", "transition_log_likelihood", "n_test"} <= set(m)
    cal = json.loads((out / "calibration.json").read_text())
    assert len(cal["primary"]["reliability"]) == 10 and cal["primary"]["roc"][0] == {"fpr": 0.0, "tpr": 0.0, "threshold": None}
    hs = pd.read_parquet(out / "hierarchy_summary.parquet")
    assert set(hs["level"]) == {"global", "group", "leaf"}
    mc = json.loads((out / "mctm_summary.json").read_text())
    assert len(mc["matrix"]) == 4 and len(mc["trajectory_current"]) == 61 and mc["holdout_log_likelihood"] is not None


def test_elapsed_term_filter_and_predictive_subsets():
    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2015-01-01", "2016-03-01", "2016-04-01", "2014-04-01", "2016-01-01"]),
            "term": [36, 36, 36, 60, 60],
        }
    )
    np.testing.assert_array_equal(elapsed_term_mask(df, "2019-03-31"), [True, True, False, False, False])
    cfg, data, _ = _data()
    fitted = fit_models(data, cfg, "2016-01-01", evaluate=False)
    hold = data.loc[data["issue_d"] >= "2016-01-01"]
    res = evaluate_predictive(fitted, hold, "2019-03-31")
    elapsed = elapsed_term_mask(hold, "2019-03-31")
    resolved = hold["y_default"].notna().to_numpy()
    assert res["default"]["primary"]["n"] == int((elapsed & resolved).sum())
    assert res["default"]["secondary"]["n"] == int(resolved.sum())
    assert res["n_elapsed_resolved"] == int((elapsed & resolved).sum())
    assert sum(v["n"] for v in res["default"]["by_vintage"].values()) == int(resolved.sum())
    assert res["recovery"]["secondary"]["n"] == int((hold["y_default"] == 1).sum())
    assert 0.0 <= res["default"]["primary"]["auc"] <= 1.0
    assert res["credit_state"]["all"]["n_transitions"] > res["credit_state"]["elapsed"]["n_transitions"]
    json.dumps(res, allow_nan=False)


def test_shadow_portfolio_scores_holdout_loans_only():
    cfg, df, _ = _data()
    fitted = fit_models(df, cfg, "2016-01-01", evaluate=False)
    per_loan, summary = run_shadow_portfolio(fitted, df, max_loans=300, seed=1, n_samples=200)
    hold_ids = set(df.loc[df["issue_d"] >= "2016-01-01", "loan_id"])
    assert len(per_loan) == 300 and set(per_loan["loan_id"]) <= hold_ids
    assert summary.n_samples == 200


def test_cli_fit_shadow_run_resume_report(tmp_path, capsys):
    cfg, df, _ = _data()
    data_path = tmp_path / "dataset.parquet"
    from bayesian_credit.data import LendingClubLoader

    LendingClubLoader.to_parquet(df, data_path)
    conf = tmp_path / "framework.json"
    conf.write_text(json.dumps({"framework": cfg.to_dict()}))
    assert cli.main(["fit", "--dataset", str(data_path), "--config", str(conf), "--out", str(tmp_path / "fit"),
                     "--train-end", "2016-01-01"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["train_end"] == "2016-01-01" and out["metrics"]["n_train"] > 0
    assert cli.main(["shadow", "--fit-dir", str(tmp_path / "fit"), "--dataset", str(data_path), "--out",
                     str(tmp_path / "shadow"), "--max-loans", "200", "--n-samples", "100"]) == 0
    shadow = json.loads(capsys.readouterr().out)
    assert shadow["summary"]["n_loans"] == 200 and (tmp_path / "shadow" / "per_loan.parquet").is_file()
    sim = SimulationConfig(start="2016-01-01", end="2016-03-31", max_loans=300, policies=("thompson", "historical"),
                           interim_months=2, checkpoint_every=1)
    rd = RunDirectory.create(tmp_path / "run", cfg, sim, {"path": str(data_path)}, tmp_path / "fit")
    assert cli.main(["run", "--run-dir", str(rd.path), "--poll-interval", "0.02"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "completed" and run["epochs_completed"] == 3
    assert cli.main(["resume", "--run-dir", str(rd.path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert cli.main(["report", "--run-dir", str(rd.path)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "final" and {p["policy"] for p in rep["policy_summary"]} == {"thompson", "historical"}
    assert cli.main(["run", "--run-dir", str(tmp_path / "missing")]) == 1


def test_cli_rejects_unknown_commands():
    with pytest.raises(SystemExit):
        cli.main(["explode"])


def test_policy_effect_and_reference_rates_round_trip_and_old_fit_dirs_still_load(tmp_path):
    cfg, df, _ = _data()
    tilted = FrameworkConfig.from_dict(cfg.to_dict())
    tilted.hierarchy.policy_effect_cfg.min_rows = 300
    fitted = fit_models(df, tilted, "2016-01-01", evaluate=False)
    assert fitted.metrics["policy_effect"]["status"] == "fitted" and fitted.policy_effect.has_effect
    out = fitted.save(tmp_path / "fit")
    loaded = FittedModels.load(out)
    assert loaded.cfg.hierarchy.policy_effect_cfg.min_rows == 300
    groups = sorted(df["y_group"].unique())[:6] + ["Z|never|SEEN"]
    for g in groups:
        for a in range(0, fitted.arm_space.n_arms, 4):
            assert loaded.two_stage.risk(g, a) == fitted.two_stage.risk(g, a)
    np.testing.assert_array_equal(
        loaded.reference_rates.for_frame(df.head(200)), fitted.reference_rates.for_frame(df.head(200))
    )
    eng = loaded.shadow_engine(n_samples=100)
    row = df.loc[df["issue_d"] >= "2016-01-01"].iloc[0]
    assert eng.evaluate_loan(row).to_dict()["eligibility"]["rate_corridor_applied"] is True
    # a fit directory written before reference rates existed: no corridor, everything else works
    (out / "reference_rates.json").unlink()
    old = FittedModels.load(out)
    assert old.reference_rates is None
    rep = old.shadow_engine(n_samples=100).evaluate_loan(row).to_dict()
    assert rep["eligibility"]["reference_rate"] is None and rep["eligibility"]["rate_corridor_applied"] is False


def test_interest_realization_is_fitted_saved_and_optional(tmp_path):
    cfg, df, _ = _data()
    fitted = fit_models(df, cfg, "2016-01-01", evaluate=False)
    tab = fitted.interest_realization
    assert tab is not None and 0.3 <= tab.factor("B", 36) <= 1.0 and tab.n_fit > 0
    m = fitted.metrics["interest_realization"]
    assert m["prepayment_adjustment"] == "empirical" and m["by_term"] and m["n_fit"] == tab.n_fit
    g = sorted(df["y_group"].unique())[0]
    a60 = fitted.arm_space.nearest(10_000, 0.13, 60)
    assert fitted.realization(g, a60) == tab.factor(g.split("|")[0], 60)
    assert fitted.realization(g, term=60) == fitted.realization(g, a60) and fitted.realization(g) == 1.0
    out = fitted.save(tmp_path / "fit")
    assert (out / "interest_realization.json").is_file()
    loaded = FittedModels.load(out)
    assert loaded.interest_realization == tab and loaded.realization(g, a60) == fitted.realization(g, a60)
    (out / "interest_realization.json").unlink()  # a fit saved before prepayment existed
    old = FittedModels.load(out)
    assert old.interest_realization is None and old.realization(g, a60) == 1.0
    assert old.shadow_engine(n_samples=50).realization(g, a60) == 1.0
