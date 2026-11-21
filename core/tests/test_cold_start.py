"""Cold start (``SimulationConfig.cold_start``): the policies run without an offline-trained
model, in the same replay world as a warm run (docs/SIMULATION.md §11)."""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, SimulationConfig
from bayesian_credit.pipeline import fit_models, prepare_dataset
from bayesian_credit.simulation import RunDirectory, SimulationRunner
from bayesian_credit.simulation.store import config_hash

POLICIES = ("thompson", "greedy", "ucb1", "historical")


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, _ = prepare_dataset("synthetic", None, cfg=cfg, n=5000, seed=3)
    fitted = fit_models(df, cfg, "2016-01-01", evaluate=False)
    return cfg, df, fitted


def sim_cfg(**kw) -> SimulationConfig:
    base = {
        "start": "2016-01-01",
        "end": "2016-08-31",
        "max_loans": 1500,
        "seed": 5,
        "policies": POLICIES,
        "interim_months": 3,
        "checkpoint_every": 3,
        "breaker_mode": "off",
    }
    base.update(kw)
    return SimulationConfig(**base)


def make_runner(path, sim: SimulationConfig, *, with_fit: bool = True, **kw) -> SimulationRunner:
    cfg, df, fitted = _data()
    kw.setdefault("install_signal_handlers", False)
    kw.setdefault("poll_interval", 0.02)
    return SimulationRunner(df, cfg, sim, path, fitted=fitted if with_fit else None, **kw)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cold")
    warm = make_runner(root / "warm", sim_cfg())
    cold = make_runner(root / "cold", sim_cfg(cold_start=True))
    return {
        "warm": (warm, warm.run(), root / "warm"),
        "cold": (cold, cold.run(), root / "cold"),
    }


def test_cold_policies_never_see_the_risk_model(runs):
    runner, res, _ = runs["cold"]
    assert res.status == "completed"
    for gov in runner.governed.values():
        assert gov.policy.prior_provider is None
        assert gov.policy.cfg.warm_start == "flat"
    warm_runner = runs["warm"][0]
    assert all(g.policy.prior_provider is not None for g in warm_runner.governed.values())


def test_cold_start_is_recorded_in_the_run(runs):
    _, _, path = runs["cold"]
    rd = RunDirectory(path)
    assert rd.read_config()["simulation"]["cold_start"] is True
    assert rd.events.read("RunStarted")[0]["payload"]["cold_start"] is True


def test_cold_greedy_never_explores_so_it_never_lends(runs):
    """Flat prior: every arm has μ = 0, ties break to the lowest index (decline), and the
    decline arm's reward is exactly 0 - pure exploitation never leaves it."""
    runner, _, path = runs["cold"]
    dec = RunDirectory(path).read_decisions()
    null = runner.arm_space.null_arm
    greedy = dec.loc[dec["policy"] == "greedy", "arm"]
    assert (greedy == null).all()
    warm = RunDirectory(runs["warm"][2]).read_decisions()
    assert (warm.loc[warm["policy"] == "greedy", "arm"] != null).any()


def test_exploring_policies_learn_to_lend_from_scratch(runs):
    _, _, path = runs["cold"]
    dec = RunDirectory(path).read_decisions()
    null = runs["cold"][0].arm_space.null_arm
    for name in ("thompson", "ucb1"):
        assert (dec.loc[dec["policy"] == name, "arm"] != null).mean() > 0.2, name


def test_cold_and_warm_runs_share_the_replay_world(runs):
    warm, cold = runs["warm"][0], runs["cold"][0]
    np.testing.assert_array_equal(warm.oracle, cold.oracle)
    np.testing.assert_array_equal(warm.elig_hist, cold.elig_hist)
    wd = RunDirectory(runs["warm"][2]).read_decisions()
    cd = RunDirectory(runs["cold"][2]).read_decisions()
    # the reference policy decides and is scored identically (its simulated-reward draws
    # come from the environment's shared per-epoch stream, so they depend on the others)
    cols = ["epoch", "loan_id", "group", "arm", "expected_reward", "oracle_reward", "realized_reward"]
    hw = wd.loc[wd["policy"] == "historical", cols].reset_index(drop=True)
    hc = cd.loc[cd["policy"] == "historical", cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(hw, hc)


def test_without_a_fit_the_world_reference_comes_from_before_the_stream(tmp_path):
    runner = make_runner(tmp_path / "nofit", sim_cfg(cold_start=True, end="2016-03-31"), with_fit=False)
    assert runner.fitted.train_end == "2016-01-01"
    assert (RunDirectory(tmp_path / "nofit").path / "fit" / "two_stage.json").is_file()
    assert runner.run().status == "completed"


class _Kill(RuntimeError):
    pass


def test_cold_kill_and_resume_reproduces_the_uninterrupted_run(tmp_path, runs):
    path = tmp_path / "killed"

    def killer(info):
        if info["epoch_index"] == 4:
            raise _Kill("simulated crash")

    with pytest.raises(_Kill):
        make_runner(path, sim_cfg(cold_start=True), progress_callback=killer).run()
    res = SimulationRunner.resume(path, install_signal_handlers=False)
    assert res.status == "completed"
    a, b = RunDirectory(runs["cold"][2]), RunDirectory(path)
    pd.testing.assert_frame_equal(a.read_decisions(), b.read_decisions(), check_exact=True)
    pd.testing.assert_frame_equal(a.read_timeseries(), b.read_timeseries(), check_exact=True)


def test_runs_from_before_cold_start_keep_their_config_hash():
    sim = sim_cfg().to_dict()
    legacy = {k: v for k, v in sim.items() if k != "cold_start"}
    assert config_hash({}, sim) == config_hash({}, legacy)
    assert config_hash({}, {**sim, "cold_start": True}) != config_hash({}, legacy)
